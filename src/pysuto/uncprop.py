"""
uncprop.py
==========
Stage 5 — Posterior uncertainty propagation.

Loads a persisted Stage-4 :class:`OptimResult` set plus its upstream
:class:`ConstraintRun`, then computes the marginal posterior standard
deviation σ_{a*}(t) and the (1-α) Gaussian confidence interval for every
edge × timestep, using the closed-form GLS expression

    σ²_{a*,i}(t) = σ²_{a,i} · (1 - σ²_{a,i} · ‖L(t)⁻¹ g_i‖²)

with ``S(t) = G · diag(σ²_a) · Gᵀ + diag(σ²_c(t)) = L(t) · L(t)ᵀ`` and
``g_i = col_i(G)``.

On the block path (``G_blocks`` present, every current run) ``S(t)`` is never
formed: ``σ²_{a,i} · g_iᵀ S⁻¹ g_i`` is the leverage of edge i in the whitened
system ``M = [G_t · diag(σ_a), diag(σ_c)]`` (``S = M Mᵀ``), read off a dense QR
of ``Mᵀ`` (:meth:`KalmanUncProp._leverage`), so it lies in [0, 1]. Forming
``S`` squares ``M``'s condition number; with σ_a spanning 1e-8 … 1e6 kt on
copper, an LU of ``S`` put the contraction outside [0, 1] on 331 edges of 1995
and 368 of 2016 in ``uncprop_20261014_140949_fcf53674`` (#27). Only the full-G
path factors ``S``: it analyses the symbolic Cholesky pattern once (G is
time-invariant there) and refreshes the numeric factor per timestep via
CHOLMOD's ``cholesky_inplace`` because σ_c is time-variant. A scipy ``splu``
fallback kicks in when scikit-sparse is not installed; it computes
``gᵀ S⁻¹ g`` directly (same value, no symbolic-pattern reuse, slower).

σ_a and a₀ are pulled through :func:`_conditioning.condition_prior`
with the same floor / ceiling rules the solver applied, so the posterior
is mathematically consistent with the ``a*`` the solver returned.
σ_c on hard rows is forced to exact zero; soft rows receive a numerical
floor only.

On the block path the posterior is the QP's linearisation at its active set
(#27): a hard inequality row (``nonneg`` cone) enters ``S(t)`` only where it
binds at ``a*`` (``G·a* − c ≥ −bind_tol``); a binding ADR 0039 lower bound
pins its edge (``σ_a² → 0``, ``σ_a* = 0``) instead of adding a row; a binding
band is an equality; slack rows drop out. Rows left without a live edge and
hard rows linearly dependent on the others (pivoted Cholesky of the scaled
hard block, ``rank_tol``) are dropped, so ``S(t)`` stays non-singular with
hard rows at exactly zero σ_c. Intervals clip at the structural zero only
(edges with a lower-bound row); ``l`` / ``u`` of the ConstraintRun carry the
prior box and the zero pin, which never reached the QP.

``exclude_scale_totals`` (#87, default off) leaves the scale totals and scrap
level anchors (:data:`SCALE_TOTAL_VARIANTS`) out of ``S(t)``: an ablation that
measures how much of the contraction comes from the model's own Σ f_agg totals,
whose information the prior already carries. ``a*`` is untouched.

Aggregates (#81, block path, in memory only). The marginals do not give the σ
of a total ``wᵀa*``: the constraints correlate the edges, so adding their σ_a*
in quadrature misstates it. With ``u = diag(σ_a)·w`` on the live edges and
``v = [u; 0]`` padded over ``Mᵀ``'s soft rows, ``var(wᵀa*) = wᵀΣ_a w −
(GΣ_a w)ᵀ S⁻¹ (GΣ_a w) = ‖u‖² − ‖Qᵀv‖²``, read off the same QR as κ.
``compute(..., aggregates=AggregateSpec)`` returns the K×K prior and posterior
covariance of K such totals; the years are separate QPs, so their contributions
add and totals of different years never covary. ``draws`` returns posterior
deviations ``diag(σ_a)·(I − QQᵀ)ξ`` (edge part, ξ ~ N(0, I) from the caller's
generator): they satisfy every kept hard row and pin exactly, and a smooth
function of a* gets its delta-method σ from them by finite differences.
``force_pinned`` / ``force_free`` override the slack rule edge by edge, for the
degenerate bounds #99 found sitting at ``bind_tol``.

See ``docs/rep/plans/stage5_uncprop_integration_plan.md`` for the full
derivation and architectural decisions.

Public API
----------
- :class:`BaseUncProp` — ABC with single :meth:`compute` method
- :class:`KalmanUncProp` — closed-form GLS concrete subclass
- :class:`UncPropResult` / :class:`UncPropTimestep` — return dataclasses
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr
from scipy.linalg.lapack import dpstrf
from scipy.stats import norm

from ._conditioning import condition_prior, condition_sigma_c
from .logging import get_logger
from .registry import ConstraintRunRegistry, OptimRunRegistry, UncPropRunRegistry
from .registry.optim import warn_if_prefix_block_run

logger = get_logger("pysuto.uncprop")


# Defaults mirror ClarabelSolver so uncprop sees the same effective prior
# the QP saw. Keep these in sync with src/pysuto/solver.py if either side
# changes. (The shared helper enforces the rules; only the *defaults*
# need to stay aligned.)
DEFAULT_SIGMA_A_FLOOR_ABS      = 1e-8
DEFAULT_SIGMA_A_FLOOR_REL      = 1e-5
DEFAULT_SIGMA_A_CEILING_FACTOR = 1e5
DEFAULT_SIGMA_C_FLOOR          = 1e-9

# Active set (#27). A hard inequality row binds when G·a* − c ≥ −bind_tol (kt,
# 1 kg). Clarabel's a* has no clean gap there (thousands of bound edges sit in
# [1e-7, 1e-6) kt), so a measure reports the pinned set's sensitivity to it.
# rank_tol is the pivot tolerance of the unit-diagonal Gram of the hard rows.
DEFAULT_BIND_TOL               = 1e-6
DEFAULT_RANK_TOL               = 1e-10

# Scale-total ablation (#87). Sum rows whose c is the model's own Σ f_agg: the
# supply / use total-output scale totals (ADR 0006) and the ICSG scrap level
# anchors (ADR 0031), read from row_meta ``type`` / ``variant``. Their prior
# information is already in σ_a, so ``exclude_scale_totals`` leaves them out of
# S(t) to measure how much of the contraction they add.
DEFAULT_EXCLUDE_SCALE_TOTALS   = False
SCALE_TOTAL_VARIANTS           = ("supplytotout", "usetotout", "smelterscrap",
                                  "directrefscrap")


# =============================================================================
# Return dataclasses
# =============================================================================

@dataclass
class UncPropTimestep:
    """Per-timestep posterior summary.

    Block path is year-local: sigma_a_star/lower/upper are the cols_t slice
    and ``cols_t`` carries the global edge index. Legacy path is full-length
    (E,) with ``cols_t = None``."""
    timestep:     str
    sigma_a_star: np.ndarray            # posterior std (cols_t or E,)
    lower:        np.ndarray            # a* − z·σ; block path: clipped at the bound's l
    upper:        np.ndarray            # a* + z·σ; legacy path: clipped to u
    diagnostics:  dict                  # trace_prior / trace_post / max_resid_hard / psd_min_diag
    cols_t:       Optional[np.ndarray] = None  # global edge cols (block path)
    bound_active: Optional[np.ndarray] = None  # pinned by a binding lower bound (block path)


@dataclass
class _PeriodReduction:
    """One period reduced at the QP's active set
    (:meth:`KalmanUncProp._reduce_at_active_set`)."""
    keep:         np.ndarray            # bool (m_t,) — the rows of S_t
    bound_active: np.ndarray            # bool (n_t,) — lower bound binds: σ_a² → 0
    lower_bound:  np.ndarray            # (n_t,) — the bound row's l, NaN without one
    counts:       dict                  # JSON-safe diagnostics


@dataclass
class AggregateSpec:
    """K named linear totals ``wᵀa`` of the edges (#81): ``W`` is (n_edges × K),
    one column per name, over the global edge index; a column may span years."""
    names: list
    W:     sp.spmatrix


@dataclass
class AggregateMoments:
    """The K totals of an :class:`AggregateSpec` at ``a*``, summed over years.
    Computed in memory; :meth:`KalmanUncProp.persist` does not store them."""
    names:            list
    value:            np.ndarray        # Wᵀa*
    cov_prior:        np.ndarray        # Wᵀ·diag(σ_a²)·W, σ_a conditioned
    cov_post:         np.ndarray        # the posterior covariance of Wᵀa*
    sigma_quadrature: np.ndarray        # √((W∘W)ᵀσ_a*²): the edges taken as independent


@dataclass
class UncPropResult:
    """Container returned by :meth:`BaseUncProp.compute`."""
    method:        str
    alpha:         float
    per_timestep:  dict[str, UncPropTimestep] = field(default_factory=dict)
    aggregates:    Optional[AggregateMoments] = None   # #81, block path
    draws:         dict = field(default_factory=dict)   # year → (k, n_t) posterior deviations


# =============================================================================
# ABC
# =============================================================================

class BaseUncProp(ABC):
    """Abstract base for posterior-uncertainty concretes.

    Every concrete owns its own algorithm (Cholesky / KKT-sensitivity /
    Monte Carlo / bounds-aware) but produces the same data contract:
    marginal σ and confidence bounds per (edge, timestep).
    """

    METHOD: str = ""

    @abstractmethod
    def compute(self, opt_run_id: str, alpha: float = 0.05) -> UncPropResult:
        """Run posterior uncertainty propagation for one optimisation run."""
        raise NotImplementedError


# =============================================================================
# Kalman / closed-form GLS concrete
# =============================================================================

class KalmanUncProp(BaseUncProp):
    """Closed-form posterior covariance: a QR of the whitened system per year on
    the block path, sparse Cholesky on the full-G (legacy) path.

    Per timestep:

      1. Pull σ_c(t) from the upstream :class:`ConstraintRun`.
      2. Build ``S(t) = G · diag(σ²_a) · Gᵀ + diag(σ²_c(t))``.
      3. Refresh numeric Cholesky (symbolic pattern cached from t₀).
      4. For each column block ``G[:, s:e]``: solve ``L⁻¹ G_blk``;
         accumulate ``‖·‖²`` column-wise to obtain ``d[s:e]``.
      5. ``σ²_{a*}(t) = σ²_a · (1 − σ²_a · d)``, clipped to ≥ 0.
      6. ``lower / upper = a*(t) ± z_{1-α/2} · σ_{a*}(t)``, clipped to
         ``[l, u]``.
      7. Diagnostics: ``trace_prior``, ``trace_post``, ``max_resid_hard``,
         ``psd_min_diag``.

    The block path replaces steps 2–5 by ``σ²_{a*} = σ²_a · (1 − κ)``, with the
    contraction ``κ`` read off a QR of the whitened system (:meth:`_leverage`).

    σ_a is conditioned with the same rules the solver used (shared
    helper in :mod:`pysuto._conditioning`). σ_c on hard rows is forced
    to exact zero so hard equalities contribute maximal variance
    reduction; soft rows receive a numerical floor only.
    """

    METHOD = "kalman"

    def __init__(
        self,
        opt_dir:     str = "data/proc/opt",
        const_dir:   str = "data/proc/const",
        uncprop_dir: str = "data/proc/uncprop",
        block_size:  int = 1024,
        sigma_a_floor_abs:      float = DEFAULT_SIGMA_A_FLOOR_ABS,
        sigma_a_floor_rel:      float = DEFAULT_SIGMA_A_FLOOR_REL,
        sigma_a_ceiling_factor: float = DEFAULT_SIGMA_A_CEILING_FACTOR,
        sigma_c_floor:          float = DEFAULT_SIGMA_C_FLOOR,
        bind_tol:               float = DEFAULT_BIND_TOL,
        rank_tol:               float = DEFAULT_RANK_TOL,
        exclude_scale_totals:   bool  = DEFAULT_EXCLUDE_SCALE_TOTALS,
    ) -> None:
        self.opt_dir      = opt_dir
        self.const_dir    = const_dir
        self.uncprop_dir  = uncprop_dir
        self.block_size   = int(block_size)
        self.sigma_a_floor_abs      = sigma_a_floor_abs
        self.sigma_a_floor_rel      = sigma_a_floor_rel
        self.sigma_a_ceiling_factor = sigma_a_ceiling_factor
        self.sigma_c_floor          = sigma_c_floor
        self.bind_tol               = bind_tol
        self.rank_tol               = rank_tol
        self.exclude_scale_totals   = bool(exclude_scale_totals)

        self._opt_registry     = OptimRunRegistry(opt_dir)
        self._const_registry   = ConstraintRunRegistry(const_dir)
        self._uncprop_registry = UncPropRunRegistry(uncprop_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        opt_run_id:   str,
        alpha:        float = 0.05,
        *,
        aggregates:   Optional[AggregateSpec] = None,
        draws:        Optional[Mapping[str, int]] = None,
        rng:          Optional[np.random.Generator] = None,
        force_pinned: Optional[np.ndarray] = None,
        force_free:   Optional[np.ndarray] = None,
    ) -> UncPropResult:
        """Posterior σ and interval of every edge × year (block path: at the QP's
        active set).

        Block path only (#81): ``aggregates`` adds the covariance of K linear
        totals (:class:`AggregateMoments`); ``draws`` maps a year to a number of
        posterior deviation draws over its ``cols_t``, taken with ``rng``;
        ``force_pinned`` / ``force_free`` are boolean masks over every edge that
        pin, or free, an edge whatever its lower bound's slack."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if draws and rng is None:
            raise ValueError("draws need a generator: pass rng=np.random.default_rng(seed)")

        # 1) Cross-stage hop: opt → const via opt's sidecar.
        opt_sidecar = self._opt_registry.load_sidecar(opt_run_id)
        const_run_id = (opt_sidecar.get("parents", {})
                                    .get("upstream", {})
                                    .get("const_run_id"))
        if not const_run_id:
            raise KeyError(
                f"OptimRun '{opt_run_id}' sidecar has no "
                f"parents.upstream.const_run_id — uncprop cannot proceed."
            )

        logger.info("Loading OptimRun '%s' (→ ConstraintRun '%s')",
                    opt_run_id, const_run_id)
        opt_results = self._opt_registry.load(opt_run_id)
        const_run   = self._const_registry.load(const_run_id)
        warn_if_prefix_block_run(opt_results, const_run)

        n_edges = const_run.G.shape[1]
        if aggregates is not None and aggregates.W.shape != (n_edges, len(aggregates.names)):
            raise ValueError(
                f"aggregates.W is {aggregates.W.shape}; it needs one row per edge and one "
                f"column per name: ({n_edges}, {len(aggregates.names)})"
            )
        for name, mask in (("force_pinned", force_pinned), ("force_free", force_free)):
            if mask is not None and np.shape(mask) != (n_edges,):
                raise ValueError(f"{name} has shape {np.shape(mask)}; it needs one entry per "
                                 f"edge: ({n_edges},)")

        if const_run.G_blocks:
            return self._compute_via_blocks(
                const_run, opt_results, alpha, aggregates=aggregates, draws=draws or {},
                rng=rng, force_pinned=force_pinned, force_free=force_free,
            )
        if (aggregates is not None or draws or force_pinned is not None
                or force_free is not None):
            raise ValueError("aggregates, draws and the force masks need the block path (a "
                             "ConstraintRun with G_blocks); the full-G path has no active set")
        if self.exclude_scale_totals:
            raise ValueError("exclude_scale_totals needs the block path (a ConstraintRun "
                             "with G_blocks); the full-G path has no active set")

        # 2) Condition σ_a once (time-invariant; same rules as solver).
        _, sigma_a = condition_prior(
            const_run.a0, const_run.sigma_a,
            floor_abs      = self.sigma_a_floor_abs,
            floor_rel      = self.sigma_a_floor_rel,
            ceiling_factor = self.sigma_a_ceiling_factor,
        )
        sigma_a2 = sigma_a ** 2

        G = const_run.G.tocsc()
        hard_mask = const_run.row_meta["is_hard"].to_numpy().astype(bool)
        if (hard_mask & (const_run.row_meta["cone_type"].to_numpy() == "nonneg")).any():
            logger.warning(
                "uncprop: the full-G path treats hard inequality rows as binding "
                "equalities (no active set, #27) — rebuild the ConstraintRun with a "
                "per-period partition to get the active-set posterior."
            )
        # ``equality_mask`` is the subset of hard rows that are exact
        # equalities (zero cone). Nonneg-cone hard rows are inequalities
        # and a non-binding nonneg row has a negative — *feasible* —
        # ``G·a − c``; only zero-cone rows are valid for an exact-residual
        # diagnostic.
        equality_mask = (const_run.row_meta["cone_type"].to_numpy() == "zero")

        # 3) Cholesky backend handle. Symbolic factor is analyzed from
        #    S at t₀; per-t refreshes reuse the pattern (CHOLMOD path).
        handle = self._init_factor(G, sigma_a2, const_run, hard_mask)

        z = float(norm.ppf(1.0 - alpha / 2.0))

        # 4) Per-timestep loop.
        per_timestep: dict[str, UncPropTimestep] = {}
        for t_label, opt_t in opt_results.items():
            t_start = time.perf_counter()

            sigma_c2_t = self._build_sigma_c2(const_run, t_label, hard_mask)
            self._refresh_factor(handle, G, sigma_a2, sigma_c2_t)

            d = self._marginal_diagonal(handle, G, n_edges)
            var_post = sigma_a2 * (1.0 - sigma_a2 * d)
            np.clip(var_post, 0.0, None, out=var_post)
            sigma_a_star = np.sqrt(var_post)

            half_width = z * sigma_a_star
            lower = np.maximum(opt_t.F_opt - half_width, const_run.l)
            upper = np.minimum(opt_t.F_opt + half_width, const_run.u)

            diagnostics = self._diagnostics(
                sigma_a2      = sigma_a2,
                sigma_a_star  = sigma_a_star,
                G             = G,
                a_star        = opt_t.F_opt,
                c_t           = np.asarray(const_run.timestep_data[t_label]["c"]),
                equality_mask = equality_mask,
            )

            per_timestep[t_label] = UncPropTimestep(
                timestep     = t_label,
                sigma_a_star = sigma_a_star,
                lower        = lower,
                upper        = upper,
                diagnostics  = diagnostics,
            )

            trace_contraction = (1.0 - diagnostics["trace_post"] / diagnostics["trace_prior"]
                                 if diagnostics["trace_prior"] > 0.0 else 0.0)
            logger.info(
                "t=%s  σ_post: min=%.3e med=%.3e max=%.3e  trace_contraction=%.3f  time=%.2fs",
                t_label,
                float(sigma_a_star.min()),
                float(np.median(sigma_a_star)),
                float(sigma_a_star.max()),
                trace_contraction,
                time.perf_counter() - t_start,
            )

        return UncPropResult(
            method       = self.METHOD,
            alpha        = float(alpha),
            per_timestep = per_timestep,
        )

    def _compute_via_blocks(
        self,
        const_run,
        opt_results:  dict,
        alpha:        float,
        *,
        aggregates:   Optional[AggregateSpec] = None,
        draws:        Optional[Mapping[str, int]] = None,
        rng:          Optional[np.random.Generator] = None,
        force_pinned: Optional[np.ndarray] = None,
        force_free:   Optional[np.ndarray] = None,
    ) -> UncPropResult:
        """Per-t posterior using ``run.G_blocks[t]`` — replaces the ``(M × M)``
        global ``S`` with one year's rows per timestep. Mathematically identical: the
        legacy global ``S`` is exactly block-diagonal by time when no
        cross-time constraint coupling exists (verified by the partition
        invariant in :func:`pysuto.constraints.engine._partition_G_by_time`).
        Non-``t`` entries of ``sigma_a_star`` inherit the prior σ_a — same
        as the legacy "deactivated row" outcome where ``σ_c=SIGMA_C_INACTIVE``
        makes d≈0 and the contraction vanishes.

        Each period is first reduced to the QP's active set at ``a*(t)``
        (:meth:`_reduce_at_active_set`, #27), so the whitened system of
        :meth:`_whitened_qr` holds the kept rows and the live (not bound-pinned)
        edges only. Aggregates and draws (#81) are read off the same QR."""
        _, sigma_a = condition_prior(
            const_run.a0, const_run.sigma_a,
            floor_abs      = self.sigma_a_floor_abs,
            floor_rel      = self.sigma_a_floor_rel,
            ceiling_factor = self.sigma_a_ceiling_factor,
        )
        sigma_a2 = sigma_a ** 2

        rm            = const_run.row_meta
        hard_mask     = rm["is_hard"].to_numpy().astype(bool)
        cone          = rm["cone_type"].to_numpy()
        equality_mask = cone == "zero"
        lower_bound   = self._lower_bound_mask(rm)
        left_out      = (self._scale_total_mask(rm) if self.exclude_scale_totals
                         else np.zeros(len(rm), dtype=bool))
        row_time      = rm["time"].astype(str).to_numpy()

        z = float(norm.ppf(1.0 - alpha / 2.0))

        if aggregates is not None:
            W   = sp.csr_matrix(aggregates.W, dtype=np.float64)
            K   = W.shape[1]
            agg = {"value": np.zeros(K), "cov_prior": np.zeros((K, K)),
                   "cov_post": np.zeros((K, K)), "quad2": np.zeros(K)}
        draws = draws or {}
        out_draws: dict[str, np.ndarray] = {}

        per_timestep: dict[str, UncPropTimestep] = {}
        for t_label, opt_t in opt_results.items():
            t_start = time.perf_counter()

            cols_t = const_run.edge_cols_by_t[t_label]
            rows_t = np.where(row_time == t_label)[0]
            self._check_partition(t_label, cols_t, rows_t, opt_t)
            G_t    = const_run.G_blocks[t_label].tocsr()
            c_t    = np.asarray(const_run.timestep_data[t_label]["c"], dtype=float)[rows_t]

            sigma_a2_t = sigma_a2[cols_t]

            raw_sc  = np.asarray(
                const_run.timestep_data[t_label]["sigma_c"], dtype=float,
            )[rows_t]
            hm_t    = hard_mask[rows_t]
            sc_t    = condition_sigma_c(
                raw_sc, floor=self.sigma_c_floor, hard_mask=hm_t,
            )

            red = self._reduce_at_active_set(
                G_t, c_t, sigma_a2_t, hm_t, cone[rows_t], lower_bound[rows_t],
                np.asarray(opt_t.residuals, dtype=float), left_out=left_out[rows_t],
                force_pinned=None if force_pinned is None else force_pinned[cols_t],
                force_free=None if force_free is None else force_free[cols_t],
            )
            sigma_a2_eff_t = np.where(red.bound_active, 0.0, sigma_a2_t)
            live = np.flatnonzero(sigma_a2_eff_t > 0.0)
            G_k  = G_t[red.keep]
            sigma_live = np.sqrt(sigma_a2_eff_t[live])

            kappa_t = np.zeros(cols_t.size, dtype=np.float64)
            qr_t = None
            if G_k.shape[0] and live.size:
                qr_t = self._whitened_qr(G_k[:, live], sigma_live, sc_t[red.keep])
                kappa_t[live] = self._leverage(*qr_t)[:live.size]

            var_post_t = sigma_a2_eff_t * (1.0 - kappa_t)
            np.clip(var_post_t, 0.0, None, out=var_post_t)
            sigma_a_star_t = np.sqrt(var_post_t)

            if aggregates is not None:
                W_t = W[cols_t]
                nz  = np.unique(W_t.indices)              # the totals this year touches
                W_t = W_t[:, nz].toarray()
                U   = sigma_live[:, None] * W_t[live]
                P   = self._project(qr_t, U)
                blk = np.ix_(nz, nz)
                agg["value"][nz]     += W_t.T @ np.asarray(opt_t.F_opt, dtype=float)
                agg["cov_prior"][blk] += W_t.T @ (sigma_a2_t[:, None] * W_t)
                agg["cov_post"][blk]  += U.T @ U - P.T @ P
                agg["quad2"][nz]      += (W_t ** 2).T @ var_post_t
            if t_label in draws:
                xi = rng.standard_normal((self._n_rows(qr_t, live.size), int(draws[t_label])))
                if qr_t is not None:
                    Q, order, _ = qr_t
                    xi[order] -= Q @ (Q.T @ xi[order])      # (I − QQᵀ)ξ
                d_t = np.zeros((xi.shape[1], cols_t.size))
                d_t[:, live] = (sigma_live[:, None] * xi[:live.size]).T
                out_draws[t_label] = d_t

            # Year-local output (REP §8.2): store only this period's cols_t
            # posterior. F_opt is already the year-local slice from the solver.
            # Clip at the bound row's own l only: ConstraintRun.l / u carry the
            # prior box and the zero pin, which reach the feasibility probe, not
            # the QP. A pinned edge within bind_tol below l keeps lower ≤ a*.
            half_width_t = z * sigma_a_star_t
            lower_t = opt_t.F_opt - half_width_t
            has_lb  = ~np.isnan(red.lower_bound)
            lower_t[has_lb] = np.minimum(
                np.maximum(lower_t[has_lb], red.lower_bound[has_lb]), opt_t.F_opt[has_lb],
            )
            upper_t = opt_t.F_opt + half_width_t

            # Per-t diagnostics: hard-row residual on the t-block only.
            em_t = equality_mask[rows_t]
            if em_t.any():
                resid_t = G_t @ opt_t.F_opt - c_t
                max_resid_hard = float(np.abs(resid_t[em_t]).max())
            else:
                max_resid_hard = 0.0
            # Traces over this year's own edges (cols_t): adding the other
            # years' untouched σ² would dilute the trace contraction ~n_t-fold.
            diagnostics = {
                "trace_prior":    float(sigma_a2_t.sum()),
                "trace_post":     float(np.square(sigma_a_star_t).sum()),
                "max_resid_hard": max_resid_hard,
                "psd_min_diag":   float(sigma_a_star_t.min()),
                **red.counts,
            }

            per_timestep[t_label] = UncPropTimestep(
                timestep     = t_label,
                sigma_a_star = sigma_a_star_t,
                lower        = lower_t,
                upper        = upper_t,
                diagnostics  = diagnostics,
                cols_t       = cols_t,
                bound_active = red.bound_active,
            )

            trace_contraction = (1.0 - diagnostics["trace_post"] / diagnostics["trace_prior"]
                                 if diagnostics["trace_prior"] > 0.0 else 0.0)
            logger.info(
                "t=%s  σ_post: min=%.3e med=%.3e max=%.3e  trace_contraction=%.3f  "
                "time=%.2fs  (per-t: %d live of %d cols × %d of %d rows; "
                "%d bound-active, %d dependent hard rows dropped)",
                t_label,
                float(sigma_a_star_t.min()),
                float(np.median(sigma_a_star_t)),
                float(sigma_a_star_t.max()),
                trace_contraction,
                time.perf_counter() - t_start,
                live.size, G_t.shape[1], G_k.shape[0], G_t.shape[0],
                red.counts["n_bound_active"], red.counts["n_hard_dependent"],
            )

        moments = None
        if aggregates is not None:
            moments = AggregateMoments(
                names            = list(aggregates.names),
                value            = agg["value"],
                cov_prior        = agg["cov_prior"],
                cov_post         = agg["cov_post"],
                sigma_quadrature = np.sqrt(agg["quad2"]),
            )
        return UncPropResult(
            method       = self.METHOD,
            alpha        = float(alpha),
            per_timestep = per_timestep,
            aggregates   = moments,
            draws        = out_draws,
        )

    # ------------------------------------------------------------------
    # Active set (#27)
    # ------------------------------------------------------------------

    @staticmethod
    def _lower_bound_mask(row_meta) -> np.ndarray:
        """Rows that are ADR 0039 lower bounds (read from ``type`` / ``variant``)."""
        if not {"type", "variant"} <= set(row_meta.columns):
            return np.zeros(len(row_meta), dtype=bool)
        is_lower = (row_meta["type"] == "bound") & (row_meta["variant"] == "lower")
        return is_lower.to_numpy()

    @staticmethod
    def _scale_total_mask(row_meta) -> np.ndarray:
        """Rows that are scale totals or scrap level anchors
        (:data:`SCALE_TOTAL_VARIANTS`, read from ``type`` / ``variant``). A
        ``row_meta`` without those columns raises: the ablation would be a no-op."""
        missing = sorted({"type", "variant"} - set(row_meta.columns))
        if missing:
            raise ValueError(f"exclude_scale_totals reads row_meta columns {missing}, "
                             "which this ConstraintRun lacks")
        is_scale = ((row_meta["type"] == "sum")
                    & row_meta["variant"].isin(SCALE_TOTAL_VARIANTS))
        return is_scale.to_numpy()

    @staticmethod
    def _lower_bound_values(G_t: sp.csr_matrix, c_t: np.ndarray,
                            lower_bound: np.ndarray) -> np.ndarray:
        """Each column's lower-bound row ``l`` (NaN without one). An ADR 0039 row
        reads ``−a ≤ −l``: one nonzero, coefficient −1, so ``l = −c``."""
        bound_rows = np.flatnonzero(lower_bound)
        bound_cols = G_t[bound_rows].tocoo()             # one nonzero per bound row
        lower = np.full(G_t.shape[1], np.nan)
        lower[bound_cols.col] = -c_t[bound_rows[bound_cols.row]]   # −a ≤ −l
        return lower

    @staticmethod
    def _check_partition(t_label: str, cols_t: np.ndarray, rows_t: np.ndarray,
                         opt_t) -> None:
        """The opt result must be year-local over this period's own partition:
        its ``residuals`` are read row by row against ``rows_t``."""
        if opt_t.cols_t is None or opt_t.rows_t is None:
            raise ValueError(
                f"t={t_label}: the opt result has no cols_t / rows_t but the "
                "ConstraintRun is partitioned — re-solve on the block path"
            )
        if not (np.array_equal(np.asarray(opt_t.cols_t), np.asarray(cols_t))
                and np.array_equal(np.asarray(opt_t.rows_t), np.asarray(rows_t))):
            raise ValueError(
                f"t={t_label}: the opt result's cols_t / rows_t differ from the "
                "ConstraintRun's partition — the two runs do not belong together"
            )

    def _reduce_at_active_set(
        self,
        G_t:         sp.csr_matrix,
        c_t:         np.ndarray,
        sigma_a2_t:  np.ndarray,
        hard:        np.ndarray,
        cone:        np.ndarray,
        lower_bound: np.ndarray,
        resid:       np.ndarray,
        left_out:    np.ndarray,
        force_pinned: Optional[np.ndarray] = None,
        force_free:   Optional[np.ndarray] = None,
    ) -> _PeriodReduction:
        """Reduce one period at the QP's active set at ``a*``.

        Kept rows: hard equalities, soft rows and binding bands, minus rows left
        without a live edge and hard rows linearly dependent on the others. A
        binding lower bound adds no row: it pins its edge (σ_a² → 0, σ_a* = 0).
        Rows flagged in ``left_out`` (the scale-total ablation; all ``False``
        otherwise) are dropped too; ``n_scale_totals_left_out`` counts those that
        would have been kept. ``force_pinned`` / ``force_free`` (per column, #81)
        pin or free an edge whatever its bound's slack; ``n_forced_pinned`` /
        ``n_forced_free`` count the edges they move.
        """
        G_t     = G_t.tocsr()
        nonneg  = hard & (cone == "nonneg")
        binding = nonneg & (resid >= -self.bind_tol)

        lower = self._lower_bound_values(G_t, c_t, lower_bound)
        bound_active = np.zeros(G_t.shape[1], dtype=bool)
        bound_active[G_t[np.flatnonzero(binding & lower_bound)].tocoo().col] = True
        n_pinned = n_freed = 0
        if force_pinned is not None:
            n_pinned = int((force_pinned & ~bound_active).sum())
            bound_active |= force_pinned
        if force_free is not None:
            n_freed = int((force_free & bound_active).sum())
            bound_active &= ~force_free
        live = ~bound_active & (sigma_a2_t > 0.0)

        keep = (hard & (cone == "zero")) | ~hard | (binding & ~lower_bound)
        keep &= np.asarray(abs(G_t) @ live.astype(float)).ravel() > 0.0
        n_left_out = int((keep & left_out).sum())
        keep &= ~left_out
        dependent = self._dependent_hard_rows(G_t, live, np.flatnonzero(keep & hard))
        keep[dependent] = False

        return _PeriodReduction(
            keep         = keep,
            bound_active = bound_active,
            lower_bound  = lower,
            counts       = {
                "n_bound_active":          int(bound_active.sum()),
                "n_bands_binding":         int((binding & ~lower_bound).sum()),
                "n_hard_dependent":        int(dependent.size),
                "n_rows_kept":             int(keep.sum()),
                "n_scale_totals_left_out": n_left_out,
                "n_forced_pinned":         n_pinned,
                "n_forced_free":           n_freed,
            },
        )

    def _dependent_hard_rows(self, G_t: sp.csr_matrix, live: np.ndarray,
                             hard_rows: np.ndarray) -> np.ndarray:
        """Hard rows linearly dependent on the others over the live edges (closed
        balance loops, a band's two rows at 0 = 0): the rows past the numerical
        rank of a pivoted Cholesky of the unit-diagonal Gram ``G_h·G_hᵀ``.

        The Gram is unweighted on purpose: weighted by σ_a², rows that differ
        only on low-σ edges look parallel and an exact pin would be lost. The
        dense Gram is m_h² floats (≈ 280 MB for copper's ≈ 5,900 hard rows).
        Dropping the dependent rows keeps ``S_t`` non-singular with every kept
        hard row at exactly zero σ_c."""
        if hard_rows.size < 2:
            return np.zeros(0, dtype=np.int64)
        G_h  = G_t[hard_rows][:, np.flatnonzero(live)]
        gram = (G_h @ G_h.T).toarray()
        scale = 1.0 / np.sqrt(np.diag(gram))
        _, piv, rank, info = dpstrf(gram * scale[:, None] * scale[None, :],
                                    lower=1, tol=self.rank_tol)
        if info < 0:
            raise ValueError(f"dpstrf: argument {-info} is invalid")
        return np.sort(hard_rows[piv[rank:] - 1])

    @staticmethod
    def _whitened_qr(G_k: sp.csr_matrix, sigma_a: np.ndarray,
                     sigma_c: np.ndarray) -> tuple:
        """``(Q, order, n_rows)`` of the whitened system ``Mᵀ = Q R``, with
        ``M = [G_k · diag(σ_a), diag(σ_c)]`` so that ``S = M Mᵀ``.

        ``Mᵀ``'s ``n_rows`` rows are the n columns of ``G_k`` followed by one row
        per soft constraint row (σ_c > 0; hard rows add none). Householder QR of
        its nonzero rows sorted by decreasing norm (``order``, indices into those
        ``n_rows``) keeps the projector ``QQᵀ`` onto ``M``'s row space exact to
        rounding; an LU of ``S`` does not, because forming ``S`` squares ``M``'s
        condition number (#27). Rows of edges in no kept row are left out of
        ``order``: their projector rows are zero.
        """
        m, n = G_k.shape
        soft = np.flatnonzero(sigma_c > 0.0)
        Mt = sp.vstack([
            (G_k @ sp.diags(sigma_a)).T,
            sp.csr_matrix((sigma_c[soft], (np.arange(soft.size), soft)),
                          shape=(soft.size, m)),
        ]).tocsr()
        row_norm = np.sqrt(np.asarray(Mt.multiply(Mt).sum(axis=1)).ravel())
        order    = np.argsort(-row_norm, kind="stable")
        order    = order[row_norm[order] > 0.0]
        if order.size < m:
            # A full-row-rank M has ≥ m nonzero rows; an economic Q of a wide
            # Mᵀ is square and would report κ = 1 on every edge.
            raise ValueError(f"_leverage: Mᵀ has {order.size} nonzero rows for {m} "
                             f"kept constraint rows — the kept rows are dependent")
        Q, _ = qr(Mt[order].toarray(order="F"), mode="economic", overwrite_a=True,
                  check_finite=False)
        return Q, order, n + soft.size

    @staticmethod
    def _leverage(Q: np.ndarray, order: np.ndarray, n_rows: int) -> np.ndarray:
        """The projector's diagonal over ``Mᵀ``'s rows; on the first n (the
        columns of ``G_k``) it is ``κ_i = σ²_{a,i} · g_iᵀ S⁻¹ g_i``, in [0, 1]."""
        lev = np.zeros(n_rows, dtype=np.float64)
        lev[order] = np.einsum("ij,ij->i", Q, Q)
        return lev

    @staticmethod
    def _project(qr_t: Optional[tuple], U: np.ndarray) -> np.ndarray:
        """``Qᵀv`` for each column ``u`` of ``U`` (n × K, over ``G_k``'s columns),
        padded to ``v = [u; 0]`` over the soft rows: ``‖Qᵀv‖² = (GΣw)ᵀS⁻¹(GΣw)``
        for ``u = diag(σ_a)·w``. No QR (no kept row) projects to nothing."""
        if qr_t is None:
            return np.zeros((0, U.shape[1]))
        Q, order, _ = qr_t
        V = np.zeros((order.size, U.shape[1]))
        on_edge = order < U.shape[0]
        V[on_edge] = U[order[on_edge]]
        return Q.T @ V

    @staticmethod
    def _n_rows(qr_t: Optional[tuple], n_live: int) -> int:
        """Length of ``Mᵀ``'s row space: the live edges plus the soft rows."""
        return n_live if qr_t is None else qr_t[2]

    def persist(
        self,
        result:     UncPropResult,
        opt_run_id: str,
        label:      str = "",
    ) -> str:
        """Persist *result* via :class:`UncPropRunRegistry`.

        Both ``opt_run_id`` and the (re-discovered) ``const_run_id`` are
        stamped into ``parents.upstream`` so downstream readers don't
        have to re-hop through the opt sidecar.
        """
        opt_sidecar = self._opt_registry.load_sidecar(opt_run_id)
        const_run_id = (opt_sidecar.get("parents", {})
                                    .get("upstream", {})
                                    .get("const_run_id", ""))
        parents = {
            "inputs":   {},
            "upstream": {
                "opt_run_id":   opt_run_id,
                "const_run_id": const_run_id,
            },
        }
        stage_config = {
            "method":                 result.method,
            "alpha":                  result.alpha,
            "block_size":             self.block_size,
            "sigma_a_floor_abs":      self.sigma_a_floor_abs,
            "sigma_a_floor_rel":      self.sigma_a_floor_rel,
            "sigma_a_ceiling_factor": self.sigma_a_ceiling_factor,
            "sigma_c_floor":          self.sigma_c_floor,
            "bind_tol":               self.bind_tol,
            "rank_tol":               self.rank_tol,
            "exclude_scale_totals":   self.exclude_scale_totals,
        }
        run_id = self._uncprop_registry.register(
            result,
            parents      = parents,
            stage_config = stage_config,
            label        = label,
        )
        logger.info("UncPropRun '%s' persisted to %s",
                    run_id, self._uncprop_registry.storage_dir)
        return run_id

    # ------------------------------------------------------------------
    # Backend internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_S(G: sp.csc_matrix, sigma_a2: np.ndarray,
                 sigma_c2_t: np.ndarray) -> sp.csc_matrix:
        """Assemble ``S = G · diag(σ²_a) · Gᵀ + diag(σ²_c)`` as CSC."""
        Da = sp.diags(sigma_a2)
        Dc = sp.diags(sigma_c2_t)
        return (G @ Da @ G.T + Dc).tocsc()

    def _build_sigma_c2(self, const_run, t_label: str,
                        hard_mask: np.ndarray) -> np.ndarray:
        raw = np.asarray(const_run.timestep_data[t_label]["sigma_c"], dtype=float)
        s = condition_sigma_c(raw, floor=self.sigma_c_floor, hard_mask=hard_mask)
        return s ** 2

    def _init_factor(self, G: sp.csc_matrix, sigma_a2: np.ndarray,
                     const_run, hard_mask: np.ndarray) -> dict:
        """Analyze symbolic factor at t₀; return a backend handle."""
        first_t = next(iter(const_run.timestep_data.keys()))
        sigma_c2_0 = self._build_sigma_c2(const_run, first_t, hard_mask)
        S0 = self._build_S(G, sigma_a2, sigma_c2_0)

        try:
            from sksparse.cholmod import cholesky as cholmod_cholesky
            factor = cholmod_cholesky(S0)
            logger.info("uncprop: CHOLMOD symbolic factor analyzed "
                        "(nnz(S)=%d, m=%d)", S0.nnz, S0.shape[0])
            return {"backend": "cholmod", "factor": factor}
        except ImportError:
            logger.warning(
                "uncprop: scikit-sparse not installed; falling back to "
                "scipy.sparse.linalg.splu. No symbolic-pattern reuse — "
                "expect noticeable per-timestep overhead on large G."
            )
            return {"backend": "scipy", "factor": None}

    def _refresh_factor(self, handle: dict, G: sp.csc_matrix,
                        sigma_a2: np.ndarray, sigma_c2_t: np.ndarray) -> None:
        """Refresh numeric factor for new σ_c(t); reuses symbolic pattern."""
        S = self._build_S(G, sigma_a2, sigma_c2_t)
        if handle["backend"] == "cholmod":
            handle["factor"].cholesky_inplace(S)
        else:
            from scipy.sparse.linalg import splu
            handle["factor"] = splu(S.tocsc())

    def _marginal_diagonal(self, handle: dict, G: sp.csc_matrix,
                           n_edges: int) -> np.ndarray:
        """Return ``d_i = ‖L⁻¹ g_i‖² = g_iᵀ S⁻¹ g_i`` for every column g_i."""
        d = np.zeros(n_edges, dtype=np.float64)
        backend = handle["backend"]

        if backend == "cholmod":
            factor = handle["factor"]
            for s in range(0, n_edges, self.block_size):
                e = min(s + self.block_size, n_edges)
                G_blk = G[:, s:e].toarray()                       # m × B
                V = factor.solve_L(G_blk, use_LDLt_decomposition=False)
                d[s:e] = np.einsum("mb,mb->b", V, V)
        else:
            # scipy fallback: g_iᵀ S⁻¹ g_i directly (= ‖L⁻¹ g_i‖² for
            # symmetric PSD S). One full system solve per column block
            # instead of one triangular solve — ~2× the work.
            lu = handle["factor"]
            for s in range(0, n_edges, self.block_size):
                e = min(s + self.block_size, n_edges)
                G_blk = G[:, s:e].toarray()
                X = lu.solve(G_blk)
                d[s:e] = np.einsum("mb,mb->b", G_blk, X)

        return d

    @staticmethod
    def _diagnostics(*, sigma_a2: np.ndarray, sigma_a_star: np.ndarray,
                     G: sp.csc_matrix, a_star: np.ndarray,
                     c_t: np.ndarray, equality_mask: np.ndarray) -> dict:
        trace_prior = float(sigma_a2.sum())
        trace_post  = float(np.square(sigma_a_star).sum())
        if equality_mask.any():
            resid = G @ a_star - c_t
            max_resid_hard = float(np.abs(resid[equality_mask]).max())
        else:
            max_resid_hard = 0.0
        return {
            "trace_prior":    trace_prior,
            "trace_post":     trace_post,
            "max_resid_hard": max_resid_hard,
            "psd_min_diag":   float(sigma_a_star.min()),
        }


