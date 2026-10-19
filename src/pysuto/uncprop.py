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
copper, an LU of ``S`` put the contraction outside [0, 1] on ~330 edges a year
(#27). Only the full-G path factors ``S``: it analyses the symbolic Cholesky
pattern once (G is time-invariant there) and refreshes the numeric factor per
timestep via CHOLMOD's ``cholesky_inplace`` because σ_c is time-variant. A
scipy ``splu`` fallback kicks in when scikit-sparse is not installed; it
computes ``gᵀ S⁻¹ g`` directly (same value, no symbolic-pattern reuse, slower).

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
from typing import Optional

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
class UncPropResult:
    """Container returned by :meth:`BaseUncProp.compute`."""
    method:        str
    alpha:         float
    per_timestep:  dict[str, UncPropTimestep] = field(default_factory=dict)


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
    """Closed-form posterior covariance via sparse Cholesky.

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

        self._opt_registry     = OptimRunRegistry(opt_dir)
        self._const_registry   = ConstraintRunRegistry(const_dir)
        self._uncprop_registry = UncPropRunRegistry(uncprop_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, opt_run_id: str, alpha: float = 0.05) -> UncPropResult:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")

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

        if const_run.G_blocks:
            return self._compute_via_blocks(const_run, opt_results, alpha)

        # 2) Condition σ_a once (time-invariant; same rules as solver).
        _, sigma_a = condition_prior(
            const_run.a0, const_run.sigma_a,
            floor_abs      = self.sigma_a_floor_abs,
            floor_rel      = self.sigma_a_floor_rel,
            ceiling_factor = self.sigma_a_ceiling_factor,
        )
        sigma_a2 = sigma_a ** 2

        G = const_run.G.tocsc()
        n_edges = G.shape[1]
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
        opt_results: dict,
        alpha:       float,
    ) -> UncPropResult:
        """Per-t Schur complement using ``run.G_blocks[t]`` — replaces the
        ``(M × M)`` global ``S`` with one ``(m_t × m_t)`` block per timestep
        (~30× smaller for the copper run). Mathematically identical: the
        legacy global ``S`` is exactly block-diagonal by time when no
        cross-time constraint coupling exists (verified by the partition
        invariant in :func:`pysuto.constraints.engine._partition_G_by_time`).
        Non-``t`` entries of ``sigma_a_star`` inherit the prior σ_a — same
        as the legacy "deactivated row" outcome where ``σ_c=SIGMA_C_INACTIVE``
        makes d≈0 and the contraction vanishes.

        Each period is first reduced to the QP's active set at ``a*(t)``
        (:meth:`_reduce_at_active_set`, #27), so the whitened system of
        :meth:`_leverage` holds the kept rows and the live (not bound-pinned)
        edges only."""
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
        row_time      = rm["time"].astype(str).to_numpy()

        z = float(norm.ppf(1.0 - alpha / 2.0))

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
                np.asarray(opt_t.residuals, dtype=float),
            )
            sigma_a2_eff_t = np.where(red.bound_active, 0.0, sigma_a2_t)
            live = np.flatnonzero(sigma_a2_eff_t > 0.0)
            G_k  = G_t[red.keep].tocsc()

            kappa_t = np.zeros(cols_t.size, dtype=np.float64)
            if G_k.shape[0] and live.size:
                kappa_t[live] = self._leverage(G_k[:, live].tocsr(),
                                               np.sqrt(sigma_a2_eff_t[live]),
                                               sc_t[red.keep])

            var_post_t = sigma_a2_eff_t * (1.0 - kappa_t)
            np.clip(var_post_t, 0.0, None, out=var_post_t)
            sigma_a_star_t = np.sqrt(var_post_t)

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

        return UncPropResult(
            method       = self.METHOD,
            alpha        = float(alpha),
            per_timestep = per_timestep,
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
    ) -> _PeriodReduction:
        """Reduce one period at the QP's active set at ``a*``.

        Kept rows: hard equalities, soft rows and binding bands, minus rows left
        without a live edge and hard rows linearly dependent on the others. A
        binding lower bound adds no row: it pins its edge (σ_a² → 0, σ_a* = 0).
        """
        G_t     = G_t.tocsr()
        nonneg  = hard & (cone == "nonneg")
        binding = nonneg & (resid >= -self.bind_tol)

        bound_rows = np.flatnonzero(lower_bound)
        bound_cols = G_t[bound_rows].tocoo()             # one nonzero per bound row
        lower = np.full(G_t.shape[1], np.nan)
        lower[bound_cols.col] = -c_t[bound_rows[bound_cols.row]]   # −a ≤ −l
        bound_active = np.zeros(G_t.shape[1], dtype=bool)
        bound_active[G_t[np.flatnonzero(binding & lower_bound)].tocoo().col] = True
        live = ~bound_active & (sigma_a2_t > 0.0)

        keep = (hard & (cone == "zero")) | ~hard | (binding & ~lower_bound)
        keep &= np.asarray(abs(G_t) @ live.astype(float)).ravel() > 0.0
        dependent = self._dependent_hard_rows(G_t, live, np.flatnonzero(keep & hard))
        keep[dependent] = False

        return _PeriodReduction(
            keep         = keep,
            bound_active = bound_active,
            lower_bound  = lower,
            counts       = {
                "n_bound_active":   int(bound_active.sum()),
                "n_bands_binding":  int((binding & ~lower_bound).sum()),
                "n_hard_dependent": int(dependent.size),
                "n_rows_kept":      int(keep.sum()),
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
    def _leverage(G_k: sp.csr_matrix, sigma_a: np.ndarray,
                  sigma_c: np.ndarray) -> np.ndarray:
        """``κ_i = σ²_{a,i} · g_iᵀ S⁻¹ g_i`` for every column ``g_i`` of ``G_k``.

        With ``M = [G_k · diag(σ_a), diag(σ_c)]``, ``S = M Mᵀ`` and ``κ_i`` is the
        i-th diagonal of the projector onto ``M``'s row space: the squared norm
        of row i of ``Q`` in ``Mᵀ = Q R``. Householder QR of ``Mᵀ`` with its rows
        sorted by decreasing norm keeps ``κ`` in [0, 1] to rounding; an LU of
        ``S`` does not, because forming ``S`` squares ``M``'s condition number
        (#27). Hard rows (σ_c = 0) add no row to ``Mᵀ``.
        """
        n, m = G_k.shape[1], G_k.shape[0]
        soft = np.flatnonzero(sigma_c > 0.0)
        Mt = sp.vstack([
            (G_k @ sp.diags(sigma_a)).T,
            sp.csr_matrix((sigma_c[soft], (np.arange(soft.size), soft)),
                          shape=(soft.size, m)),
        ]).tocsr()
        row_norm = np.sqrt(np.asarray(Mt.multiply(Mt).sum(axis=1)).ravel())
        order    = np.argsort(-row_norm, kind="stable")
        Q, _ = qr(Mt[order].toarray(), mode="economic", overwrite_a=True,
                  check_finite=False)
        lev = np.empty(n + soft.size, dtype=np.float64)
        lev[order] = np.einsum("ij,ij->i", Q, Q)
        return lev[:n]

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


