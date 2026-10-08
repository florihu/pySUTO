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

On the block path (``G_blocks`` present, every current run) ``S(t)`` is
analysed and factored once per timestep from that year's ``G_t``. Only the
full-G path analyses the symbolic Cholesky pattern once (G is time-invariant
there) and refreshes the numeric factor per timestep via CHOLMOD's
``cholesky_inplace`` because σ_c is time-variant. A scipy ``splu``
fallback kicks in when scikit-sparse is not installed; it computes
``gᵀ S⁻¹ g`` directly (same value, no symbolic-pattern reuse, slower).

σ_a and a₀ are pulled through :func:`_conditioning.condition_prior`
with the same floor / ceiling rules the solver applied, so the posterior
is mathematically consistent with the ``a*`` the solver returned.
σ_c on hard rows is forced to exact zero; soft rows receive a numerical
floor only.

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
    lower:        np.ndarray            # a* − z·σ, clipped to l
    upper:        np.ndarray            # a* + z·σ, clipped to u
    diagnostics:  dict                  # trace_prior / trace_post / max_resid_hard / psd_min_diag
    cols_t:       Optional[np.ndarray] = None  # global edge cols (block path)


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
    ) -> None:
        self.opt_dir      = opt_dir
        self.const_dir    = const_dir
        self.uncprop_dir  = uncprop_dir
        self.block_size   = int(block_size)
        self.sigma_a_floor_abs      = sigma_a_floor_abs
        self.sigma_a_floor_rel      = sigma_a_floor_rel
        self.sigma_a_ceiling_factor = sigma_a_ceiling_factor
        self.sigma_c_floor          = sigma_c_floor

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

            shrinkage = (1.0 - diagnostics["trace_post"] / diagnostics["trace_prior"]
                         if diagnostics["trace_prior"] > 0.0 else 0.0)
            logger.info(
                "t=%s  σ_post: min=%.3e med=%.3e max=%.3e  shrinkage=%.3f  time=%.2fs",
                t_label,
                float(sigma_a_star.min()),
                float(np.median(sigma_a_star)),
                float(sigma_a_star.max()),
                shrinkage,
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
        makes d≈0 and shrinkage vanishes."""
        _, sigma_a = condition_prior(
            const_run.a0, const_run.sigma_a,
            floor_abs      = self.sigma_a_floor_abs,
            floor_rel      = self.sigma_a_floor_rel,
            ceiling_factor = self.sigma_a_ceiling_factor,
        )
        sigma_a2 = sigma_a ** 2

        hard_mask     = const_run.row_meta["is_hard"].to_numpy().astype(bool)
        equality_mask = (const_run.row_meta["cone_type"].to_numpy() == "zero")
        row_time      = const_run.row_meta["time"].astype(str).to_numpy()

        z = float(norm.ppf(1.0 - alpha / 2.0))

        per_timestep: dict[str, UncPropTimestep] = {}
        for t_label, opt_t in opt_results.items():
            t_start = time.perf_counter()

            cols_t = const_run.edge_cols_by_t[t_label]
            rows_t = np.where(row_time == t_label)[0]
            G_t    = const_run.G_blocks[t_label].tocsc()

            sigma_a2_t = sigma_a2[cols_t]

            raw_sc  = np.asarray(
                const_run.timestep_data[t_label]["sigma_c"], dtype=float,
            )[rows_t]
            hm_t    = hard_mask[rows_t]
            sc_t    = condition_sigma_c(
                raw_sc, floor=self.sigma_c_floor, hard_mask=hm_t,
            )
            sigma_c2_t = sc_t ** 2

            S_t = self._build_S(G_t, sigma_a2_t, sigma_c2_t)
            d_t = self._marginal_diagonal_block(S_t, G_t, sigma_a2_t.size)

            var_post_t = sigma_a2_t * (1.0 - sigma_a2_t * d_t)
            np.clip(var_post_t, 0.0, None, out=var_post_t)
            sigma_a_star_t = np.sqrt(var_post_t)

            # Year-local output (REP §8.2): store only this period's cols_t
            # posterior. F_opt is already the year-local slice from the solver.
            half_width_t = z * sigma_a_star_t
            lower_t = np.maximum(opt_t.F_opt - half_width_t, const_run.l[cols_t])
            upper_t = np.minimum(opt_t.F_opt + half_width_t, const_run.u[cols_t])

            # Per-t diagnostics: hard-row residual on the t-block only.
            em_t = equality_mask[rows_t]
            c_t_full = np.asarray(const_run.timestep_data[t_label]["c"])
            if em_t.any():
                resid_t = G_t @ opt_t.F_opt - c_t_full[rows_t]
                max_resid_hard = float(np.abs(resid_t[em_t]).max())
            else:
                max_resid_hard = 0.0
            diagnostics = {
                "trace_prior":    float(sigma_a2.sum()),
                "trace_post":     float((sigma_a2.sum() - sigma_a2_t.sum())
                                        + np.square(sigma_a_star_t).sum()),
                "max_resid_hard": max_resid_hard,
                "psd_min_diag":   float(sigma_a_star_t.min()),
            }

            per_timestep[t_label] = UncPropTimestep(
                timestep     = t_label,
                sigma_a_star = sigma_a_star_t,
                lower        = lower_t,
                upper        = upper_t,
                diagnostics  = diagnostics,
                cols_t       = cols_t,
            )

            shrinkage = (1.0 - diagnostics["trace_post"] / diagnostics["trace_prior"]
                         if diagnostics["trace_prior"] > 0.0 else 0.0)
            logger.info(
                "t=%s  σ_post: min=%.3e med=%.3e max=%.3e  shrinkage=%.3f  "
                "time=%.2fs  (per-t: %d cols × %d rows)",
                t_label,
                float(sigma_a_star_t.min()),
                float(np.median(sigma_a_star_t)),
                float(sigma_a_star_t.max()),
                shrinkage,
                time.perf_counter() - t_start,
                G_t.shape[1], G_t.shape[0],
            )

        return UncPropResult(
            method       = self.METHOD,
            alpha        = float(alpha),
            per_timestep = per_timestep,
        )

    def _marginal_diagonal_block(
        self,
        S_t: sp.csc_matrix,
        G_t: sp.csc_matrix,
        n_t: int,
    ) -> np.ndarray:
        """``d_i = g_iᵀ S_t⁻¹ g_i`` for every column ``g_i`` of ``G_t``.
        Uses CHOLMOD when available (one analyze+factor per t — the per-t
        path doesn't reuse symbolic patterns across t because G_blocks can
        differ); falls back to scipy ``splu`` otherwise."""
        d = np.zeros(n_t, dtype=np.float64)
        try:
            from sksparse.cholmod import cholesky as cholmod_cholesky
            factor = cholmod_cholesky(S_t)
            for s in range(0, n_t, self.block_size):
                e = min(s + self.block_size, n_t)
                G_blk = G_t[:, s:e].toarray()
                V = factor.solve_L(G_blk, use_LDLt_decomposition=False)
                d[s:e] = np.einsum("mb,mb->b", V, V)
        except ImportError:
            from scipy.sparse.linalg import splu
            lu = splu(S_t)
            for s in range(0, n_t, self.block_size):
                e = min(s + self.block_size, n_t)
                G_blk = G_t[:, s:e].toarray()
                X = lu.solve(G_blk)
                d[s:e] = np.einsum("mb,mb->b", G_blk, X)
        return d

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


