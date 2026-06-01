"""
uncprop.py
==========
Stage 5 — Posterior uncertainty propagation under the structural model.

Loads a persisted Stage-4 :class:`OptimResult` set plus its upstream
:class:`ConstraintRun`, then computes the marginal posterior standard
deviation σ_{a*}(t) and the (1-α) Gaussian confidence interval for every
edge × period, using the closed-form GLS expression

    σ²_{a*,i}(t) = σ²_{a,i}(t) · (1 - σ²_{a,i}(t) · ‖L(t)⁻¹ g_i(t)‖²)

with ``S(t) = G(t) · diag(σ²_a(t)) · G(t)ᵀ + diag(σ²_c(t))`` and
``g_i(t) = col_i(G(t))``. Only active rows at period ``t`` contribute to
``S(t)`` — inactive rows would inject a "no information" identity block
that adds no shrinkage, so they are dropped from the per-period system.

Under the structural model :class:`StructG` shares one sparsity pattern
across every period; the symbolic Cholesky pattern remains stable while
only the per-period data tape and active row subset vary, so CHOLMOD's
``cholesky_inplace`` keeps the symbolic factor across periods when the
active row set is unchanged. The scipy ``splu`` fallback computes
``gᵀ S⁻¹ g`` directly (same value, no symbolic-pattern reuse).

σ_a and a₀ are pulled through :func:`_conditioning.condition_prior` with
the same floor / ceiling rules the solver applied, so the posterior is
mathematically consistent with the ``a*`` the solver returned. σ_c on
hard rows is forced to exact zero; soft rows receive a numerical floor
only.

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

logger = get_logger("pysuto.uncprop")


# Defaults mirror ClarabelSolver so uncprop sees the same effective prior
# the QP saw.
DEFAULT_SIGMA_A_FLOOR_ABS      = 1e-8
DEFAULT_SIGMA_A_FLOOR_REL      = 1e-5
DEFAULT_SIGMA_A_CEILING_FACTOR = 1e5
DEFAULT_SIGMA_C_FLOOR          = 1e-9


# =============================================================================
# Return dataclasses
# =============================================================================

@dataclass
class UncPropTimestep:
    """Per-period posterior summary.

    Under the structural model every array is full-length ``(n_edges,)``;
    the pre-refactor year-local ``cols_t`` slice is gone with StructG.
    """
    timestep:     str
    sigma_a_star: np.ndarray            # (n_edges,) posterior std
    lower:        np.ndarray            # (n_edges,) a* − z·σ, clipped to l
    upper:        np.ndarray            # (n_edges,) a* + z·σ, clipped to u
    diagnostics:  dict


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
    """Abstract base for posterior-uncertainty concretes."""

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

    Per period:

      1. Pull the period slice ``σ_c(t)`` from :class:`ConstraintRun` and
         keep only the active rows.
      2. Build ``S(t) = G_active(t) · diag(σ²_a(t)) · G_active(t)ᵀ +
                        diag(σ²_c_active(t))``.
      3. Refresh the numeric Cholesky factor.
      4. For each column block ``G_active(t)[:, s:e]``: solve
         ``L⁻¹ G_blk``; accumulate ``‖·‖²`` column-wise to obtain
         ``d[s:e]``.
      5. ``σ²_{a*}(t) = σ²_a(t) · (1 − σ²_a(t) · d)``, clipped to ≥ 0.
      6. ``lower / upper = a*(t) ± z_{1-α/2} · σ_{a*}(t)``, clipped to
         ``[l, u]``.
      7. Diagnostics: ``trace_prior``, ``trace_post``, ``max_resid_hard``,
         ``psd_min_diag``.
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

        hard_mask     = const_run.row_meta["is_hard"].to_numpy().astype(bool)
        equality_mask = (const_run.row_meta["cone_type"].to_numpy() == "zero")

        z = float(norm.ppf(1.0 - alpha / 2.0))

        # 2) Per-period loop. σ_a is conditioned per period since it is
        #    2-D ``(n_edges, n_timesteps)`` under the structural model.
        per_timestep: dict[str, UncPropTimestep] = {}
        t_idx_for = {lbl: i for i, lbl in enumerate(const_run.timesteps)}

        for t_label, opt_t in opt_results.items():
            t_start = time.perf_counter()
            t_idx = t_idx_for.get(t_label)
            if t_idx is None:
                raise KeyError(
                    f"OptimRun period {t_label!r} not in ConstraintRun.timesteps "
                    f"{const_run.timesteps}"
                )

            sigma_a_t_raw = const_run.sigma_a[:, t_idx]
            _, sigma_a_t = condition_prior(
                const_run.a0, sigma_a_t_raw,
                floor_abs      = self.sigma_a_floor_abs,
                floor_rel      = self.sigma_a_floor_rel,
                ceiling_factor = self.sigma_a_ceiling_factor,
            )
            sigma_a2 = sigma_a_t ** 2

            G_t = const_run.G.at(t_idx).tocsc()
            n_edges = G_t.shape[1]
            active  = np.asarray(const_run.active[:, t_idx], dtype=bool)

            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                # No active rows → posterior = prior.
                sigma_a_star = sigma_a_t.copy()
                d = np.zeros(n_edges)
            else:
                G_act = G_t[active_idx]
                sc_raw = np.asarray(const_run.sigma_c[active_idx, t_idx], dtype=float)
                hm_act = hard_mask[active_idx]
                sc_act = condition_sigma_c(
                    sc_raw, floor=self.sigma_c_floor, hard_mask=hm_act,
                )
                S = self._build_S(G_act, sigma_a2, sc_act ** 2)
                d = self._marginal_diagonal(S, G_act, n_edges)
                var_post = sigma_a2 * (1.0 - sigma_a2 * d)
                np.clip(var_post, 0.0, None, out=var_post)
                sigma_a_star = np.sqrt(var_post)

            half_width = z * sigma_a_star
            lower = np.maximum(opt_t.a_star - half_width, const_run.l)
            upper = np.minimum(opt_t.a_star + half_width, const_run.u)

            diagnostics = self._diagnostics(
                sigma_a2      = sigma_a2,
                sigma_a_star  = sigma_a_star,
                G_t           = G_t,
                active        = active,
                equality_mask = equality_mask,
                a_star        = opt_t.a_star,
                c_t           = np.asarray(const_run.c[:, t_idx], dtype=float),
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

    def persist(
        self,
        result:     UncPropResult,
        opt_run_id: str,
        label:      str = "",
    ) -> str:
        """Persist *result* via :class:`UncPropRunRegistry`."""
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
                 sigma_c2: np.ndarray, ridge: float = 1e-12) -> sp.csc_matrix:
        """Assemble ``S = G · diag(σ²_a) · Gᵀ + diag(σ²_c) + ridge·I`` as CSC.

        ``ridge`` is a tiny diagonal regularisation that keeps ``S`` strictly
        PSD when hard equality rows dominate (``σ_c² = 0`` after
        ``condition_sigma_c``). Without it ``G·diag(σ_a²)·Gᵀ`` can be
        rank-deficient — e.g. when the only active rows are mass-balance
        equalities on a graph whose constraint rows aren't linearly
        independent — and ``splu`` reports ``Factor is exactly singular``.
        The ridge changes shrinkage by less than one part in 10¹² of σ_a².
        """
        Da = sp.diags(sigma_a2)
        Dc = sp.diags(sigma_c2)
        m = G.shape[0]
        return (G @ Da @ G.T + Dc + ridge * sp.eye(m, format="csc")).tocsc()

    def _marginal_diagonal(self, S: sp.csc_matrix, G: sp.csc_matrix,
                           n_edges: int) -> np.ndarray:
        """Return ``d_i = g_iᵀ S⁻¹ g_i`` for every column ``g_i`` of ``G``."""
        d = np.zeros(n_edges, dtype=np.float64)
        try:
            from sksparse.cholmod import cholesky as cholmod_cholesky
            factor = cholmod_cholesky(S)
            for s in range(0, n_edges, self.block_size):
                e = min(s + self.block_size, n_edges)
                G_blk = G[:, s:e].toarray()
                V = factor.solve_L(G_blk, use_LDLt_decomposition=False)
                d[s:e] = np.einsum("mb,mb->b", V, V)
        except ImportError:
            from scipy.sparse.linalg import splu
            lu = splu(S)
            for s in range(0, n_edges, self.block_size):
                e = min(s + self.block_size, n_edges)
                G_blk = G[:, s:e].toarray()
                X = lu.solve(G_blk)
                d[s:e] = np.einsum("mb,mb->b", G_blk, X)
        return d

    @staticmethod
    def _diagnostics(*, sigma_a2: np.ndarray, sigma_a_star: np.ndarray,
                     G_t: sp.csc_matrix, active: np.ndarray,
                     equality_mask: np.ndarray, a_star: np.ndarray,
                     c_t: np.ndarray) -> dict:
        trace_prior = float(sigma_a2.sum())
        trace_post  = float(np.square(sigma_a_star).sum())
        # Only active equality rows contribute to the hard-residual check.
        em_active = equality_mask & active
        if em_active.any():
            resid = G_t @ a_star - c_t
            max_resid_hard = float(np.abs(resid[em_active]).max())
        else:
            max_resid_hard = 0.0
        return {
            "trace_prior":    trace_prior,
            "trace_post":     trace_post,
            "max_resid_hard": max_resid_hard,
            "psd_min_diag":   float(sigma_a_star.min()),
        }
