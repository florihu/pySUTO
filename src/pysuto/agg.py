"""
agg.py
======
AggregationResult  –  dataclass holding the output of Stage 2 collection
                       and aggregation (and later imputation).

AggregationMath    –  inverse-variance weighted aggregation of N_f feeds
                       onto the (n_edges, n_timesteps) (structural edge × time)
                       grid.

Shape convention (post-structural refactor)
-------------------------------------------
* a_agg, sigma_agg, coverage : 2-D (n_edges, n_timesteps)
* Omega, F, Sigma            : 3-D (N_f, n_edges, n_timesteps)
* feed_ids                   : length-N_f list

Mathematical summary (see README § Stage 2a)
--------------------------------------------
    P[k, j, t]     = 1 / sigma[k, j, t]²        (precision; 0 where not covered)
    Omega[k, j, t] = P[k, j, t] / sum_k P[k, j, t]  (normalised weight)
    a_agg[j, t]    = sum_k Omega[k, j, t] * F[k, j, t]              (BLUE estimator)
    sigma_agg[j, t]= ( sum_k |Omega[k, j, t] · sigma[k, j, t]|^p )^(1/p)

Invariants
----------
  • Every covered (j, t) slice of Omega sums to 1 over feeds k.
  • For p ∈ (1, 2]: sigma_agg[j, t] ≤ min_k sigma[k, j, t] (single-cover);
    when feeds overlap, σ_agg ≤ Lp({Ω·σ}) ≤ max_k σ_k.
  • Uncovered cells (no feed contributes): a_agg = NaN, sigma_agg = NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from pysuto.logging import get_logger

logger = get_logger("agg")


# =============================================================================
# AggregationResult
# =============================================================================

@dataclass
class AggregationResult:
    """
    Output of FeedCollector.aggregate(), updated in-place by ImputationEngine.

    Attributes
    ----------
    a_agg       : (n_edges, n_timesteps)       inverse-variance weighted mean; NaN = not covered
    sigma_agg   : (n_edges, n_timesteps)       combined uncertainty; NaN = not covered
    Omega       : (N_f, n_edges, n_timesteps)  provenance weight; 0 where feed does not cover
    F           : (N_f, n_edges, n_timesteps)  raw observations; NaN where feed does not cover
    Sigma       : (N_f, n_edges, n_timesteps)  per-feed σ; NaN where not covered
    feed_ids    : list of feed ID strings (length-N_f labels for axis 0 of F/Sigma)
    coverage    : (n_edges, n_timesteps) bool  True where ≥1 feed has a finite observation
    imputation_log : attached by ImputationEngine after imputation (None until then)
    """

    a_agg:    np.ndarray
    sigma_agg:np.ndarray
    Omega:    np.ndarray
    F:        np.ndarray
    Sigma:    np.ndarray
    feed_ids: List[str]
    coverage: np.ndarray
    imputation_log:      Optional[object] = field(default=None)
    imputation_metadata: Optional[object] = field(default=None)
    # Provenance: set by ImputationRunRegistry.register() so downstream stages
    # (ConstraintEngine) can record the upstream imputation run.
    _imp_run_id:         Optional[str]    = field(default=None)

    @property
    def n_feeds(self) -> int:
        return len(self.feed_ids)

    @property
    def n_edges(self) -> int:
        return self.a_agg.shape[0]

    @property
    def n_timeps(self) -> int:
        return self.a_agg.shape[1]

    @property
    def n_covered(self) -> int:
        return int(self.coverage.sum())

    @property
    def n_missing(self) -> int:
        return int(self.a_agg.size) - self.n_covered


# =============================================================================
# AggregationMath
# =============================================================================

class AggregationMath:
    """
    Vectorised inverse-variance aggregation.  All operations run over the
    full (N_f, n_edges, n_timesteps) tensors without loops.
    """

    @staticmethod
    def run(
        F:        np.ndarray,
        Sigma:    np.ndarray,
        feed_ids: List[str],
        p:        float = 1.3,
    ) -> AggregationResult:
        """
        Compute inverse-variance weighted aggregation with Lp σ propagation.

        Parameters
        ----------
        F        : (N_f, n_edges, n_timesteps)  observations; NaN where not covered
        Sigma    : (N_f, n_edges, n_timesteps)  standard deviations; NaN where not covered
        feed_ids : length-N_f list of feed ID strings
        p        : power-mean exponent for σ_agg (default 1.3). p=2 recovers
                   Gaussian quadrature σ_agg = 1/sqrt(Σ 1/σ_k²).

        Returns
        -------
        AggregationResult with 2-D a_agg/sigma_agg/coverage.
        """
        N_f, n_edges, n_timeps = F.shape
        logger.info(
            "aggregating %d feed(s) over %d struct edges × %d timesteps (p=%.2f)",
            N_f, n_edges, n_timeps, p,
        )
        for k, fid in enumerate(feed_ids):
            n_cov = int(np.isfinite(Sigma[k]).sum())
            logger.debug("  feed '%s': %d / %d cells covered", fid, n_cov, n_edges * n_timeps)

        # Precision: 1/sigma²; uncovered cells (NaN sigma) → 0 precision
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(np.isfinite(Sigma) & (Sigma > 0), 1.0 / Sigma**2, 0.0)

        P_sum = P.sum(axis=0)          # (n_edges, n_timesteps)
        coverage = P_sum > 0           # (n_edges, n_timesteps) bool

        # Normalised weight tensor Omega; 0 where not covered
        with np.errstate(divide="ignore", invalid="ignore"):
            Omega = np.where(coverage[np.newaxis, :, :], P / P_sum[np.newaxis, :, :], 0.0)

        # a_agg = sum_k(Omega ⊙ F):  replace NaN in F with 0 before sum
        # Safe because Omega[k,j,t]=0 everywhere F[k,j,t]=NaN.
        F_filled = np.where(np.isnan(F), 0.0, F)
        a_agg    = np.sum(Omega * F_filled, axis=0)
        a_agg    = np.where(coverage, a_agg, np.nan)

        # σ_agg[j,t] = ( Σ_k |Ω_kjt · σ_kjt|^p )^(1/p); NaN where uncovered.
        Sigma_safe = np.where(np.isfinite(Sigma), Sigma, 0.0)
        terms = np.abs(Omega * Sigma_safe) ** p          # (N_f, n_edges, n_timesteps)
        sum_p = terms.sum(axis=0)                         # (n_edges, n_timesteps)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_agg = np.where(
                coverage,
                np.power(np.maximum(sum_p, 0.0), 1.0 / p),
                np.nan,
            )

        n_covered = int(coverage.sum())
        n_cells   = n_edges * n_timeps
        logger.info(
            "aggregation done: %d covered, %d missing out of %d cells",
            n_covered, n_cells - n_covered, n_cells,
        )

        return AggregationResult(
            a_agg     = a_agg,
            sigma_agg = sigma_agg,
            Omega     = Omega,
            F         = F,
            Sigma     = Sigma,
            feed_ids  = list(feed_ids),
            coverage  = coverage,
        )
