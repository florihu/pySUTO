"""
agg.py
======
AggregationResult  –  dataclass holding the output of Stage 2 collection
                       and aggregation (and later imputation).

AggregationMath    –  inverse-variance weighted aggregation of N_f feeds
                       onto the N_r root-edge axis.

Mathematical summary (see README § Stage 2a)
--------------------------------------------
    P[k, j]     = 1 / sigma[k, j]²        (precision; 0 where not covered)
    Omega[k, j] = P[k, j] / sum_k P[k, j] (normalised weight; 0 where not covered)
    f_agg[j]    = sum_k  Omega[k, j] * F[k, j]
    sigma_agg[j]= 1 / sqrt(sum_k P[k, j])

Invariants
----------
  • Every covered column of Omega sums to 1.
  • sigma_agg[j] ≤ min_k sigma[k, j]  (combining feeds never increases uncertainty).
  • Uncovered edges (no feed contributes): f_agg = NaN, sigma_agg = NaN.
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
    f_agg       : [N_r]       inverse-variance weighted mean; NaN = not covered
    sigma_agg   : [N_r]       combined uncertainty; NaN = not covered
    Omega       : [N_f × N_r] provenance weight matrix; 0 where feed does not cover
    F           : [N_f × N_r] raw observations; NaN where feed does not cover
    Sigma       : [N_f × N_r] per-feed standard deviations; NaN where not covered
    feed_ids    : list of feed ID strings (row labels for F and Sigma)
    coverage    : [N_r] bool  True where at least one feed has a valid observation
    imputation_log : attached by ImputationEngine after imputation (None until then)
    """

    f_agg:    np.ndarray
    sigma_agg:np.ndarray
    Omega:    np.ndarray
    F:        np.ndarray
    Sigma:    np.ndarray
    feed_ids: List[str]
    coverage: np.ndarray
    imputation_log: Optional[object] = field(default=None)

    @property
    def n_feeds(self) -> int:
        return len(self.feed_ids)

    @property
    def n_edges(self) -> int:
        return self.f_agg.shape[0]

    @property
    def n_covered(self) -> int:
        return int(self.coverage.sum())

    @property
    def n_missing(self) -> int:
        return self.n_edges - self.n_covered


# =============================================================================
# AggregationMath
# =============================================================================

class AggregationMath:
    """
    Vectorised inverse-variance aggregation.  All operations run over the
    full [N_f × N_r] matrices without loops.
    """

    @staticmethod
    def run(
        F:        np.ndarray,
        Sigma:    np.ndarray,
        feed_ids: List[str],
    ) -> AggregationResult:
        """
        Compute inverse-variance weighted aggregation.

        Parameters
        ----------
        F        : [N_f × N_r]  observations; NaN where feed does not cover edge
        Sigma    : [N_f × N_r]  standard deviations; NaN where feed does not cover
        feed_ids : length-N_f list of feed ID strings

        Returns
        -------
        AggregationResult
        """
        N_f, N_r = F.shape
        logger.info("aggregating %d feed(s) over %d root edges", N_f, N_r)
        for k, fid in enumerate(feed_ids):
            n_cov = int(np.isfinite(Sigma[k]).sum())
            logger.debug("  feed '%s': %d / %d edges covered", fid, n_cov, N_r)

        # Precision: 1/sigma²; uncovered entries (NaN sigma) → 0 precision
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(np.isfinite(Sigma) & (Sigma > 0), 1.0 / Sigma**2, 0.0)

        P_sum = P.sum(axis=0)          # [N_r]
        coverage = P_sum > 0           # [N_r] bool

        # Normalised weight matrix Omega; 0 where not covered
        with np.errstate(divide="ignore", invalid="ignore"):
            Omega = np.where(coverage, P / P_sum[np.newaxis, :], 0.0)

        # f_agg = 1ᵀ(Omega ⊙ F):  replace NaN in F with 0 before dot product
        # Safe because Omega[k,j]=0 everywhere F[k,j]=NaN.
        F_filled = np.where(np.isnan(F), 0.0, F)
        f_agg    = np.sum(Omega * F_filled, axis=0)
        f_agg    = np.where(coverage, f_agg, np.nan)

        # sigma_agg = 1 / sqrt(P_sum); NaN where uncovered
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_agg = np.where(coverage, 1.0 / np.sqrt(P_sum), np.nan)

        n_covered = int(coverage.sum())
        logger.info(
            "aggregation done: %d covered, %d missing out of %d edges",
            n_covered, N_r - n_covered, N_r,
        )

        return AggregationResult(
            f_agg     = f_agg,
            sigma_agg = sigma_agg,
            Omega     = Omega,
            F         = F,
            Sigma     = Sigma,
            feed_ids  = list(feed_ids),
            coverage  = coverage,
        )
