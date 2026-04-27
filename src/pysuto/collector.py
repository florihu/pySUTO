"""
collector.py
============
FeedCollector  –  selects feeds from a DataFeedRegistry and aligns their
                  F_root / sigma_root vectors onto the common root-edge index,
                  then delegates to AggregationMath for inverse-variance
                  weighted aggregation.

Usage
-----
    collector = FeedCollector(registry=registry, acc_struct=acc)
    collector.select(["baci", "crude_ore"])          # by ID (latest version)
    collector.select([("baci", 1), "crude_ore"])     # or specific version tuple

    agg_result = collector.aggregate()
    # agg_result.f_agg    [N_r]       inverse-variance weighted mean
    # agg_result.sigma_agg[N_r]       combined uncertainty
    # agg_result.Omega    [N_f × N_r] provenance weight matrix
    # agg_result.coverage [N_r] bool  True where ≥1 feed covers the edge

Convention for "not covered"
-----------------------------
sigma_root = 0 in a result.npz means no source edge contributed to that
root edge.  The collector converts sigma_root = 0 to NaN so that
AggregationMath treats the entry as missing.  The corresponding F_root
entry is also set to NaN regardless of its stored value.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from pysuto.registry import DataFeedRegistry
from pysuto.agg import AggregationMath, AggregationResult
from pysuto.logging import get_logger

logger = get_logger("collector")


# =============================================================================
# FeedCollector
# =============================================================================

class FeedCollector:
    """
    Loads a selection of feeds from the registry and assembles their
    F_root / sigma_root vectors into aligned [N_f × N_r] matrices.

    Parameters
    ----------
    registry   : DataFeedRegistry with pre-loaded index
    acc_struct : AccountingStructure whose edge_index defines the N_r axis
    """

    def __init__(self, registry: DataFeedRegistry, acc_struct, min_flow_kt: float = 0.0) -> None:
        self._registry    = registry
        self._acc_struct  = acc_struct
        self._feed_ids:   List[str] = []
        self._min_flow_kt = min_flow_kt

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        feed_ids: List[Union[str, tuple]],
    ) -> "FeedCollector":
        """
        Specify which feeds to collect.

        Each entry is either:
          * a feed_id string  → latest registered version is used
          * a (feed_id, version) tuple  → version field is recorded but the
            registry currently stores one entry per ID; future versioning
            will use this to disambiguate.

        Returns self so calls can be chained.

        Raises
        ------
        KeyError  if a feed_id is not present in the registry
        """
        resolved: List[str] = []
        for item in feed_ids:
            if isinstance(item, tuple):
                fid = item[0]
            else:
                fid = item
            if fid not in self._registry:
                raise KeyError(
                    f"Feed '{fid}' not found in registry. "
                    f"Available: {self._registry.ids()}"
                )
            resolved.append(fid)
        self._feed_ids = resolved
        logger.info("selected %d feed(s): %s", len(resolved), resolved)
        return self

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load F_root and sigma_root for every selected feed and stack into
        [N_f × N_r] matrices with NaN where a feed does not cover an edge.

        Returns
        -------
        F     : [N_f × N_r]  float64, NaN = not covered
        Sigma : [N_f × N_r]  float64, NaN = not covered
        """
        N_r = len(self._acc_struct.edge_index)
        N_f = len(self._feed_ids)

        F     = np.full((N_f, N_r), np.nan, dtype=float)
        Sigma = np.full((N_f, N_r), np.nan, dtype=float)

        for k, fid in enumerate(self._feed_ids):
            res     = self._registry.load_result(fid)
            f_k     = res["F_root"].astype(float)
            sigma_k = res["sigma_root"].astype(float)

            # sigma_root = 0 → no source edge contributed → not covered
            covered = sigma_k > 0
            if self._min_flow_kt > 0:
                below = covered & (np.abs(f_k) < self._min_flow_kt)
                covered &= ~below
                if below.any():
                    logger.info(
                        "'%s': suppressed %d near-zero edges (|f| < %.4g kt)",
                        fid, int(below.sum()), self._min_flow_kt,
                    )
            F[k, covered]     = f_k[covered]
            Sigma[k, covered] = sigma_k[covered]

            logger.debug(
                "loaded '%s': %d / %d edges covered",
                fid, int(covered.sum()), N_r,
            )

        return F, Sigma

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregationResult:
        """
        Align selected feeds and compute inverse-variance aggregation.

        Returns
        -------
        AggregationResult with f_agg, sigma_agg, Omega, F, Sigma, coverage.

        Raises
        ------
        RuntimeError  if select() has not been called
        """
        if not self._feed_ids:
            raise RuntimeError("No feeds selected. Call select() first.")

        logger.info("starting aggregation for feeds: %s", self._feed_ids)
        F, Sigma = self._load_arrays()
        return AggregationMath.run(F, Sigma, self._feed_ids)
