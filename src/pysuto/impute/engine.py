"""
engine.py
=========
``ImputationEngine`` — orchestrator that runs registered imputers in
registration order and stamps the resulting log + metadata onto the
``AggregationResult``. Includes the two log-shape dataclasses
(:class:`ImputationLog`, :class:`ImputationMetadata`).
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from pysuto.agg import AggregationResult
from pysuto.logging import get_logger
from pysuto.param_relation import get_edge_cache

from .base import BaseImputer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ImputationLog
# ---------------------------------------------------------------------------

class ImputationLog:
    """
    Provenance table for every imputed edge.

    Each entry is a dict with keys:
        edge_id, step, method, source_edges, coefficients,
        f_before, f_after, sigma_before, sigma_after,
        description, param_id
    """

    _COLS = [
        "edge_id", "step", "method", "source_edges", "coefficients",
        "f_before", "f_after", "sigma_before", "sigma_after",
        "description", "param_id",
    ]

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def append(self, entry: dict) -> None:
        self._entries.append(entry)

    def extend(self, entries: list[dict]) -> None:
        self._entries.extend(entries)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._entries, columns=self._COLS)

    def for_edge(self, edge_id: int) -> pd.DataFrame:
        return self.as_dataframe().query("edge_id == @edge_id").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ImputationLog({len(self._entries)} entries)"


# ---------------------------------------------------------------------------
# ImputationMetadata
# ---------------------------------------------------------------------------

class ImputationMetadata:
    """Run-level summary: one record per imputer step."""

    _COLS = [
        "run_timestamp", "step", "imputer_type", "input_paths",
        "elapsed_seconds", "edges_filled",
        "edges_missing_after", "pct_additionally_filled",
    ]

    def __init__(self) -> None:
        self._records: list[dict] = []

    def append(self, record: dict) -> None:
        self._records.append(record)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=self._COLS)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"ImputationMetadata({len(self._records)} steps)"

# ---------------------------------------------------------------------------
# ImputationEngine
# ---------------------------------------------------------------------------

class ImputationEngine:
    """
    Runs registered imputers in registration order.

    Each step receives the state (f, σ) left by the previous step.
    Stops early if no missing edges remain.
    Attaches the completed ImputationLog to agg_result.imputation_log.
    """

    def __init__(self, acc_struct) -> None:
        self._acc_struct = acc_struct
        self._imputers: list[BaseImputer] = []

    @property
    def imputers(self) -> list[BaseImputer]:
        """Read-only view of registered imputers (in registration order)."""
        return list(self._imputers)

    def register(self, imputer: BaseImputer) -> None:
        if hasattr(imputer, "prepare"):
            imputer.prepare(self._acc_struct)
        self._imputers.append(imputer)

    def run(self, agg_result: AggregationResult) -> AggregationResult:
        f     = agg_result.a_agg.copy()
        sigma = agg_result.sigma_agg.copy()
        log   = ImputationLog()
        meta  = ImputationMetadata()
        run_ts = datetime.now().isoformat(timespec="seconds")

        n_total = len(f)
        for step, imputer in enumerate(self._imputers):
            missing = np.isnan(f)
            n_missing_before = int(missing.sum())
            if n_missing_before == 0:
                logger.info("step %d: no missing edges — stopping early", step)
                break
            cov_before = (n_total - n_missing_before) / n_total

            # Hand the (read-only) aggregation context to any imputer that
            # opts in via attach_agg_result(); FlatPriorImputer uses this
            # to read per-edge coverage for density-aware CV.
            if hasattr(imputer, "attach_agg_result"):
                imputer.attach_agg_result(agg_result)

            t0 = time.perf_counter()
            entries = imputer.impute(f, sigma, missing, step, self._acc_struct)
            elapsed = time.perf_counter() - t0
            log.extend(entries)

            # Post-step bounds enforcement: warn and clip any newly-filled out-of-bounds values
            newly_filled = np.where(missing & ~np.isnan(f))[0]
            if newly_filled.size > 0:
                bl = self._acc_struct.bounds.l[newly_filled]
                bu = self._acc_struct.bounds.u[newly_filled]
                fv = f[newly_filled]
                clip_needed = (fv < bl) | (fv > bu)
                if clip_needed.any():
                    _, _, id_to_tuple, _, _, _ = get_edge_cache(self._acc_struct)
                    for e_id in newly_filled[clip_needed]:
                        lo  = float(self._acc_struct.bounds.l[e_id])
                        hi  = float(self._acc_struct.bounds.u[e_id])
                        val = float(f[e_id])
                        clipped_val = float(np.clip(val, lo, hi))
                        tup = id_to_tuple[int(e_id)]
                        warnings.warn(
                            f"step {step} ({type(imputer).__name__}): "
                            f"edge {e_id} ({tup[0]}[{tup[2]}]→{tup[3]}[{tup[5]}], "
                            f"t={tup[6]}, {tup[7]}): "
                            f"imputed {val:.4g} clipped to {clipped_val:.4g} "
                            f"(bounds [{lo:.4g}, {hi:.4g}])",
                            UserWarning,
                            stacklevel=2,
                        )
                        f[e_id] = clipped_val

            n_missing_after = int(np.isnan(f).sum())
            n_filled = n_missing_before - n_missing_after
            cov_after = (n_total - n_missing_after) / n_total
            logger.info(
                "step %d (%s): filled %d edges  →  coverage %.1f%% → %.1f%%  (△+%.1f%%)",
                step, type(imputer).__name__, n_filled,
                100 * cov_before, 100 * cov_after, 100 * (cov_after - cov_before),
            )
            meta.append({
                "run_timestamp": run_ts,
                "step": step,
                "imputer_type": type(imputer).__name__,
                "input_paths": "; ".join(imputer.source_paths),
                "elapsed_seconds": round(elapsed, 4),
                "edges_filled": n_filled,
                "edges_missing_after": n_missing_after,
                "pct_additionally_filled": round(100 * n_filled / n_total, 2),
            })

        agg_result.a_agg               = f
        agg_result.sigma_agg           = sigma
        agg_result.imputation_log      = log
        agg_result.imputation_metadata = meta
        return agg_result
