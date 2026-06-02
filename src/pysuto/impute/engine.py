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
        "edge_id", "step", "imputer_id", "pass", "method",
        "source_edges", "coefficients",
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
        "run_timestamp", "step", "imputer_id", "pass", "imputer_type",
        "input_paths", "elapsed_seconds", "edges_filled",
        "edges_missing_after", "pct_additionally_filled", "config",
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
    Runs imputers in two phases:

    1. **Cascade** (registered via :meth:`register_cascade`) — iterated to a
       fixpoint: the whole group is re-run until a full sweep fills no new
       edges (or ``max_iter`` is hit). Lets a value derived at one
       metallurgical stage propagate to the next on the following pass.
    2. **Terminal** (registered via :meth:`register`) — run once each, in
       registration order, after the cascade converges.

    Each step receives the state (f, σ) left by the previous step and stops
    early if no missing edges remain. Every fill is stamped with the
    imputer's ``id`` and the cascade ``pass`` number (None for terminals) in
    both the ImputationLog and the per-step metadata.
    """

    def __init__(self, acc_struct) -> None:
        self._acc_struct = acc_struct
        self._imputers: list[BaseImputer] = []       # terminal (run once)
        self._cascade:  list[BaseImputer] = []       # iterated to fixpoint
        self._cascade_max_iter: int = 1
        self._step: int = 0                          # running committed step id

    @property
    def imputers(self) -> list[BaseImputer]:
        """Read-only view of every registered imputer (cascade then terminal)."""
        return list(self._cascade) + list(self._imputers)

    def register(self, imputer: BaseImputer) -> None:
        """Register a terminal imputer (run once, after the cascade)."""
        if hasattr(imputer, "prepare"):
            imputer.prepare(self._acc_struct)
        self._imputers.append(imputer)

    def register_cascade(self, imputers: list[BaseImputer], max_iter: int = 10) -> None:
        """Register the cascade group, iterated together to a fixpoint.

        The group is re-run in order until a complete sweep fills zero new
        edges, capped at ``max_iter`` passes (fills are monotone, so the loop
        always terminates).
        """
        for imp in imputers:
            if hasattr(imp, "prepare"):
                imp.prepare(self._acc_struct)
            self._cascade.append(imp)
        self._cascade_max_iter = int(max_iter)

    def run(self, agg_result: AggregationResult) -> AggregationResult:
        f     = agg_result.f_agg.copy()
        sigma = agg_result.sigma_agg.copy()
        log   = ImputationLog()
        meta  = ImputationMetadata()
        run_ts = datetime.now().isoformat(timespec="seconds")
        n_total = len(f)
        self._step = 0

        # ── Phase 1: cascade to fixpoint ────────────────────────────────────
        converged = False
        for pass_no in range(1, self._cascade_max_iter + 1):
            if int(np.isnan(f).sum()) == 0:
                converged = True
                break
            missing_before_pass = int(np.isnan(f).sum())
            log_len = len(log)
            meta_len = len(meta)
            step_before = self._step
            for imputer in self._cascade:
                if int(np.isnan(f).sum()) == 0:
                    break
                self._run_one(imputer, pass_no, f, sigma, agg_result,
                              log, meta, n_total, run_ts)
            if int(np.isnan(f).sum()) == missing_before_pass:
                # Dead pass: a full sweep filled nothing new → fixpoint reached.
                # Discard its (all zero-fill) records so provenance stays clean.
                del log._entries[log_len:]
                del meta._records[meta_len:]
                self._step = step_before
                converged = True
                break
        if not converged and self._cascade and int(np.isnan(f).sum()) > 0:
            logger.warning(
                "cascade hit max_iter=%d without reaching a fixpoint; "
                "%d edges still missing",
                self._cascade_max_iter, int(np.isnan(f).sum()),
            )

        # ── Phase 2: terminal imputers, once each ───────────────────────────
        for imputer in self._imputers:
            if int(np.isnan(f).sum()) == 0:
                logger.info("step %d: no missing edges — stopping early", self._step)
                break
            self._run_one(imputer, None, f, sigma, agg_result,
                          log, meta, n_total, run_ts)

        agg_result.f_agg               = f
        agg_result.sigma_agg           = sigma
        agg_result.imputation_log      = log
        agg_result.imputation_metadata = meta
        return agg_result

    def _run_one(
        self, imputer, pass_no, f, sigma, agg_result, log, meta, n_total, run_ts,
    ) -> None:
        """Run one imputer once: impute, clip, stamp provenance, log + record."""
        step = self._step
        missing = np.isnan(f)
        n_missing_before = int(missing.sum())
        cov_before = (n_total - n_missing_before) / n_total

        # Hand the (read-only) aggregation context to any imputer that opts in
        # via attach_agg_result(); FlatPriorImputer uses this to read per-edge
        # coverage for density-aware CV.
        if hasattr(imputer, "attach_agg_result"):
            imputer.attach_agg_result(agg_result)

        t0 = time.perf_counter()
        entries = imputer.impute(f, sigma, missing, step, self._acc_struct)
        elapsed = time.perf_counter() - t0
        for e in entries:
            e["imputer_id"] = imputer.id
            e["pass"] = pass_no
        log.extend(entries)

        # Post-step bounds enforcement: warn and clip any newly-filled
        # out-of-bounds values.
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
            "step %d [%s pass %s] (%s): filled %d edges  →  "
            "coverage %.1f%% → %.1f%%  (△+%.1f%%)",
            step, imputer.id, pass_no if pass_no is not None else "-",
            type(imputer).__name__, n_filled,
            100 * cov_before, 100 * cov_after, 100 * (cov_after - cov_before),
        )
        meta.append({
            "run_timestamp": run_ts,
            "step": step,
            "imputer_id": imputer.id,
            "pass": pass_no,
            "imputer_type": type(imputer).__name__,
            "input_paths": "; ".join(imputer.source_paths),
            "elapsed_seconds": round(elapsed, 4),
            "edges_filled": n_filled,
            "edges_missing_after": n_missing_after,
            "pct_additionally_filled": round(100 * n_filled / n_total, 2),
            "config": imputer.config() if hasattr(imputer, "config") else {},
        })
        self._step += 1
