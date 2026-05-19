"""
imputation.py
=============
``ImputationRunRegistry`` — Stage-2 output registry.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ._sidecar import _now_iso, build_sidecar
from .base import BaseOutputRegistry


class ImputationRunRegistry(BaseOutputRegistry):
    """Stage-2 registry.

    Per-run directory layout::

        data/proc/imp/
        ├── registry.json
        └── imp_<ts>_<code>/
            ├── metadata.json          ← harmonized sidecar
            ├── agg_result.npz         ← f_agg, sigma_agg, Omega, F, Σ, coverage, feed_ids
            └── imputation_log.parquet
    """

    STAGE = "imp"

    def __init__(self, storage_dir: str = "data/proc/imp") -> None:
        super().__init__(storage_dir=storage_dir)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.load_index()

    # ── register ──────────────────────────────────────────────────────

    def register(
        self,
        agg_result,
        *,
        label:         str,
        parents:       dict,
        imputers:      Optional[List[Any]] = None,
        stage_config:  Optional[dict] = None,
        stage_summary: Optional[dict] = None,
    ) -> str:
        """Persist a Stage-2 imputation run and return its run-id.

        Parameters
        ----------
        agg_result
            AggregationResult after ``ImputationEngine.run()`` — must
            carry an ``imputation_log``.
        parents
            Harmonized parents block.  Expects
            ``parents.inputs = {"acc": {...}, "params": [...]}`` and
            ``parents.upstream = {"feed_run_ids": [...]}``.
        imputers
            Optional list of imputer instances; their ``.config()`` and
            the per-step metadata are folded into the sidecar.

        Returns
        -------
        str
            ``run_id``.  Also stamped onto ``agg_result._imp_run_id`` so
            downstream constraint builds can auto-resolve the parent.
        """
        if agg_result.imputation_log is None:
            raise ValueError(
                "AggregationResult.imputation_log is None — register only "
                "after ImputationEngine.run()."
            )
        if "inputs" not in parents:
            raise ValueError("parents must contain an 'inputs' block")

        run_id     = self._new_run_id()
        created_at = _now_iso()
        run_dir    = self._make_run_dir(run_id)

        # 1. arrays
        np.savez(
            os.path.join(run_dir, "agg_result.npz"),
            f_agg     = agg_result.f_agg,
            sigma_agg = agg_result.sigma_agg,
            Omega     = agg_result.Omega,
            F         = agg_result.F,
            Sigma     = agg_result.Sigma,
            coverage  = agg_result.coverage,
            feed_ids  = np.array(agg_result.feed_ids, dtype=object),
        )

        # 2. imputation log
        log_df = agg_result.imputation_log.as_dataframe()
        log_df.to_parquet(
            os.path.join(run_dir, "imputation_log.parquet"),
            index=False,
        )

        # 3. assemble per-imputer step records (matches old metadata.json shape)
        meta_records: list[dict] = []
        if agg_result.imputation_metadata is not None:
            meta_records = agg_result.imputation_metadata.as_dataframe().to_dict("records")
        if imputers:
            for rec in meta_records:
                step = int(rec.get("step", -1))
                if 0 <= step < len(imputers):
                    imp = imputers[step]
                    rec["config"] = imp.config() if hasattr(imp, "config") else {}

        # 4. summary numbers
        n_total   = int(agg_result.f_agg.shape[0])
        n_imputed = int(len(log_df))
        n_after   = int(np.isfinite(agg_result.f_agg).sum())
        n_before  = int(agg_result.coverage.sum())

        summary = (stage_summary or {}) | {
            "n_total":          n_total,
            "n_covered_before": n_before,
            "n_covered_after":  n_after,
            "n_imputed":        n_imputed,
            "n_steps":          len(meta_records),
        }

        # 5. sidecar
        sidecar = build_sidecar(
            run_id        = run_id,
            stage         = self.STAGE,
            label         = label,
            created_at    = created_at,
            parents       = parents,
            stage_config  = stage_config,
            stage_summary = summary,
            extra         = {"imputers": meta_records},
        )
        self._write_sidecar(run_dir, sidecar)

        # 6. index + save (imp registry auto-saves on register, like before)
        self._add_entry(run_id=run_id, label=label, created_at=created_at)
        self.save()

        # 7. stamp run_id back for downstream provenance
        agg_result._imp_run_id = run_id

        return run_id

    # ── loaders ───────────────────────────────────────────────────────

    def load(self, run_id: str):
        """Reconstruct an ``AggregationResult`` with attached log."""
        from pysuto.agg import AggregationResult
        from pysuto.impute import ImputationLog

        if run_id not in self._index:
            raise KeyError(f"Imputation run '{run_id}' not in registry")

        run_dir = os.path.join(self.storage_dir, run_id)
        data    = np.load(os.path.join(run_dir, "agg_result.npz"), allow_pickle=True)

        log_path = os.path.join(run_dir, "imputation_log.parquet")
        log_obj  = ImputationLog()
        if os.path.exists(log_path):
            df = pd.read_parquet(log_path)
            log_obj.extend(df.to_dict("records"))

        result = AggregationResult(
            f_agg     = data["f_agg"],
            sigma_agg = data["sigma_agg"],
            Omega     = data["Omega"],
            F         = data["F"],
            Sigma     = data["Sigma"],
            feed_ids  = list(data["feed_ids"]),
            coverage  = data["coverage"].astype(bool),
        )
        result.imputation_log = log_obj
        result._imp_run_id    = run_id
        return result

    def load_sidecar(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"Imputation run '{run_id}' not in registry")
        return self._read_sidecar(run_id)

    def ids(self) -> List[str]:
        return list(self._index.keys())

    def summary_df(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame()
        rows = []
        for run_id, entry in self._index.items():
            row = entry.to_index_dict()
            try:
                sidecar = self._read_sidecar(run_id)
                summary = sidecar.get("stage_summary", {})
                row.update({
                    "n_total":          summary.get("n_total"),
                    "n_imputed":        summary.get("n_imputed"),
                    "n_covered_after": summary.get("n_covered_after"),
                    "feed_run_ids":     (sidecar.get("parents", {})
                                                 .get("upstream", {})
                                                 .get("feed_run_ids", [])),
                })
            except Exception as exc:
                warnings.warn(f"Could not load sidecar for '{run_id}': {exc}")
            rows.append(row)
        return pd.DataFrame(rows)
