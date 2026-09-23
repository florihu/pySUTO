"""
psut.py
=======
``PsutRunRegistry`` — Stage-6 output registry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ._sidecar import _now_iso, build_sidecar
from .base import BaseOutputRegistry

if TYPE_CHECKING:                                  # break Stage-6 import cycle
    from pysuto.psut import PSUT


class PsutRunRegistry(BaseOutputRegistry):
    """Stage-6 registry — one run = one persisted tidy :class:`~pysuto.psut.PSUT`.

    Per-run directory layout::

        data/proc/psut/
        ├── registry.json
        └── psut_<ts>_<code>/
            ├── metadata.json            ← harmonized sidecar
            ├── nodes.parquet            ← node universe the views are labelled from
            └── timesteps/
                └── t_{period}.parquet   ← the tidy rows of that period

    ``stage_config`` records the valuation and the blocks materialised;
    ``stage_summary.per_timestep`` the row count and value total per block,
    so ``load_sidecar`` alone says what a run holds. ``parents.upstream``
    carries ``opt_run_id`` (and ``const_run_id``), ``parents.inputs.acc``
    the accounting structure id.
    """

    STAGE = "psut"

    def __init__(self, storage_dir: str = "data/proc/psut") -> None:
        super().__init__(storage_dir=storage_dir)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.load_index()

    def register(
        self,
        psut: "PSUT",
        *,
        parents:       dict,
        stage_config:  Optional[dict] = None,
        stage_summary: Optional[dict] = None,
        label:         str = "",
    ) -> str:
        """Mint run_id, write one parquet per period + the node universe,
        write the sidecar, update the index.

        ``parents`` should look like
        ``{"inputs": {"acc": {"id": ...}}, "upstream": {"opt_run_id": ...,
        "const_run_id": ...}}``.
        """
        run_id     = self._new_run_id()
        created_at = _now_iso()
        run_dir    = self._make_run_dir(run_id)
        ts_dir     = os.path.join(run_dir, "timesteps")
        Path(ts_dir).mkdir(parents=True, exist_ok=True)

        # ── 1. per-period tidy rows + node universe ───────────────────
        periods = psut.periods()
        per_timestep = {}
        for t in periods:
            rows = psut.table[psut.table["period"] == t]
            rows.to_parquet(os.path.join(ts_dir, f"t_{t}.parquet"), index=False)
            per_timestep[t] = {
                "n_rows":       int(len(rows)),
                "val_by_block": {b: float(v) for b, v in
                                 rows.groupby("block")["val"].sum().items()},
            }
        psut.nodes.to_parquet(os.path.join(run_dir, "nodes.parquet"), index=False)

        # ── 2. config + summary ───────────────────────────────────────
        config  = (stage_config or {}) | {
            "valuation": psut.valuation,
            "blocks":    sorted(psut.table["block"].unique().tolist()),
        }
        summary = (stage_summary or {}) | {"timesteps": periods}
        if "per_timestep" not in summary:
            summary["per_timestep"] = per_timestep

        # ── 3. sidecar ────────────────────────────────────────────────
        sidecar = build_sidecar(
            run_id        = run_id,
            stage         = self.STAGE,
            label         = label,
            created_at    = created_at,
            parents       = parents,
            stage_config  = config,
            stage_summary = summary,
        )
        self._write_sidecar(run_dir, sidecar)

        # ── 4. index ──────────────────────────────────────────────────
        self._add_entry(run_id=run_id, label=label, created_at=created_at)
        self.save()
        return run_id

    def load(self, run_id: str) -> "PSUT":
        """Reconstruct a :class:`~pysuto.psut.PSUT` from the parquet files + sidecar."""
        from pysuto.psut import PSUT                 # lazy: break cycle
        if run_id not in self._index:
            raise KeyError(f"PSUT run '{run_id}' not in registry")

        run_dir = os.path.join(self.storage_dir, run_id)
        ts_dir  = os.path.join(run_dir, "timesteps")
        sidecar = self._read_sidecar(run_id)
        summary = sidecar.get("stage_summary", {})
        config  = sidecar.get("stage_config", {})
        parents = sidecar.get("parents", {})

        frames = [pd.read_parquet(os.path.join(ts_dir, f"t_{t}.parquet"))
                  for t in summary.get("timesteps", [])]
        nodes  = pd.read_parquet(os.path.join(run_dir, "nodes.parquet"))

        return PSUT(
            pd.concat(frames, ignore_index=True),
            nodes,
            valuation    = str(config.get("valuation", "")),
            acc_id       = str(parents.get("inputs", {}).get("acc", {}).get("id", "")),
            opt_run_id   = parents.get("upstream", {}).get("opt_run_id"),
            const_run_id = parents.get("upstream", {}).get("const_run_id"),
        )

    def load_sidecar(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"PSUT run '{run_id}' not in registry")
        return self._read_sidecar(run_id)
