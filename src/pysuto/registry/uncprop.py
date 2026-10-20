"""
uncprop.py
==========
``UncPropRunRegistry`` — Stage-5 output registry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ._sidecar import _now_iso, build_sidecar
from .base import BaseOutputRegistry

if TYPE_CHECKING:                                  # break Stage-5 import cycle
    from pysuto.uncprop import UncPropResult


class UncPropRunRegistry(BaseOutputRegistry):
    """Stage-5 registry — one run = one persisted ``UncPropResult``.

    Per-run directory layout::

        data/proc/uncprop/
        ├── registry.json
        └── uncprop_<ts>_<code>/
            ├── metadata.json                ← harmonized sidecar
            └── timesteps/
                └── t_{label}.npz            (sigma_a_star, lower, upper
                                                  [, cols_t, bound_active])

    Diagnostics scalars (``trace_prior``, ``trace_post``, ``max_resid_hard``,
    ``psd_min_diag``) live in the sidecar's ``stage_summary.per_timestep``
    so ``load_sidecar`` alone reconstructs a trace-contraction dashboard without
    touching the arrays. ``a*`` is *not* persisted here — readers join on
    the upstream ``opt_run_id`` recorded in ``parents.upstream``.
    """

    STAGE = "uncprop"

    def __init__(self, storage_dir: str = "data/proc/uncprop") -> None:
        super().__init__(storage_dir=storage_dir)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.load_index()

    def register(
        self,
        result: "UncPropResult",
        *,
        parents:       dict,
        stage_config:  Optional[dict] = None,
        stage_summary: Optional[dict] = None,
        label:         str = "",
    ) -> str:
        """Mint run_id, write per-timestep .npz, write sidecar, update index.

        ``parents`` should look like
        ``{"inputs": {}, "upstream": {"opt_run_id": ..., "const_run_id": ...}}``.
        """
        run_id     = self._new_run_id()
        created_at = _now_iso()
        run_dir    = self._make_run_dir(run_id)
        ts_dir     = os.path.join(run_dir, "timesteps")
        Path(ts_dir).mkdir(parents=True, exist_ok=True)

        # ── 1. per-timestep arrays ────────────────────────────────────
        for t, ts in result.per_timestep.items():
            arrays = {
                "sigma_a_star": ts.sigma_a_star,
                "lower":        ts.lower,
                "upper":        ts.upper,
            }
            # Block path (year-local) carries cols_t and its active set; legacy omits them.
            if ts.cols_t is not None:
                arrays["cols_t"] = ts.cols_t
            if ts.bound_active is not None:
                arrays["bound_active"] = ts.bound_active
            np.savez(os.path.join(ts_dir, f"t_{t}.npz"), **arrays)

        # ── 2. summary (caller's per_timestep + ordered keys) ─────────
        summary = (stage_summary or {}) | {"timesteps": list(result.per_timestep.keys())}
        if "per_timestep" not in summary:
            summary["per_timestep"] = {
                t: dict(ts.diagnostics) for t, ts in result.per_timestep.items()
            }

        # ── 3. sidecar ────────────────────────────────────────────────
        config = (stage_config or {}) | {
            "method": result.method,
            "alpha":  result.alpha,
        }
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

    def load(self, run_id: str) -> "UncPropResult":
        """Reconstruct an ``UncPropResult`` from .npz + sidecar."""
        from pysuto.uncprop import UncPropResult, UncPropTimestep   # lazy: break cycle
        if run_id not in self._index:
            raise KeyError(f"UncProp run '{run_id}' not in registry")

        run_dir = os.path.join(self.storage_dir, run_id)
        ts_dir  = os.path.join(run_dir, "timesteps")
        sidecar = self._read_sidecar(run_id)
        summary = sidecar.get("stage_summary", {})
        config  = sidecar.get("stage_config", {})
        per_ts  = summary.get("per_timestep", {})

        per_timestep: dict[str, UncPropTimestep] = {}
        for t in summary.get("timesteps", []):
            npz = np.load(os.path.join(ts_dir, f"t_{t}.npz"))
            per_timestep[t] = UncPropTimestep(
                timestep     = t,
                sigma_a_star = npz["sigma_a_star"],
                lower        = npz["lower"],
                upper        = npz["upper"],
                diagnostics  = dict(per_ts.get(t, {})),
                cols_t       = npz["cols_t"] if "cols_t" in npz.files else None,
                bound_active = (npz["bound_active"] if "bound_active" in npz.files
                                else None),
            )

        return UncPropResult(
            method       = str(config.get("method", "")),
            alpha        = float(config.get("alpha", 0.0)),
            per_timestep = per_timestep,
        )

    def load_sidecar(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"UncProp run '{run_id}' not in registry")
        return self._read_sidecar(run_id)
