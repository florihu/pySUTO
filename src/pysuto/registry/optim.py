"""
optim.py
========
``OptimRunRegistry`` — Stage-4 output registry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ._sidecar import _now_iso, build_sidecar
from .base import BaseOutputRegistry

if TYPE_CHECKING:                                  # break Stage-4 import cycle
    from pysuto.solver import OptimResult


def warn_if_prefix_block_run(results: dict, const_run) -> None:
    """No-op under the structural model.

    The pre-refactor ``G_blocks`` partition and its cross-year placeholder
    failure mode (REP §8.2) are gone: :class:`pysuto.constraints.StructG` has
    no time axis on the column side, so every per-period optimum is naturally
    year-local. The function is retained as a stable import name; older
    callers see no warnings."""
    return


class OptimRunRegistry(BaseOutputRegistry):
    """Stage-4 registry — one run = one persisted ``dict[ts, OptimResult]``.

    Per-run directory layout::

        data/proc/opt/
        ├── registry.json
        └── opt_<ts>_<code>/
            ├── metadata.json          ← harmonized sidecar
            └── timesteps/
                └── t_{label}.npz      (F_opt, eps_opt, c_star, residuals)

    Per-timestep diagnostics (objective, iterations, solver_status,
    convergence) live in the sidecar's ``stage_summary.per_timestep``
    so ``load_sidecar`` is enough to rebuild the dashboard without
    touching the arrays.
    """

    STAGE = "opt"

    def __init__(self, storage_dir: str = "data/proc/opt") -> None:
        super().__init__(storage_dir=storage_dir)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.load_index()

    def register(
        self,
        results: "dict[str, OptimResult]",
        *,
        parents:       dict,
        stage_config:  Optional[dict] = None,
        stage_summary: Optional[dict] = None,
        label:         str = "",
    ) -> str:
        """Mint run_id, write per-timestep .npz, write sidecar, update index.

        ``parents`` should look like
        ``{"inputs": {}, "upstream": {"const_run_id": "..."}}``.

        ``stage_summary`` is merged with ``{timesteps, per_timestep}`` —
        callers typically pass ``per_timestep`` themselves and let the
        registry append the ordered ``timesteps`` list.
        """
        run_id     = self._new_run_id()
        created_at = _now_iso()
        run_dir    = self._make_run_dir(run_id)
        ts_dir     = os.path.join(run_dir, "timesteps")
        Path(ts_dir).mkdir(parents=True, exist_ok=True)

        # ── 1. per-timestep arrays ────────────────────────────────────
        for t, res in results.items():
            arrays = {
                "a_star":    res.a_star,
                "eps_opt":   res.eps_opt,
                "c_star":    res.c_star,
                "residuals": res.residuals,
            }
            if res.iter_history is not None:
                arrays["iter_history"] = res.iter_history
            if res.active is not None:
                arrays["active"] = res.active
            np.savez(os.path.join(ts_dir, f"t_{t}.npz"), **arrays)

        # ── 2. summary (caller's per_timestep + ordered keys) ─────────
        summary = (stage_summary or {}) | {"timesteps": list(results.keys())}
        if "per_timestep" not in summary:
            summary["per_timestep"] = {
                t: {
                    "objective":     res.objective,
                    "iterations":    res.iterations,
                    "solve_time_s":  res.solve_time_s,
                    "solver_status": res.solver_status,
                    "convergence":   res.convergence,
                }
                for t, res in results.items()
            }

        # ── 3. sidecar ────────────────────────────────────────────────
        sidecar = build_sidecar(
            run_id        = run_id,
            stage         = self.STAGE,
            label         = label,
            created_at    = created_at,
            parents       = parents,
            stage_config  = stage_config,
            stage_summary = summary,
        )
        self._write_sidecar(run_dir, sidecar)

        # ── 4. index ──────────────────────────────────────────────────
        self._add_entry(run_id=run_id, label=label, created_at=created_at)
        self.save()
        return run_id

    def load(self, run_id: str) -> "dict[str, OptimResult]":
        """Reconstruct ``{timestep: OptimResult}`` from .npz + sidecar."""
        from pysuto.solver import OptimResult        # lazy: break cycle
        if run_id not in self._index:
            raise KeyError(f"Optim run '{run_id}' not in registry")

        run_dir = os.path.join(self.storage_dir, run_id)
        ts_dir  = os.path.join(run_dir, "timesteps")
        sidecar = self._read_sidecar(run_id)
        summary = sidecar.get("stage_summary", {})
        per_ts  = summary.get("per_timestep", {})

        results: Dict[str, "OptimResult"] = {}
        for t in summary.get("timesteps", []):
            npz = np.load(os.path.join(ts_dir, f"t_{t}.npz"))
            per = per_ts.get(t, {})
            # ``a_star`` is the canonical field name under the structural
            # refactor; older runs persisted ``F_opt`` — read either.
            a_star = npz["a_star"] if "a_star" in npz.files else npz["F_opt"]
            results[t] = OptimResult(
                timestep      = t,
                a_star        = a_star,
                eps_opt       = npz["eps_opt"],
                c_star        = npz["c_star"],
                residuals     = npz["residuals"],
                objective     = float(per.get("objective", 0.0)),
                iterations    = int(per.get("iterations", 0)),
                solve_time_s  = float(per.get("solve_time_s", 0.0)),
                solver_status = str(per.get("solver_status", "")),
                convergence   = dict(per.get("convergence", {})),
                iter_history  = npz["iter_history"] if "iter_history" in npz.files else None,
                active        = npz["active"] if "active" in npz.files else None,
            )
        return results

    def load_sidecar(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"Optim run '{run_id}' not in registry")
        return self._read_sidecar(run_id)
