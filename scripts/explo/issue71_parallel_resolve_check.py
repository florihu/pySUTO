"""
issue71_parallel_resolve_check.py
=================================
#71 measurement of #70: the parallel cold solve against the serial baseline on the same
constraint run. Per period it compares status, objective, iterations, the ``F_opt`` slice
(``np.array_equal``, together with ``cols_t``) and ``solve_time_s``; it prints Stage 4's
phases (load, probe, solves, persist + diag) from each run's
``logs/init_run_solver_*.log``, and each run's solve configuration from its sidecar.
Further opt run ids get the sidecar check only (#71's maintainer comment: the milestone
run's ``n_workers``, ``max_threads``, ``direct_solve_method``, statuses and
``solve_wall_s``).

    python scripts/explo/issue71_parallel_resolve_check.py <opt_new> <opt_base> \\
                                                           [<opt> ...]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OPT  = Path("data/proc/opt")
LOGS = Path("logs")
OK   = {"Solved", "AlmostSolved"}
TS   = "%Y-%m-%d %H:%M:%S,%f"
MARK = {"start":     "init_run_solver START",
        "probe":     "Feasibility pre-check on",
        "passed":    "Feasibility pre-check passed",
        "persisted": "persisted to data/proc/opt",
        "done":      "init_run_solver DONE"}


def _meta(opt: str) -> dict:
    return json.loads((OPT / opt / "metadata.json").read_text())


def _periods(meta: dict) -> pd.DataFrame:
    """Per period: status, objective, iterations and solve time, from a run's sidecar."""
    cols = ("solver_status", "objective", "iterations", "solve_time_s")
    per = meta["stage_summary"]["per_timestep"]
    return pd.DataFrame({t: {k: v[k] for k in cols} for t, v in per.items()}).T


def _log(opt: str) -> Path:
    """The per-run solver log that saved ``opt``."""
    hits = [p for p in sorted(LOGS.glob("init_run_solver_*.log"))
            if f"OptimRun saved as '{opt}'" in p.read_text(errors="replace")]
    if len(hits) != 1:
        raise SystemExit(f"{opt}: {len(hits)} solver logs name it, expected 1")
    return hits[0]


def _phases(opt: str) -> dict:
    """Stage 4's wall-clock phases in seconds, from the run's log markers."""
    log = _log(opt)
    stamps, periods = {}, []
    for line in log.read_text(errors="replace").splitlines():
        match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) \|", line)
        if not match:
            continue
        t = datetime.strptime(match.group(1), TS)
        if re.search(r"\| t=\d{4}\s+status=", line):
            periods.append(t)
        for name, text in MARK.items():
            if text in line:
                stamps[name] = t
    missing = [name for name in MARK if name not in stamps]
    missing += [] if periods else ["periods"]
    if missing:
        raise SystemExit(f"{opt}: solver log {log} lacks {missing}")
    last = max(periods)

    def sec(a: datetime, b: datetime) -> float:
        return (b - a).total_seconds()

    return {"load":           sec(stamps["start"], stamps["probe"]),
            "probe":          sec(stamps["probe"], stamps["passed"]),
            "solves":         sec(stamps["passed"], last),
            "persist + diag": sec(last, stamps["done"]),
            "of it persist":  sec(last, stamps["persisted"]),
            "total":          sec(stamps["start"], stamps["done"])}


def sidecar(opt: str) -> None:
    """The run's constraint run, solve configuration, statuses and solve wall-clock."""
    meta = _meta(opt)
    cfg, summary = meta.get("stage_config", {}), meta["stage_summary"]
    status = _periods(meta).solver_status
    const = meta["parents"]["upstream"].get("const_run_id")
    print(f"   {opt}: const {const}, prior_mode {cfg.get('prior_mode')}, n_workers "
          f"{cfg.get('n_workers')}, max_threads {cfg.get('max_threads')}, "
          f"direct_solve_method {cfg.get('direct_solve_method')}; statuses "
          f"{status.value_counts().to_dict()} (all Solved/AlmostSolved: "
          f"{bool(status.isin(OK).all())}); solve_wall_s {summary.get('solve_wall_s')}")


def compare(new: str, base: str) -> None:
    """Per period: status, objective, iterations, F_opt bit-identity, solve time."""
    meta_new, meta_base = _meta(new), _meta(base)
    const_new = meta_new["parents"]["upstream"].get("const_run_id")
    const_base = meta_base["parents"]["upstream"].get("const_run_id")
    if const_new != const_base:
        raise SystemExit(f"{new} solves {const_new}, {base} solves {const_base}")
    p_new, p_base = _periods(meta_new), _periods(meta_base)
    rows = []
    for t in sorted(p_base.index):
        t_new = np.load(OPT / new / "timesteps" / f"t_{t}.npz", allow_pickle=True)
        t_base = np.load(OPT / base / "timesteps" / f"t_{t}.npz", allow_pickle=True)
        same = (np.array_equal(t_new["cols_t"], t_base["cols_t"])
                and np.array_equal(t_new["F_opt"], t_base["F_opt"]))
        rows.append({
            "period":              t,
            "status":              f"{p_base.solver_status[t]} / "
                                   f"{p_new.solver_status[t]}",
            "objective equal":     p_base.objective[t] == p_new.objective[t],
            "iterations":          f"{p_base.iterations[t]} / {p_new.iterations[t]}",
            "F_opt bit-identical": same,
            "solve_time_s":        f"{p_base.solve_time_s[t]:.1f} / "
                                   f"{p_new.solve_time_s[t]:.1f}"})
    table = pd.DataFrame(rows)
    print(f"\n== per period on {const_new}, {base} / {new}")
    print(table.to_string(index=False))
    n, s_base, s_new = len(table), p_base.solve_time_s, p_new.solve_time_s
    print(f"\n   periods bit-identical {int(table['F_opt bit-identical'].sum())}/{n}; "
          f"objectives equal {int(table['objective equal'].sum())}/{n}\n   solve_time_s "
          f"Σ {s_base.sum():,.0f} / {s_new.sum():,.0f}, per period "
          f"{s_base.min():.1f}–{s_base.max():.1f} / {s_new.min():.1f}–{s_new.max():.1f}; "
          f"Σ new / Σ base {s_new.sum() / s_base.sum():.3f}")
    phases = pd.DataFrame({base: _phases(base), new: _phases(new)}).T
    print("\n== Stage 4 phases, s (from the runs' solver logs)")
    print(phases.round(1).to_string())


def main(new: str, base: str, others: list) -> None:
    print("== sidecars")
    for opt in [new, base] + others:
        sidecar(opt)
    compare(new, base)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
