"""
issue71_parallel_resolve_check.py
=================================
#71 measurement of #70: the parallel cold solve against the serial baseline on the same
constraint run. Per period it compares status, objective, iterations, the ``F_opt`` slice
(``np.array_equal``, together with ``cols_t``) and ``solve_time_s``; it prints Stage 4's
phases (load, probe, solves, persist + diag) from each run's ``logs/init_run_solver_*.log``,
and each run's solve configuration from its sidecar. Further opt run ids get the sidecar
check only (#71's maintainer comment: the milestone run's ``n_workers``, ``max_threads``,
``direct_solve_method``, statuses and ``solve_wall_s``).

    python scripts/explo/issue71_parallel_resolve_check.py <opt_new> <opt_base> [<opt> ...]
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


def _meta(opt: str) -> dict:
    return json.loads((OPT / opt / "metadata.json").read_text())


def _periods(opt: str) -> pd.DataFrame:
    pt = _meta(opt)["stage_summary"]["per_timestep"]
    return pd.DataFrame({t: {k: v[k] for k in ("solver_status", "objective", "iterations",
                                                "solve_time_s")} for t, v in pt.items()}).T


def _log(opt: str) -> Path:
    """The per-run solver log that saved ``opt``."""
    hits = [p for p in sorted(LOGS.glob("init_run_solver_*.log"))
            if f"OptimRun saved as '{opt}'" in p.read_text(errors="replace")]
    if len(hits) != 1:
        raise SystemExit(f"{opt}: {len(hits)} solver logs name it, expected 1")
    return hits[0]


def _phases(opt: str) -> dict:
    """Stage 4's wall-clock phases in seconds, from the run's log markers."""
    stamps = {"start": None, "probe": None, "passed": None, "done": None}
    periods = []
    for line in _log(opt).read_text(errors="replace").splitlines():
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) \|", line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), TS)
        if "init_run_solver START" in line:
            stamps["start"] = t
        elif "Feasibility pre-check on" in line:
            stamps["probe"] = t
        elif "Feasibility pre-check passed" in line:
            stamps["passed"] = t
        elif re.search(r"\| t=\d{4}\s+status=", line):
            periods.append(t)
        elif "init_run_solver DONE" in line:
            stamps["done"] = t
    last = max(periods)
    sec = lambda a, b: (b - a).total_seconds()
    return {"load": sec(stamps["start"], stamps["probe"]),
            "probe": sec(stamps["probe"], stamps["passed"]),
            "solves": sec(stamps["passed"], last),
            "persist + diag": sec(last, stamps["done"]),
            "total": sec(stamps["start"], stamps["done"])}


def sidecar(opt: str) -> None:
    """The run's solve configuration, its statuses and its solve wall-clock."""
    m = _meta(opt)
    cfg, summ = m.get("stage_config", {}), m["stage_summary"]
    st = _periods(opt).solver_status
    print(f"   {opt}: prior_mode {cfg.get('prior_mode')}, n_workers {cfg.get('n_workers')}, "
          f"max_threads {cfg.get('max_threads')}, direct_solve_method "
          f"{cfg.get('direct_solve_method')}; statuses {st.value_counts().to_dict()} "
          f"(all Solved/AlmostSolved: {bool(st.isin(OK).all())}); solve_wall_s "
          f"{summ.get('solve_wall_s')}")


def compare(new: str, base: str) -> None:
    """Per period: status, objective, iterations, F_opt bit-identity, solve time."""
    a, b = _periods(new), _periods(base)
    rows = []
    for t in sorted(b.index):
        za = np.load(OPT / new / "timesteps" / f"t_{t}.npz", allow_pickle=True)
        zb = np.load(OPT / base / "timesteps" / f"t_{t}.npz", allow_pickle=True)
        same = (np.array_equal(za["cols_t"], zb["cols_t"])
                and np.array_equal(za["F_opt"], zb["F_opt"]))
        rows.append({"period": t, "status": f"{b.solver_status[t]} / {a.solver_status[t]}",
                     "objective equal": b.objective[t] == a.objective[t],
                     "iterations": f"{b.iterations[t]} / {a.iterations[t]}",
                     "F_opt bit-identical": same,
                     "solve_time_s": f"{b.solve_time_s[t]:.1f} / {a.solve_time_s[t]:.1f}"})
    d = pd.DataFrame(rows)
    print(f"\n== per period, {base} / {new}")
    print(d.to_string(index=False))
    n = len(d)
    print(f"\n   periods bit-identical {int(d['F_opt bit-identical'].sum())}/{n}; objectives "
          f"equal {int(d['objective equal'].sum())}/{n}; Σ solve_time_s "
          f"{b.solve_time_s.sum():,.0f} / {a.solve_time_s.sum():,.0f}")
    ph = pd.DataFrame({base: _phases(base), new: _phases(new)}).T
    print("\n== Stage 4 phases, min (from the runs' solver logs)")
    print((ph / 60).round(2).to_string())


def main(new: str, base: str, others: list) -> None:
    print("== sidecars")
    for o in [new, base] + others:
        sidecar(o)
    compare(new, base)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
