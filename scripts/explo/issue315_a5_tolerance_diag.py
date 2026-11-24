"""
issue315_a5_tolerance_diag.py
=============================
#315: termination accuracy on the reference const. Three opt runs share one const run:
loose (flows form, gap_rel 1e-3, the settings of ``opt_20260916_124721_c4d1d944``), tight
(flows form, gap_rel 1e-6, as ``opt_20260930_093914_523d57d7``) and the reference (defects
form, 1e-8). Reads persisted run artefacts only and prints #315's pre-registered items:

    T1  solve status per year and the tolerances each sidecar records
    A5  |Δ retention| per stage (30-year pool), loose vs tight, against 0.02 (Table 3)
    Q1  the same, tight vs reference (committee Q1, 2026-11-10); reported, not gating
    J   Σ_t objective per run against the reference (one const, so the prior constant
        cancels), Chile's concentrate export retention, the Round-E items and the
        common-set sum-row z² of each run against the reference

    python scripts/explo/issue315_a5_tolerance_diag.py <opt_loose> <opt_tight> <opt_reference>

Run from the repository root of the checkout that holds ``data/proc``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
sys.path.insert(0, "scripts/report")
from pysuto import acc_store                                # noqa: E402
from pysuto.acc import AccountingStructure                  # noqa: E402
import copper_metabolism_insights as CMI                    # noqa: E402
import issue69_aggregate_gap_diag as P69                    # noqa: E402
from results_figs import THRESH, chile_export_retention, retention_by_stage  # noqa: E402

ROOT     = Path("data/proc")
TOL_KEYS = ("qp_variables", "tol_feas", "tol_gap_rel", "tol_gap_abs", "reduced_tol_feas",
            "reduced_tol_gap_rel", "reduced_tol_gap_abs", "max_iter")


def _sidecar(opt: str) -> dict:
    with open(ROOT / "opt" / opt / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def solve(opt: str) -> tuple:
    """T1: status per year, Σ_t objective, the const run and the recorded tolerances."""
    m   = _sidecar(opt)
    pt  = m["stage_summary"]["per_timestep"]
    st  = pd.Series({t: v["solver_status"] for t, v in pt.items()})
    obj = sum(v["objective"] for v in pt.values())
    cfg = {k: m["stage_config"].get(k, "flows (not recorded)" if k == "qp_variables" else None)
           for k in TOL_KEYS}
    const = m["parents"]["upstream"]["const_run_id"]
    print(f"\n== T1 {opt} ({m.get('label')}): {st.value_counts().to_dict()} over {len(st)} "
          f"years; solve_wall_s {m['stage_summary'].get('solve_wall_s', float('nan')):.0f}; "
          f"Σ_t objective {obj:,.1f}; const {const}\n   {cfg}")
    return obj, const


def pair(ret: dict, a: str, b: str, item: str) -> None:
    """|Δ retention| per stage on the 30-year pool, a → b, against the A5 band."""
    pa = ret[a][ret[a].year == "all"].set_index("stage")
    pb = ret[b][ret[b].year == "all"].set_index("stage")
    d  = pd.DataFrame({"retention_a": pa.retention, "retention_b": pb.retention})
    d["abs_delta"] = (d.retention_b - d.retention_a).abs()
    n_fail = int((d.abs_delta >= THRESH["A5_delta"]).sum())
    print(f"\n== {item}: a = {a}, b = {b}\n   {len(d) - n_fail}/{len(d)} stages within "
          f"{THRESH['A5_delta']:.2f}; max {d.abs_delta.max():.3f} ({d.abs_delta.idxmax()}) "
          f"→ {'PASS' if n_fail == 0 else 'FAIL'}")
    print(d.round(4).to_string())


def main(loose: str, tight: str, ref: str) -> None:
    runs = (loose, tight, ref)
    out  = {o: solve(o) for o in runs}
    if len({c for _, c in out.values()}) != 1:
        raise ValueError(f"the three runs solve different const runs: {out}")

    edges = {o: CMI.load_run(o) for o in runs}
    ret   = {o: retention_by_stage(e) for o, e in edges.items()}
    pair(ret, loose, tight, "A5 (loose vs tight)")
    pair(ret, tight, ref, "Q1 (tight vs reference)")
    pair(ret, loose, ref, "loose vs reference")

    print(f"\n== J vs the reference {ref}")
    for o in runs:
        d = out[o][0] - out[ref][0]
        print(f"   {o}: Σ_t objective {out[o][0]:,.1f} (Δ {d:+,.1f}); Chile concentrate "
              f"export retention 2010–2019 {chile_export_retention(edges[o]):.3f}")

    AccountingStructure.load = staticmethod(acc_store.load)   # the older helpers call it
    for o in (loose, tight):
        print(f"\n###### {o} vs the reference {ref}")
        E, S, j = P69.load(ref, o)
        P69.round_e(E, S, j, ref, o)
        P69.common_set(j, ref)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
