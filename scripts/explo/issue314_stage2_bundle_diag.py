# Frozen record: last ran against commit 723c526; rerun by checking out that commit (see README.md).
"""
issue314_stage2_bundle_diag.py
==============================
#314 measurement of #312: a later imputer writes no primary a relaxed recipe close counted
``assumed_absent``; only ``flat_prior`` (``last_resort``) may. Reads persisted run artefacts
only and prints #314's pre-registered items for <new> against <base>:

    R1  solve status per year, the chains' ids
    D1  the new imp's assumed-absent primaries and who wrote them after the relaxed close;
        the base imp's time_interp writes on them
    D2  every edge-year whose prior (opt ``a_prior``) differs from the base, by cause
    D3  the free-sink guard: posterior a* of the assumed-absent edges against their base
        prior, and the sum for the edges ``time_interp`` filled in the base
    D4  the Round-E items (ADR 0025) against the base
    D5  common-set sum-row z² against the base and against <reference>

    python scripts/explo/issue314_stage2_bundle_diag.py <opt_base> <opt_new> <opt_reference>

Run from the repository root of the checkout that holds ``data/proc``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
from pysuto import acc_store                                # noqa: E402
from pysuto.acc import AccountingStructure                  # noqa: E402
from pysuto.config import load_config                       # noqa: E402
import issue69_aggregate_gap_diag as P69                    # noqa: E402

ROOT         = Path("data/proc")
SINK_TOL_KT  = 1.0
BASE_SUM_KT  = None     # the base's time_interp sum, derived in D1


def _sidecar(stage: str, run: str) -> dict:
    with open(ROOT / stage / run / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _imp_of(opt: str) -> str:
    const = _sidecar("opt", opt)["parents"]["upstream"]["const_run_id"]
    return _sidecar("const", const)["parents"]["upstream"]["imp_run_id"]


def solve(opt: str) -> None:
    """R1: solve status per year and the chain."""
    m  = _sidecar("opt", opt)
    ss = m["stage_summary"]
    st = pd.Series({t: v["solver_status"] for t, v in ss["per_timestep"].items()})
    const = m["parents"]["upstream"]["const_run_id"]
    print(f"\n== R1 {opt}: {st.value_counts().to_dict()} over {len(st)} years; "
          f"solve_wall_s {ss['solve_wall_s']:.0f}; const {const}; imp {_imp_of(opt)}")


def _log(imp: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")


def _absent(d: pd.DataFrame) -> set:
    out: set = set()
    for a in d["assumed_absent"]:
        out.update(int(i) for i in a)
    return out


def _label(acc, edge_ids) -> pd.DataFrame:
    ei  = acc.edge_index.set_index("edge_id").loc[list(edge_ids)]
    nid = acc.node_index.reset_index(drop=True)
    return pd.DataFrame({"from": nid.sector.values[ei.from_id.values],
                         "to": nid.sector.values[ei.to_id.values],
                         "region": nid.region.values[ei.to_id.values],
                         "time": ei.time.values, "val": ei.valuation.values},
                        index=list(edge_ids))


def ledger(acc, imp_base: str, imp_new: str) -> tuple:
    """D1: who writes the assumed-absent primaries after the relaxed close."""
    out = {}
    for lab, imp in (("base", imp_base), ("new", imp_new)):
        d = _log(imp)
        ab = _absent(d)
        first = int(d[d.relaxed].step.min())
        later = d[(d.step > first) & d.edge_id.isin(ab)]
        w = later.groupby(["imputer_id", "method"]).size()
        print(f"\n== D1 {lab} {imp}: {len(ab)} assumed-absent primaries "
              f"({int(d.relaxed.sum())} relaxed steps, first at step {first}); writes on "
              f"them after it: {w.to_dict()}")
        ti = later[later.imputer_id != "flat_prior"]
        if len(ti):
            t = ti[["edge_id", "imputer_id", "method", "step", "f_after"]].copy()
            tag = _label(acc, t.edge_id.values)
            for c in tag.columns:
                t[c] = tag[c].to_numpy()
            print(t.round(4).to_string(index=False))
        out[lab] = (ab, ti)
    return out


def prior_diff(E: dict, ab_new: set, base: str, new: str) -> pd.DataFrame:
    """D2: edge-years whose prior differs from the base."""
    k = ["edge_id", "time"]
    j = E[base][k + ["prior", "post", "sector_from", "sector_to", "region_to",
                    "valuation"]].merge(E[new][k + ["prior", "post"]], on=k, how="outer",
                                        suffixes=("_base", "_new"))
    diff = j[~np.isclose(j.prior_base.fillna(-1), j.prior_new.fillna(-1), rtol=0.0,
                         atol=1e-12)]
    print(f"\n== D2 prior (opt a_prior) differs on {len(diff)} of {len(j):,} edge-years; "
          f"of them assumed-absent edges: {int(diff.edge_id.isin(ab_new).sum())}")
    if len(diff):
        print(diff.sort_values("prior_new").round(4).head(40).to_string(index=False))
    return j


def sink(j: pd.DataFrame, ab_new: set, ti_base: pd.DataFrame, base: str, new: str) -> None:
    """D3: the free-sink guard."""
    a = j[j.edge_id.isin(ab_new)].copy()
    a["dpost"] = a.post_new - a.prior_base.fillna(0.0)
    cc = a[a.valuation == "CC"]
    print(f"\n== D3 {len(a)} assumed-absent edge-years ({len(cc)} CC): posterior a* new "
          f"minus base prior; largest rises:")
    print(a.sort_values("dpost", ascending=False).head(8)[
        ["edge_id", "time", "sector_from", "sector_to", "region_to", "valuation",
         "prior_base", "prior_new", "post_base", "post_new", "dpost"]]
        .round(4).to_string(index=False))
    f = ti_base.edge_id.tolist()
    # the rule: an edge time_interp filled in the base may not exceed that value + 1 kt
    sub = a[a.edge_id.isin(f)]
    print(f"   the rule (edges time_interp filled in the base): breaches of a* new > base "
          f"prior + {SINK_TOL_KT:g} kt: {int((sub.dpost > SINK_TOL_KT).sum())} of {len(sub)}")
    # the broader screen: any assumed-absent edge whose a* rose against the base's a*
    a["rise"] = a.post_new - a.post_base
    print(f"   screen: assumed-absent edge-years whose a* new exceeds a* base by more than "
          f"{SINK_TOL_KT:g} kt: {int((a.rise > SINK_TOL_KT).sum())}; largest rise "
          f"{a.rise.max():+.4f} kt CC; edge-years with a* > {SINK_TOL_KT:g} kt in the base "
          f"already: {int((a.post_base > SINK_TOL_KT).sum())}")
    s = j[j.edge_id.isin(f)]
    print(f"   edges time_interp filled in the base: {f}; base prior Σ "
          f"{s.prior_base.sum():.3f}, base a* Σ {s.post_base.sum():.3f}, new prior Σ "
          f"{s.prior_new.sum():.3f}, new a* Σ {s.post_new.sum():.3f} kt CC")
    print(f"   all assumed-absent CC edge-years: Σ a* base {cc.post_base.sum():.3f}, "
          f"new {cc.post_new.sum():.3f} kt CC; Σ prior base {cc.prior_base.sum():.3f}, "
          f"new {cc.prior_new.sum():.3f}")


def main(base: str, new: str, ref: str) -> None:
    AccountingStructure.load = staticmethod(acc_store.load)   # the older helpers call it
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    for o in (ref, base, new):
        solve(o)
    out = ledger(acc, _imp_of(base), _imp_of(new))
    E, S, j = P69.load(base, new)
    pj = prior_diff(E, out["new"][0], base, new)
    sink(pj, out["new"][0], out["base"][1], base, new)
    print(f"\n###### D4, D5 vs the base {base}")
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    P69.attribution(j)
    print(f"\n###### D5 vs the reference {ref}")
    E2, S2, j2 = P69.load(ref, new)
    P69.round_e(E2, S2, j2, ref, new)
    P69.common_set(j2, ref)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
