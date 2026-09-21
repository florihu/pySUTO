"""Round P.0 pre-solver gate (ADR 0031) — run on the const artifacts, before Stage 4.

The Round O `--feed-only` discipline: everything checkable without a solve is
checked without one, because a solve costs ~25 minutes and ~0.5 GB.

    python scripts/explo/roundP_gate.py <const_run_id>

Exits non-zero if any gate fails.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/proc")


def main(const: str) -> int:
    rm = pd.read_parquet(ROOT / "const" / const / "row_meta.parquet")
    imp = json.load(open(ROOT / "const" / const / "metadata.json"))[
        "parents"]["upstream"]["imp_run_id"]
    z = np.load(ROOT / "imp" / imp / "agg_result.npz", allow_pickle=True)
    f = np.nan_to_num(np.asarray(z["f_agg"], dtype=float))
    cov = np.asarray(z["coverage"], dtype=bool)

    sys.path.insert(0, "src")
    from pysuto.acc import AccountingStructure
    from pysuto.config import load_config
    from pysuto.param_relation import expand_edge_index
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)

    scrap = rm[rm.variant.isin(["smelterscrap", "directrefscrap"])]
    ok = True

    def check(tag, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"   {'PASS' if cond else '*** FAIL':>8}  {tag}  {detail}")

    print(f"Round P.0 gate — const {const}\n  imp {imp}")
    n = scrap.variant.value_counts().to_dict()
    check("G1 row counts by variant", len(scrap) > 0, str(n))

    check("G2 CC only", set(scrap.valuation) == {"CC"},
          f"valuations={sorted(set(scrap.valuation))}")

    # G3: c re-derived independently from f_agg, per (route, region, year)
    m = ei[ei.sector_from.eq("Scrap") & ei.valuation.eq("CC")
           & ei.sector_to.isin(["Smelting", "CFR"])].copy()
    m["f"] = f[m.edge_id.values]
    m["c"] = cov[m.edge_id.values]
    obs = m[m["c"] & (m.f.abs() > 1e-9)]
    want = obs.groupby(["sector_to", "region_to", "time"]).f.sum()
    check("G3 expected row population", len(want) == len(scrap),
          f"f_agg says {len(want)}, row_meta has {len(scrap)}")
    tot = obs.groupby("sector_to").f.sum().round(0).to_dict()
    print(f"             Σ reported: {tot}")

    # G5: row exists  <=>  a feed observed a non-zero level
    key_rows = set(zip(scrap.sector.astype(str), scrap.region.astype(str),
                       scrap.time.astype(str)))
    key_feed = set((a, b, str(c)) for a, b, c in want.index)
    check("G5 row exists <=> feed-observed", key_rows == key_feed,
          f"rows-not-feed {len(key_rows - key_feed)}, feed-not-rows {len(key_feed - key_rows)}")
    regions_all = set(ei.region_to.unique())
    regions_anchored = set(scrap.region.astype(str))
    print(f"             {len(regions_all - regions_anchored)} of {len(regions_all)} "
          f"regions carry no scrap anchor at all (unstated, not forbidden)")

    # G4/G6 need the G matrix
    import scipy.sparse as sp
    G = sp.load_npz(ROOT / "const" / const / "G.npz").tocsr()
    idx = scrap.index.to_numpy()
    mins = []
    for tf in sorted((ROOT / "const" / const / "timesteps").iterdir()):
        sc = np.asarray(np.load(tf)["sigma_c"])[idx]
        act = sc[sc < 1e9]            # year-locked rows are deactivated at 1e10
        if act.size:
            mins.append(float(act.min()))
    check("G4 ADR 0025 σ floor (all years)", bool(mins) and min(mins) >= 0.3 - 1e-9,
          f"min active σ_c over 30 y = {min(mins):.4f}" if mins else "no active rows")

    smelt_e = set(ei[ei.sector_to.eq("Smelting")].edge_id)
    cfr_e = set(ei[ei.sector_to.eq("CFR")].edge_id)
    spans = 0
    for r in idx:
        cols = set(G.indices[G.indptr[r]:G.indptr[r + 1]].tolist())
        if (cols & smelt_e) and (cols & cfr_e):
            spans += 1
    check("G6 no row spans both routes", spans == 0, f"{spans} rows span both")

    print(f"\n   G7 sum rows now {int((rm.type == 'sum').sum()):,} "
          f"(+{len(scrap):,} scrap-level rows)")
    print("\n" + ("GATE PASSED — safe to solve" if ok else "GATE FAILED — do not solve"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
