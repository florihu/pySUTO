"""
roundL_reexport_diag.py
=======================
Round L measurement: L.0 (bilateral legs are not time-interpolated, ADR 0026)
and L.1 (re-exports re-attributed at feed time, ADR 0027), each against its
predecessor. Reads persisted run artifacts only — every upstream id is derived
from the run's own sidecar, never hard-coded (the Round C hygiene lesson: a
pinned feed id silently measured the wrong chain for three rounds).

    python scripts/explo/roundL_reexport_diag.py <opt_base> <opt_L0> [<opt_L1>]

Stop rule (plan 2026-09-15):
  L.0 — cross-region `time_linear` fills must reach 0; exporter retention
        (Chile/Peru/Indonesia) and world retention within ±0.01; 30/30 Solved.
  L.1 — phantom production at P=0 hubs falls by ≈ the applied T where the
        ledger says pass-through; hub∪fabricated z² on the BASELINE class set
        falls WITH commensurate moved mass (Round E: metric without mass is a
        stop); Zambia→CH/SG/LU retention stays 1–6 %; exporter and world
        retention hold; 30/30 Solved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402
import india_concentrate_diag as D                      # noqa: E402

ROOT  = Path("data/proc")
YEARS = list(range(1995, 2025))
CONC  = {"Concentrate": 0.275, "Matte": 0.525, "Anode": 0.94, "Cathode": 0.9975}
EXPORTERS = ["Chile", "Peru", "Indonesia"]
# P = 0 hubs the plan names, with their pooled baseline posterior production.
PHANTOM = [("Cathode", "Netherlands"), ("Cathode", "Singapore"),
           ("Cathode", "France"), ("Cathode", "United Arab Emirates"),
           ("Cathode", "Malaysia"), ("Cathode", "United Kingdom"),
           ("Anode", "Netherlands"), ("Anode", "Slovakia"),
           ("Concentrate", "Germany"), ("Concentrate", "Korea, Rep.")]
PRODUCER_OF = {"Concentrate": "MF", "Matte": "Smelting", "Anode": "CFR",
               "Cathode": "CP"}
ZAMBIA_HUBS = ["Switzerland", "Singapore", "Luxembourg"]
# Round C classification, recomputed on the BASE run and held fixed, so class
# shrinkage is visible instead of the classes following the change.
X_MIN_CC, HUB_X_OVER_P, HUB_X_OVER_REF, FAB_X_OVER_REF = 1.0, 3.0, 1.5, 3.0


def _chain(opt):
    const, imp = D._lineage(opt)
    feeds = json.load(open(ROOT / "imp" / imp / "metadata.json"))["parents"]["upstream"]["feed_run_ids"]
    return const, imp, {f.split("_")[1]: f for f in feeds}


def _status(opt):
    s = json.load(open(ROOT / "opt" / opt / "metadata.json"))["stage_summary"]["per_timestep"]
    bad = {k: v["solver_status"] for k, v in s.items() if v["solver_status"] != "Solved"}
    return bad or "30/30 Solved"


def _edges(opt, ei):
    rows = []
    for y in YEARS:
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    e = pd.concat(rows, ignore_index=True)
    return e.merge(ei[["edge_id", "sector_from", "region_from", "sector_to",
                       "region_to", "entity_from", "entity_to", "valuation",
                       "time"]], on="edge_id")


def _ret(e, mask):
    sub = e[mask]
    p = sub.prior.sum()
    return sub.post.sum() / p if p else np.nan


def _world_retention(e):
    return {
        "CO Mass":        _ret(e, (e.sector_from == "CO") & (e.valuation == "Mass")),
        "Concentrate CC": _ret(e, (e.sector_from == "MF") & (e.sector_to == "Concentrate") & (e.valuation == "CC")),
        "Cathode CC":     _ret(e, (e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.valuation == "CC")),
    }


def _time_fills(imp, ei):
    log = pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")
    tl = log[log.method.str.startswith("time_")].merge(
        ei[["edge_id", "region_from", "region_to", "valuation"]], on="edge_id")
    cross = tl[tl.region_from != tl.region_to]
    m = cross[cross.valuation == "Mass"]
    return len(tl), len(cross), len(m), float(m.f_after.abs().sum())


def _hub_classes(e, acc, ei):
    """Round C hub / fabricated / ordinary on the PRIOR of the given run."""
    bil = e[(e.entity_from == "Flow") & (e.region_from != e.region_to)
            & (e.valuation == "Mass") & e.sector_from.isin(CONC)].copy()
    bil["cc"] = bil.prior * bil.sector_from.map(CONC)
    X = bil.groupby(["sector_from", "region_from", "time"]).cc.sum()
    I = bil.groupby(["sector_from", "region_to", "time"]).cc.sum()
    prod = e[(e.entity_from == "Process") & (e.entity_to == "Flow")
             & (e.valuation == "CC")]
    P = prod.groupby(["sector_to", "region_to", "time"]).prior.sum()
    idx = sorted(set(X.index) | set(I.index))
    t = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        idx, names=["sector", "region", "time"]))
    t["X"] = X.reindex(t.index).fillna(0.0)
    t["I"] = I.reindex(t.index).fillna(0.0)
    t["P"] = P.reindex(t.index).fillna(0.0)
    t = t[t.X > X_MIN_CC].reset_index()
    t["time"] = t["time"].astype(int)     # sum rows carry the year as int
    ti = np.where(t.P > 0, t.X / t.P.replace(0, np.nan), np.inf)
    ref = t.X / (t.P + t.I + 1e-9)
    t["klass"] = "ordinary"
    t.loc[(ti > HUB_X_OVER_P) & (ref <= HUB_X_OVER_REF), "klass"] = "hub"
    t.loc[ref > FAB_X_OVER_REF, "klass"] = "fabricated"
    return t


def _sums(opt):
    """Soft **scale-total** rows only — the family every round is measured on
    (ADR 0006). `D._rows` returns every constraint row, so the ratio family
    would otherwise inflate the headline against the documented baselines."""
    const, _ = D._lineage(opt)
    c = D._rows(opt, const)
    return c[c.type == "sum"].copy()


def main(base, l0, l1=None):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    runs = [("base", base), ("L.0", l0)] + ([("L.1", l1)] if l1 else [])

    print("=" * 78)
    for lab, r in runs:
        const, imp, feeds = _chain(r)
        print(f"{lab:5s} {r}\n      const {const}\n      imp   {imp}"
              f"\n      baci  {feeds.get('baci')}\n      {_status(r)}")

    E = {lab: _edges(r, ei) for lab, r in runs}
    S = {lab: _sums(r) for lab, r in runs}

    print("\n== sum-row z² (soft scale totals; ratio rows excluded)")
    for lab, r in runs:
        tot = S[lab].z2.sum()
        d = f"  ({(tot / S['base'].z2.sum() - 1) * 100:+.1f} % vs base)" if lab != "base" else ""
        print(f"   {lab:5s} {tot:12,.0f}{d}")

    print("\n== L.0 check: time_linear fills (cross-region must be 0 after ADR 0026)")
    for lab, r in runs:
        n, nc, nm, kt = _time_fills(_chain(r)[1], ei)
        print(f"   {lab:5s} total {n:6,}   cross-region {nc:6,}   "
              f"cross Mass {nm:5,} ({kt:,.0f} kt)")

    print("\n== world retention (posterior / prior)")
    R = {lab: _world_retention(E[lab]) for lab, _ in runs}
    for k in R["base"]:
        line = f"   {k:16s}"
        for lab, _ in runs:
            d = f" (Δ {R[lab][k] - R['base'][k]:+.3f})" if lab != "base" else ""
            line += f"  {lab} {R[lab][k]:.3f}{d}"
        print(line)

    print("\n== exporter concentrate retention, cross-region CC, 2010–2019")
    for lab, _ in runs:
        e = E[lab]
        m = ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
             & (e.region_from != e.region_to) & (e.valuation == "CC")
             & e.time.astype(int).between(2010, 2019))
        line = f"   {lab:5s} world {_ret(e, m):.3f}"
        for x in EXPORTERS:
            line += f"   {x} {_ret(e, m & (e.region_from == x)):.3f}"
        print(line)

    print("\n== Zambia anode retention into the three hubs (must stay 1–6 %:"
          " the finding is not hidden)")
    for lab, _ in runs:
        e = E[lab]
        m = ((e.sector_from == "Anode") & (e.region_from == "Zambia")
             & (e.region_to.isin(ZAMBIA_HUBS)) & (e.valuation == "CC"))
        o = ((e.sector_from == "Anode") & (e.region_from == "Zambia")
             & (~e.region_to.isin(ZAMBIA_HUBS)) & (e.region_from != e.region_to)
             & (e.valuation == "CC"))
        print(f"   {lab:5s} hubs {_ret(e, m):.3f}   other destinations {_ret(e, o):.3f}")

    print("\n== phantom production at the named P=0 hubs "
          "(Σ posterior Process→Flow CC, pooled)")
    hdr = "   " + f"{'sector / region':32s}" + "".join(f"{lab:>13s}" for lab, _ in runs)
    print(hdr)
    for sec, reg in PHANTOM:
        line = f"   {sec + ' / ' + reg:32s}"
        for lab, _ in runs:
            e = E[lab]
            m = ((e.sector_from == PRODUCER_OF[sec]) & (e.sector_to == sec)
                 & (e.region_from == reg) & (e.region_to == reg)
                 & (e.valuation == "CC"))
            line += f"{e[m].post.sum():13,.0f}"
        print(line)

    print("\n== Round C classes, fixed on the BASE run's prior "
          "(so shrinkage is visible)")
    cls = _hub_classes(E["base"], acc, ei)
    key = cls.set_index(["sector", "region", "time"])["klass"]
    print("   class sizes on base: " +
          "  ".join(f"{k} {v}" for k, v in cls.klass.value_counts().items()))
    for lab, _ in runs:
        s = S[lab].copy()
        s["klass"] = key.reindex(
            pd.MultiIndex.from_arrays([s.sector, s.region, s.time])).values
        g = s.groupby("klass", dropna=False).z2.sum()
        hub_fab = g.get("hub", 0.0) + g.get("fabricated", 0.0)
        print(f"   {lab:5s} hub {g.get('hub', 0):10,.0f}   fabricated "
              f"{g.get('fabricated', 0):10,.0f}   hub∪fab {hub_fab:10,.0f}"
              f"   ({hub_fab / s.z2.sum() * 100:.1f} % of total)")

    print("\n== mass moved vs the previous run (Σ|Δposterior| on CC edges)")
    prev = "base"
    for lab, _ in runs[1:]:
        m = E[prev][["edge_id", "post", "valuation"]].merge(
            E[lab][["edge_id", "post"]], on="edge_id", suffixes=("_a", "_b"))
        m = m[m.valuation == "CC"]
        moved = (m.post_b - m.post_a).abs().sum()
        print(f"   {prev:5s} → {lab:5s}   moved {moved:12,.0f} kt CC "
              f"({moved / m.post_a.abs().sum() * 100:.2f} % of posterior)")
        prev = lab

    print("\n== largest cells on the last run, with every run beside it")
    last = runs[-1][0]
    top = S[last].groupby(["sector", "region"]).z2.sum().sort_values(
        ascending=False).head(14)
    print("   " + f"{'sector / region':34s}" + "".join(f"{lab:>13s}" for lab, _ in runs))
    for (sec, reg) in top.index:
        line = f"   {str(sec) + ' / ' + str(reg):34s}"
        for lab, _ in runs:
            g = S[lab]
            line += f"{g[(g.sector == sec) & (g.region == reg)].z2.sum():13,.0f}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:4])
