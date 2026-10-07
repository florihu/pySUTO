"""
roundY_psc_diag.py
==================
Round Y measurement (spec #41; ADR 0041): the producer stock change CP→PSC gets a
Mass prior and the net-stock seed, and no one-signed hard band sits on a signed edge.
Reads persisted run artefacts only — the opt run's year-local timesteps and its
per-year constraint tables, its const run's prior σ and row_meta, its imp run's
imputation log, coverage and sidecar — and prints the pre-registered metrics of the
round, one block per run:

    P1  PSC prior classes per valuation: observed / twin / seed / pinned / free
        (counts over the 1,410 region-years; the classes overlap)
    P2  Σ|F_Mass − F_CC/0.9975| over the PSC legs, per year
    P3  mass moved Σ|a* − a_prior| on PSC per year and valuation, United States
        excluded (the garbled ±2 Mt rows, #35) and reported on its own
    P4  cumulative drift Σ_t(ΣF − Σa_prior) on PSC 1995–2024 per valuation, US
        excluded and reported on its own
    P5  CP→Cathode CC retention per year: world, Chile, China, United States
    P6  common-set sum-row z² vs the base
    P7  the Round-E items (ADR 0025)
    P8  the PSC stock-bound rows: max |z|, n |z| > 3 and Σz² per year
    P9  solver status
    P10 USC / MESC sign: negative posterior edge-years and the per-year drift
        Σ(F − a_prior) of each family

    python scripts/explo/roundY_psc_diag.py <opt_base> [<opt_new> ...]

Needs the per-year tables (``scripts/solver_stats.py <opt> --years all``) for
every run named.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure               # noqa: E402
from pysuto.config import load_config                    # noqa: E402
from pysuto.param_relation import expand_edge_index      # noqa: E402
import india_concentrate_diag as D                       # noqa: E402
import roundL_reexport_diag as L                         # noqa: E402
import roundX_stock_sink_diag as X                       # noqa: E402

ROOT        = Path("data/proc")
CONC_PSC    = 0.9975          # the cathode concentration the PSC row carries
PINNED      = 1e-5            # σ_a below this is the Stage-3 floor on a zero prior
FREE        = 1e5             # σ_a at or above this is the flat prior (noisy zero)
US          = "United States"
RET_REGIONS = ["Chile", "China", US]
P2_LIMIT    = 10.0            # kt per year
P3_LIMIT    = 100.0           # kt per year
P4_LIMIT    = 1000.0          # kt over 1995–2024
P10_LIMIT   = 16.0            # kt per year, |mean drift| on each of USC / MESC
PSC_ROW     = "netstockboundpsc"


def _psc(e):
    return e[(e.sector_from == "CP") & (e.sector_to == "PSC")].copy()


def _classes(opt, psc):
    """P1: one count per class and valuation; a twin may also be pinned."""
    const, imp, _ = L._chain(opt)
    cov = np.load(ROOT / "imp" / imp / "agg_result.npz", allow_pickle=True)["coverage"]
    log = pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")
    sig = np.load(ROOT / "const" / const / "sigma_a.npy")
    meth = log.groupby("edge_id").method.last()
    out = {}
    for val, s in psc.groupby("valuation"):
        ids = s.edge_id.to_numpy()
        m = meth.reindex(ids).fillna("")
        out[val] = {
            "observed": int((cov[ids] > 0).sum()),
            "twin":     int(m.str.startswith(("parametric", "network_mass_twin")).sum()),
            "seed":     int((m == "share_seed").sum()),
            "pinned":   int((sig[ids] < PINNED).sum()),
            "free":     int((sig[ids] >= FREE).sum()),
        }
    return out


def _gap(psc):
    """P2: Σ|F_Mass − F_CC/conc| per year over the PSC legs."""
    m = psc[psc.valuation == "Mass"].set_index(["region_from", "time"]).post
    c = psc[psc.valuation == "CC"].set_index(["region_from", "time"]).post
    d = (m - c / CONC_PSC).abs()
    return d.groupby(level="time").sum()


def _moved_and_drift(psc):
    """P3 / P4 per valuation, US excluded; the US on its own row."""
    rows = {}
    for val, s in psc.groupby("valuation"):
        rest, us = s[s.region_from != US], s[s.region_from == US]
        g = rest.groupby("time").agg(prior=("prior", "sum"), post=("post", "sum"),
                                     moved=("delta", lambda d: d.abs().sum()))
        g["drift"] = (g.post - g.prior).cumsum()
        rows[val] = {
            "moved_max":   float(g.moved.max()),
            "moved_mean":  float(g.moved.mean()),
            "drift_final": float(g.drift.iloc[-1]),
            "drift_max":   float(g.drift.abs().max()),
            "us_moved":    float(us.delta.abs().sum()),
            "us_drift":    float((us.post - us.prior).sum()),
            "table":       g,
        }
    return rows


def _cathode_retention(e):
    """P5: CP→Cathode CC retention per year, world and the three named regions."""
    p = e[(e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.valuation == "CC")]

    def _ret(sub):
        g = sub.groupby("time")[["prior", "post"]].sum()
        return g.post / g.prior

    out = {"world": _ret(p)}
    for reg in RET_REGIONS:
        out[reg] = _ret(p[p.region_from == reg])
    return pd.DataFrame(out)


def _psc_rows(opt):
    """P8: the soft PSC stock-bound rows per year."""
    const, _, _ = L._chain(opt)
    c = D._rows(opt, const)
    c = c[c.constraint_id.str.contains(PSC_ROW)]
    return c.groupby("time").agg(n=("z", "size"),
                                 max_abs_z=("z", lambda z: z.abs().max()),
                                 n_gt3=("z", lambda z: int((z.abs() > 3).sum())),
                                 sum_z2=("z2", "sum"))


def _stock_sign(e):
    """P10: negative posterior edge-years and the per-year drift of USC / MESC."""
    out = {}
    for ind in X.STOCKS:
        s = e[(e.sector_to == ind) & (e.valuation == "CC")]
        y = s.groupby("time")[["prior", "post"]].sum()
        g = y.post - y.prior
        out[ind] = {
            "neg_edge_years": int((s.post < -1e-6).sum()),
            "neg_prior":      int((s.prior < -1e-6).sum()),
            "drift_mean":     float(g.mean()),
            "drift_min":      float(g.min()),
            "drift_max":      float(g.max()),
            "years_up":       int((g > 0).sum()),
        }
    return out


def _header(lab, r):
    const, imp, feeds = L._chain(r)
    im = json.load(open(ROOT / "imp" / imp / "metadata.json"))
    cm = json.load(open(ROOT / "const" / const / "metadata.json"))
    params = {p["id"]: p["version"] for p in im["parents"]["inputs"].get("params", [])}
    icfg   = im.get("stage_config", {})
    ccfg   = cm.get("stage_config", {})
    ts = "?"
    fmeta = ROOT / "feed" / "registry" / feeds.get("icsg", "") / "metadata.json"
    if fmeta.is_file():
        fj = json.load(open(fmeta))
        ts = (fj.get("parents", {}).get("inputs", {}).get("trade_shares", {}) or {}
              ).get("version", "?")
    print(f"{lab:5s} {r}  {L._status(r)}\n      const {const}  imp {imp}")
    print(f"      conc v{params.get('conc', '?')}  stock_bounds in imp parents: "
          f"{'stock_bounds' in params}  stock_sectors={icfg.get('stock_sectors')}")
    print(f"      icsg trade_shares v{ts}  "
          f"enforce_lower_bounds={ccfg.get('enforce_lower_bounds')}")


def _print_run(lab, r, e):
    psc = _psc(e)
    print(f"\n{'#' * 78}\n# {lab}  {r}\n{'#' * 78}")

    print("\n== P1 PSC prior classes (region-years; classes overlap)")
    for val, c in _classes(r, psc).items():
        print(f"   {val:4s} " + "  ".join(f"{k} {v}" for k, v in c.items()))

    gap = _gap(psc)
    print(f"\n== P2 Σ|F_Mass − F_CC/{CONC_PSC}| on PSC per year (kt): "
          f"min {gap.min():.1f}  max {gap.max():.1f}  "
          f"years > {P2_LIMIT:.0f}: {int((gap > P2_LIMIT).sum())}")
    print("   " + "  ".join(f"{t} {v:.0f}" for t, v in gap.items()))

    md = _moved_and_drift(psc)
    print(f"\n== P3 mass moved on PSC per year (kt, US excluded; limit {P3_LIMIT:.0f}) "
          f"and P4 cumulative drift (kt; limit ±{P4_LIMIT:.0f})")
    for val, d in md.items():
        print(f"   {val:4s} moved max {d['moved_max']:,.1f} mean {d['moved_mean']:,.1f}"
              f"   drift final {d['drift_final']:+,.1f} "
              f"max|drift| {d['drift_max']:,.1f}")
        print(f"        US alone: moved {d['us_moved']:,.1f}, "
              f"Σ(F − a0) {d['us_drift']:+,.1f}")
    both = pd.concat({v: d["table"][["prior", "post", "moved", "drift"]]
                      for v, d in md.items()}, axis=1)
    print(both.round(1).to_string())

    ret = _cathode_retention(e)
    print("\n== P5 CP→Cathode CC retention Σa*/Σa_prior per year")
    print(ret.round(3).T.to_string())

    print("\n== P7 Round-E items (ADR 0025)")
    print("   world retention: "
          + "  ".join(f"{k} {v:.3f}" for k, v in L._world_retention(e).items()))
    med, p90, n = X._small_nodes(r)
    print(f"   small-node (|c| < {X.SMALL_NODE:.0f} kt) posterior/prior: "
          f"median {med:.3f}  p90 {p90:.3f}  (n={n:,})")
    print("   exporter concentrate retention (cross-region Concentrate→Smelting CC): "
          + "  ".join(f"{k} {v:.3f}" for k, v in X._exporter_retention(e).items()))
    cc = e[e.valuation == "CC"]
    print(f"   mass moved, all CC edges: {cc.delta.abs().sum():,.0f} kt")

    pr = _psc_rows(r)
    print(f"\n== P8 PSC stock-bound rows (soft, CC): max |z| {pr.max_abs_z.max():.2f}, "
          f"rows |z| > 3 over all years {int(pr.n_gt3.sum())}, Σz² {pr.sum_z2.sum():,.1f}")
    print("   " + "  ".join(f"{t} {v:.1f}" for t, v in pr.max_abs_z.items()))

    print(f"\n== P9 solver status: {L._status(r)}")

    print(f"\n== P10 USC / MESC sign (CC; limit |mean drift| < {P10_LIMIT:.0f} kt/yr)")
    for ind, d in _stock_sign(e).items():
        print(f"   {ind:4s} negative posterior edge-years {d['neg_edge_years']:,} "
              f"(negative priors {d['neg_prior']:,})")
        print(f"        drift per year mean {d['drift_mean']:+.1f} "
              f"[{d['drift_min']:+.1f}, {d['drift_max']:+.1f}]   "
              f"years with drift > 0: {d['years_up']} of 30")


def main(base, news):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    runs = [("base", base)] + [(f"Y.{i}", r) for i, r in enumerate(news)]

    print("=" * 78)
    E, S = {}, {}
    for lab, r in runs:
        _header(lab, r)
        E[lab] = X._edges_with_sigma(r, ei)
        S[lab] = L._sums(r)

    if len(runs) > 1:
        common = set.intersection(*(set(s.constraint_id) for s in S.values()))
        print(f"\n== P6 common-set sum-row z² ({len(common):,} rows present in every run)")
        zb = float(S["base"][S["base"].constraint_id.isin(common)].z2.sum())
        for lab, _ in runs:
            z = float(S[lab][S[lab].constraint_id.isin(common)].z2.sum())
            print(f"   {lab:5s} {z:,.0f}   vs base {(z / zb - 1) * 100:+.1f} %")

    for lab, r in runs:
        _print_run(lab, r, E[lab])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
