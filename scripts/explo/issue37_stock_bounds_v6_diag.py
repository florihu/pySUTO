"""
issue37_stock_bounds_v6_diag.py
===============================
#37 measurement: stock_bounds regenerated on icsg v15 (v5 → v6, PSC p95 0.0489 → 0.0276),
the 0.1 kt net-stock scale floor kept. Reads both chains' imp ``agg_result.npz``, both
constraint runs' per-year σ_c and row_meta, and the runs' persisted artefacts; builds the
sum-row and constraint tables in memory (no per-year CSVs, #46). Prints #37's pass-rule
items for <new> against <base>:

    I   scope: the edges whose f or σ_a differ between the imp runs, by sector pair; the
        rows whose σ_c differ between the const runs, by family; the PSC rows' factor on
        the stored σ_c and with the engine's σ_c Lp lift inverted (item 1)
    P   the PSC rows: Σz², |z| > 3, and the near-pin rows (σ_c < 0.01 kt) with the hazard
        check (CP→Cathode prior < 1 kt beside a nonzero PSC prior)
    E   the Round-E items (ADR 0025) (item 2)
    K   world Σ CP→PSC (CC) by year against ICSG's reported PSC: Σ|post − ICSG| (item 3)
    S   stock legs newly above |z| 5 (item 4); the 1995 stock legs by family
    C   common-set sum-row z² by region, sector and year; rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue37_stock_bounds_v6_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue65_icsg_stock_parse_diag as P65               # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import issue79_baci490_taiwan_diag as P79                 # noqa: E402
import issue86_congo_sacu_diag as P86                     # noqa: E402
import issue92_refined_scope_diag as P92                  # noqa: E402
import issue94_relative_ratio_sigma_diag as P94           # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.config import load_config                     # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic  # noqa: E402
from pysuto.param_relation import expand_edge_index       # noqa: E402

ROOT     = Path("data/proc")
ICSG_IN  = Path("data/input/feed/icsg")
PSC_ROW  = "netstockbound"
PIN      = 0.01                                           # kt; a near-pin PSC row
YEARS_K  = [str(y) for y in range(1996, 2025)]            # years ICSG can report a change


def scope(acc, base: str, new: str) -> None:
    """I: which edges' f/σ_a and which rows' σ_c differ between the two chains."""
    (cb, ib, _), (cn, in_, _) = L._chain(base), L._chain(new)
    print(f"\n== I scope, {ib} → {in_} and {cb} → {cn} (item 1)")
    ab, an = (np.load(ROOT / "imp" / i / "agg_result.npz") for i in (ib, in_))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    for k in ("f_agg", "sigma_agg"):
        x, y = ab[k], an[k]
        d = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        pairs = ei.loc[d, ["sector_from", "sector_to"]].astype(str).agg("→".join, axis=1)
        print(f"   {k}: {int(d.sum()):,} edges differ; by sector pair {pairs.value_counts().to_dict()}")
        if k == "sigma_agg" and d.any():
            r = y[d] / x[d]
            print(f"     σ_a factor new/base: min {np.nanmin(r):.4f}  median {np.nanmedian(r):.4f}  "
                  f"max {np.nanmax(r):.4f}")
    mb = pd.read_parquet(ROOT / "const" / cb / "row_meta.parquet")
    mn = pd.read_parquet(ROOT / "const" / cn / "row_meta.parquet")
    same = len(mb) == len(mn) and (mb.constraint_id.to_numpy() == mn.constraint_id.to_numpy()).all()
    print(f"   rows {len(mb):,} → {len(mn):,}; same rows in the same order: {same}")
    if not same:
        return
    sb, sn, c_b, c_n = (np.zeros(len(mn)) for _ in range(4))
    for f in sorted((ROOT / "const" / cn / "timesteps").glob("t_*.npz")):
        rows = (mn.time.astype(str) == f.stem[2:]).to_numpy()
        zb, zn = np.load(ROOT / "const" / cb / "timesteps" / f.name), np.load(f)
        sb[rows], c_b[rows] = zb["sigma_c"][rows], zb["c"][rows]
        sn[rows], c_n[rows] = zn["sigma_c"][rows], zn["c"][rows]
    changed = ~np.isclose(sb, sn, rtol=1e-12, atol=0.0)
    fam = P94._family(mn.variant.astype(str))
    print(f"   σ_c: {int(changed.sum()):,} rows differ; by family {fam[changed].value_counts().to_dict()}; "
          f"c differs on {int((~np.isclose(c_b, c_n, rtol=1e-12, atol=0.0)).sum()):,} rows")
    other = changed & (fam != PSC_ROW).to_numpy()
    if other.any():
        print("   σ_c changes outside the PSC rows:")
        print(mn[other].assign(base=sb[other], new=sn[other])
              [["constraint_id", "type", "variant", "time", "base", "new"]].head(20).to_string())
    psc = changed & (fam == PSC_ROW).to_numpy()
    cfg_b, cfg_n = P94._stage_config(cb), P94._stage_config(cn)
    stored = sn[psc] / sb[psc]
    unl = P94._lp_unlift(sn[psc], c_n[psc], cfg_n) / P94._lp_unlift(sb[psc], c_b[psc], cfg_b)
    print(f"   PSC rows' σ_c factor: stored {stored.min():.4f}–{stored.max():.4f}; "
          f"Lp lift inverted {unl.min():.6f}–{unl.max():.6f} ({int(psc.sum()):,} rows)")


def _psc_frame(opt: str, acc) -> pd.DataFrame:
    d = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    return cf[cf["variant"].astype(str).str.startswith(PSC_ROW)].reset_index(drop=True)


def psc_rows(acc, E: dict, base: str, new: str) -> None:
    """P: the PSC rows' z², the near-pin rows, and the hazard check."""
    print("\n== P PSC net-stock rows (base → new)")
    f = {o: _psc_frame(o, acc) for o in (base, new)}
    for o in (base, new):
        x, pin = f[o], f[o][f[o].sigma_c < PIN]
        print(f"   {o}: {len(x):,} rows, Σz² {(x.z ** 2).sum():,.1f}, |z| > 3 {int((x.z.abs() > 3).sum())}, "
              f"max |z| {x.z.abs().max():.2f}; σ_c < {PIN} kt: {len(pin)} rows, Σz² "
              f"{(pin.z ** 2).sum():.2f}, max |z| {pin.z.abs().max():.3f}, min σ_c {pin.sigma_c.min():.4f}")
        top = x.reindex(x.z.abs().sort_values(ascending=False).index).head(5)
        print("     largest |z|: " + "  ".join(f"{r.constraint_id} {r.z:+.2f}" for r in top.itertuples()))
    for o in (base, new):
        e = E[o]
        cp = (e[(e.sector_from == "CP") & (e.sector_to == "Cathode")]
              .groupby(["region_to", "time", "valuation"]).prior.sum())
        ps = (e[(e.sector_from == "CP") & (e.sector_to == "PSC")]
              .groupby(["region_to", "time", "valuation"]).prior.sum())
        j = pd.concat([cp.rename("cp"), ps.rename("psc")], axis=1).fillna(0.0)
        bad = j[(j.cp.abs() < 1.0) & (j.psc.abs() > 1e-9)]
        print(f"   {o}: region-years with CP→Cathode prior < 1 kt and a nonzero PSC prior: {len(bad)}")


def icsg_psc(E: dict, base: str, new: str) -> None:
    """K: world Σ CP→PSC (CC) against ICSG's reported world PSC, 1996–2024."""
    v = P75._icsg(new)[1]
    raw = pd.read_csv(ICSG_IN / f"v{v}" / "values.csv")
    icsg = raw[raw.Type == "PSC"].groupby(raw.Year.astype(str)).Value.sum()
    icsg = icsg.reindex(YEARS_K).fillna(0.0)
    print(f"\n== K world PSC (CC, kt) against ICSG's reported PSC (icsg v{v}), 1996–2024 (item 3)")
    t = {}
    for o in (base, new):
        e = E[o][(E[o].sector_from == "CP") & (E[o].sector_to == "PSC") & (E[o].valuation == "CC")]
        g = e.groupby(e.time.astype(str))[["prior", "post"]].sum().reindex(YEARS_K).fillna(0.0)
        t[o] = g
        gap = (g.post - icsg).abs()
        print(f"   {o}: Σ|post − ICSG| {gap.sum():,.1f}; Σ|prior − ICSG| {(g.prior - icsg).abs().sum():,.1f};"
              f" Σ post {g.post.sum():+,.1f}, Σ ICSG {icsg.sum():+,.1f}")
    gb, gn = ((t[o].post - icsg).abs().sum() for o in (base, new))
    print(f"   Σ|post − ICSG| {gb:,.1f} → {gn:,.1f} ({100 * (gn / gb - 1):+.1f} % vs {base})")
    tab = pd.DataFrame({"ICSG": icsg, "post_base": t[base].post, "post_new": t[new].post})
    tab["d"] = (tab.post_new - tab.ICSG).abs() - (tab.post_base - tab.ICSG).abs()
    print("   years whose gap moves most:")
    print(tab.reindex(tab.d.abs().sort_values(ascending=False).index).head(6).round(1).to_string())


def year_1995(E: dict, base: str, new: str) -> None:
    """S: the 1995 stock legs by family (no ICSG net addition exists for 1995)."""
    print("\n== S 1995 stock legs (CC), posterior by family")
    for o in (base, new):
        e = P65._family(E[o]).query("valuation == 'CC'")
        y = e[e.time.astype(str) == "1995"]
        g = y.groupby("family").post.sum()
        print(f"   {o}: Σ {g.sum():+,.1f} kt CC; " + "  ".join(f"{k} {v:+,.1f}" for k, v in g.items()))


def main(base: str, new: str) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    scope(acc, base, new)
    E, S, j = P69.load(base, new)
    psc_rows(acc, E, base, new)
    P69.round_e(E, S, j, base, new)
    icsg_psc(E, base, new)
    P79.stock_z(E, base, new)
    P92.stock_sizes(E, base, new)
    year_1995(E, base, new)
    P69.common_set(j, base)
    P86.by_year(j)
    by = j.groupby(P92._sector(j.label)).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
