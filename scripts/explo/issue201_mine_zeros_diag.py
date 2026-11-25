# Frozen record: last ran against commit 7b13e32; rerun by checking out that commit (see README.md).
"""
issue201_mine_zeros_diag.py
===========================
#201 measurement of #200: a covered region-year with no ICSG mine production says zero
on its mine output (MF→Concentrate CC joins ``PRODUCTION_EDGES``). Reads persisted run
artefacts only and prints #201's pre-registered items for <new> against <base>:

    F   the two ICSG feed runs: every array that differs and on which edges, the newly
        stamped edges by sector pair, stamped producers and guarded region-years
        (item 1, P1)
    Y   solve status per year (item 2, P2)
    M   CO→Mining and MF→Concentrate CC in the stamped region-years, prior / posterior
        by region (item 3, P3)
    E   the Round-E items (ADR 0025) (item 4, P5)
    C   common-set sum-row z² by year and region, z² by row family, attribution
        (item 5, P4)
    K   stock legs newly above |z| 5 (item 6, P6)

    python scripts/explo/issue201_mine_zeros_diag.py <opt_base> <opt_new>

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
import copper_feed as CF                                    # noqa: E402
from pysuto.acc import AccountingStructure                  # noqa: E402
from pysuto.config import load_config                       # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic   # noqa: E402
from pysuto.feedprep.reexport import icsg_aggregate_members  # noqa: E402
from pysuto.feedprep.trade_shares import decoded_edges      # noqa: E402
from pysuto.input_registry import InputRegistries           # noqa: E402
import issue65_icsg_stock_parse_diag as P65                 # noqa: E402
import issue69_aggregate_gap_diag as P69                    # noqa: E402

ROOT     = Path("data/proc")
FEED_DIR = ROOT / "feed" / "registry"
Z_STOCK  = 5.0
HUBS     = ("United Arab Emirates", "Switzerland", "Singapore", "Taiwan", "Netherlands",
            "France")


def _sidecar(stage: str, run: str) -> dict:
    with open(ROOT / stage / run / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _icsg_feed_of(opt: str) -> str:
    const = _sidecar("opt", opt)["parents"]["upstream"]["const_run_id"]
    imp   = _sidecar("const", const)["parents"]["upstream"]["imp_run_id"]
    feeds = _sidecar("imp", imp)["parents"]["upstream"]["feed_run_ids"]
    return next(f for f in feeds if f.startswith("feed_icsg_"))


def _icsg_inputs(acc, edges: pd.DataFrame):
    """Coverage, other-input reports and raw ICSG production, as the feed's hook reads
    them (``copper_feed._stamp_feed_observed_zeros``), from the registered inputs."""
    reader  = InputRegistries.load(root="data/input")
    fdir    = reader.feed.resolve("icsg")
    raw_df  = CF.ICSG_SPEC.load_raw(str(fdir / "values.csv"))
    tables  = CF.ICSG_SPEC.load_tables(str(fdir / "conc.xlsx"))
    ore_dir = reader.feed.resolve(CF.CRUDE_ORE_ID)
    covered = CF.icsg_coverage(
        raw_df, tables["region_map"], sorted(edges["root_time"].unique()),
        icsg_aggregate_members(reader.params.resolve(CF.AGGREGATES_ID) / "params.csv",
                               CF.AGGREGATES_ID))
    elsewhere = CF.reported_elsewhere(
        pd.read_csv(ore_dir / "values.csv"),
        pd.read_excel(ore_dir / "conc.xlsx", sheet_name="region_map"),
        pd.read_csv(reader.params.resolve(CF.ORE_GRADE_ID) / "params.csv",
                    sep=None, engine="python", dtype=str))
    return covered, elsewhere, CF.raw_production(reader, edges)


def feeds(acc, base: str, new: str) -> pd.DataFrame:
    """F: what differs between the two ICSG feed runs. Returns the newly stamped edges."""
    fb, fn = _icsg_feed_of(base), _icsg_feed_of(new)
    print(f"\n== F ICSG feed {fb} → {fn}")
    a = np.load(FEED_DIR / fb / "result.npz", allow_pickle=True)
    b = np.load(FEED_DIR / fn / "result.npz", allow_pickle=True)
    edges = decoded_edges(acc)
    new_zero = np.where(b["observed_zero"] & ~a["observed_zero"])[0]
    lost     = np.where(a["observed_zero"] & ~b["observed_zero"])[0]
    stamped  = set(new_zero.tolist())
    for k in a.files:
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            print(f"   {k}: shape/dtype {x.shape} {x.dtype} → {y.shape} {y.dtype}")
            continue
        same = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else x == y
        diff = set(np.where(~same)[0].tolist()) if same.ndim else set()
        print(f"   {k}: {len(diff):,} entries differ; outside the newly stamped edges "
              f"{len(diff - stamped):,}")
    print(f"   observed_zero: {len(new_zero):,} newly stamped, {len(lost):,} unstamped")
    s = edges[edges.edge_id.isin(stamped)]
    print("   newly stamped by sector pair and valuation:")
    print(s.groupby(["root_sector_from", "root_sector_to", "valuation"]).size().to_string())
    print(f"   σ_root on them: {np.unique(b['sigma_root'][new_zero]).round(6).tolist()}; "
          f"F_root {np.unique(b['F_root'][new_zero]).tolist()}")
    covered, elsewhere, production = _icsg_inputs(acc, edges)
    p = production[production.root_sector_from == "Concentrate"].set_index(
        ["root_region_to", "root_time"])["production_cc"]
    ry = list(zip(s.root_region_to, s.root_time))
    print(f"   region-years {len(set(ry))} in {s.root_region_to.nunique()} regions; "
          f"stamped producers (ICSG mine > 0) {sum(p.get(k, 0.0) > 0 for k in ry)}; "
          f"stamped where another input reports a mine "
          f"{sum(('Concentrate', r, t) in elsewhere for r, t in ry)}; "
          f"outside the coverage set {sum(k not in covered for k in ry)}")
    return s


def years(opt: str) -> None:
    """Y: solve status per year."""
    ss = _sidecar("opt", opt)["stage_summary"]
    st = pd.Series({t: v["solver_status"] for t, v in ss["per_timestep"].items()})
    print(f"\n== Y {opt}: {st.value_counts().to_dict()} over {len(st)} years; "
          f"solve_wall_s {ss['solve_wall_s']:.0f}")


def mines(E: dict, stamped: pd.DataFrame, base: str, new: str) -> None:
    """M: CO→Mining and MF→Concentrate CC in the stamped region-years, base vs new."""
    ry = set(zip(stamped.root_region_to, stamped.root_time.astype(str)))
    for f, t in (("CO", "Mining"), ("MF", "Concentrate")):
        g = {}
        for o in (base, new):
            e = E[o]
            e = e[(e.valuation == "CC") & (e.sector_from == f) & (e.sector_to == t)
                  & (e.region_from == e.region_to)]
            on = np.array([k in ry for k in zip(e.region_to, e.time.astype(str))])
            g[o] = e[on].groupby("region_to")[["prior", "post"]].sum()
        m = g[base].join(g[new], lsuffix="_base", rsuffix="_new", how="outer").fillna(0.0)
        m["d_post"] = m.post_new - m.post_base
        tot = m.sum()
        pct = 100 * (tot.post_new / tot.post_base - 1) if tot.post_base else float("nan")
        print(f"\n== M {f}→{t} CC in the {len(ry)} stamped region-years, Σ 1995–2024 "
              f"(kt CC): prior {tot.prior_base:,.1f} → {tot.prior_new:,.1f}, posterior "
              f"{tot.post_base:,.1f} → {tot.post_new:,.1f} ({pct:+.1f} % vs {base})")
        print(m.sort_values("post_base", ascending=False).round(1).to_string())
        hub = m.reindex(list(HUBS)).fillna(0.0)
        print("   #183's hubs, posterior new: "
              + ", ".join(f"{r} {v:,.1f}" for r, v in hub.post_new.items()))
        w = {o: E[o][(E[o].valuation == "CC") & (E[o].sector_from == f)
                     & (E[o].sector_to == t)].post.sum() for o in (base, new)}
        print(f"   world posterior {w[base]:,.0f} → {w[new]:,.0f} kt CC "
              f"({100 * (w[new] / w[base] - 1):+.2f} % vs {base})")


def _frame(opt: str, acc) -> pd.DataFrame:
    d  = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    cf = cf[~cf.is_hard.astype(bool)]
    fam = cf["type"].astype(str)   # a ratio's variant names its region-year: too fine
    return pd.DataFrame({"constraint_id": cf.constraint_id.astype(str).to_numpy(),
                         "year": cf.year.astype(str).to_numpy(), "family": fam.to_numpy(),
                         "z2": (cf.z ** 2).to_numpy()})


def families(acc, base: str, new: str) -> None:
    """C: common-set z² of every soft row, by row type."""
    k = ["constraint_id", "year"]
    j = _frame(base, acc).merge(_frame(new, acc)[k + ["z2"]], on=k,
                                suffixes=("_base", "_new"))
    g = j.groupby("family")[["z2_base", "z2_new"]].sum()
    g["d"] = g.z2_new - g.z2_base
    g["pct"] = 100 * g.d / g.z2_base
    print(f"\n== C z² of every soft row by type, {len(j):,} common rows "
          f"(% vs {base}):")
    print(g.round(3).to_string())


def regions(j: pd.DataFrame, base: str) -> None:
    """C: common-set sum-row Σz² by region, largest |Δ|."""
    g = j.groupby("region")[["z2_base", "z2_new"]].sum()
    g["dz2"] = g.z2_new - g.z2_base
    print(f"\n== C regions, common-set sum-row Σz² base → new (vs {base}), largest |Δ|")
    print(g.reindex(g.dz2.abs().sort_values(ascending=False).index[:15]).round(0)
          .to_string())


def stock_z(E: dict, base: str, new: str) -> None:
    """K: stock-leg edge-years newly above |z| 5."""
    f = {o: P65._family(P69._with_sigma(E[o], o)) for o in (base, new)}
    k = ["edge_id"]
    j = f[base][k + ["family", "valuation", "region_to", "time", "z", "post"]].merge(
        f[new][k + ["z", "post"]], on=k, suffixes=("_base", "_new"))
    hit = j[(j.z_new.abs() > Z_STOCK) & ~(j.z_base.abs() > Z_STOCK)]
    gone = j[~(j.z_new.abs() > Z_STOCK) & (j.z_base.abs() > Z_STOCK)]
    print(f"\n== K stock legs above |z| {Z_STOCK:g}: {int((j.z_base.abs() > Z_STOCK).sum())}"
          f" → {int((j.z_new.abs() > Z_STOCK).sum())}; newly above {len(hit)}, "
          f"no longer above {len(gone)}")
    if len(hit):
        g = hit.groupby(["family", "valuation", "region_to"]).agg(
            n=("edge_id", "size"), d_post=("post_new", "sum"))
        g["d_post"] -= hit.groupby(["family", "valuation", "region_to"]).post_base.sum()
        print(g.sort_values("n", ascending=False).head(15).round(1).to_string())


def main(base: str, new: str) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    stamped = feeds(acc, base, new)
    years(base)
    years(new)
    E, S, j = P69.load(base, new)
    mines(E, stamped, base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    regions(j, base)
    families(acc, base, new)
    P69.attribution(j)
    stock_z(E, base, new)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
