"""
Round C transit/re-export diagnosis (read-only; REP feeder).

Measures — on the shipped γ=0.5 chain (imp_20260903_122719_f13ea10b →
opt_20260903_124655_4b23df2f) — how much of the remaining Mass–CC tension the
missing re-export concept explains, before Round D designs the structural fix:

  A. Hub table: per (region, traded sector, year) covered production basis P,
     implied imports I / exports X (covered bilateral Mass legs ×
     conc_nominal). ICSG maps refined/smelted production onto the producing
     process's FEED (Refining_Matte/Refining_Scrap keys, icsg_feed.py), so
     Process→Flow CC coverage exists only for Concentrate (MF→) and CSXEW
     (SXEW→); P therefore = direct Process→Flow CC + covered feed of the
     producing chain (Smelting feed → Matte; CFR feed → Anode; CFR feed +
     ER feed + SXEW cathode → Cathode, since ICSG 'Refining' lands on
     Matte→CFR and CC yields ≈ 1 down the refining chain; imported feed
     legitimately counts — Japan's smelter chain). Transit index X/P; hub iff
     X >> P but X <~ P+I (imports explain the exports); fabricated iff
     X >> P+I.
  B. Sum-row z² per (region, sector, variant) pooled 1995–2024 from the opt
     constraint diags, joined against A → explained-z² bracket.
  C. Feed-time clip ledger: recompute the trade-share fan's domestic residual
     clip max(0, usage − imports) (build_trade_shares.py:105) per consumer
     group — mass silently discarded at feed time; independent hub indicator.
  D. FD-faucet ledger: posterior flows into MESC/USC/FU SB sectors whose
     priors are flat fills per the imp-log method rule (method ==
     'flat_prior_noisy_zero' on the edge's final record — NOT f==0), plus the
     netstockbound σ pricing (z) on those faucets.

Outputs: CSVs under data/proc/opt/<OPT_RUN>/diag/roundC_transit/ + a printed
summary. Findings land in
docs/rep/analysis_opt_20260903_124655_4b23df2f_20260907_roundC-transit-diagnosis.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from pysuto.acc import AccountingStructure                    # noqa: E402
from pysuto.config import load_config                         # noqa: E402
from pysuto.param_relation import expand_edge_index           # noqa: E402
from build_trade_shares import (                              # noqa: E402
    TRADED_SECTORS, _conc_by_sector, build_edges_frame,
)

# Defaults are the Round-C reference chain; pass "<imp_run_id> <opt_run_id>" to
# re-measure another chain (Round D re-runs this against the recovered fans).
IMP_RUN = ROOT / "data/proc/imp" / (sys.argv[1] if len(sys.argv) > 2
                                    else "imp_20260903_122719_f13ea10b")
OPT_RUN = ROOT / "data/proc/opt" / (sys.argv[2] if len(sys.argv) > 2
                                    else "opt_20260903_124655_4b23df2f")
FEED_REG = ROOT / "data/proc/feed/registry"
# exact feed runs behind the γ=0.5 chain (imp metadata upstream.feed_run_ids)
FEED_ICSG = "feed_icsg_20260902_085421_899e69f8"
FEED_BACI = "feed_baci_20260902_085153_de9e393f"

OUT_DIR = OPT_RUN / "diag" / "roundC_transit"
YEARS = range(1995, 2025)

X_MIN_CC = 1.0        # kt CC floor — below this an exporter-year is noise
HUB_X_OVER_P = 3.0    # X >> P (mirrors OUTLIER_LEG_K)
HUB_X_OVER_REF = 1.5  # X <~ this × (P+I): imports explain the exports
FABRICATED_X_OVER_REF = 3.0   # X >> P+I even with imports (PNG class)
EPS = 1e-9


def _load_agg_frame():
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "data/proc/acc"))
    agg = np.load(IMP_RUN / "agg_result.npz", allow_pickle=True)
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg["coverage"], dtype=bool)
    f = np.nan_to_num(np.asarray(agg["f_agg"], dtype=float))
    return acc, ei, cov, f


# producing processes whose covered feed is the production basis of each
# traded flow sector (ICSG keys production data by process input — see
# docstring). CFR feed backs Anode AND Cathode: refining-chain CC yield ≈ 1.
PROC_OF = {"Smelting": ["Matte"], "CFR": ["Anode", "Cathode"],
           "ER": ["Cathode"]}
DIRECT_AS = {"CSXEW": "Cathode"}   # electrowon cathode counts as Cathode


def hub_table(ei, cov, f, conc) -> pd.DataFrame:
    """A — P/I/X per (sector, region, year); bilat frame as in the screen."""
    eid = ei["edge_id"].to_numpy()
    bilat = (cov[eid]
             & ei["entity_from"].eq("Flow").to_numpy()
             & (ei["region_from"] != ei["region_to"]).to_numpy()
             & ei["valuation"].eq("Mass").to_numpy()
             & ei["sector_from"].isin(conc).to_numpy())

    legs = ei[bilat].copy()
    legs["cc"] = f[legs["edge_id"].to_numpy()] \
        * legs["sector_from"].map(conc).to_numpy()
    X = legs.groupby(["sector_from", "region_from", "time"])["cc"].sum()
    I = legs.groupby(["sector_from", "region_to", "time"])["cc"].sum()
    I.index.names = X.index.names

    # direct covered production (Process→Flow CC): Concentrate, CSXEW
    prodm = (cov[eid]
             & ei["entity_from"].eq("Process").to_numpy()
             & ei["entity_to"].eq("Flow").to_numpy()
             & ei["valuation"].eq("CC").to_numpy())
    prod = ei[prodm].copy()
    prod["cc"] = f[prod["edge_id"].to_numpy()]
    prod["sector"] = prod["sector_to"].replace(DIRECT_AS)
    P_dir = prod.groupby(["sector", "region_to", "time"])["cc"].sum()

    # covered feed of the producing process (CC preferred, else Mass×conc)
    infl = ei[ei["entity_to"].eq("Process")
              & ei["sector_to"].isin(PROC_OF)].copy()
    infl["covered"] = cov[infl["edge_id"].to_numpy()]
    infl["val"] = f[infl["edge_id"].to_numpy()]
    key = ["from_id", "to_id", "time"]
    cc_cov = (infl[infl["valuation"].eq("CC")]
              .set_index(key)["covered"])
    ccp = infl[infl["valuation"].eq("CC") & infl["covered"]].copy()
    ccp["basis"] = ccp["val"]
    ma = infl[infl["valuation"].eq("Mass") & infl["covered"]
              & infl["sector_from"].isin(conc)].copy()
    ma["basis"] = ma["val"] * ma["sector_from"].map(conc)
    ma = ma[~ma.set_index(key).index.map(cc_cov).fillna(False).to_numpy()]
    feed = pd.concat([ccp, ma])
    feed["sector"] = feed["sector_to"].map(PROC_OF)
    feed = feed.explode("sector")
    P_feed = feed.groupby(["sector", "region_to", "time"])["basis"].sum()

    P = P_dir.add(P_feed, fill_value=0.0)
    P.index.names = X.index.names

    t = pd.concat([X.rename("X"), I.rename("I"), P.rename("P")],
                  axis=1).fillna(0.0).reset_index()
    t.columns = ["sector", "region", "year", "X", "I", "P"]
    t = t[t["X"] > X_MIN_CC].copy()
    t["transit_index"] = np.where(t["P"] > 0, t["X"] / t["P"], np.inf)
    t["x_over_ref"] = t["X"] / (t["P"] + t["I"] + EPS)
    t["klass"] = "ordinary"
    t.loc[(t["transit_index"] > HUB_X_OVER_P)
          & (t["x_over_ref"] <= HUB_X_OVER_REF), "klass"] = "hub"
    t.loc[t["x_over_ref"] > FABRICATED_X_OVER_REF, "klass"] = "fabricated"
    return t.sort_values(["sector", "region", "year"])


def sum_z2() -> pd.DataFrame:
    """B — sum-row z² per (region, sector, variant), pooled years+valuations."""
    frames = []
    for y in YEARS:
        c = pd.read_csv(OPT_RUN / "diag" / f"constraints_{y}.csv")
        s = c[c["type"] == "sum"].copy()
        mid = s["constraint_id"].str.split("_")
        s["sector"] = mid.str[2]                      # sectors have no "_"
        s["region"] = mid.str[3:-2].str.join("_")
        s["year"] = y
        frames.append(s[["variant", "sector", "region", "year",
                         "valuation", "z"]])
    s = pd.concat(frames)
    s["z2"] = s["z"] ** 2
    return (s.groupby(["region", "sector", "variant"])["z2"].sum()
             .reset_index())


def clip_ledger(acc, conc) -> pd.DataFrame:
    """C — feed-time domestic-residual clip: max(0, imports − usage) per group."""
    icsg = np.load(FEED_REG / FEED_ICSG / "result.npz")["F_root"]
    baci = np.load(FEED_REG / FEED_BACI / "result.npz")["F_root"]
    edges = build_edges_frame(acc, icsg, baci, TRADED_SECTORS)
    df = edges.copy()
    df["baci_mass"] = df["baci_mass"].fillna(0.0)
    dom = df["root_region_from"] == df["root_region_to"]
    df["imp_cc"] = np.where(
        dom, 0.0, df["root_sector_from"].map(conc) * df["baci_mass"])
    g = df.groupby(["root_sector_from", "root_sector_to",
                    "root_region_to", "root_time"])
    per_grp = g.agg(usage=("icsg", "sum"), imports=("imp_cc", "sum"))
    per_grp = per_grp[(per_grp["imports"] > 0) & (per_grp["usage"] > 0)]
    per_grp["clip_cc"] = np.maximum(0.0, per_grp["imports"] - per_grp["usage"])
    out = per_grp.reset_index().rename(columns={
        "root_sector_from": "sector", "root_sector_to": "consumer",
        "root_region_to": "region", "root_time": "year"})
    return out[out["clip_cc"] > 0].sort_values("clip_cc", ascending=False)


def faucet_ledger() -> tuple[pd.DataFrame, pd.DataFrame]:
    """D — posterior inflows into MESC/USC/FU by prior method + bound z."""
    log = pd.read_parquet(IMP_RUN / "imputation_log.parquet",
                          columns=["edge_id", "step", "method"])
    last = (log.sort_values("step").drop_duplicates("edge_id", keep="last")
               .set_index("edge_id")["method"])

    rows, nsb = [], []
    for y in YEARS:
        e = pd.read_csv(OPT_RUN / "diag" / f"edges_{y}.csv")
        fd = e[e["to_entity"].eq("SB")
               & e["to_sector"].isin(["MESC", "USC", "FU"])].copy()
        fd["prior_method"] = (fd["edge_id"].map(last)
                              .fillna("covered_or_untouched"))
        fd["year"] = y
        rows.append(fd[["year", "from_sector", "from_region", "to_sector",
                        "valuation", "a_prior", "a_post", "prior_method"]])

        c = pd.read_csv(OPT_RUN / "diag" / f"constraints_{y}.csv")
        b = c[c["variant"].str.startswith("netstockbound", na=False)].copy()
        b["year"] = y
        b["sb"] = b["variant"].str.extract(r"netstockbound(mesc|usc|psc)")[0]
        nsb.append(b[["year", "sb", "constraint_id", "lhs_post",
                      "sigma_c", "z"]])
    return (pd.concat(rows, ignore_index=True),
            pd.concat(nsb, ignore_index=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    acc, ei, cov, f = _load_agg_frame()
    conc = _conc_by_sector()
    known_sectors = set(acc.node_index["sector"])

    # A ---------------------------------------------------------------------
    hubs = hub_table(ei, cov, f, conc)
    hubs.to_csv(OUT_DIR / "hub_table.csv", index=False)
    rank = (hubs.groupby(["sector", "region"])
                .agg(X=("X", "sum"), I=("I", "sum"), P=("P", "sum"),
                     n_years=("year", "size"),
                     n_hub=("klass", lambda k: (k == "hub").sum()),
                     n_fab=("klass", lambda k: (k == "fabricated").sum()))
                .reset_index())
    rank["transit_excess"] = np.maximum(0.0, rank["X"] - rank["P"])
    rank["transit_index"] = np.where(rank["P"] > 0,
                                     rank["X"] / rank["P"], np.inf)
    rank = rank.sort_values("transit_excess", ascending=False)
    rank.to_csv(OUT_DIR / "hub_ranking.csv", index=False)

    print("=" * 72)
    print("A  HUB RANKING (pooled 1995–2024, kt CC) — top 20 by transit excess")
    print(rank.head(20).to_string(index=False, float_format="%.1f"))

    # B ---------------------------------------------------------------------
    z2 = sum_z2()
    assert set(z2["sector"]) <= known_sectors, "sum-row sector parse broke"
    z2.to_csv(OUT_DIR / "sum_z2.csv", index=False)
    total_z2 = z2["z2"].sum()

    hub_rs = rank[rank["n_hub"] > 0][["sector", "region"]]
    fab_rs = rank[rank["n_fab"] > 0][["sector", "region"]]
    any_rs = rank[(rank["n_hub"] + rank["n_fab"]) > 0][["sector", "region"]]
    use_z2 = z2[z2["variant"] == "usetotout"]
    narrow_hub = use_z2.merge(hub_rs, on=["sector", "region"])["z2"].sum()
    narrow_fab = use_z2.merge(fab_rs, on=["sector", "region"])["z2"].sum()
    narrow_any = use_z2.merge(any_rs, on=["sector", "region"])["z2"].sum()
    hub_regions = set(rank.loc[rank["n_hub"] > 0, "region"])
    broad_hub = z2[z2["region"].isin(hub_regions)]["z2"].sum()

    reg_z2 = z2.groupby("region")["z2"].sum().rename("z2")
    reg_hub = (rank.groupby("region")["transit_excess"].sum()
                   .rename("transit_excess"))
    join = pd.concat([reg_z2, reg_hub], axis=1).fillna(0.0)
    rho = join["z2"].corr(join["transit_excess"], method="spearman")
    join.sort_values("z2", ascending=False).to_csv(
        OUT_DIR / "region_z2_vs_transit.csv")

    print("\n" + "=" * 72)
    print("B  SUM-ROW z² ATTRIBUTION (pooled)")
    print(f"   total sum-row z²                  {total_z2:12.0f}")
    print(f"   usetotout z² on hub sectors       {narrow_hub:12.0f}  "
          f"({narrow_hub / total_z2:.1%})")
    print(f"   usetotout z² on fabricated       {narrow_fab:12.0f}  "
          f"({narrow_fab / total_z2:.1%})")
    print(f"   usetotout z² hub∪fab (no dbl cnt) {narrow_any:12.0f}  "
          f"({narrow_any / total_z2:.1%})")
    print(f"   ALL sum z² in hub regions (broad) {broad_hub:12.0f}  "
          f"({broad_hub / total_z2:.1%})")
    print(f"   Spearman(region z², transit excess) = {rho:.3f}")
    top = (use_z2.merge(rank, on=["sector", "region"], how="left")
                 .sort_values("z2", ascending=False).head(15))
    print(top[["region", "sector", "z2", "transit_index", "n_hub",
               "n_fab"]].to_string(index=False, float_format="%.1f"))

    # C ---------------------------------------------------------------------
    clips = clip_ledger(acc, conc)
    clips.to_csv(OUT_DIR / "clip_ledger.csv", index=False)
    clip_rs = (clips.groupby(["sector", "region"])["clip_cc"].sum()
                    .sort_values(ascending=False))
    print("\n" + "=" * 72)
    print(f"C  FEED-TIME CLIP (imports > usage; kt CC discarded at "
          f"build_trade_shares:105)\n   total clipped {clips['clip_cc'].sum():.0f} kt CC "
          f"over {len(clips)} group-years")
    print(clip_rs.head(15).to_string(float_format="%.1f"))

    # D ---------------------------------------------------------------------
    fd, nsb = faucet_ledger()
    fd.to_csv(OUT_DIR / "faucet_ledger.csv", index=False)
    nsb.to_csv(OUT_DIR / "netstockbound_z.csv", index=False)
    cc = fd[fd["valuation"].eq("CC")]
    print("\n" + "=" * 72)
    print("D  FD FAUCETS (Flow→SB into MESC/USC/FU, CC kt)")
    print(f"   posterior inflow total            {cc['a_post'].sum():12.0f}")
    print("   posterior + adaptation |Δ| by prior method:")
    by_m = (cc.assign(adapt=(cc["a_post"] - cc["a_prior"]).abs())
              .groupby("prior_method")
              .agg(a_prior=("a_prior", "sum"), a_post=("a_post", "sum"),
                   adapt=("adapt", "sum"), n=("a_post", "size"))
              .sort_values("adapt", ascending=False))
    print(by_m.to_string(float_format="%.1f"))
    seeded = cc[cc["prior_method"].isin(["share_seed",
                                         "flat_prior_noisy_zero"])]
    top_f = (seeded.assign(adapt=(seeded["a_post"] - seeded["a_prior"]).abs())
                   .groupby(["from_region", "to_sector"])["adapt"].sum()
                   .sort_values(ascending=False).head(15))
    print("\n   top data-free faucets by |Δ| (share_seed + flat fills):")
    print(top_f.to_string(float_format="%.1f"))
    chn = (cc[cc["from_region"].eq("China") & cc["to_sector"].eq("MESC")]
           .groupby("year")[["a_prior", "a_post"]].sum())
    print("\n   China Cathode→MESC series (kt CC):")
    print(chn.to_string(float_format="%.1f"))
    print("\n   netstockbound z: |z| p50/p95/max = "
          f"{nsb['z'].abs().quantile(0.5):.2f} / "
          f"{nsb['z'].abs().quantile(0.95):.2f} / {nsb['z'].abs().max():.2f}")
    print(nsb.reindex(nsb["z"].abs().sort_values(ascending=False).index)
             .head(10)[["year", "constraint_id", "lhs_post", "sigma_c", "z"]]
             .to_string(index=False, float_format="%.2f"))

    print(f"\nCSV outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
