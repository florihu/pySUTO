"""
build_trade_shares.py
=====================
Build BACI-trade-share weights for ICSG's single-endpoint producer fan-out
(ADR 0002 option D) and emit them as a ``params``-kind input artifact
``trade_shares`` that the ICSG feed build consumes as ``user_weights_df``.

Why
---
ICSG ``Use`` rows classify only the consumer (producer region open), so each
fans across producer regions. The pre-existing fix (ADR 0002) split that fan
*uniformly*; on a bilateral trade edge X→Y the uniform ICSG-CC value then
disagrees with the real BACI-Mass observation, breaking the ``conc_*_supply``
ratio. Option D reweights the fan by BACI trade shares so ICSG-CC inherits
BACI's bilateral geography.

The weighting (per sector_from, sector_to, consumer region Y, year t), in
copper-content (CC) units so the domestic/import split is unit-consistent:

    import leg  X≠Y : raw weight = conc_sector · BACI_Mass_Froot(X→Y, t)
    domestic    X==Y: raw weight = max(0, ICSG_usage_CC(Y, t)
                                          − Σ_X conc_sector · BACI_Mass(X→Y, t))

``ConcordanceTable._impute_weights`` then normalises each src fan's raw weights
to sum 1, so ``F_root_CC(import) = conc·BACI_Mass`` and
``F_root_CC(domestic) = usage − imports`` (usage conserved).

Groups with **no** recorded imports (ADR 0018): if BACI observes the consumer
region at all, ``Σ imports = 0`` is a real observation of domestic sourcing and
the residual ``usage − 0 = usage`` yields an all-domestic fan. If BACI is blind
to the region (Taiwan — BACI 490 "Other Asia, nes" folds into ROW, ADR 0014),
no rows are emitted and the fan stays undistributed per ADR 0008. Before ADR
0018 every import-free group was unemitted, which silently discarded 469 ICSG
fans (~45 Mt CC) — see REP roundD-fan-recovery.

``ICSG_usage_CC`` is read from the *existing* ICSG feed ``F_root``: the fan total
Σ_X F_root(X→Y) equals the raw ICSG usage regardless of the weights used to
build it (mass-preserving), so it is the same number the raw carries.

SCOPE — traded refined/intermediate products only:
    Cathode, Concentrate, Anode, Matte.
Scrap is an ICSG Use type but is NOT traded → deliberately excluded.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from pysuto.acc import AccountingStructure
from pysuto.config import load_config

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# Traded supplier sectors whose ICSG producer fan is BACI-weighted. Scrap is
# NOT traded and must never appear here. All are uniform producer fans
# (verified): Cathode→FU/USC, Concentrate→Smelting, Anode→ER, Matte→CFR.
TRADED_SECTORS = ["Cathode", "Concentrate", "Anode", "Matte"]
# Sectors actually emitted this run.
SECTORS = list(TRADED_SECTORS)

_INPUT_ROOT = os.path.join("data", "input")
ACC_DIR     = os.path.join("data", "proc", "acc")
FEED_REG    = os.path.join("data", "proc", "feed", "registry")
OUT_DIR     = os.path.join(_INPUT_ROOT, "params", "trade_shares", "current")

assert "Scrap" not in TRADED_SECTORS and "Scrap" not in SECTORS, "Scrap is not traded"


# ---------------------------------------------------------------------------
# Pure share computation (unit-tested)
# ---------------------------------------------------------------------------

def compute_trade_share_weights(
    edges: pd.DataFrame,
    conc_by_sector: dict[str, float],
    baci_observed_regions: set[str] | None = None,
    usage_fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute raw fan weights from a decoded CC-edge frame.

    ``edges`` columns (one row per CC fan-target edge):
        root_sector_from, root_sector_to, root_region_from, root_region_to,
        root_time, icsg (ICSG F_root CC), baci_mass (BACI F_root Mass on the
        paired Mass edge; NaN/0 where no trade).

    ``baci_observed_regions`` — consumer regions BACI observes at all (any
    cross-region import, any sector, any year). For those, ``Σ imports == 0``
    is an *observation* of domestic sourcing, so the ADR-0002-option-D residual
    ``max(0, usage − Σ imports) = usage`` applies and the fan is emitted
    all-domestic (ADR 0018). For a region BACI never observes (BACI 490 "Other
    Asia, nes" → ROW leaves Taiwan trade-blind, ADR 0014), zero imports carries
    no information: the group is left unemitted so the fan stays undistributed
    — the honest "total known, split unknown" state (ADR 0008). ``None`` means
    "no region is known-observed", i.e. the pre-ADR-0018 behaviour.

    ``usage_fallback`` — per-group ICSG usage read from the *raw* ICSG rows
    (the four group keys + ``usage_raw``). Needed only to break a bootstrap:
    ``usage`` is normally the ICSG ``F_root`` fan total, but a fan a previous
    build left **undistributed** has ``F_root == 0`` on every leg, so its usage
    reads as 0 and the group could never be recovered. Applied strictly where
    the ``F_root`` total is 0, leaving distributed groups bit-for-bit unchanged.

    Returns rows with columns
        root_sector_from, root_sector_to, root_region_from, root_region_to,
        root_time, weight
    for consumer groups with ICSG usage that BACI either covers (≥1 import) or
    observes. The raw weights are NOT normalised here —
    ConcordanceTable._impute_weights does that per src fan.
    """
    df = edges.copy()
    df["baci_mass"] = df["baci_mass"].fillna(0.0)
    df["is_domestic"] = df["root_region_from"] == df["root_region_to"]
    conc = df["root_sector_from"].map(conc_by_sector)
    df["imp_cc"] = np.where(df["is_domestic"], 0.0, conc * df["baci_mass"])

    grp_cols = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
    g = df.groupby(grp_cols, sort=False)
    usage   = g["icsg"].transform("sum").to_numpy()
    imp_sum = g["imp_cc"].transform("sum").to_numpy()

    if usage_fallback is not None:
        # many-to-one left merge preserves df's row order
        raw_u = (df[grp_cols].merge(usage_fallback, on=grp_cols, how="left")
                 ["usage_raw"].fillna(0.0).to_numpy())
        usage = np.where(usage > 0, usage, raw_u)

    df["weight"] = np.where(
        df["is_domestic"],
        np.maximum(0.0, usage - imp_sum),   # domestic residual
        df["imp_cc"],                       # import leg (0 where untraded)
    )

    # Emit groups ICSG fans into (usage>0) that BACI either covers (≥1 import)
    # or observes (zero imports is then a real observation → all-domestic fan,
    # ADR 0018). Trade-blind regions stay unemitted → fan undistributed.
    observed = df["root_region_to"].isin(baci_observed_regions or set()).to_numpy()
    covered = (usage > 0) & ((imp_sum > 0) | observed)
    out = df.loc[covered, grp_cols[:2] + ["root_region_from", "root_region_to",
                                          "root_time", "weight"]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _latest_feed_froot(label: str) -> np.ndarray:
    dirs = sorted(glob.glob(os.path.join(FEED_REG, f"feed_{label}_*")))
    if not dirs:
        raise FileNotFoundError(f"no feed_{label}_* run under {FEED_REG}")
    run = dirs[-1]
    print(f"   {label:10s} ← {os.path.basename(run)}")
    return np.load(os.path.join(run, "result.npz"))["F_root"]


def _conc_by_sector() -> dict[str, float]:
    inputs = _input_registries()
    csv = inputs.params.resolve("conc") / "params.csv"
    # delimiter-agnostic: the conc params CSV moved to comma-delimited
    p = pd.read_csv(csv, sep=None, engine="python", comment="#", dtype=str)
    # param_ids are underscore-free since conc v2 (naming grammar), e.g.
    # 'concmattesupply'
    p = p[(p["param_type"] == "concentration")
          & (p["param_id"].str.endswith("supply"))]
    out = {row["a_sector_from"]: float(row["value_nominal"]) for _, row in p.iterrows()}
    return {s: out[s] for s in TRADED_SECTORS if s in out}


def _input_registries():
    from pysuto.input_registry import InputRegistries
    return InputRegistries.initialize(root=_INPUT_ROOT)


def _decoded_edges(acc: AccountingStructure) -> pd.DataFrame:
    ei = acc.edge_index.copy()
    nidx = acc.node_index.reset_index(drop=True)
    ei["root_sector_from"] = nidx["sector"].values[ei["from_id"].values]
    ei["root_sector_to"]   = nidx["sector"].values[ei["to_id"].values]
    ei["root_region_from"] = nidx["region"].values[ei["from_id"].values]
    ei["root_region_to"]   = nidx["region"].values[ei["to_id"].values]
    ei["root_time"]        = ei["time"]
    return ei


def _icsg_sector_key(proc: str, typ: str, supply_use: str) -> str:
    """Mirror of ``icsg_feed._make_sector_key`` for Use rows (process+material
    disambiguation). Kept local so this script does not import the feed driver."""
    if supply_use == "Supply":
        return proc
    if proc == "Smelting" and typ in ("Scrap", "Srap"):
        return "Smelting_Scrap"
    if proc == "Smelting":
        return "Smelting_Concentrate"
    if proc == "Refining" and typ == "Scrap":
        return "Refining_Scrap"
    if proc == "Refining":
        return "Refining_Matte"
    return proc


def icsg_raw_usage(sectors) -> pd.DataFrame:
    """Per-fan-group ICSG Use totals straight from the raw ICSG rows.

    Group key is the fan's ``(root_sector_from, root_sector_to,
    root_region_to, root_time)``: the input product (sector_map ``icsg_type``),
    the consumer process/boundary (``root_sector``), the mapped consumer region
    and the year. Verified to reproduce the ICSG ``F_root`` fan total exactly
    (max relative difference 3e-16 over all 2,610 distributed groups), so it is
    a faithful stand-in where ``F_root`` is 0 because the fan was dropped.
    """
    inputs = _input_registries()
    fdir = inputs.feed.resolve("icsg")
    raw = pd.read_csv(fdir / "values.csv")
    smap_df = pd.read_excel(fdir / "conc.xlsx", sheet_name="sector_map")
    rmap_df = pd.read_excel(fdir / "conc.xlsx", sheet_name="region_map")
    regmap = dict(zip(rmap_df["src_region"], rmap_df["root_region"]))
    smap = {r["src_sector"]: (r["icsg_type"], r["root_sector"])
            for _, r in smap_df[smap_df["supply_use"] == "Use"].iterrows()}

    u = raw[raw["Supply_Use"] == "Use"].copy()
    key = [_icsg_sector_key(p, t, su) for p, t, su
           in zip(u["Process"], u["Type"], u["Supply_Use"])]
    u["root_sector_from"] = [smap.get(k, (None, None))[0] for k in key]
    u["root_sector_to"]   = [smap.get(k, (None, None))[1] for k in key]
    u["root_region_to"]   = u["Region"].map(regmap)
    u["root_time"]        = u["Year"].astype(str)
    u = u[u["root_sector_from"].isin(sectors) & u["root_region_to"].notna()]
    return (u.groupby(["root_sector_from", "root_sector_to",
                       "root_region_to", "root_time"])["Value"]
             .sum().rename("usage_raw").reset_index())


def baci_observed_regions(acc, baci_froot) -> set[str]:
    """Consumer regions BACI observes at all (any cross-region import, any
    sector/year). Zero imports is informative only for these — see ADR 0018."""
    ei = _decoded_edges(acc)
    cross = ei[(ei["root_region_from"] != ei["root_region_to"])
               & (ei["valuation"] == "Mass")]
    v = baci_froot[cross["edge_id"].to_numpy()]
    return set(cross.loc[v > 0, "root_region_to"].unique())


def build_edges_frame(acc, icsg_froot, baci_froot, sectors) -> pd.DataFrame:
    """One row per CC fan-target edge for `sectors`, with icsg + paired baci_mass."""
    ei = _decoded_edges(acc)
    ei = ei[ei["root_sector_from"].isin(sectors)]
    cc   = ei[ei["valuation"] == "CC"].copy()
    mass = ei[ei["valuation"] == "Mass"][["from_id", "to_id", "time", "edge_id"]]
    mass = mass.rename(columns={"edge_id": "mass_edge_id"})
    cc = cc.merge(mass, on=["from_id", "to_id", "time"], how="left")
    cc["icsg"]      = icsg_froot[cc["edge_id"].values]
    cc["baci_mass"] = np.where(cc["mass_edge_id"].notna(),
                               baci_froot[cc["mass_edge_id"].fillna(0).astype(int).values],
                               np.nan)
    return cc[["root_sector_from", "root_sector_to", "root_region_from",
               "root_region_to", "root_time", "icsg", "baci_mass"]]


def main() -> None:
    print("=" * 60)
    print("Building BACI trade-share weights (ADR 0002 option D)")
    print(f"   sectors: {SECTORS}  (Scrap excluded — not traded)")

    acc_id = load_config().pipeline.acc_id
    acc = AccountingStructure.load(id=acc_id, storage_dir=ACC_DIR)
    conc_by_sector = _conc_by_sector()
    print(f"   conc: {conc_by_sector}")
    icsg = _latest_feed_froot("icsg")
    baci = _latest_feed_froot("baci")

    edges = build_edges_frame(acc, icsg, baci, SECTORS)
    observed = baci_observed_regions(acc, baci)
    usage_raw = icsg_raw_usage(SECTORS)
    all_regions = set(acc.node_index["region"])
    blind = sorted(all_regions - observed)
    print(f"   BACI-observed regions: {len(observed)}/{len(all_regions)}"
          f"   trade-blind (fans stay undistributed): {blind or 'none'}")
    shares = compute_trade_share_weights(edges, conc_by_sector, observed,
                                         usage_raw)

    # ADR 0018 diagnostic: which groups are emitted only because of the
    # import-free domestic fallback, and how much ICSG tonnage they carry.
    pre = compute_trade_share_weights(edges, conc_by_sector, None, None)
    gcols = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
    new_grp = (shares[gcols].drop_duplicates()
               .merge(pre[gcols].drop_duplicates(), how="left", indicator=True))
    n_new = int((new_grp["_merge"] == "left_only").sum())
    dom = shares[shares["root_region_from"] == shares["root_region_to"]]
    recovered = dom.merge(new_grp[new_grp["_merge"] == "left_only"][gcols],
                          on=gcols)["weight"].sum()
    print(f"   ADR 0018 domestic fallback: {n_new:,} groups recovered, "
          f"{recovered:,.0f} kt CC of ICSG usage that previously went "
          f"undistributed")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Named params.csv because the params input-registry only hashes that file
    # for content-versioning (input_registry _LOAD_BEARING["params"]).
    out_csv = os.path.join(OUT_DIR, "params.csv")
    shares.to_csv(out_csv, index=False)
    print(f"   wrote {len(shares):,} weight rows → {out_csv}")
    print(f"   consumer groups: "
          f"{shares.groupby(['root_sector_from','root_sector_to','root_region_to','root_time']).ngroups:,}")

    # Snapshot/version the artifact via the params registry (provenance).
    inputs = _input_registries()
    entry = inputs.params.entry("trade_shares")
    print(f"   registered trade_shares v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    main()
