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
to the region (Taiwan was, until #77 mapped BACI 490 "Other Asia, nes" to it),
no rows are emitted and the fan stays undistributed per ADR 0008. Before ADR
0018 every import-free group was unemitted, which silently discarded 469 ICSG
fans (~45 Mt CC) — see REP roundD-fan-recovery.

Domestic leg bounded by production (ADR 0024). The residual "needs no ICSG
production data" — and that is the defect: where BACI imports in CC fall short
of what the consumer's process used, the shortfall is booked as *domestic*
supply the region never produced (India: 100–456 kt/yr of concentrate against
23–48 kt mined). For sectors in ``BOUND_DOMESTIC_SECTORS`` the domestic raw
weight is therefore capped at the consumer region's own ICSG production of
the product, and the remainder of the usage is spread over the import legs in
BACI proportion (usage still conserved):

    dom_raw   = min(max(0, usage − Σ imp_cc), production_CC(Y, t))
    imp_raw_X = imp_cc_X · (usage − dom_raw) / Σ imp_cc

Bounded only where the recorded imports can carry the remainder,
``Σ imp_cc ≥ residual − production`` (the import legs are at most doubled).
An import-free fan has no leg to carry it and keeps the ADR 0018
all-domestic form; a near-self-sufficient smelter (United States 1997–2003:
usage 996, mine 886, imports 26 kt CC) has legs that would need to grow
fourfold — that gap is ICSG-internal, not BACI's, and spreading it would
fabricate imports instead of domestic supply (measured: it cut US mine
output 10–30 pp in those years through the interpolated neighbours).

``ICSG_usage_CC`` is read from the registered ICSG input's raw ``Use`` rows
(``icsg_raw_usage``), never from an ICSG feed run: that run is built from these
very weights, so reading its ``F_root`` handed the builder the *previous* ICSG
data and the documented order (trade shares → ICSG feed) was exact only on a
second pass (#66). The weights therefore depend on registered inputs alone —
the BACI feed run, built first, and the ICSG input.

SCOPE — traded refined/intermediate products only:
    Cathode, Concentrate, Anode, Matte.
Scrap is an ICSG Use type but is NOT traded → deliberately excluded.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from pysuto.acc import AccountingStructure
from pysuto.config import load_config
from pysuto.input_registry import PARAMS_PARENTS, InputRegistries
from pysuto.param_relation import conc_for, conc_nominal_lookup

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# Traded supplier sectors whose ICSG producer fan is BACI-weighted. Scrap is
# NOT traded and must never appear here. All are uniform producer fans
# (verified): Cathode→FU/USC, Concentrate→Smelting, Anode→ER, Matte→CFR.
TRADED_SECTORS = ["Cathode", "Concentrate", "Anode", "Matte"]
# Sectors actually emitted this run.
SECTORS = list(TRADED_SECTORS)
# Destinations of the signed stock-change legs (GLOSSARY "Stock-change leg";
# CP→PSC is not fanned).  The same membership is kept in three other places, #39.
SIGNED_SECTORS_TO = ("MESC", "USC")

_INPUT_ROOT = os.path.join("data", "input")
ACC_DIR     = os.path.join("data", "proc", "acc")
FEED_REG    = os.path.join("data", "proc", "feed", "registry")
OUT_DIR     = os.path.join(_INPUT_ROOT, "params", "trade_shares", "current")

assert "Scrap" not in TRADED_SECTORS and "Scrap" not in SECTORS, "Scrap is not traded"

# ADR 0024: sectors whose fan's domestic leg is bounded by the consumer
# region's own ICSG production of the product (read from the ICSG F_root on
# the producing-process → product edge). Concentrate first — 21.7 Mt of
# residual above mine output over 210 region-years; Matte (1.3 Mt) is the
# follow-up once the rule is measured.
# ADR 0027 amends this: Concentrate alone leaves the *cathode* hubs' residual
# unbounded, so the re-export netting merely moves their phantom refinery from
# the export side to the use side (France: posterior CP→Cathode 1,848 → 1,818
# across L.1, against Netherlands 1,369 → 328). Cathode and Anode production
# cannot be read off the ICSG F_root — there is no ICSG-anchored CP→Cathode
# edge — so `icsg_raw_production` derives all three from the raw ICSG rows.
BOUND_DOMESTIC_SECTORS = ["Concentrate", "Cathode", "Anode"]
PRODUCER_OF = {"Concentrate": "MF", "Matte": "Smelting", "Anode": "CFR"}

# ADR 0020's withhold, retained but disabled by ADR 0021 (the ADR-0019 pattern:
# keep the detector, drop the remedy).  The gate existed because a refining
# total below its own region's smelting made the Matte node's hard balance
# unsatisfiable.  ADR 0021 removes that at source — refined production no
# longer anchors Matte→CFR — so the same comparison now flags an *anode
# exporter*, which the model can finally represent.  Withholding there would
# discard exactly the data the re-placement is designed to accommodate.
WITHHOLD_CHAIN_INCONSISTENT = False


# ---------------------------------------------------------------------------
# Pure share computation (unit-tested)
# ---------------------------------------------------------------------------

def compute_trade_share_weights(
    edges: pd.DataFrame,
    usage_raw: pd.DataFrame,
    conc_lookup: dict,
    baci_observed_regions: set[str] | None = None,
    withhold_fallback: set | None = None,
    production: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute raw fan weights from a decoded CC-edge frame.

    ``edges`` columns (one row per CC fan-target edge):
        root_sector_from, root_sector_to, root_region_from, root_region_to,
        root_time, baci_mass (BACI F_root Mass on the paired Mass edge; NaN/0
        where no trade).

    ``usage_raw`` — per fan group ``(root_sector_from, root_sector_to,
    root_region_to, root_time)`` the ICSG Use total in CC (column ``usage_raw``),
    read from the registered ICSG input by ``icsg_raw_usage``. A group without a row
    has no usage and is not emitted. Never the ICSG feed run's ``F_root`` fan
    total: that run is built from these weights, so after an ICSG change it
    carries the previous data (#66).

    ``baci_observed_regions`` — consumer regions BACI observes at all (any
    cross-region import, any sector, any year). For those, ``Σ imports == 0``
    is an *observation* of domestic sourcing, so the ADR-0002-option-D residual
    ``max(0, usage − Σ imports) = usage`` applies and the fan is emitted
    all-domestic (ADR 0018). For a region BACI never observes (Taiwan was one
    until #77 mapped BACI 490 "Other Asia, nes" to it), zero imports carries
    no information: the group is left unemitted so the fan stays undistributed
    — the honest "total known, split unknown" state (ADR 0008). ``None`` means
    "no region is known-observed", i.e. the pre-ADR-0018 behaviour.

    ``withhold_fallback`` — group keys ``(sector_from, sector_to, region_to,
    time)`` for which the import-free domestic fallback must **not** fire, so
    the fan stays undistributed exactly as it was before ADR 0018. Groups BACI
    covers are unaffected. Used by ADR 0020 to withhold a restored refining
    total that contradicts its own region-year's smelting series.

    ``production`` — per ``(root_sector_from, root_region_to, root_time)`` the
    consumer region's own ICSG production of the input product in CC
    (``production_cc``). Where given, and the group has ≥1 recorded import,
    the domestic residual is bounded by it and the remainder of the usage is
    spread over the import legs in BACI proportion (ADR 0024) — provided the
    recorded imports can carry that remainder (``Σ imp_cc ≥ residual −
    production``). Groups without a production row, import-free groups, and
    groups whose imports are too small to carry the remainder are unchanged.

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
    # An import leg is priced with its EXPORTER's concentration: the grade is a
    # property of what that region produced, not of where the cargo lands.
    conc = conc_for(conc_lookup, df["root_sector_from"], df["root_region_from"])
    df["imp_cc"] = np.where(df["is_domestic"], 0.0, conc * df["baci_mass"])

    grp_cols = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
    imp_sum = df.groupby(grp_cols, sort=False)["imp_cc"].transform("sum").to_numpy()
    # many-to-one left merge preserves df's row order
    usage = (df[grp_cols].merge(usage_raw, on=grp_cols, how="left",
                                validate="many_to_one")
             ["usage_raw"].astype(float).fillna(0.0).to_numpy())

    # A stock change is signed: a draw-down fans like a build of the same size,
    # and the sign travels with the value, not the weights.  Without this, only
    # usage > 0 was emitted and every negative MESC/USC row stayed undistributed
    # (ADR 0008, #40).
    signed = df["root_sector_to"].isin(SIGNED_SECTORS_TO).to_numpy()
    usage = np.where(signed, np.abs(usage), usage)

    dom_raw = np.maximum(0.0, usage - imp_sum)      # domestic residual (ADR 0002)
    imp_raw = df["imp_cc"].to_numpy()               # import leg (0 where untraded)
    prod = np.full(len(df), np.nan)

    if production is not None and len(production):
        # ADR 0024: a region cannot supply itself more of the product than it
        # produces; bound the residual and hand the remainder to the imports.
        pcols = ["root_sector_from", "root_region_to", "root_time"]
        prod = (df[pcols].merge(production[pcols + ["production_cc"]], on=pcols, how="left")
                ["production_cc"].to_numpy())
        remainder = dom_raw - np.maximum(np.nan_to_num(prod, nan=np.inf), 0.0)
        bounded = ~np.isnan(prod) & (imp_sum > 0) & (imp_sum >= remainder)
        dom_b = np.where(bounded, np.minimum(dom_raw, np.maximum(prod, 0.0)), dom_raw)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(bounded, (usage - dom_b) / imp_sum, 1.0)
        dom_raw = dom_b
        imp_raw = np.where(bounded, imp_raw * scale, imp_raw)

    df["weight"] = np.where(df["is_domestic"], dom_raw, imp_raw)

    # A stock change is drawn from the region's cathode supply in proportion:
    # its own production on the domestic leg, each partner's exports to it on
    # the import legs.  A region that refines nothing holds no stock change of
    # its own cathode (#40: 48 hub use totals at z 58.6), and one with neither
    # production nor imports leaves the change to the net-stock seed (ADR 0040).
    own = np.maximum(np.nan_to_num(prod, nan=0.0), 0.0)
    df["weight"] = np.where(signed, np.where(df["is_domestic"], own, df["imp_cc"]),
                            df["weight"])

    # Emit groups ICSG fans into (usage>0) that BACI either covers (≥1 import)
    # or observes (zero imports is then a real observation → all-domestic fan,
    # ADR 0018). Trade-blind regions stay unemitted → fan undistributed.
    observed = df["root_region_to"].isin(baci_observed_regions or set()).to_numpy()
    if withhold_fallback:
        held = np.array([k in withhold_fallback for k in
                         zip(df["root_sector_from"], df["root_sector_to"],
                             df["root_region_to"], df["root_time"])])
        observed = observed & ~held
    nowhere = signed & (imp_sum <= 0) & (own <= 0)
    covered = (usage > 0) & ((imp_sum > 0) | observed) & ~nowhere
    out = df.loc[covered, grp_cols[:2] + ["root_region_from", "root_region_to",
                                          "root_time", "weight"]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _latest_feed_run(label: str) -> str:
    dirs = sorted(glob.glob(os.path.join(FEED_REG, f"feed_{label}_*")))
    if not dirs:
        raise FileNotFoundError(f"no feed_{label}_* run under {FEED_REG}")
    return dirs[-1]


def _latest_feed_froot(label: str) -> np.ndarray:
    run = _latest_feed_run(label)
    print(f"   {label:10s} ← {os.path.basename(run)}")
    return np.load(os.path.join(run, "result.npz"))["F_root"]


# The weights' parents (#53), in a sidecar's vocabulary: the BACI feed run they
# read (``upstream``) and the registered inputs the lookups below resolve
# (``inputs``).  Written beside params.csv, where the params registry hashes it,
# so a rebuild from another run is a new version.
def parents(inputs: InputRegistries) -> dict:
    """The parents the weights are built from, as they stand now."""
    return {
        "upstream": {"feed_run_ids": [os.path.basename(_latest_feed_run("baci"))]},
        "inputs":   {"feed":   {i: inputs.feed.entry(i)["version"] for i in ("baci", "icsg")},
                     "params": {i: inputs.params.entry(i)["version"] for i in ("conc", "eff")}},
    }


def write_parents(out_dir: str, inputs: InputRegistries) -> dict:
    """Record the weights' parents beside their params.csv and return them."""
    built_from = parents(inputs)
    with open(os.path.join(out_dir, PARAMS_PARENTS), "w", encoding="utf-8") as f:
        json.dump(built_from, f, indent=2, sort_keys=True)
    return built_from


def check_parents(version_dir: os.PathLike, inputs: InputRegistries) -> None:
    """Raise unless the weights at *version_dir* were built from the BACI feed run
    and input versions now in force.

    Any ICSG or BACI change invalidates the weights, and four rounds fanned ICSG
    with stale ones before anyone noticed (6.2 pp of sum-row z², ADR 0024, 0027);
    ``icsg_feed`` calls this before it fans (#53).
    """
    path = os.path.join(version_dir, PARAMS_PARENTS)
    if not os.path.isfile(path):
        raise ValueError(
            f"trade_shares at {version_dir} records no {PARAMS_PARENTS}, so its weights "
            f"cannot be checked against the BACI feed run and inputs in force — rebuild "
            f"them: python scripts/build_trade_shares.py")
    with open(path, encoding="utf-8") as f:
        built_from = json.load(f)
    now = parents(inputs)
    if built_from != now:
        raise ValueError(
            f"trade_shares at {version_dir} is stale: built from {built_from}, but {now} "
            f"is in force — rebuild them (python scripts/build_trade_shares.py), then "
            f"the ICSG feed")


def _conc_lookup() -> dict:
    """``(traded sector, exporter region) -> conc nominal``; ``"*"`` is the fallback.

    Keyed by region rather than by sector alone so that a region-resolved family
    cannot be silently collapsed to one arbitrary row by the caller. Every row
    is region-wildcarded today, so this resolves to one value per sector.
    """
    inputs = _input_registries()
    lookup = conc_nominal_lookup(inputs.params.resolve("conc") / "params.csv")
    return {(s, r): v for (s, r), v in lookup.items() if s in TRADED_SECTORS}


def _input_registries():
    """Read-only: a lookup never versions a ``current/`` (#93)."""
    return InputRegistries.load(root=_INPUT_ROOT)


def _register_inputs():
    """Version every edited ``current/`` — ``main``'s call, as a pipeline step's."""
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
        # ADR 0021: the refining Use row is total refined production placed on
        # the anode entering ER, not the matte entering CFR.
        return "Refining_Anode"
    return proc


def icsg_raw_usage(sectors) -> pd.DataFrame:
    """Per-fan-group ICSG Use totals straight from the raw ICSG rows.

    Group key is the fan's ``(root_sector_from, root_sector_to,
    root_region_to, root_time)``: the input product (sector_map ``icsg_type``),
    the consumer process/boundary (``root_sector``), the mapped consumer region
    and the year. The builder's only source of usage (#66): a distributed fan's
    ``F_root`` total reproduces it to 3.5e-16 relative (all 3,113 groups of
    ``feed_icsg_20261012_130637_fa562177``, 599 of them signed stock groups),
    but that feed run is built from these weights, so reading it after an ICSG
    change hands back the previous data. Signed stock rows (MESC, USC) keep
    their sign here; the caller fans them by magnitude.
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


def _eff_smelting() -> float:
    """`effsmelting` nominal from the live eff params (ADR 0005: pooled inputs)."""
    csv = _input_registries().params.resolve("eff") / "params.csv"
    p = pd.read_csv(csv, sep=None, engine="python", comment="#", dtype=str)
    row = p[(p["param_type"] == "efficiency") & (p["a_sector_to"] == "Smelting")]
    return float(row["value_nominal"].iloc[0])


def chain_inconsistent_refining(k: float = 0.5) -> set[tuple[str, str]]:
    """Region-years whose ICSG refining feed contradicts their own smelting.

    A region-year's smelted matte must be refined at home or exported; ICSG
    reports both sides, and where

        refining_feed < k · eff_smelting · (smelting concentrate + scrap)

    the two series cannot both be right. Restoring the refining side as a
    covered prior there makes the Matte node's *hard* mass balance
    unsatisfiable at the priors, and the solver resolves it by collapsing the
    smelting chain (Russia 1995: 545.0 kt CC in vs 50.8 out; Iran 1996:
    116.0 vs 3.8). ADR 0020 therefore withholds the ADR-0018 fallback for
    those groups, leaving the fan undistributed as it was before.

    Read straight from the raw ICSG rows — no model state, so it can gate the
    feed build that produces that state. Returns ``{(root_region, year)}``.

    **Disabled by ADR 0021** (``WITHHOLD_CHAIN_INCONSISTENT``). Since refined
    production anchors ``Anode→ER`` rather than ``Matte→CFR``, a refining total
    below the region's smelting no longer makes any node unsatisfiable — it is
    the signature of an anode/blister exporter, and the matte's outlet is the
    observed anode export leg. Kept as a diagnostic, and so the remedy can be
    revived from one constant if the measurement calls for it.
    """
    fdir = _input_registries().feed.resolve("icsg")
    raw = pd.read_csv(fdir / "values.csv")
    rmap = pd.read_excel(fdir / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw["Region"].map(dict(zip(rmap["src_region"],
                                            rmap["root_region"])))
    raw["yr"] = raw["Year"].astype(str)
    raw = raw[raw["reg"].notna()]
    smelt = raw[raw["Process"] == "Smelting"].groupby(["reg", "yr"])["Value"].sum()
    refine = (raw[(raw["Process"] == "Refining") & (raw["Type"] == "Anode")]
              .groupby(["reg", "yr"])["Value"].sum())
    d = pd.concat([(smelt * _eff_smelting()).rename("matte_in"),
                   refine.rename("matte_out")], axis=1).fillna(0.0)
    d = d[d["matte_in"] > 1.0]
    return set(d.index[d["matte_out"] < k * d["matte_in"]])


def imports_exhausted_groups(edges: pd.DataFrame) -> set:
    """Fan groups whose ``Σ imports = 0`` was *constructed*, not observed.

    [ADR 0027](../docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md)'s
    netting can consume every import leg of a hub-year while the region still
    has ICSG usage to cover. The resulting zero looks exactly like ADR 0018's
    "BACI observed the region and it recorded no imports", but it is an artefact
    of the re-attribution, not evidence of domestic sourcing — so the fan must
    not hand those groups an all-domestic split. They are withheld instead,
    leaving the fan undistributed: ADR 0008's honest "total known, split
    unknown". Reads the ledger the netting wrote beside the feed; absent ledger
    (an un-netted feed) means an empty set and unchanged behaviour.
    """
    path = _input_registries().feed.resolve("baci") / "reexport_ledger.csv"
    if not path.is_file():
        return set()
    led = pd.read_csv(path)
    flagged = led[led["imports_exhausted"].astype(bool)]
    if not len(flagged):
        return set()
    key = edges[["root_sector_from", "root_sector_to", "root_region_to",
                 "root_time"]].drop_duplicates()
    key["root_time"] = key["root_time"].astype(str)
    hit = key.merge(
        flagged[["sector", "region", "time"]].astype({"time": str}).rename(
            columns={"sector": "root_sector_from", "region": "root_region_to",
                     "time": "root_time"}),
        on=["root_sector_from", "root_region_to", "root_time"], how="inner")
    return set(zip(hit["root_sector_from"], hit["root_sector_to"],
                   hit["root_region_to"], hit["root_time"]))


def baci_observed_regions(acc, baci_froot) -> set[str]:
    """Consumer regions BACI observes at all (any cross-region import, any
    sector/year). Zero imports is informative only for these — see ADR 0018."""
    ei = _decoded_edges(acc)
    cross = ei[(ei["root_region_from"] != ei["root_region_to"])
               & (ei["valuation"] == "Mass")]
    v = baci_froot[cross["edge_id"].to_numpy()]
    return set(cross.loc[v > 0, "root_region_to"].unique())


def icsg_production(acc, icsg_froot, sectors) -> pd.DataFrame:
    """Per (product sector, region, year): the region's own ICSG production of
    the product in CC, read off the producing-process → product edge of the
    ICSG ``F_root`` (``PRODUCER_OF``). Rows for ``sectors`` only; keyed the way
    ``compute_trade_share_weights`` expects (``root_sector_from`` = product)."""
    ei = _decoded_edges(acc)
    frames = []
    for sec in sectors:
        m = ((ei["root_sector_from"] == PRODUCER_OF[sec]) & (ei["root_sector_to"] == sec)
             & (ei["root_region_from"] == ei["root_region_to"]) & (ei["valuation"] == "CC"))
        sub = ei.loc[m, ["root_region_to", "root_time", "edge_id"]].copy()
        sub["production_cc"] = np.nan_to_num(icsg_froot[sub["edge_id"].values], nan=0.0)
        g = sub.groupby(["root_region_to", "root_time"])["production_cc"].sum().reset_index()
        g.insert(0, "root_sector_from", sec)
        frames.append(g)
    cols = ["root_sector_from", "root_region_to", "root_time", "production_cc"]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def icsg_raw_production(sectors, edges: pd.DataFrame) -> pd.DataFrame:
    """Per (product, consumer region, year) that region's own ICSG production, CC.

    `icsg_production` reads the producing-process → product edge of the ICSG
    ``F_root``, which exists only where ICSG anchors that edge: there is no
    ICSG-anchored ``CP→Cathode`` edge, so cathode production is invisible to it.
    This reads the raw ICSG rows instead, through the definitions ADR 0027 states
    (cathode = refined + SX-EW; anode = smelter intake · eff_Smelting · eff_CFR),
    and is the *same* helper the re-export netting uses for its ``P`` — imported
    rather than restated, so the two can never drift apart.

    **A region ICSG does not report is a region that produces none, and the row
    must say so explicitly.** ADR 0024's guard reads a *missing* row as "unknown"
    and declines to bound; a raw-ICSG lookup returns nothing at all for Germany
    or Korea, which mine no concentrate — so emitting only the rows ICSG carries
    silently un-bounds exactly the phantom-mine regions the bound exists for
    (measured: Concentrate/Germany 0 → 38,731 z², Korea's phantom mine 149 →
    2,173 kt). The frame therefore spans every group the fan covers, defaulting
    to 0.0, which is what the `F_root` reading did implicitly.
    """
    from build_baci_reexport import icsg_intake_production   # deferred: cycle

    _, production, _ = icsg_intake_production(sectors)
    grid = (edges.loc[edges["root_sector_from"].isin(sectors),
                      ["root_sector_from", "root_region_to", "root_time"]]
            .drop_duplicates().copy())
    grid["root_time"] = grid["root_time"].astype(str)
    grid["production_cc"] = [
        max(0.0, float(production.get((s, r, t), 0.0)))
        for s, r, t in zip(grid["root_sector_from"], grid["root_region_to"],
                           grid["root_time"])]
    cols = ["root_sector_from", "root_region_to", "root_time", "production_cc"]
    return grid[cols].reset_index(drop=True)


def stock_legs_take_the_cathode_trade(edges: pd.DataFrame) -> pd.DataFrame:
    """A stock leg's BACI mass is the cathode trade on the same route's FU leg.

    BACI books cathode trade on the FU legs only (sector-pair fan 100/0/0, #36),
    so a stock fan reads its import pattern there; its own BACI legs are empty.
    """
    fu = edges[edges["root_sector_to"] == "FU"]
    key = ["root_sector_from", "root_region_from", "root_region_to", "root_time"]
    trade = fu.set_index(key)["baci_mass"]
    out = edges.copy()
    stock = out["root_sector_to"].isin(SIGNED_SECTORS_TO).to_numpy()
    idx = pd.MultiIndex.from_frame(out.loc[stock, key])
    out.loc[stock, "baci_mass"] = trade.reindex(idx).to_numpy()
    return out


def build_edges_frame(acc, baci_froot, sectors) -> pd.DataFrame:
    """One row per CC fan-target edge for `sectors`, with the paired baci_mass."""
    ei = _decoded_edges(acc)
    ei = ei[ei["root_sector_from"].isin(sectors)]
    cc   = ei[ei["valuation"] == "CC"].copy()
    mass = ei[ei["valuation"] == "Mass"][["from_id", "to_id", "time", "edge_id"]]
    mass = mass.rename(columns={"edge_id": "mass_edge_id"})
    cc = cc.merge(mass, on=["from_id", "to_id", "time"], how="left")
    cc["baci_mass"] = np.where(cc["mass_edge_id"].notna(),
                               baci_froot[cc["mass_edge_id"].fillna(0).astype(int).values],
                               np.nan)
    return stock_legs_take_the_cathode_trade(
        cc[["root_sector_from", "root_sector_to", "root_region_from",
            "root_region_to", "root_time", "baci_mass"]])


def main() -> None:
    print("=" * 60)
    print("Building BACI trade-share weights (ADR 0002 option D)")
    print(f"   sectors: {SECTORS}  (Scrap excluded — not traded)")
    _register_inputs()

    acc_id = load_config().pipeline.acc_id
    acc = AccountingStructure.load(id=acc_id, storage_dir=ACC_DIR)
    conc_lookup = _conc_lookup()
    _wild = {s_: v for (s_, r_), v in conc_lookup.items() if r_ == "*"}
    _reg = len(conc_lookup) - len(_wild)
    print(f"   conc: {_wild}" + (f"  (+{_reg} region-resolved rows)" if _reg else ""))
    baci = _latest_feed_froot("baci")

    edges = build_edges_frame(acc, baci, SECTORS)
    observed = baci_observed_regions(acc, baci)
    usage_raw = icsg_raw_usage(SECTORS)
    # ADR 0020's withhold, disabled by ADR 0021 — see WITHHOLD_CHAIN_INCONSISTENT.
    inconsistent = chain_inconsistent_refining()
    withhold = ({("Anode", "ER", r, t) for r, t in inconsistent}
                if WITHHOLD_CHAIN_INCONSISTENT else set())
    print(f"   chain-inconsistent refining: {len(inconsistent)} region-years"
          f"   withheld: {len(withhold)} (ADR 0021 disables the ADR-0020 gate)")
    # ADR 0027: a hub whose imports the netting exhausted has a *constructed*
    # zero import sum; it must not be read as observed domestic sourcing.
    exhausted = imports_exhausted_groups(edges)
    withhold = withhold | exhausted
    print(f"   imports exhausted by the ADR 0027 netting: {len(exhausted)} fan "
          f"groups withheld (left undistributed, ADR 0008)")
    all_regions = set(acc.node_index["region"])
    blind = sorted(all_regions - observed)
    print(f"   BACI-observed regions: {len(observed)}/{len(all_regions)}"
          f"   trade-blind (fans stay undistributed): {blind or 'none'}")
    production = icsg_raw_production(BOUND_DOMESTIC_SECTORS, edges)
    print(f"   production rows (raw ICSG, ADR 0027 definitions): {len(production):,} "
          f"for {BOUND_DOMESTIC_SECTORS}")
    shares = compute_trade_share_weights(edges, usage_raw, conc_lookup, observed,
                                         withhold, production)

    # ADR 0024 diagnostic: how many groups the production bound touched and
    # how much usage moved from the domestic leg onto the import legs.
    unbounded = compute_trade_share_weights(edges, usage_raw, conc_lookup, observed,
                                            withhold, None)
    dcols = ["root_sector_from", "root_sector_to", "root_region_from", "root_region_to", "root_time"]
    both = (unbounded.merge(shares, on=dcols, suffixes=("_free", "_bound")))
    dom_both = both[both["root_region_from"] == both["root_region_to"]]
    moved = (dom_both["weight_free"] - dom_both["weight_bound"])
    print(f"   ADR 0024 production bound ({BOUND_DOMESTIC_SECTORS}): "
          f"{int((moved > 1e-9).sum()):,} domestic legs bounded, "
          f"{moved.sum():,.0f} kt CC moved from domestic residual onto import legs")

    # ADR 0018 diagnostic: which groups are emitted only because of the
    # import-free domestic fallback, and how much ICSG tonnage they carry.
    pre = compute_trade_share_weights(edges, usage_raw, conc_lookup, None)
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
    built_from = write_parents(OUT_DIR, _input_registries())
    print(f"   built from : {built_from['upstream']['feed_run_ids']}, {built_from['inputs']}")
    print(f"   consumer groups: "
          f"{shares.groupby(['root_sector_from','root_sector_to','root_region_to','root_time']).ngroups:,}")

    # Snapshot/version the artifact via the params registry (provenance).
    entry = _register_inputs().params.entry("trade_shares")
    print(f"   registered trade_shares v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    main()
