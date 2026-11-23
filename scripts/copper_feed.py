"""copper_feed.py
===============
Copper's Stage 1: the three feeds as FeedSpecs (``ICSG_FEED``, ``BACI_FEED``,
``CRUDE_ORE_FEED``, #143) with their loaders, the labels and registered
inputs the feed-side algorithms of ``pysuto.feedprep`` are bound to, the raw-ICSG readers
(usage, intake, production), the observed-zero stamping of the ICSG feed (#76–#97), and the
**complete ICSG FeedSpec**, ``ICSG_SPEC`` (#147): ``ICSG_FEED`` plus the BACI-trade-share
weights (ADR 0002 option D; #53's refusal of stale weights) and the observed-zero hook.
``scripts/icsg_feed.py`` is ``run_feed(ICSG_SPEC)``; copper's Stage-1 hook,
:func:`run_stage1` (#109), runs the feed builds in their load-bearing order.

Every reader takes the ``InputRegistries`` it resolves from, read-only (#93); nothing here
reads the working directory.

Public API
----------
ICSG_SPEC                                         The complete copper ICSG feed
run_stage1                                        Copper's Stage-1 hook (#109)
TRADED_SECTORS, SIGNED_SECTORS_TO, ...            The labels the algorithms are bound to
ORE_GRADE_LABELS                                  The ore-grade triangle's labels
icsg_raw_usage, icsg_intake_production, icsg_raw_production, production_recipes
chain_inconsistent_refining                       Raw-ICSG readers
trade_share_parents, check_trade_share_parents    #53
observed_zero_sigma, icsg_coverage, reported_elsewhere, raw_production,
non_producer_mask, production_zero_mask, stamp_observed_zeros   The observed zeros
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pysuto.conc import ConcordanceTable
from pysuto.feedpipe import FeedSpec, check_file, run_feed
from pysuto.feedprep.ore_grade import OreGradeLabels
from pysuto.feedprep.trade_shares import (check_parents_record, decoded_edges,
                                          trade_share_parents as _trade_share_parents)
from pysuto.feedprep.reexport import MAX_CV_SCALE, icsg_aggregate_members
from pysuto.input_registry import InputRegistries
from pysuto.logging import get_logger
from pysuto.relation.param_relation import expand_edge_index
from pysuto.relation.params import read_params

_log = get_logger(__name__)

# The acc id the copper feeds are built on (the case's, scripts/copper.py).
ACC_ID = "copper"

# =============================================================================
# Stage 1 — the feeds (#143)
# =============================================================================
# The three copper feeds as :class:`pysuto.feedpipe.FeedSpec`s, run by
# ``feedpipe.run_feed``. The loaders and hooks moved here unchanged from
# ``scripts/{icsg,baci,crude_ore}_feed.py``. ``ICSG_FEED`` is the loaders' part of ICSG's
# spec: ``ICSG_SPEC`` adds the trade-share weights and the
# observed-zero stamping (#147).

# The case's ROW region, for the ROW pedigree penalty (#119), and the
# Supply_Use column's vocabulary: each value mapped to the endpoint its rows classify.
ROW_REGION      = "ROW"
ICSG_AUX_SIDES  = {"Supply": "from", "Use": "to"}


def icsg_sector_key(proc: str, typ: str, supply_use: str) -> str:
    """Composite key for ICSG's ``sector_map``: disambiguates process+material combos."""
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


def _concordance_maps(sheets: dict, sector_cols: list, var_cols: list) -> dict:
    """The five concordance maps of a feed's workbook ``sheets``; ``sector_cols`` and
    ``var_cols`` name the columns of ``sector_map`` and ``variable_map`` it reads."""
    sector_map    = sheets["sector_map"][sector_cols].drop_duplicates()
    region_map    = sheets["region_map"][
        ["src_region", "root_region"]].drop_duplicates()
    time_map      = sheets["time_map"][
        ["src_time", "root_time"]].drop_duplicates()
    valuation_map = sheets["valuation_map"][
        ["src_valuation", "root_valuation"]].drop_duplicates()
    var_map       = sheets["variable_map"][var_cols].copy()

    time_map["src_time"]  = time_map["src_time"].astype(str)
    time_map["root_time"] = time_map["root_time"].astype(str)

    return dict(
        sector_map=sector_map,
        region_map=region_map,
        time_map=time_map,
        valuation_map=valuation_map,
        var_map=var_map,
    )


def load_icsg_raw(path: str) -> pd.DataFrame:
    """The ICSG input's raw rows, ``Year`` as text and ``_sector_key`` (``icsg_sector_key``)
    added: the key ``sector_map`` looks the rows up by."""
    check_file(path, "raw flow data")
    df = pd.read_csv(path)
    df["_sector_key"] = df.apply(
        lambda row: icsg_sector_key(row["Process"], row["Type"], row["Supply_Use"]),
        axis=1)
    df["Year"] = df["Year"].astype(str)
    return df


def load_icsg_tables(path: str) -> dict:
    """The five concordance maps of the ICSG workbook, its ``sector_map`` with the
    ``icsg_type`` and ``supply_use`` columns the fan policy reads."""
    check_file(path, "concordance workbook")
    sheets = pd.read_excel(path, sheet_name=None)
    # icsg_type / supply_use feed the fan-policy leg filter (ADR 0008)
    tables = _concordance_maps(
        sheets, ["src_sector", "root_sector", "icsg_type", "supply_use"],
        ["raw_var", "src_var", "Supply_Use"])
    tables["var_map"]["Supply_Use"] = tables["var_map"]["Supply_Use"].replace("", pd.NA)
    return tables


def icsg_leg_filter(tables: dict) -> dict:
    """Fan policy (ADR 0008): each ICSG Use variable names its input product
    (sector_map icsg_type) — its fan may only land on legs of that supplier
    sector. Without this, the '<any>' open side fans across every incoming
    edge of the consumer node (e.g. Smelting_Scrap value on Concentrate
    legs) and route-keyed trade-share weights bleed across products."""
    sm = tables["sector_map"]
    return dict(
        sm.loc[sm["supply_use"].astype(str).str.lower() == "use",
               ["src_sector", "icsg_type"]].itertuples(index=False, name=None)
    )


def load_baci_raw(path: str) -> pd.DataFrame:
    """The BACI input's raw rows, ``Year`` as text."""
    check_file(path, "raw flow data")
    df = pd.read_csv(path)
    # Year must be str for time_map lookup; Sector stays int for sector_map lookup
    df["Year"] = df["Year"].astype(str)
    return df


def load_crude_ore_raw(path: str) -> pd.DataFrame:
    """The crude-ore input's raw rows, ``Year`` as text."""
    check_file(path, "raw extraction data")
    df = pd.read_csv(path)
    # Year must be str for time_map lookup
    df["Year"] = df["Year"].astype(str)
    return df


def load_plain_tables(path: str) -> dict:
    """The concordance maps of a feed whose ``sector_map`` carries no fan-policy columns
    (BACI, crude ore)."""
    check_file(path, "concordance workbook")
    sheets = pd.read_excel(path, sheet_name=None)
    return _concordance_maps(sheets, ["src_sector", "root_sector"],
                             ["raw_var", "src_var"])


def baci_user_weights(build):
    """Load optional user weight table; returns ``(None, {})`` if the file is absent.

    The committed baci weights.csv is the legacy sector-pair format
    (root_sector_from, root_sector_to, weight — e.g. the Cathode→FU/USC/MESC
    100/0/0 to-side fan split; a zero weight leaves a leg uncovered, #40). Since the
    route+time-keyed user_weights seam (commit 8faf5a4) ``ConcordanceTable.from_raw``
    requires the full (sectors, regions, time) key, so a sector-pair row — which by the
    old semantics applied to every route and year — is expanded across all region
    pairs × years here, preserving those semantics.
    """
    weights_file = os.path.join(build.feed_dir, "weights.csv")
    if not os.path.isfile(weights_file):
        _log.info("   weights file not found — using equal allocation")
        return None, {}
    wdf = pd.read_csv(weights_file)
    _log.info("   weights file : %s  (%d entries)", weights_file, len(wdf))
    route_cols = {"root_region_from", "root_region_to", "root_time"}
    if route_cols.isdisjoint(wdf.columns):
        acc = build.acc
        regions = sorted(acc.node_index["region"].unique())
        times = [str(t) for t in acc.time_range]
        key = pd.MultiIndex.from_product(
            [regions, regions, times],
            names=["root_region_from", "root_region_to", "root_time"],
        ).to_frame(index=False)
        wdf = wdf.merge(key, how="cross")
        _log.info("   weights expanded: legacy sector-pair rows → %d "
                  "full-key rows (all routes × years)", len(wdf))
    return wdf, {}


# ``MAX_CV_SCALE`` (imported from ``feedprep.reexport``) is ADR 0027's cap on the σ
# multiplier; past ~4.8 ADR 0017's damping ceiling (cv 1.0 on a ~0.209 pedigree) makes
# further inflation inert.

# Per-root-edge σ multipliers, applied through the one ``cv_root_scale`` seam
# ADR 0003's ROW penalty already uses. Each file is one root leg per row
# (sector, from-region, to-region, year) and the factor by which that leg's σ
# must grow because part of it is an inference rather than an observation.
CV_SCALE_FILES = {
    "reexport_cv_scale.csv": "ADR 0027 re-attributed legs",
    "quantity_cv_scale.csv": "ADR 0033 reconstructed quantities",
}


def load_cv_scales(acc, feed_dir: str) -> "np.ndarray | None":
    """Combined per-root-edge CV multiplier from every scale file in ``feed_dir``.

    The factors **multiply**: 3,593 of 6,706 Concentrate legs carry an ADR 0027
    re-export scale, and the ADR 0033 reconstruction lands on many of the same
    legs, so assigning would silently undo the netting's σ on every overlap.
    The *product* is capped at ``MAX_CV_SCALE`` rather than each factor, so two
    corrections cannot compound past the point where ADR 0017's damping ceiling
    makes them inert.

    Each factor is applied to **every** cross-region Flow-origin edge of that
    leg in both valuations — for cathode that is the FU/USC/MESC triplet, which
    the feed's own fan splits 100/0/0 (#40).

    Returns ``None`` when no file is present, so the build is bit-identical to
    a feed that has been through neither transformation.
    """
    found = {n: os.path.join(feed_dir, n)
             for n in CV_SCALE_FILES if os.path.isfile(
                 os.path.join(feed_dir, n))}
    if not found:
        _log.info("   cv scales: none (feed is neither netted nor reconstructed)")
        return None

    ei = expand_edge_index(acc.edge_index, acc.node_index)
    ei = ei[(ei["region_from"] != ei["region_to"])
            & (ei["entity_from"] == "Flow")].copy()
    ei["root_time"] = ei["time"].astype(str)

    scale = np.ones(len(acc.edge_index), dtype=float)
    for name, path in found.items():
        df = pd.read_csv(path)
        df["root_time"] = df["time"].astype(str)
        j = ei.merge(
            df[["sector", "region_from", "region_to", "root_time", "cv_scale"]],
            left_on=["sector_from", "region_from", "region_to", "root_time"],
            right_on=["sector", "region_from", "region_to", "root_time"],
            how="inner")
        ids = j["edge_id"].to_numpy(dtype=int)
        scale[ids] *= j["cv_scale"].to_numpy(dtype=float)
        _log.info("   cv scale %-24s %s: %d root legs → %d edges "
                  "(median %.3f, max %.1f)", name, CV_SCALE_FILES[name],
                  len(df), len(j), float(j["cv_scale"].median()),
                  float(j["cv_scale"].max()))
    np.clip(scale, 1.0, MAX_CV_SCALE, out=scale)
    _log.info("   cv scale combined: %d edges scaled, max %.2f",
              int((scale > 1.0).sum()), float(scale.max()))
    return scale


ICSG_FEED = FeedSpec(
    feed_id               = "icsg",
    load_raw              = load_icsg_raw,
    load_tables           = load_icsg_tables,
    row_region            = ROW_REGION,
    aux_col               = "Supply_Use",
    aux_sides             = ICSG_AUX_SIDES,
    leg_filter            = icsg_leg_filter,
    # BACI-trade-share weights (ADR 0002 option D): None ⇒ uniform fan, which the fan
    # policy then refuses to distribute (ADR 0008). Attached by ICSG_SPEC,
    # with the observed-zero stamping (post_propagate).
    drop_unweighted_fans  = True,
)

# BACI has explicit from/to columns, so no aux discriminator; it is scoped to
# cross_region (mapped self-trade exclusion, ADR 0014).
BACI_FEED = FeedSpec(
    feed_id      = "baci",
    load_raw     = load_baci_raw,
    load_tables  = load_plain_tables,
    row_region   = ROW_REGION,
    edge_scope   = True,
    user_weights = baci_user_weights,
    cv_scale     = load_cv_scales,
    knobs        = {"MAX_CV_SCALE": MAX_CV_SCALE, "CV_SCALE_FILES": CV_SCALE_FILES},
)

# A single-region domestic feed: no aux discriminator.
CRUDE_ORE_FEED = FeedSpec(
    feed_id     = "crude_ore",
    load_raw    = load_crude_ore_raw,
    load_tables = load_plain_tables,
    row_region  = ROW_REGION,
)


# =============================================================================
# Labels of the feed-side algorithms
# =============================================================================

# Traded supplier sectors whose ICSG producer fan is BACI-weighted. Scrap is NOT traded
# and must never appear here. All are uniform producer fans (verified):
# Cathode→FU/USC, Concentrate→Smelting, Anode→ER, Matte→CFR.
TRADED_SECTORS    = ["Cathode", "Concentrate", "Anode", "Matte"]
# Destinations of the signed stock-change legs (GLOSSARY "Stock-change leg"; CP→PSC is
# not fanned). The same membership is kept in three other places, #39.
SIGNED_SECTORS_TO = ("MESC", "USC")
# The sector BACI books cathode trade on (sector-pair fan 100/0/0, #36): a stock fan reads
# its import pattern on that leg.
CATHODE_CARRIER_SECTOR_TO = "FU"
# ADR 0024: sectors whose fan's domestic leg is bounded by the consumer region's own ICSG
# production of the product. ADR 0027 amends it to all three: Cathode and Anode production
# cannot be read off the ICSG F_root, so `icsg_raw_production` derives all three from the
# raw ICSG rows.
BOUND_DOMESTIC_SECTORS = ["Concentrate", "Cathode", "Anode"]
# ADR 0020's withhold, retained but disabled by ADR 0021 (the ADR-0019 pattern: keep the
# detector, drop the remedy).
WITHHOLD_CHAIN_INCONSISTENT = False

# The traded products the ADR 0027 netting handles, and the HS6 code stamped on a *new* raw
# row (any code mapping to the root sector would do; this is the dominant one per product).
NETTED_SECTORS = ["Concentrate", "Matte", "Anode", "Cathode"]
HS_OF          = {"Concentrate": 260300, "Matte": 740110, "Anode": 740200,
                  "Cathode": 740311}
# An aggregate region has no warehouse.
HUB_EXCLUDE = {ROW_REGION}
# The copper acc's gross-mass valuation: the conc rows' a_valuation (#119).
GROSS_VALUATION = "Mass"

if "Scrap" in TRADED_SECTORS or "Scrap" in NETTED_SECTORS:
    raise ValueError("Scrap is not traded: it belongs in neither TRADED_SECTORS nor "
                     "NETTED_SECTORS")

# Registered inputs.
BACI_ID           = "baci"
ICSG_ID           = "icsg"
TRADE_SHARES_ID   = "trade_shares"     # params id of the BACI-trade-share producer weights
AGGREGATES_ID     = "icsg_aggregates"  # which root regions an ICSG aggregate row holds (#67)
CRUDE_ORE_ID      = "crude_ore"
ORE_GRADE_ID      = "ore_grade"
# The weights' parents (#53): the inputs their lookups resolve.
TRADE_SHARES_PARENT_FEEDS  = (BACI_ID, ICSG_ID)
TRADE_SHARES_PARENT_PARAMS = ("conc", "eff")
TRADE_SHARES_REBUILD       = "python scripts/build_trade_shares.py"

# The ore-grade triangle (ADR 0015): three conc-family roles, region/time-resolved, each
# filling the a-side selectors; the b-side is "same" except b_valuation (a = the gross
# edge, b = its content twin).
ORE_GRADE_LABELS = OreGradeLabels(
    roles = (
        ("cosupply", {"a_sector_from": "CO", "a_entity_from": "SB",
                      "region_col": "a_region_from"},
         "Cu in-situ grade of CO (SB) — supply edges (from-node=CO/SB)"),
        ("oreuse", {"a_sector_to": "Ore", "a_entity_to": "Flow",
                    "region_col": "a_region_to"},
         "Cu in-situ grade of Ore (Flow) — use edges (to-node=Ore/Flow)"),
        ("oresupply", {"a_sector_from": "Ore", "a_entity_from": "Flow",
                       "region_col": "a_region_from"},
         "Cu in-situ grade of Ore (Flow) — supply edges (from-node=Ore/Flow)"),
    ),
    route_sector   = {"mf": "MF", "sxew": "SXEW"},
    process_entity = "Process",
    tail_sector    = "Tailings",
    tail_entity    = "SB",
    conc_edge      = ("MF", "Concentrate"),
    sxew_edge      = ("SXEW", "CSXEW"),
    ore_edge       = ("CO", "Mining"),
)

# =============================================================================
# The observed zeros of the ICSG feed (#76–#97)
# =============================================================================

# #76: a weighted fan's domestic leg at trade-share weight 0 is an observed zero, covered
# at 0 with the net-stock seed's pricing (ADR 0040) on the usage U its fans distribute:
# σ = max(OBSERVED_ZERO_CV·|U|, OBSERVED_ZERO_FLOOR_KT). Mirrors collect_aggregate_impute's
# STOCK_SEED_CV / STOCK_SEED_ABS_FLOOR_KT: change both or neither. The edges are on ICSG's
# content valuation, so the floor is in kt CC.
OBSERVED_ZERO_CV       = 0.01
OBSERVED_ZERO_FLOOR_KT = 0.3
# The stock-change legs: their zero-weight domestic leg is a supply-mix share (ADR 0043)
# whose zero the net-stock seed prices (ADR 0040), not an observed zero.
STOCK_SECTORS_TO       = SIGNED_SECTORS_TO
# The products whose zero-weight domestic leg is an observed zero in a region that makes
# none of them, by its raw ICSG production (ADR 0024's bound): concentrate (#78) only.
# Cathode (#96) and anode are not stamped: #96's Cathode legs were measured and rejected
# (z² +37.8 % with the mine-output zeros, opt_20261021_105350_b1fea245; ADR 0024 am.),
# since they expose ICSG-usage gaps a zero cannot close.
OBSERVED_ZERO_SECTORS  = ("Concentrate",)
# The edges a region's own mine and SX-EW output leave on, as (process, product): the
# edges this feed books ICSG's Mining/Concentrate and SXEW/Cathode supply rows on. Where
# the region makes none of the product, the edge is an observed zero (#97, #31). The mine
# output zero, rejected on measurement in 2026-10, is the maintainer's decision since
# #200: a region-year ICSG covers and lists no mine for mines nothing (measured by #201,
# opt_20261119_130302_b20f0dba; ADR 0024 am.).
PRODUCTION_EDGES       = (("MF", "Concentrate"), ("SXEW", "CSXEW"))
# Regions whose zero is never a report: ROW is the sum of the members ICSG lists, with its
# "Others" rows among them.
AGGREGATE_REGIONS      = (ROW_REGION,)
# Other registered inputs that report a mine or an SX-EW plant: where one does, ICSG's
# missing row is missing data (a small producer under "Others"), and the output edge is
# not stamped (ADR 0019). The crude-ore feed reports mining wherever its ore mass is > 0;
# an ore-grade row reports the product its from-sector names, by region-year.
ORE_GRADE_ACTIVITY     = {"CO": "Concentrate", "SXEW": "CSXEW"}


# =============================================================================
# Registered-input readers
# =============================================================================

def eff_nominals(inputs: InputRegistries) -> dict:
    """``{a_sector_to: nominal}`` of the live eff params' efficiency rows (ADR 0005: pooled
    inputs); the last row of a sector wins."""
    out = {}
    for row in read_params(inputs.params.resolve("eff") / "params.csv", ("efficiency",)):
        out[row.selectors["a_sector_to"]] = float(row.value_nominal)
    return out


def eff_smelting(inputs: InputRegistries) -> float:
    """`effsmelting` nominal from the live eff params (ADR 0005: pooled inputs); the first
    efficiency row into Smelting."""
    for row in read_params(inputs.params.resolve("eff") / "params.csv", ("efficiency",)):
        if row.selectors["a_sector_to"] == "Smelting":
            return float(row.value_nominal)
    raise ValueError("eff params: no efficiency row into Smelting")


def _icsg_raw(inputs: InputRegistries):
    """The registered ICSG input's raw rows and its concordance dir."""
    fdir = inputs.feed.resolve(ICSG_ID)
    return pd.read_csv(fdir / "values.csv"), fdir


def icsg_raw_usage(inputs: InputRegistries, sectors) -> pd.DataFrame:
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
    raw, fdir = _icsg_raw(inputs)
    smap_df = pd.read_excel(fdir / "conc.xlsx", sheet_name="sector_map")
    rmap_df = pd.read_excel(fdir / "conc.xlsx", sheet_name="region_map")
    regmap = dict(zip(rmap_df["src_region"], rmap_df["root_region"]))
    smap = {r["src_sector"]: (r["icsg_type"], r["root_sector"])
            for _, r in smap_df[smap_df["supply_use"] == "Use"].iterrows()}

    u = raw[raw["Supply_Use"] == "Use"].copy()
    key = [icsg_sector_key(p, t, su) for p, t, su
           in zip(u["Process"], u["Type"], u["Supply_Use"])]
    u["root_sector_from"] = [smap.get(k, (None, None))[0] for k in key]
    u["root_sector_to"]   = [smap.get(k, (None, None))[1] for k in key]
    u["root_region_to"]   = u["Region"].map(regmap)
    u["root_time"]        = u["Year"].astype(str)
    u = u[u["root_sector_from"].isin(sectors) & u["root_region_to"].notna()]
    return (u.groupby(["root_sector_from", "root_sector_to",
                       "root_region_to", "root_time"])["Value"]
             .sum().rename("usage_raw").reset_index())


def chain_inconsistent_refining(inputs: InputRegistries,
                                k: float = 0.5) -> set[tuple[str, str]]:
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
    raw, fdir = _icsg_raw(inputs)
    rmap = pd.read_excel(fdir / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw["Region"].map(dict(zip(rmap["src_region"],
                                            rmap["root_region"])))
    raw["yr"] = raw["Year"].astype(str)
    raw = raw[raw["reg"].notna()]
    smelt = raw[raw["Process"] == "Smelting"].groupby(["reg", "yr"])["Value"].sum()
    refine = (raw[(raw["Process"] == "Refining") & (raw["Type"] == "Anode")]
              .groupby(["reg", "yr"])["Value"].sum())
    d = pd.concat([(smelt * eff_smelting(inputs)).rename("matte_in"),
                   refine.rename("matte_out")], axis=1).fillna(0.0)
    d = d[d["matte_in"] > 1.0]
    return set(d.index[d["matte_out"] < k * d["matte_in"]])


def production_recipes(mining, conc_in, scrap_sm, anode_in, scrap_cfr, sxew,
                       cathode_use, eff) -> tuple:
    """``(prod, use)`` per sector from the ICSG input series, all keyed (region, year).

        Concentrate  U = Smelting/Concentrate use       P = Mining/Concentrate supply
        Anode        U = Refining/Anode use (ADR 0021)  P = (matte + scrap into CFR) · eff_CFR
        Cathode      U = Semi + USC + MESC use          P = Refining/Anode use · eff_ER
                                                          + SXEW/Cathode supply
        Matte        U = max(P_matte, anode use ÷ eff_CFR − scrap into CFR)
                     — no ICSG series; a converter consumes at least its own
                     smelter's matte, and its anode output nets off the scrap it
                     also pools.       P = smelter intake · eff_Smelting
        CSXEW        no intake         P = SXEW/Cathode supply (the ICSG feed's
                                           non-producer test, #97)

    CFR pools matte **and** scrap (ADR 0005), so anode production must count both:
    Belgium 2015 smelts 146 kt of scrap but feeds its converter a further 155 kt
    of scrap directly, and a matte-only reading puts its anode output at 139 kt
    against the 380 kt its own refinery consumes — a secondary refiner made to
    look like a pure importer.

    Cathode production carries NO second scrap term: ``anode_in`` is the total
    refined output ÷ eff_ER (ADR 0021), so the secondary refined copper is
    already inside it. The former ``+ Refining/Scrap · eff_CFR · eff_ER`` was
    right only while the Anode row was primary-only, and counted secondary
    refined twice — 17 % of world cathode production in 2024 (ADR 0030).
    """
    smelt_in = conc_in.add(scrap_sm, fill_value=0.0)
    matte_p  = smelt_in * eff["Smelting"]
    prod = {
        "Concentrate": mining,
        "Matte":       matte_p,
        "Anode":       matte_p.add(scrap_cfr, fill_value=0.0) * eff["CFR"],
        "Cathode":     (anode_in * eff["ER"]).add(sxew, fill_value=0.0),
        "CSXEW":       sxew,
    }
    use = {
        "Concentrate": conc_in,
        "Anode":       anode_in,
        "Cathode":     cathode_use,
        "Matte":       pd.concat(
            [matte_p,
             (anode_in / eff["CFR"]).subtract(scrap_cfr, fill_value=0.0).clip(lower=0.0)],
            axis=1).max(axis=1),
    }
    return prod, use


def icsg_intake_production(inputs: InputRegistries, sectors=NETTED_SECTORS) -> tuple:
    """ICSG intake ``U``, production ``P`` and the reported ``(sector, region)`` set.

    Read straight from the raw ICSG rows — no model state — so this can gate a
    feed build that produces that state, and so the trade-share builder can reuse
    the same definitions (ADR 0024's bound needs exactly this ``P``). The
    arithmetic lives in ``production_recipes``.
    """
    raw, fdir = _icsg_raw(inputs)
    rmap = pd.read_excel(fdir / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw["Region"].map(dict(zip(rmap["src_region"], rmap["root_region"])))
    raw = raw[raw["reg"].notna()].copy()
    raw["yr"] = raw["Year"].astype(str)
    eff = eff_nominals(inputs)

    def ser(process, typ, supply_use):
        x = raw[(raw["Process"] == process) & (raw["Type"] == typ)
                & (raw["Supply_Use"] == supply_use)]
        return x.groupby(["reg", "yr"])["Value"].sum()

    prod, use = production_recipes(
        mining=ser("Mining", "Concentrate", "Supply"),
        conc_in=ser("Smelting", "Concentrate", "Use"),
        scrap_sm=ser("Smelting", "Scrap", "Use"),
        anode_in=ser("Refining", "Anode", "Use"),
        scrap_cfr=ser("Refining", "Scrap", "Use"),     # ADR 0021: scrap enters CFR
        sxew=ser("SXEW", "Cathode", "Supply"),
        cathode_use=ser("Semi", "Cathode", "Use")
                    .add(ser("USC", "Cathode", "Use"), fill_value=0.0)
                    .add(ser("MESC", "Cathode", "Use"), fill_value=0.0),
        eff=eff)

    intake, production, reported = {}, {}, set()
    for s in sectors:
        for (reg, yr), v in use.get(s, pd.Series(dtype=float)).items():
            intake[(s, reg, yr)] = float(v)
            reported.add((s, reg))
        for (reg, yr), v in prod[s].items():
            production[(s, reg, yr)] = float(v)
            reported.add((s, reg))
    return intake, production, reported


def icsg_raw_production(inputs: InputRegistries, sectors,
                        edges: pd.DataFrame) -> pd.DataFrame:
    """Per (product, consumer region, year) that region's own ICSG production, CC.

    The ICSG ``F_root`` carries a producing-process → product edge only where ICSG
    anchors one: there is no ICSG-anchored ``CP→Cathode`` edge, so cathode production
    is invisible to it (the ICSG feed's non-producer test reads this frame too, #96).
    This reads the raw ICSG rows instead, through the definitions ADR 0027 states
    (cathode = refined + SX-EW; anode = smelter intake · eff_Smelting · eff_CFR),
    and is the *same* helper the re-export netting uses for its ``P`` — one function,
    so the two can never drift apart.

    **A region ICSG does not report is a region that produces none, and the row
    must say so explicitly.** ADR 0024's guard reads a *missing* row as "unknown"
    and declines to bound; a raw-ICSG lookup returns nothing at all for Germany
    or Korea, which mine no concentrate — so emitting only the rows ICSG carries
    silently un-bounds exactly the phantom-mine regions the bound exists for
    (measured: Concentrate/Germany 0 → 38,731 z², Korea's phantom mine 149 →
    2,173 kt). The frame therefore spans every group the fan covers, defaulting
    to 0.0, which is what the `F_root` reading did implicitly.
    """
    _, production, _ = icsg_intake_production(inputs, sectors)
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


# =============================================================================
# The trade-share weights' parents (#53)
# =============================================================================

def latest_feed_run(registry_dir: str, label: str) -> str:
    """The newest ``feed_<label>_*`` run directory under ``registry_dir``."""
    dirs = sorted(glob.glob(os.path.join(registry_dir, f"feed_{label}_*")))
    if not dirs:
        raise FileNotFoundError(f"no feed_{label}_* run under {registry_dir}")
    return dirs[-1]


def trade_share_parents(inputs: InputRegistries, registry_dir: str) -> dict:
    """The parents the trade-share weights are built from, as they stand now: the newest
    BACI feed run under ``registry_dir`` and the input versions in force."""
    run = latest_feed_run(registry_dir, BACI_ID)
    return _trade_share_parents(os.path.basename(run), inputs,
                                TRADE_SHARES_PARENT_FEEDS, TRADE_SHARES_PARENT_PARAMS)


def check_trade_share_parents(version_dir, inputs: InputRegistries,
                              registry_dir: str) -> None:
    """Raise unless the weights at *version_dir* were built from the BACI feed run and
    input versions now in force (#53)."""
    check_parents_record(version_dir, lambda: trade_share_parents(inputs, registry_dir),
                         TRADE_SHARES_REBUILD)


# =============================================================================
# Observed zeros
# =============================================================================

def observed_zero_sigma(legs: pd.DataFrame, F_src: np.ndarray,
                        sigma_root: np.ndarray) -> np.ndarray:
    """σ of every observed zero, NaN on every other root edge (#76).

    ``legs`` lists the zero-weight domestic legs (``src_id``, ``edge_id``) of
    ``ConcordanceTable.zero_weight_domestic_legs``. Each edge is priced on the
    usage its fans distribute, ``U = Σ |F_src[src_id]|``: an aggregate region
    (ROW) fans each member country's row onto the same legs. A leg another fan
    feeds (``sigma_root > 0``) is a flow, not an observed zero.
    """
    edge = legs["edge_id"].to_numpy()
    usage = np.zeros(len(sigma_root))
    np.add.at(usage, edge, np.abs(F_src[legs["src_id"].to_numpy()]))
    out = np.full(len(sigma_root), np.nan)
    out[edge] = np.maximum(OBSERVED_ZERO_CV * usage[edge], OBSERVED_ZERO_FLOOR_KT)
    out[sigma_root > 0] = np.nan
    return out


def icsg_coverage(raw_df: pd.DataFrame, region_map_df: pd.DataFrame, years,
                  aggregate_of: dict) -> set[tuple[str, str]]:
    """The ``(region, year)`` pairs ICSG covers, where a missing production is a 0 (#97).

    A region is covered when any ICSG row maps to it, in any table and any year. ICSG's
    mine, smelter, refinery and SX-EW tables list every producing country by name, the
    small ones under "Others", so a region ICSG lists anywhere and not in a table that
    year produces none of that table's product. Years without any ICSG row for the region
    count too: Mauritania before 2006, Panama before 2019 and Botswana in 2019–20 had no
    mine. Not covered: ``AGGREGATE_REGIONS``, whose production is a sum of members; a root
    region no row maps to (Luxembourg, reported jointly with Belgium); and a region-year
    ICSG reports inside one of its own aggregate rows, ``aggregate_of``
    (``feedprep.reexport.icsg_aggregate_members``), a reporting gap (#67).
    """
    root = raw_df["Region"].map(dict(zip(region_map_df["src_region"],
                                         region_map_df["root_region"])))
    regions = set(root.dropna()) - set(AGGREGATE_REGIONS)
    return {(r, str(t)) for r in regions for t in years} - set(aggregate_of)


def reported_elsewhere(crude_ore_df: pd.DataFrame, crude_ore_region_map: pd.DataFrame,
                       ore_grade_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    """``(product, region, year)`` where another registered input reports the output (#97).

    Mining (``Concentrate``) wherever the crude-ore feed's ore mass is > 0, and wherever
    an ore-grade row starts at ``CO``; SX-EW (``CSXEW``) wherever one starts at
    ``SXEW`` (``ORE_GRADE_ACTIVITY``). There ICSG's missing row is a gap, and a zero
    stamped on it would contradict that input (ADR 0019).
    """
    region = crude_ore_df["Region"].map(dict(zip(crude_ore_region_map["src_region"],
                                                 crude_ore_region_map["root_region"])))
    ore = crude_ore_df.assign(region=region, year=crude_ore_df["Year"].astype(str))
    ore = ore.groupby(["region", "year"])["Value"].sum()
    out = {("Concentrate", r, y) for (r, y), v in ore.items() if v > 0}
    rows = ore_grade_df[ore_grade_df["a_sector_from"].isin(list(ORE_GRADE_ACTIVITY))]
    out |= set(zip(rows["a_sector_from"].map(ORE_GRADE_ACTIVITY), rows["a_region_from"],
                   rows["a_time"].astype(str)))
    return out


def raw_production(inputs: InputRegistries, edges: pd.DataFrame) -> pd.DataFrame:
    """Raw ICSG production, CC, of every product the observed zeros test (#96, #97).

    ``icsg_raw_production`` reads it by ADR 0027's definitions, the ADR 0024 bound's own
    (cathode = refined + SX-EW, CSXEW = SX-EW supply), with an explicit 0 for every
    product in every region-year of ``edges``, so a row never depends on which edges the
    frame holds. Keyed ``root_sector_from`` = product.
    """
    products = sorted(set(OBSERVED_ZERO_SECTORS) | {p for _, p in PRODUCTION_EDGES})
    region_years = edges[["root_region_to", "root_time"]].drop_duplicates()
    grid = pd.concat([region_years.assign(root_sector_from=p) for p in products],
                     ignore_index=True)
    return icsg_raw_production(inputs, products, grid)


def _produces_none(product, region, time, production: pd.DataFrame,
                   covered: set) -> np.ndarray:
    """True where a covered region-year's ICSG production of ``product`` is 0.

    "A non-producer must say zero, not nothing" (ADR 0024). A missing production row is 0.
    """
    p = production.set_index(["root_sector_from", "root_region_to",
                              "root_time"])["production_cc"]
    key = pd.MultiIndex.from_arrays([np.asarray(product), np.asarray(region),
                                     np.asarray(time)])
    own = p.reindex(key).fillna(0.0).to_numpy()
    seen = pd.MultiIndex.from_arrays([np.asarray(region), np.asarray(time)]).isin(covered)
    return seen & (own == 0.0)


def non_producer_mask(edges: pd.DataFrame, production: pd.DataFrame, covered: set,
                      n_edges: int) -> np.ndarray:
    """Per root edge: True where a fan's zero-weight domestic leg may be an observed zero.

    The edge carries a product in ``OBSERVED_ZERO_SECTORS`` into a covered region-year
    that produces none of it (#78, #96). A zero-weight domestic leg anywhere else stays
    open to the imputer: in a producer it is the fan's residual where imports exceed
    usage, not an observation.

    ``edges``: ``edge_id, root_sector_from, root_region_to, root_time``; ``production``:
    ``raw_production``; ``covered``: ``icsg_coverage``.
    """
    none = _produces_none(edges["root_sector_from"], edges["root_region_to"],
                          edges["root_time"], production, covered)
    mask = np.zeros(n_edges, dtype=bool)
    mask[edges["edge_id"].to_numpy()] = (edges["root_sector_from"]
                                         .isin(OBSERVED_ZERO_SECTORS).to_numpy() & none)
    return mask


def production_zero_mask(edges: pd.DataFrame, production: pd.DataFrame, covered: set,
                         n_edges: int, elsewhere: set,
                         content: str = "CC") -> np.ndarray:
    """Per root edge: True on a covered non-producer's mine or SX-EW output (#97, #31).

    The edge is a ``PRODUCTION_EDGES`` pair in the content valuation (``content``, ``CC``:
    the valuation ICSG reports) in a covered region-year that produces none of the
    product, and no other input reports that output there (``elsewhere``, from
    ``reported_elsewhere``). Only the output edge is stamped; hard rows carry the zero up
    the chain. The efficiency bands (output ≥ eff_min · input) hold Ore→MF and Ore→SXEW at
    the output's zero, and the mass balances follow: MF's and SXEW's own leave their
    tailings nothing, the Ore node's takes Mining→Ore, Mining's takes CO→Mining, and
    CSXEW's takes CSXEW→CP. The conc relations take the Mass twins. Stamping any of these
    edges as well would price the same zero twice.

    ``edges``: ``edge_id, root_sector_from, root_sector_to, root_region_to, root_time,
    valuation``; ``production``: ``raw_production``; ``covered``: ``icsg_coverage``.
    """
    pair = pd.MultiIndex.from_frame(edges[["root_sector_from", "root_sector_to"]])
    out = edges[pair.isin(PRODUCTION_EDGES) & (edges["valuation"] == content).to_numpy()]
    key = [out["root_sector_to"], out["root_region_to"], out["root_time"]]
    mask = np.zeros(n_edges, dtype=bool)
    mask[out["edge_id"].to_numpy()] = (
        _produces_none(*key, production, covered)
        & ~pd.MultiIndex.from_arrays([k.to_numpy() for k in key]).isin(elsewhere))
    return mask


def stamp_observed_zeros(ct: ConcordanceTable, F_src: np.ndarray, result,
                         non_producer: np.ndarray, production_zero: np.ndarray) -> int:
    """Mark the feed's observed zeros on ``result`` and give them their σ (#76–#97).

    Two kinds: the zero-weight domestic legs ``non_producer`` allows, priced on their fans'
    usage (``observed_zero_sigma``), and the output edges ``production_zero`` marks. No fan
    distributes usage onto an output edge, so #78's rule leaves it the floor,
    ``OBSERVED_ZERO_FLOOR_KT``, the resolution ICSG reports at (ADR 0025). An edge the
    feed fills (``sigma_root > 0``) is a flow, never an observed zero. Sets
    ``result.observed_zero`` and overwrites ``result.sigma_root`` there, after
    propagation, so no per-edge CV multiplier reaches them. Returns their count.
    """
    legs = ct.zero_weight_domestic_legs(STOCK_SECTORS_TO)
    legs = legs[non_producer[legs["edge_id"].to_numpy()]]
    sigma0 = observed_zero_sigma(legs, F_src, result.sigma_root)
    sigma0[production_zero & ~(result.sigma_root > 0)] = OBSERVED_ZERO_FLOOR_KT
    result.observed_zero = ~np.isnan(sigma0)
    result.sigma_root[result.observed_zero] = sigma0[result.observed_zero]
    return int(result.observed_zero.sum())


# =============================================================================
# The complete ICSG feed
# =============================================================================

def load_trade_shares(inputs: InputRegistries, registry_dir: str):
    """Load the optional BACI-trade-share weights as user_weights_df.

    Returns (df, entry) or (None, None) when the artifact is not registered, so
    the ICSG build falls back to the uniform producer fan (unchanged behaviour).
    Raises ValueError when the weights were built from another BACI feed run or
    other input versions than those in force (#53). Run
    scripts/build_trade_shares.py to (re)generate it.
    """
    if TRADE_SHARES_ID not in inputs.params.ids():
        _log.info("   trade_shares : not registered — uniform fan")
        return None, None
    version_dir = inputs.params.resolve(TRADE_SHARES_ID)
    check_trade_share_parents(version_dir, inputs, registry_dir)
    df = pd.read_csv(version_dir / "params.csv")
    entry = inputs.params.entry(TRADE_SHARES_ID)
    _log.info("   trade_shares : %s v%s (code=%s, %s weight rows)", TRADE_SHARES_ID,
              entry["version"], entry["code"], f"{len(df):,}")
    return df, entry


def _trade_share_weights(build):
    """``FeedSpec.user_weights``: the BACI-trade-share weights and, when registered, the
    input entry they are recorded under in the run's parents."""
    df, entry = load_trade_shares(build.inputs, build.registry_dir)
    return df, ({} if entry is None else {TRADE_SHARES_ID: entry})


def _stamp_feed_observed_zeros(build) -> None:
    """``FeedSpec.post_propagate``: the observed zeros (#76–#97), after propagation."""
    acc, raw_df, tables = build.acc, build.raw_df, build.tables
    ct, F_src, result, input_ref = build.ct, build.F_src, build.result, build.input_ref
    edges      = decoded_edges(acc)
    reader     = InputRegistries.load(root=build.input_root)
    ore_dir    = reader.feed.resolve(CRUDE_ORE_ID)
    input_ref[AGGREGATES_ID] = reader.params.entry(AGGREGATES_ID)
    input_ref[CRUDE_ORE_ID]  = reader.feed.entry(CRUDE_ORE_ID)
    input_ref[ORE_GRADE_ID]  = reader.params.entry(ORE_GRADE_ID)
    covered    = icsg_coverage(
        raw_df, tables["region_map"], sorted(edges["root_time"].unique()),
        icsg_aggregate_members(reader.params.resolve(AGGREGATES_ID) / "params.csv",
                               AGGREGATES_ID))
    elsewhere  = reported_elsewhere(
        pd.read_csv(ore_dir / "values.csv"),
        pd.read_excel(ore_dir / "conc.xlsx", sheet_name="region_map"),
        pd.read_csv(reader.params.resolve(ORE_GRADE_ID) / "params.csv",
                    sep=None, engine="python", dtype=str))
    production = raw_production(reader, edges)
    non_producer    = non_producer_mask(edges, production, covered, len(result.F_root))
    production_zero = production_zero_mask(edges, production, covered, len(result.F_root),
                                           elsewhere, acc.valuation_roles.content)
    n_zero = stamp_observed_zeros(ct, F_src, result, non_producer, production_zero)
    n_out  = int((result.observed_zero & production_zero).sum())
    _log.info("   observed zeros: %s in region-years ICSG covers that produce none of "
              "the product: %s domestic legs at trade-share weight 0 (#76, #78, #96), "
              "%s mine and SX-EW output edges (#97, #31)",
              f"{n_zero:,}", f"{n_zero - n_out:,}", f"{n_out:,}")


# The complete copper ICSG feed: ICSG_FEED with the BACI-trade-share weights
# (user_weights) and the observed-zero stamping (post_propagate), whose module constants
# it records as its knobs (#210); they are defined below ICSG_FEED, so they are set here.
ICSG_SPEC = dataclasses.replace(
    ICSG_FEED, user_weights=_trade_share_weights,
    post_propagate=_stamp_feed_observed_zeros,
    knobs={"OBSERVED_ZERO_CV":       OBSERVED_ZERO_CV,
           "OBSERVED_ZERO_FLOOR_KT": OBSERVED_ZERO_FLOOR_KT,
           "STOCK_SECTORS_TO":       STOCK_SECTORS_TO,
           "OBSERVED_ZERO_SECTORS":  OBSERVED_ZERO_SECTORS,
           "PRODUCTION_EDGES":       PRODUCTION_EDGES,
           "AGGREGATE_REGIONS":      AGGREGATE_REGIONS,
           "ORE_GRADE_ACTIVITY":     ORE_GRADE_ACTIVITY})


# =============================================================================
# Stage 1 — the hook (#109)
# =============================================================================

def baci_source_version(data_dir: str) -> int:
    """The registered gross BACI version the re-export netting reads (ADR 0027): the one
    the ICSG input of ``data_dir`` was parsed against, as its ``parents.json`` records it
    (``icsg_clean --baci-version N --to-feed``). ``build_baci_reexport`` refuses any
    other (#63), so the hook reads the record instead of pinning a number.

    Raises
    ------
    ValueError
        If the ICSG input records no BACI version.
    """
    path = os.path.join(str(data_dir), "input", "feed", "icsg", "current", "parents.json")
    if not os.path.isfile(path):
        raise ValueError(f"stage 1: {path} is missing — the ICSG input records no BACI "
                         "version; run icsg_clean --baci-version N --to-feed first")
    with open(path, encoding="utf-8") as f:
        version = json.load(f).get("inputs", {}).get("feed", {}).get("baci")
    if version is None:
        raise ValueError(f"stage 1: {path} names no baci version")
    return int(version)

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _run_script(script: str, *args: str, data_dir: str) -> None:
    """Run ``scripts/<script>`` as a subprocess from the working directory, its output
    appended to ``<data_dir>/logs/<script>.log``; raise ``RuntimeError`` with its tail
    if it fails."""
    cmd = [sys.executable, str(_SCRIPTS_DIR / script), *args]
    os.makedirs(os.path.join(data_dir, "logs"), exist_ok=True)
    log_path = os.path.join(data_dir, "logs", f"{Path(script).stem}.log")
    _log.info("stage 1: %s", " ".join(cmd[1:]))
    with open(log_path, "a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-30:])
        raise RuntimeError(f"stage 1: {script} exited {rc} (log {log_path}):\n{tail}")


def run_stage1(data_dir: str) -> "dict[str, str]":
    """Copper's Stage 1: the feed builds in their load-bearing order (ADR 0024, 0027,
    0030), from the *registered* inputs of ``data_dir``.

    ``build_baci_reexport --source-version N`` (the gross BACI snapshot vN the ICSG input
    was parsed against, never the raw data) -> ``baci_feed`` -> ``build_trade_shares`` -> ``icsg_feed`` ->
    ``crude_ore_feed``. The BACI parse, the gross registration and ``icsg_clean
    --to-feed`` read ``data/input/raw`` and stay manual; ``build_ore_grade`` reads the
    finished feed runs and stays a separate step (the ICSG feed reads its params, so
    rerunning it here would make them stale against the run). A byte-identical rebuild
    bumps no input version: the registries version a ``current/`` by its sha256 only.
    The one input that does change is ``trade_shares``, whose ``parents.json`` names the
    BACI feed run, so every run of this hook registers a new ``trade_shares`` version.

    The caller holds the stage lock (the pipeline's ``feed`` step does), so the drivers
    run one at a time. The two ``build_*`` scripts run as subprocesses (their ``main``
    takes no lock) and the feeds in-process through ``run_feed`` (no lock either); the
    feed scripts' own ``__main__`` locks would refuse while the caller holds it.

    Returns
    -------
    dict
        ``{feed_id: feed_run_id}`` for ``baci``, ``icsg`` and ``crude_ore``.
    """
    data_dir = str(data_dir)
    _run_script("build_baci_reexport.py", "--source-version",
                str(baci_source_version(data_dir)), "--data-dir", data_dir,
                data_dir=data_dir)
    run_ids = {}

    def feed(spec) -> None:
        run_log = os.path.join(data_dir, "logs",
                               f"{spec.feed_id}_{datetime.now():%Y%m%d_%H%M%S}.log")
        run_ids[spec.feed_id] = run_feed(spec, acc_id=ACC_ID, data_dir=data_dir,
                                         run_log=run_log).run_id

    feed(BACI_FEED)
    _run_script("build_trade_shares.py", "--data-dir", data_dir, data_dir=data_dir)
    feed(ICSG_SPEC)
    feed(CRUDE_ORE_FEED)
    return run_ids
