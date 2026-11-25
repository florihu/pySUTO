"""trade_shares.py
================
The BACI-trade-share weights of ICSG's single-endpoint producer fan (ADR 0002 option D,
ADR 0018, ADR 0024, ADR 0027), as library functions with the case's labels as parameters.

``compute_trade_share_weights`` is the pure share computation. The frame builders around
it decode an accounting structure's edges into the CC fan-target frame it reads
(:func:`build_edges_frame`), the consumer regions BACI observes
(:func:`baci_observed_regions`) and the fan groups the ADR 0027 netting exhausted
(:func:`imports_exhausted_groups`); :func:`check_parents_record` is #53's refusal of
stale weights. Reading the registered inputs (raw ICSG usage and production) is the
case's business, ``scripts/copper_feed.py``; ``scripts/build_trade_shares.py`` is
argument parsing and calls.

Public API
----------
compute_trade_share_weights        The weights of a decoded CC-edge frame
stock_legs_take_the_cathode_trade  A stock leg's BACI mass is the carrier leg's
decoded_edges, build_edges_frame   Edge frames of an accounting structure
baci_observed_regions              Consumer regions BACI observes at all
imports_exhausted_groups           Fan groups the netting left without imports
traded_conc_lookup                 The conc nominals of the traded sectors
trade_share_parents                The parents record the weights are built from
check_parents_record               Raise unless a recorded parents block is current
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from pysuto.acc import AccountingStructure
from pysuto.input_registry import PARAMS_PARENTS
from pysuto.relation.param_relation import conc_for, conc_nominal_lookup


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
    *,
    signed_sectors_to,
) -> pd.DataFrame:
    """Compute raw fan weights from a decoded CC-edge frame.

    ``edges`` columns (one row per CC fan-target edge):
        root_sector_from, root_sector_to, root_region_from, root_region_to,
        root_time, baci_mass (BACI F_root Mass on the paired Mass edge; NaN/0
        where no trade).

    ``usage_raw`` — per fan group ``(root_sector_from, root_sector_to,
    root_region_to, root_time)`` the ICSG Use total in CC (column ``usage_raw``),
    read from the registered ICSG input by the case's raw-usage reader. A group without
    a row has no usage and is not emitted. Never the ICSG feed run's ``F_root`` fan
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

    ``signed_sectors_to`` — the destination sectors of the signed stock-change legs
    (keyword-only; the case's own, copper's MESC and USC).

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
    signed = df["root_sector_to"].isin(signed_sectors_to).to_numpy()
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
# Edge frames
# ---------------------------------------------------------------------------

def decoded_edges(acc: AccountingStructure) -> pd.DataFrame:
    """``acc.edge_index`` with the root sector and region of both endpoints and ``root_time``."""
    ei = acc.edge_index.copy()
    nidx = acc.node_index.reset_index(drop=True)
    ei["root_sector_from"] = nidx["sector"].values[ei["from_id"].values]
    ei["root_sector_to"]   = nidx["sector"].values[ei["to_id"].values]
    ei["root_region_from"] = nidx["region"].values[ei["from_id"].values]
    ei["root_region_to"]   = nidx["region"].values[ei["to_id"].values]
    ei["root_time"]        = ei["time"]
    return ei


def baci_observed_regions(acc, baci_froot) -> set[str]:
    """Consumer regions BACI observes at all (any cross-region import, any
    sector/year). Zero imports is informative only for these — see ADR 0018."""
    ei = decoded_edges(acc)
    cross = ei[(ei["root_region_from"] != ei["root_region_to"])
               & (ei["valuation"] == acc.valuation_roles.gross)]
    v = baci_froot[cross["edge_id"].to_numpy()]
    return set(cross.loc[v > 0, "root_region_to"].unique())


def stock_legs_take_the_cathode_trade(edges: pd.DataFrame, signed_sectors_to,
                                      carrier_sector_to: str) -> pd.DataFrame:
    """A stock leg's BACI mass is the cathode trade on the same route's carrier leg.

    BACI books cathode trade on the FU legs only (sector-pair fan 100/0/0, #36), so a
    stock fan reads its import pattern there; its own BACI legs are empty.
    ``carrier_sector_to`` is that leg's destination sector (copper's FU).
    """
    fu = edges[edges["root_sector_to"] == carrier_sector_to]
    key = ["root_sector_from", "root_region_from", "root_region_to", "root_time"]
    trade = fu.set_index(key)["baci_mass"]
    out = edges.copy()
    stock = out["root_sector_to"].isin(signed_sectors_to).to_numpy()
    idx = pd.MultiIndex.from_frame(out.loc[stock, key])
    out.loc[stock, "baci_mass"] = trade.reindex(idx).to_numpy()
    return out


def build_edges_frame(acc, baci_froot, sectors, signed_sectors_to,
                      carrier_sector_to: str) -> pd.DataFrame:
    """One row per CC fan-target edge for `sectors`, with the paired baci_mass."""
    roles = acc.valuation_roles
    ei = decoded_edges(acc)
    ei = ei[ei["root_sector_from"].isin(sectors)]
    cc   = ei[ei["valuation"] == roles.content].copy()
    mass = ei[ei["valuation"] == roles.gross][["from_id", "to_id", "time", "edge_id"]]
    mass = mass.rename(columns={"edge_id": "mass_edge_id"})
    cc = cc.merge(mass, on=["from_id", "to_id", "time"], how="left")
    cc["baci_mass"] = np.where(cc["mass_edge_id"].notna(),
                               baci_froot[cc["mass_edge_id"].fillna(0).astype(int).values],
                               np.nan)
    return stock_legs_take_the_cathode_trade(
        cc[["root_sector_from", "root_sector_to", "root_region_from",
            "root_region_to", "root_time", "baci_mass"]],
        signed_sectors_to, carrier_sector_to)


def imports_exhausted_groups(edges: pd.DataFrame, ledger_path) -> set:
    """Fan groups whose ``Σ imports = 0`` was *constructed*, not observed.

    [ADR 0027](../docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md)'s
    netting can consume every import leg of a hub-year while the region still
    has ICSG usage to cover. The resulting zero looks exactly like ADR 0018's
    "BACI observed the region and it recorded no imports", but it is an artefact
    of the re-attribution, not evidence of domestic sourcing — so the fan must
    not hand those groups an all-domestic split. They are withheld instead,
    leaving the fan undistributed: ADR 0008's honest "total known, split
    unknown". Reads the ledger the netting wrote beside the feed (``ledger_path``);
    absent ledger (an un-netted feed) means an empty set and unchanged behaviour.
    """
    path = ledger_path
    if not os.path.isfile(path):
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


def traded_conc_lookup(conc_csv, gross: str, traded_sectors) -> dict:
    """``(traded sector, exporter region) -> conc nominal``; ``"*"`` is the fallback.
    ``gross`` is the acc's gross-mass valuation label (``acc.valuation_roles.gross``).

    Keyed by region rather than by sector alone so that a region-resolved family
    cannot be silently collapsed to one arbitrary row by the caller. Every row
    is region-wildcarded today, so this resolves to one value per sector.
    """
    lookup = conc_nominal_lookup(conc_csv, gross=gross)
    return {(s, r): v for (s, r), v in lookup.items() if s in traded_sectors}


# ---------------------------------------------------------------------------
# The weights' parents (#53)
# ---------------------------------------------------------------------------

def trade_share_parents(baci_run_id: str, inputs, feed_ids, params_ids) -> dict:
    """The parents the weights are built from, in a sidecar's vocabulary: the BACI feed
    run they read (``upstream``) and the registered inputs the lookups resolve
    (``inputs``: their versions)."""
    return {
        "upstream": {"feed_run_ids": [baci_run_id]},
        "inputs":   {"feed":   {i: inputs.feed.entry(i)["version"] for i in feed_ids},
                     "params": {i: inputs.params.entry(i)["version"] for i in params_ids}},
    }


def check_parents_record(version_dir, now, rebuild: str) -> None:
    """Raise unless the weights at *version_dir* were built from the parents ``now()``.

    ``now`` is a callable, called only once the record is found: a version without one
    is refused before the parents in force are looked up.

    Any ICSG or BACI change invalidates the weights, and four rounds fanned ICSG
    with stale ones before anyone noticed (6.2 pp of sum-row z², ADR 0024, 0027);
    the ICSG feed calls this before it fans (#53). ``rebuild`` is the command that
    rebuilds them, named in the error.
    """
    path = os.path.join(version_dir, PARAMS_PARENTS)
    if not os.path.isfile(path):
        raise ValueError(
            f"trade_shares at {version_dir} records no {PARAMS_PARENTS}, so its weights "
            f"cannot be checked against the BACI feed run and inputs in force — rebuild "
            f"them: {rebuild}")
    with open(path, encoding="utf-8") as f:
        built_from = json.load(f)
    in_force = now()
    if built_from != in_force:
        raise ValueError(
            f"trade_shares at {version_dir} is stale: built from {built_from}, but {in_force} "
            f"is in force — rebuild them ({rebuild}), then the ICSG feed")
