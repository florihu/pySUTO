"""
toy.py
======
The **toy case** (#59): a small synthetic case of the fictional metal **orium**, generated
from a planted, perfectly balanced **ground truth**, observed through three **synthetic
feeds**, and run through all six stages into a run tree the caller chooses.

The world
---------
Three regions over the years ``"2001"``–``"2005"``, a gross-mass and an orium-content
valuation, and no ``ROW``: the world is closed.

* **Montara** mines about 1,000 kt of ore a year at about 1 % grade, mills it and exports
  most of the concentrate; its smelter is small.
* **Portua** is the port: smelters that run on imported concentrate and scrap, and no
  mines, a structural zero.
* **Isla** is self-sufficient and modest in everything.

Thirteen sectors (``ACC_SPEC``): ``Deposit → Mining → Ore → Milling → Concentrate →
Smelting → Metal → {Use, Stock}``, with ``WasteRock`` (a satellite of Mining, outside its
mass balance), ``Tailings``, ``Slag`` and ``Scrap`` as system-boundary nodes.
Concentrate is the one tradable product and ``Metal → Stock`` the one signed leg: 42
edges per year and valuation, 420 in all.

The sources
-----------
Three synthetic feeds report on the truth, each in its own vocabulary and each noisy at
exactly the coefficient of variation its pedigree implies (Laner et al. 2016, as Stage 1
computes it). The **Survey** reports ore extraction in gross mass, one row per mining
region-year. The **Bureau** reports the chain in orium content (mine, concentrate and
refined output, scrap feed, consumption and the signed stock change), with primary and
secondary refined output as two series that one root edge pools, and two region-years it
did not publish. The **Customs** feed reports the bilateral concentrate legs in gross
mass, under partner codes rather than region names. One of the three carries one
deliberate error; :func:`reveal` names it.

Stage 2 runs copper's cascade order, ``trade → network → conc``. The per-pair Milling
recovery relation is closed by the network (``network_backward``, from the Bureau's
concentrate output), so copper's ``eff`` ParametricImputer, which runs after the network,
would write no edge here, as it writes none in copper: it is not registered (a note for
#124's ``CaseSpec``).

Limitations, stated once
------------------------
* **Its own labels** (ADR 0034, #123). The toy declares ``Gross`` and ``Content`` as its
  valuations and ``Raw``, ``Sec`` and ``Demand`` as its boundary subs, bound to the
  framework's roles in ``ACC_SPEC`` (a ``role`` on each ``SB`` sector and each valuation);
  they are defined once, beside ``ACC_SPEC``, and none of copper's labels appears here.
* **No hard bands** (#27). No ``RatioBoundBuilder`` and no ``FlowBoundsBuilder`` are
  registered: every hard row is an equality, so Stage 5's posterior has no active set to
  choose and its intervals can be scored against the truth. The price is the band group
  the manuscript describes for copper (Method §3.1, ADR 0039), which carries the
  non-negativity of every flow: without it a posterior flow may go below zero. On the
  seed-59 chain Portua, which mines nothing, books a negative ore extraction in 2003
  (``Deposit → Mining`` −2.4 kt, the free slack of its mine-less chain).
* **A closed world.** No ``ROW`` region, so the ROW pedigree penalty never fires.
* **One fixed seed.** This module is ``src/``'s exception to "no RNG" (CLAUDE.md): the
  truth's year-to-year wiggle and the feeds' noise come from ``np.random.default_rng``.
  Stage 1 records the seed in every feed run's sidecar; :func:`scoreboard` reads it back.

Public API
----------
``ACC_SPEC``, ``ACC_REF``, :func:`build_acc` — the accounting structure (the #57 acc
spec) and the acc version every run names; ``GROSS``, ``CONTENT``, ``RAW``, ``SEC``,
``DEMAND``, ``UNITS`` — its labels.
:func:`truth` — the ground truth; :func:`make_feeds` — the three synthetic feeds;
:func:`write_params` — the ``conc``, ``eff`` and ``stripping_ratio`` CSVs.
:func:`run_stage1` … :func:`run_stage6` — the six stages into ``run_root/{acc, params,
feed, imp, const, opt, uncprop, psut}``, each returning its run id(s).
:func:`soft_residuals`, :func:`scoreboard` — the solver's soft residuals, and the
estimates of every stage scored against the truth (:class:`pysuto.diag.truth.TruthDiag`).
``check_<chapter>_<n>`` and :func:`solution` — the tutorial's exercise hooks (#60).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from pysuto._conditioning import condition_prior, condition_sigma_c
from pysuto.acc import AccountingStructure, write_build_record
from pysuto.collector import FeedCollector
from pysuto.conc import WeightMatrix
from pysuto.constraints import (
    ConstraintEngine,
    MassBalanceBuilder,
    RatioConstraintBuilder,
    SumConstraintBuilder,
)
from pysuto.diag.truth import TruthDiag, prior_z
from pysuto.feedpipe import (
    build_F_src,
    propagate_with_row_penalty,
    register_feed,
    step_B_build_concordance,
    step_C_build_weight_matrix,
)
from pysuto.impute import (
    FlatPriorImputer,
    ImputationEngine,
    NetworkImputer,
    ParametricImputer,
    TimeImputer,
    TradeImputer,
)
from pysuto.logging import get_logger
from pysuto.meta import DataFeedMetadata
from pysuto.psut import PSUT
from pysuto.registry import (
    ConstraintRunRegistry,
    DataFeedRegistry,
    ImputationRunRegistry,
    OptimRunRegistry,
    PsutRunRegistry,
    UncPropRunRegistry,
)
from pysuto.solver import ClarabelSolver
from pysuto.uncprop import KalmanUncProp

_log = get_logger("pysuto.cases.toy")

PathLike = Union[str, os.PathLike]


# =============================================================================
# The accounting structure
# =============================================================================

SEED    = 59
ACC_ID  = "toy"
REGIONS = ("Montara", "Portua", "Isla")
YEARS   = ("2001", "2002", "2003", "2004", "2005")

# The toy's own labels, defined once; the roles they are bound to are in ACC_SPEC (ADR 0034).
GROSS   = "Gross"     # the gross-mass valuation
CONTENT = "Content"   # the orium-content valuation
RAW     = "Raw"       # boundary sub of extraction (Deposit)
SEC     = "Sec"       # boundary sub of secondary inputs and residual outputs
DEMAND  = "Demand"    # boundary sub of final demand
# The bureau feed Supply_Use column: each value mapped to the endpoint its rows classify.
AUX_SIDES = {"Supply": "from", "Use": "to"}
UNITS   = {GROSS: "kt", CONTENT: f"kt {CONTENT}"}    # "kt" vs "kt Content" in every report

ACC_SPEC: dict = {
    "sectors": [
        {"name": "Deposit",     "entity": "SB",      "sub": RAW, "role": "nature"},
        {"name": "Mining",      "entity": "Process"},
        {"name": "Ore",         "entity": "Flow"},
        {"name": "WasteRock",   "entity": "SB",      "sub": SEC, "role": "secondary"},
        {"name": "Milling",     "entity": "Process"},
        {"name": "Concentrate", "entity": "Flow"},
        {"name": "Tailings",    "entity": "SB",      "sub": SEC, "role": "secondary"},
        {"name": "Smelting",    "entity": "Process"},
        {"name": "Scrap",       "entity": "SB",      "sub": SEC, "role": "secondary"},
        {"name": "Metal",       "entity": "Flow"},
        {"name": "Slag",        "entity": "SB",      "sub": SEC, "role": "secondary"},
        {"name": "Use",         "entity": "SB",      "sub": DEMAND, "role": "final_demand"},
        {"name": "Stock",       "entity": "SB",      "sub": DEMAND, "role": "final_demand"},
    ],
    "regions":    list(REGIONS),
    "valuations": [{"name": GROSS,   "role": "gross"},
                   {"name": CONTENT, "role": "content"}],
    "times":      list(YEARS),
    "supply":     {"Mining":   ["Ore", "WasteRock"],
                   "Milling":  ["Concentrate", "Tailings"],
                   "Smelting": ["Metal", "Slag"]},
    "use":        {"Deposit":     ["Mining"],
                   "Ore":         ["Milling"],
                   "Concentrate": ["Smelting"],
                   "Scrap":       ["Smelting"],
                   "Metal":       ["Use", "Stock"]},
    "tradable":   ["Concentrate"],
    "accounting": [
        {"from": "Process", "to": "Flow",    "same_region": True},
        {"from": "Process", "to": "SB",      "same_region": True},
        {"from": "Flow",    "to": "Process", "same_region": False},
        {"from": "Flow",    "to": "SB",      "same_region": False},
        {"from": "SB",      "to": "Process", "same_region": True},
    ],
    # the signed stock leg; every other edge keeps the default [0, 1e9]
    "bounds": [
        {"Sector_from": "Metal", "Region_from": "*", "Sector_to": "Stock",
         "Region_to": "*", "Time": "*", "Valuation": "*",
         "Lower": "-1000000", "Upper": "1000000"},
    ],
    # waste rock is a satellite mass flow outside the Mining balance (ADR 0038)
    "masks": [
        {"mask_name": "massbal_scope", "logic": "include",
         "sector_from": "*",      "sector_to": "*"},
        {"mask_name": "massbal_scope", "logic": "exclude",
         "sector_from": "Mining", "sector_to": "WasteRock"},
    ],
}

# The toy's acc has no input-registry entry; this ref stands in for one, so every run's
# sidecar names an acc version and the built structure's build record (#53) is checked
# against it as copper's is. ``code`` is a hash of ACC_SPEC.
ACC_REF = {"id": ACC_ID, "version": 1,
           "code": hashlib.sha256(json.dumps(ACC_SPEC, sort_keys=True).encode()
                                  ).hexdigest()[:8]}


def build_acc(storage_dir: PathLike) -> AccountingStructure:
    """Build the toy's accounting structure from ``ACC_SPEC`` and save it under
    ``storage_dir/toy`` with its build record (``ACC_REF``). Returns the structure
    (420 admissible edges)."""
    acc = AccountingStructure.from_spec(ACC_SPEC, ACC_ID, storage_dir=str(storage_dir))
    acc.save()
    write_build_record(str(storage_dir), ACC_REF)
    return acc


# =============================================================================
# The ground truth (numbers are the spec author's call, #59)
# =============================================================================

_ORE_KT    = {"Montara": (1000.0, 25.0), "Isla": (400.0, 10.0)}  # kt ore in 2003, kt/yr
_GRADE     = {"Montara": 0.010, "Isla": 0.008}                  # ore grade, kt Content per kt
_EFF_MILL  = 0.90                                               # milling recovery
_EFF_SMELT = 0.97                                               # smelting recovery
_C_CONC    = 0.28                                               # concentrate grade
_C_METAL   = 0.99                                               # metal grade
_C_SCRAP   = 0.70                                               # scrap grade
_K_STRIP   = 2.0                                                # waste rock per ore
# where each region's concentrate goes (shares of its output; Portua mills nothing)
_SHARES    = {"Montara": {"Montara": 0.20, "Portua": 0.70, "Isla": 0.10},
              "Isla":    {"Isla": 0.85, "Portua": 0.15}}
_SCRAP_KT  = {"Montara": (0.5, 0.02), "Portua": (3.0, 0.10), "Isla": (0.8, 0.03)}
# the signed leg: linear in time, crossing zero in every region (kt Content)
_STOCK_KT  = {"Montara": (0.05, -0.15), "Portua": (0.10, -0.40), "Isla": (-0.05, 0.10)}
_WIGGLE    = 0.01      # relative year-to-year noise on ore and scrap (the truth's draw)

_TRUTH_COLUMNS = ["time", "valuation", "from_region", "from_sector", "from_entity",
                  "to_region", "to_sector", "to_entity", "value"]
_EDGE_KEY      = ["time", "valuation", "from_sector", "from_region", "to_sector",
                  "to_region"]
_ENTITY        = {s["name"]: s["entity"] for s in ACC_SPEC["sectors"]}


def truth(seed: int = SEED) -> pd.DataFrame:
    """The ground truth: every edge's value in both valuations and every year.

    Ore and scrap follow a linear trend with a 1 % seeded wiggle; everything else follows
    from the nominal parameters of :func:`write_params` (grades, recoveries, the
    stripping ratio, the concentrate shares), so every ``Process`` and ``Flow`` node
    balances in both valuations, waste rock excluded from Mining's balance (ADR 0038).

    Returns
    -------
    DataFrame
        420 rows, columns ``time, valuation, from_region, from_sector, from_entity,
        to_region, to_sector, to_entity, value`` (in ``UNITS`` of the valuation).
    """
    rng  = np.random.default_rng([seed, 0])
    rows = []

    def edge(t, fs, fr, ts, tr, mass, cc):
        for valuation, value in ((GROSS, mass), (CONTENT, cc)):
            rows.append((t, valuation, fr, fs, _ENTITY[fs], tr, ts, _ENTITY[ts],
                         float(value)))

    for t in YEARS:
        c     = int(t) - 2003
        ore   = {r: 0.0 for r in REGIONS}
        scrap = {}
        for r in REGIONS:
            if r in _ORE_KT:
                lvl, slope = _ORE_KT[r]
                ore[r] = (lvl + slope * c) * (1.0 + _WIGGLE * rng.standard_normal())
            lvl, slope = _SCRAP_KT[r]
            scrap[r] = (lvl + slope * c) * (1.0 + _WIGGLE * rng.standard_normal())

        conc_cc = {}
        for r in REGIONS:
            m, g = ore[r], _GRADE.get(r, 0.0)
            conc_cc[r] = _EFF_MILL * g * m
            conc_mass  = conc_cc[r] / _C_CONC
            edge(t, "Deposit",  r, "Mining",      r, m,             g * m)
            edge(t, "Mining",   r, "Ore",         r, m,             g * m)
            edge(t, "Mining",   r, "WasteRock",   r, _K_STRIP * m,  0.0)
            edge(t, "Ore",      r, "Milling",     r, m,             g * m)
            edge(t, "Milling",  r, "Concentrate", r, conc_mass,     conc_cc[r])
            edge(t, "Milling",  r, "Tailings",    r, m - conc_mass, g * m - conc_cc[r])

        for s in REGIONS:
            in_cc, in_mass = 0.0, 0.0
            for r in REGIONS:
                leg = _SHARES.get(r, {}).get(s, 0.0) * conc_cc[r]
                edge(t, "Concentrate", r, "Smelting", s, leg / _C_CONC, leg)
                in_cc   += leg
                in_mass += leg / _C_CONC
            edge(t, "Scrap", s, "Smelting", s, scrap[s] / _C_SCRAP, scrap[s])
            in_cc      += scrap[s]
            in_mass    += scrap[s] / _C_SCRAP
            metal_cc    = _EFF_SMELT * in_cc
            metal_mass  = metal_cc / _C_METAL
            lvl, slope  = _STOCK_KT[s]
            stock       = lvl + slope * c
            edge(t, "Smelting", s, "Metal", s, metal_mass,           metal_cc)
            edge(t, "Smelting", s, "Slag",  s, in_mass - metal_mass, in_cc - metal_cc)
            edge(t, "Metal",    s, "Stock", s, stock / _C_METAL,     stock)
            edge(t, "Metal",    s, "Use",   s, metal_mass - stock / _C_METAL,
                 metal_cc - stock)

    return pd.DataFrame(rows, columns=_TRUTH_COLUMNS)


def truth_vector(frame: pd.DataFrame, acc: AccountingStructure) -> np.ndarray:
    """``frame``'s ``value`` column on the structure's edge ids (length ``acc.n_edges``).

    ``frame`` has the edge-key columns of :func:`truth`; every edge of ``acc`` must
    appear exactly once, else ``ValueError``.
    """
    merged = acc.edge_meta[["edge_id"] + _EDGE_KEY].merge(
        frame[_EDGE_KEY + ["value"]], on=_EDGE_KEY, how="left")
    if len(merged) != acc.n_edges or merged["value"].isna().any():
        raise ValueError(
            f"truth_vector: {int(merged['value'].isna().sum())} of {acc.n_edges} edges "
            f"have no value, or an edge appears twice ({len(merged)} rows)")
    out = np.empty(acc.n_edges, dtype=float)
    out[merged["edge_id"].to_numpy(dtype=np.int64)] = merged["value"].to_numpy(float)
    return out


# =============================================================================
# The three synthetic feeds
# =============================================================================

FEED_IDS = ("survey", "bureau", "customs")

# reliability, completeness, temporal, geographical, other correlation (Laner et al. 2016)
_PEDIGREE = {"survey":  (2, 3, 2, 2, 2),
             "bureau":  (2, 1, 1, 1, 1),
             "customs": (3, 2, 1, 1, 2)}
_MINING_REGIONS = ("Montara", "Isla")
_BUREAU_GAPS    = (("Isla", "2003"), ("Montara", "2004"))   # region-years not published
_BUREAU_UNIT    = "kt orium"                                # the Bureau's content unit
_CUSTOMS_CODE   = {"Montara": "101", "Portua": "202", "Isla": "303"}

# ── Private: the planted defect (Q27 b: named only by reveal()) ───────────────
_PLANTED = {
    "feed":        "customs",
    "from_sector": "Concentrate",
    "from_region": "Montara",
    "to_sector":   "Smelting",
    "to_region":   "Portua",
    "time":        "2003",
    "valuation":   GROSS,
    "factor":      1000.0,
    "story":       "a gross mass booked in kg where the feed reports t, read as kt: "
                   "the leg arrives ×1000 (the PNG case of ADR 0033)",
}
_PLANTED_EDGE = ("from_sector", "from_region", "to_sector", "to_region", "time",
                 "valuation")


def reveal() -> dict:
    """The planted defect: which feed, which edge, by what factor, and the story."""
    return dict(_PLANTED)


@dataclass
class SyntheticFeed:
    """A datafeed derived from the ground truth (GLOSSARY.md "Synthetic feed").

    Attributes
    ----------
    feed_id : str
        ``"survey"``, ``"bureau"`` or ``"customs"``.
    raw : DataFrame
        The feed's rows in its own vocabulary: ``Value``, ``Year``, ``Process``,
        ``Region`` and whatever else the feed needs (``FeedDiagnostics`` draws it).
    maps : dict
        The five concordance maps, keyed as ``ConcordanceTable.from_raw`` names them
        (``sector_map_df``, ``region_map_df``, ``time_map_df``, ``valuation_map_df``,
        ``var_map_df``).
    metadata : DataFeedMetadata
        The pedigree scores, as an Excel sidecar would declare them.
    aux_col : str or None
        The raw column that tells supply rows from use rows, where the feed has both.
    """
    feed_id:  str
    raw:      pd.DataFrame
    maps:     dict
    metadata: DataFeedMetadata
    aux_col:  Optional[str] = None


def _metadata(feed_id: str) -> DataFeedMetadata:
    rel, comp, temp, geo, oth = _PEDIGREE[feed_id]
    return DataFeedMetadata(
        feed_name                = feed_id,
        iedc_type                = "1_F_",
        reliability              = rel,
        completeness             = comp,
        temporal_correlation     = temp,
        geographical_correlation = geo,
        other_correlation        = oth,
        version                  = 0,
        person                   = "pysuto.cases.toy",
        dataset_link             = "synthetic (pysuto.cases.toy.make_feeds)",
    )


def _maps(sectors: dict, regions: dict, units: dict, variables: list,
          aux_col: Optional[str] = None) -> dict:
    var_columns = ["raw_var", "src_var"] + ([aux_col] if aux_col else [])
    return {
        "sector_map_df":    pd.DataFrame(sorted(sectors.items()),
                                         columns=["src_sector", "root_sector"]),
        "region_map_df":    pd.DataFrame(sorted(regions.items()),
                                         columns=["src_region", "root_region"]),
        "time_map_df":      pd.DataFrame({"src_time":  list(YEARS),
                                          "root_time": list(YEARS)}),
        "valuation_map_df": pd.DataFrame(sorted(units.items()),
                                         columns=["src_valuation", "root_valuation"]),
        "var_map_df":       pd.DataFrame(variables, columns=var_columns),
    }


def _noisy(values: np.ndarray, cv: float, rng: np.random.Generator,
           keep: np.ndarray) -> np.ndarray:
    """``values · (1 + cv·ε)`` with ε drawn N(0, 1) and rescaled so that its RMS over the
    ``keep`` rows is exactly 1: the feed's empirical CV against the truth is its
    pedigree CV, not a draw near it."""
    eps = rng.standard_normal(values.size)
    eps = eps / np.sqrt(np.mean(eps[keep] ** 2))
    return values * (1.0 + cv * eps)


def _finish(feed_id: str, frame: pd.DataFrame, rng: np.random.Generator,
            planted: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Replace the ``_truth`` column by the noisy ``Value``; multiply planted rows."""
    cv      = _metadata(feed_id).to_pedigree_scores().total_cv()
    planted = np.zeros(len(frame), dtype=bool) if planted is None else planted
    value   = _noisy(frame["_truth"].to_numpy(dtype=float), cv, rng, ~planted)
    value[planted] *= _PLANTED["factor"]
    return (frame.drop(columns="_truth").assign(Value=value)
                 .reset_index(drop=True))


def make_feeds(ground_truth: pd.DataFrame,
               seed: int = SEED) -> "dict[str, SyntheticFeed]":
    """Derive the three synthetic feeds from ``ground_truth`` (:func:`truth`):
    reclassified into each feed's vocabulary, noisy at exactly the pedigree CV, with the
    Bureau's two unpublished region-years left out and one deliberate error planted.

    Returns
    -------
    dict
        ``{"survey": SyntheticFeed, "bureau": ..., "customs": ...}``.
    """
    rng = np.random.default_rng([seed, 1])
    tv  = ground_truth.set_index(_EDGE_KEY)["value"]

    def val(t, v, fs, fr, ts, tr) -> float:
        return float(tv.loc[(t, v, fs, fr, ts, tr)])

    # ── Survey: ore extraction (Deposit → Mining), gross mass, 1:1 ───────
    survey = pd.DataFrame(
        [(t, r, "extraction", "crude ore", "kt",
          val(t, GROSS, "Deposit", r, "Mining", r))
         for t in YEARS for r in _MINING_REGIONS],
        columns=["Year", "Region", "Process", "Material", "Unit", "_truth"])
    feeds = {"survey": SyntheticFeed(
        feed_id  = "survey",
        raw      = _finish("survey", survey, rng),
        maps     = _maps({"crude ore": "Deposit"}, {r: r for r in REGIONS},
                         {"kt": GROSS},
                         [("Year",     "src_time"),
                          ("Material", "src_sector_from"),
                          ("Region",   "src_region_from"),
                          ("Region",   "src_region_to"),
                          ("Unit",     "src_valuation")]),
        metadata = _metadata("survey"),
    )}

    # ── Bureau: the chain in content, two refined series, two gaps ───────
    rows = []
    for t in YEARS:
        for r in REGIONS:
            if (r, t) in _BUREAU_GAPS:
                continue
            if r in _MINING_REGIONS:
                rows.append((t, r, "Supply", "mine", "ore",
                             val(t, CONTENT, "Mining", r, "Ore", r)))
                rows.append((t, r, "Supply", "concentrator", "concentrate",
                             val(t, CONTENT, "Milling", r, "Concentrate", r)))
            metal   = val(t, CONTENT, "Smelting", r, "Metal", r)
            scrap   = val(t, CONTENT, "Scrap",    r, "Smelting", r)
            ore_in  = sum(val(t, CONTENT, "Concentrate", q, "Smelting", r)
                          for q in REGIONS)
            primary = metal * ore_in / (ore_in + scrap)
            rows.append((t, r, "Supply", "primary refinery",   "refined", primary))
            rows.append((t, r, "Supply", "secondary refinery", "refined",
                         metal - primary))
            rows.append((t, r, "Use",    "secondary refinery", "scrap",   scrap))
            rows.append((t, r, "Use",    "fabrication",        "refined",
                         val(t, CONTENT, "Metal", r, "Use", r)))
            rows.append((t, r, "Use",    "stocks",             "refined",
                         val(t, CONTENT, "Metal", r, "Stock", r)))
    bureau = pd.DataFrame(rows, columns=["Year", "Region", "Supply_Use", "Process",
                                         "Item", "_truth"]).assign(Unit=_BUREAU_UNIT)
    feeds["bureau"] = SyntheticFeed(
        feed_id  = "bureau",
        raw      = _finish("bureau", bureau, rng),
        maps     = _maps({"mine": "Mining", "concentrator": "Milling",
                          "primary refinery": "Smelting",
                          "secondary refinery": "Smelting",
                          "fabrication": "Use", "stocks": "Stock",
                          "ore": "Ore", "concentrate": "Concentrate",
                          "refined": "Metal", "scrap": "Scrap"},
                         {r: r for r in REGIONS}, {_BUREAU_UNIT: CONTENT},
                         [("Year",    "src_time",        None),
                          ("Unit",    "src_valuation",   None),
                          ("Region",  "src_region_from", None),
                          ("Region",  "src_region_to",   None),
                          ("Process", "src_sector_from", "Supply"),
                          ("Item",    "src_sector_to",   "Supply"),
                          ("Item",    "src_sector_from", "Use"),
                          ("Process", "src_sector_to",   "Use")],
                         aux_col="Supply_Use"),
        metadata = _metadata("bureau"),
        aux_col  = "Supply_Use",
    )

    # ── Customs: bilateral concentrate legs, gross mass, partner codes ───
    rows = []
    for t in YEARS:
        for r in REGIONS:
            for s in REGIONS:
                leg = val(t, GROSS, "Concentrate", r, "Smelting", s)
                if r != s and leg > 0.0:
                    rows.append((t, "261790", _CUSTOMS_CODE[r], _CUSTOMS_CODE[s], leg))
    customs = pd.DataFrame(rows, columns=["Year", "Process", "Region", "Partner",
                                          "_truth"]).assign(Unit="kt")
    planted = ((customs["Year"] == _PLANTED["time"])
               & (customs["Region"] == _CUSTOMS_CODE[_PLANTED["from_region"]])
               & (customs["Partner"] == _CUSTOMS_CODE[_PLANTED["to_region"]])
               ).to_numpy()
    feeds["customs"] = SyntheticFeed(
        feed_id  = "customs",
        raw      = _finish("customs", customs, rng, planted),
        maps     = _maps({"261790": "Concentrate"},
                         {code: r for r, code in _CUSTOMS_CODE.items()},
                         {"kt": GROSS},
                         [("Year",    "src_time"),
                          ("Process", "src_sector_from"),
                          ("Region",  "src_region_from"),
                          ("Partner", "src_region_to"),
                          ("Unit",    "src_valuation")]),
        metadata = _metadata("customs"),
    )
    return feeds


# =============================================================================
# The parameters
# =============================================================================

_PARAM_COLUMNS = [
    "param_id", "param_type", "agg", "description",
    "a_sector_from", "a_entity_from", "a_region_from", "a_sector_to", "a_entity_to",
    "a_region_to", "a_time", "a_valuation",
    "b_sector_from", "b_entity_from", "b_region_from", "b_sector_to", "b_entity_to",
    "b_region_to", "b_time", "b_valuation",
    "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
    "pedigree_reliability", "pedigree_completeness", "pedigree_temporal_correlation",
    "pedigree_geographical_correlation", "pedigree_other_correlation",
    "epsilon", "bound_lower", "bound_upper", "source", "comment",
]
_SOURCE = "pysuto.cases.toy: a synthetic case; its ground truth uses this nominal value"

# A concentration row ties an edge's content to its gross mass: "into" rows cover the
# edges into the sector, "out" rows the edges out of it (the template's use and supply).
#    (param_id,                edges,  sector,        region,    value,             sd)
_CONC_ROWS = (
    ("concoreusemontara",     "into", "Ore",         "Montara", _GRADE["Montara"], 1e-3),
    ("concoresupplymontara",  "out",  "Ore",         "Montara", _GRADE["Montara"], 1e-3),
    ("concoreuseisla",        "into", "Ore",         "Isla",    _GRADE["Isla"],    8e-4),
    ("concoresupplyisla",     "out",  "Ore",         "Isla",    _GRADE["Isla"],    8e-4),
    ("concconcentrateuse",    "into", "Concentrate", "*",       _C_CONC,           0.02),
    ("concconcentratesupply", "out",  "Concentrate", "*",       _C_CONC,           0.02),
    ("concmetaluse",          "into", "Metal",       "*",       _C_METAL,          5e-3),
    ("concmetalsupply",       "out",  "Metal",       "*",       _C_METAL,          5e-3),
    ("concscrapsupply",       "out",  "Scrap",       "*",       _C_SCRAP,          0.05),
    ("concslaguse",           "into", "Slag",        "*",       0.015,             3e-3),
    ("conctailingsuse",       "into", "Tailings",    "*",       0.001,             2e-4),
    ("concwasterockuse",      "into", "WasteRock",   "*",       0.0,               0.0),
)

# An efficiency row ties a process's content product to its input; agg=true pools every
# input.
#    (param_id,      agg,    input, process,    product,       value,      min,  max)
_EFF_ROWS = (
    ("effmilling",  "",     "Ore", "Milling",  "Concentrate", _EFF_MILL,  0.85, 0.95),
    ("effsmelting", "true", "*",   "Smelting", "Metal",       _EFF_SMELT, 0.95, 0.99),
)


def _param_row(**fields) -> dict:
    row = dict.fromkeys(_PARAM_COLUMNS, "")
    row.update(source=_SOURCE, **fields)
    return row


def _a_b(a: tuple, b: tuple) -> dict:
    keys = _PARAM_COLUMNS[4:20]
    return dict(zip(keys, a + b))


def write_params(params_dir: PathLike) -> "dict[str, Path]":
    """Write the toy's ``conc``, ``eff`` and ``stripping_ratio`` CSVs as
    ``params_dir/<param_id>/params.csv`` in the project's params schema
    (``data/input/params/params_template.csv``). Returns ``{param_id: path}``."""
    same_content = ("same",) * 7 + (CONTENT,)
    conc = []
    for pid, edges, sector, region, value, sd in _CONC_ROWS:
        entity = _ENTITY[sector]
        a = (("*", "*", "*", sector, entity, region, "*", GROSS) if edges == "into"
             else (sector, entity, region, "*", "*", "*", "*", GROSS))
        conc.append(_param_row(
            param_id=pid, param_type="concentration",
            description=f"orium content of {sector} ({entity}) on the edges "
                        f"{'into' if edges == 'into' else 'out of'} it, "
                        f"{'every region' if region == '*' else region}",
            value_nominal=value, unc_type="sd", value_sd=sd, bound_lower=0,
            bound_upper=1, **_a_b(a, same_content)))
    eff = []
    for pid, agg, inp, process, product, value, lo, hi in _EFF_ROWS:
        entity = "*" if inp == "*" else _ENTITY[inp]
        eff.append(_param_row(
            param_id=pid, param_type="efficiency", agg=agg,
            description=f"orium recovery of {process}: {inp} in, {product} out"
                        + (" (pooled inputs, scrap included)" if agg else ""),
            value_nominal=value, unc_type="min_max", value_min=lo, value_max=hi,
            bound_lower=0, bound_upper=1,
            **_a_b((inp, entity, "*", process, "Process", "same", "*", CONTENT),
                   (process, "Process", "same", product, "Flow", "same", "same",
                    "same"))))
    strip = [_param_row(
        param_id="strippingratio", param_type="stripping_ratio",
        description="waste rock per ore mass at Mining, a satellite outside the Mining "
                    "balance (ADR 0038)",
        value_nominal=_K_STRIP, unc_type="min_max", value_min=1.5, value_max=2.5,
        **_a_b(("Mining", "Process", "*", "Ore", "Flow", "same", "*", GROSS),
               ("Mining", "Process", "same", "WasteRock", "SB", "same", "same",
                "same")))]

    out = {}
    for pid, rows in (("conc", conc), ("eff", eff), ("stripping_ratio", strip)):
        path = Path(params_dir) / pid / "params.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=_PARAM_COLUMNS).to_csv(path, index=False)
        out[pid] = path
    return out


# =============================================================================
# The six stages
# =============================================================================
# Knobs mirror the copper drivers' USER CONFIGURATION blocks unless a comment says why
# the toy departs from them (#124 turns them into CaseSpec fields).

P_PROP          = 1.3       # Lp exponent, every stage
Q_REG           = 3.0       # NetworkImputer / ParametricImputer backward-branch knee
ALPHA_MIN_REG   = 0.05      # inverse-multiplier sensitivity floor
FLAT_SIGMA_HIGH = 1.0e6     # noisy-zero flat prior σ; the waste-rock fill's σ (ADR 0038)
# Copper's cascade order without its eff ParametricImputer, which writes no edge after
# the network (see the module docstring).
CASCADE         = ("trade", "network", "conc")
STOCK_SECTORS   = ("Stock",)    # never written by the network's balance (ADR 0040)
# scripts/collect_aggregate_impute.py: CASCADE_MAX_ITER, TIME_WINDOW, TIME_MIN_OBS,
# TIME_GAP_INFLATION, TIME_METHOD and TIME_INTERPOLATE_ONLY by name; NETWORK_CV_CAP is
# the cv_cap=1.5 literal of its NetworkImputer call.
CASCADE_MAX_ITER      = 10
TIME_WINDOW           = 3
TIME_MIN_OBS          = 2
TIME_GAP_INFLATION    = 1.05
TIME_METHOD           = "linear"
TIME_INTERPOLATE_ONLY = True
NETWORK_CV_CAP        = 1.5
SIGMA_A_CV_MIN    = 0.001
SIGMA_A_ABS_FLOOR = 1.0e-6
SIGMA_C_CV_MIN    = 0.01
SIGMA_C_ABS_FLOOR = 1.0e-3
NODATA_ANCHOR_CV  = 0.05
# Copper spares its extraction chain from the no-data anchor where a grade row sets its
# magnitude (ADR 0013 am. 3). The toy's only no-data edges are Portua's mine-less chain
# and the concentrate legs no feed reports, none of which a relation sets: none is spared.
NODATA_ANCHOR_EXCLUDE = ()
SUM_SIGMA_FRACTION    = 0.05
# Ratio rows' flow-space scale floor. Copper's 100 kt would swamp the toy, whose content
# edges are 0.1–10 kt Content; the builder's default 1 kt is used.
RATIO_SIGMA_REL_FLOOR = 0.02
RATIO_SCALE_ABS_FLOOR = 1.0
# Solver: the ADR 0039 tolerances of scripts/run_solver.py, the cold prior.
TOL_FEAS, TOL_GAP_REL, TOL_GAP_ABS = 1e-8, 1e-8, 1e-8
REDUCED_TOL_FEAS, REDUCED_TOL_GAP_REL, REDUCED_TOL_GAP_ABS = 1e-6, 1e-4, 10.0
MAX_ITER     = 2000
# The solver's own default, not the copper driver's "defects" (#33): the toy's QP has no
# inequality cone, so Clarabel solves its KKT system directly (0 iterations) and the
# gap tolerance #33 tuned never binds. Solving for the defects subtracts G·a_prior from
# c first, and with one prior of ≈ 2·10⁴ kt among values of 1–10³ kt that costs digits:
# on the seed-59 chain the largest hard-row residual is 9.1e-11 kt under "defects" and
# 4.5e-13 kt under "flows", for the same a*.
QP_VARIABLES = "flows"
ALPHA        = 0.05         # Stage 5: (1 − α) intervals
BIND_TOL     = 1e-6
RANK_TOL     = 1e-10
LABEL        = "toy"        # every run's registry label; _latest selects by it

_TREE = ("acc", "params", "feed", "imp", "const", "opt", "uncprop", "psut")


def _tree(run_root: PathLike) -> "dict[str, Path]":
    tree = {k: Path(run_root) / k for k in _TREE}
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree


def _latest(registry) -> str:
    """The latest run of ``registry`` labelled ``LABEL``: the module's one rule for
    "the latest run", read off the registry's own ``latest_by_label``."""
    try:
        return registry.latest_by_label(LABEL).run_id
    except KeyError:
        raise KeyError(f"no {LABEL!r} run registered in {registry.storage_dir}; "
                       f"run the stage first") from None


def _load_acc(tree: dict) -> AccountingStructure:
    return AccountingStructure.load(ACC_ID, storage_dir=str(tree["acc"]))


def _upstream(sidecar: dict, key: str) -> str:
    value = sidecar.get("parents", {}).get("upstream", {}).get(key)
    if not value:
        raise KeyError(f"sidecar of {sidecar.get('run_id')!r} names no upstream {key!r}")
    return value


def _feed_steps(feed: SyntheticFeed, acc: AccountingStructure) -> tuple:
    """Stage 1's steps B–D for one synthetic feed: concordance, weight matrix, the
    source vector and the propagated root values. Returns ``(ct, wm, F_src, result,
    row_summary)``.

    ``feedpipe.run_feed_pipeline`` does not fit: it reads the pedigree from an Excel
    sidecar (step A) and takes ``F_src`` as an argument although ``F_src`` is aligned to
    the concordance it builds itself. The toy's feeds carry their metadata in memory, so
    the steps run as ``scripts/crude_ore_feed.py`` runs them.
    """
    ct = step_B_build_concordance(
        feed.raw, acc, **feed.maps, feed_id=feed.feed_id, aux_col=feed.aux_col,
        aux_sides=AUX_SIDES if feed.aux_col else None, allow_unmatched_edges=False)
    wm    = step_C_build_weight_matrix(ct)
    F_src = build_F_src(feed.raw, ct, label=feed.feed_id)
    result, row_summary = propagate_with_row_penalty(wm, F_src, feed.metadata, acc,
                                                     p=P_PROP)
    return ct, wm, F_src, result, row_summary


def run_stage1(run_root: PathLike, seed: int = SEED) -> "dict[str, str]":
    """Stage 1: build the accounting structure, then register the three synthetic feeds,
    each run's sidecar recording ``seed``.

    Returns
    -------
    dict
        ``{feed_id: feed_run_id}`` for ``survey``, ``bureau`` and ``customs``.
    """
    tree     = _tree(run_root)
    acc      = build_acc(tree["acc"])
    registry = DataFeedRegistry(odir=str(tree["feed"]))
    registry.load_index()
    run_ids  = {}
    for feed_id, feed in make_feeds(truth(seed), seed).items():
        _, wm, F_src, result, row_summary = _feed_steps(feed, acc)
        wm.odir = str(tree["feed"])         # WeightMatrix.save creates its odir
        entry = register_feed(
            wm, feed.metadata, result, registry,
            parents       = {"inputs":   {"acc": dict(ACC_REF), "feed": {"id": feed_id}},
                             "upstream": {}},
            F_src         = F_src,
            label         = feed_id,
            stage_config  = {"p": P_PROP, "seed": seed},
            stage_summary = row_summary,
        )
        run_ids[feed_id] = entry.run_id
        _log.info("stage 1: %s → %s (%d rows, cv %.4f)", feed_id, entry.run_id,
                  len(feed.raw), result.cv_src)
    return run_ids


def run_stage2(run_root: PathLike, feed_run_ids: Optional[dict] = None) -> str:
    """Stage 2: collect and aggregate the feeds, then impute every edge.

    Cascade (to a fixpoint) ``trade → network → conc``, then the terminal
    ``TimeImputer`` and ``FlatPriorImputer``; see ``CASCADE``. ``feed_run_ids`` pins the
    feed runs (default: the latest run of each feed). Returns the imp run id.
    """
    tree     = _tree(run_root)
    acc      = _load_acc(tree)
    params   = write_params(tree["params"])
    registry = DataFeedRegistry(odir=str(tree["feed"]))
    registry.load_index()
    keys      = list(feed_run_ids.values()) if feed_run_ids else list(FEED_IDS)
    collector = FeedCollector(registry=registry, acc_struct=acc)
    agg       = collector.select(keys).aggregate(p=P_PROP)

    conc, eff = str(params["conc"]), str(params["eff"])
    cascade = {
        "trade":   TradeImputer(conc_csv=conc, p=P_PROP, id="trade"),
        "network": NetworkImputer(eff_csv=eff, conc_csv=conc,
                                  stripping_ratio_csv=str(params["stripping_ratio"]),
                                  stripping_sigma=FLAT_SIGMA_HIGH, p=P_PROP, q=Q_REG,
                                  alpha_min=ALPHA_MIN_REG, cv_cap=NETWORK_CV_CAP,
                                  id="network", stock_sectors=STOCK_SECTORS),
        "conc":    ParametricImputer.from_csv(conc, acc, p=P_PROP, q=Q_REG,
                                              alpha_min=ALPHA_MIN_REG, id="conc"),
    }
    engine = ImputationEngine(acc_struct=acc)
    engine.register_cascade([cascade[k] for k in CASCADE], max_iter=CASCADE_MAX_ITER)
    engine.register(TimeImputer(window=TIME_WINDOW, method=TIME_METHOD,
                                min_obs=TIME_MIN_OBS, gap_inflation=TIME_GAP_INFLATION,
                                interpolate_only=TIME_INTERPOLATE_ONLY, p=P_PROP,
                                id="time_interp"))
    engine.register(FlatPriorImputer(acc_struct=acc, mode="noisy_zero",
                                     sigma_high=FLAT_SIGMA_HIGH, p=P_PROP,
                                     id="flat_prior"))
    agg = engine.run(agg)

    run_id = ImputationRunRegistry(storage_dir=str(tree["imp"])).register(
        agg,
        label    = LABEL,
        parents  = {"inputs":   {"acc": dict(ACC_REF),
                                 "params": [{"id": k, "path": str(p)}
                                            for k, p in params.items()]},
                    "upstream": {"feed_run_ids": [s["feed_run_id"]
                                                  for s in collector.selected]}},
        imputers = engine.imputers,
        stage_config = {"p": P_PROP, "q": Q_REG, "alpha_min": ALPHA_MIN_REG,
                        "cascade_order": list(CASCADE),
                        "cascade_max_iter": CASCADE_MAX_ITER,
                        "network_cv_cap": NETWORK_CV_CAP,
                        "flat_sigma_high": FLAT_SIGMA_HIGH,
                        "stock_sectors": list(STOCK_SECTORS),
                        "time": {"window": TIME_WINDOW, "min_obs": TIME_MIN_OBS,
                                 "gap_inflation": TIME_GAP_INFLATION,
                                 "method": TIME_METHOD,
                                 "interpolate_only": TIME_INTERPOLATE_ONLY}},
    )
    _log.info("stage 2: %s (%d edges covered by a feed, %d imputation log entries)",
              run_id, agg.n_covered, len(agg.imputation_log))
    return run_id


def run_stage3(run_root: PathLike, imp_run_id: Optional[str] = None) -> str:
    """Stage 3: hard mass balances (``massbal_scope``), the ``from_acc_struct`` soft sum
    rows, soft ``conc`` and ``eff`` ratio rows and the hard stripping ratio; no
    ``RatioBoundBuilder``, no ``FlowBoundsBuilder`` (#27). Returns the const run id."""
    tree       = _tree(run_root)
    acc        = _load_acc(tree)
    imp_reg    = ImputationRunRegistry(storage_dir=str(tree["imp"]))
    imp_run_id = imp_run_id or _latest(imp_reg)
    agg        = imp_reg.load(imp_run_id)
    params     = write_params(tree["params"])

    engine = ConstraintEngine(
        acc_struct        = acc,
        storage_dir       = str(tree["const"]),
        p                 = P_PROP,
        sigma_a_cv_min    = SIGMA_A_CV_MIN,
        sigma_a_abs_floor = SIGMA_A_ABS_FLOOR,
        sigma_c_cv_min    = SIGMA_C_CV_MIN,
        sigma_c_abs_floor = SIGMA_C_ABS_FLOOR,
        nodata_anchor_cv  = NODATA_ANCHOR_CV,
        nodata_anchor_exclude_sectors = NODATA_ANCHOR_EXCLUDE,
    )
    engine.register(MassBalanceBuilder(mask="massbal_scope"))
    # #59's soft sum rows: from_acc_struct writes one per Flow node, Σ supply − Σ use = 0
    # at 5 % of the supply, which the hard balance already holds. Copper's self-anchored
    # scale totals (from_mask_anchor at 1 %, ADR 0006) were tried on the seed-59 chain:
    # they restate every prior at 1 %, an erroneous one too, so a Concentrate usetotout
    # row became the top soft residual (z −58.6) and coverage fell 0.936 → 0.674.
    engine.register(SumConstraintBuilder.from_acc_struct(
        acc, sigma_fraction=SUM_SIGMA_FRACTION))
    for pid, hard in (("conc", False), ("eff", False), ("stripping_ratio", True)):
        engine.register(RatioConstraintBuilder.from_csv(
            str(params[pid]), acc_struct=acc, sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
            scale_abs_floor=RATIO_SCALE_ABS_FLOOR, hard=hard))
    run = engine.run(
        agg_result = agg,
        timesteps  = list(YEARS),
        label      = LABEL,
        persist    = True,
        parents    = {"inputs":   {"acc": dict(ACC_REF),
                                   "params": [{"id": k, "path": str(p)}
                                              for k, p in params.items()]},
                      "upstream": {"imp_run_id": imp_run_id}},
        stage_config = {"sigma_a_cv_min": SIGMA_A_CV_MIN,
                        "sigma_a_abs_floor": SIGMA_A_ABS_FLOOR,
                        "sigma_c_cv_min": SIGMA_C_CV_MIN,
                        "sigma_c_abs_floor": SIGMA_C_ABS_FLOOR,
                        "nodata_anchor_cv": NODATA_ANCHOR_CV,
                        "nodata_anchor_exclude": list(NODATA_ANCHOR_EXCLUDE),
                        "sum_sigma_fraction": SUM_SIGMA_FRACTION,
                        "ratio_sigma_rel_floor": RATIO_SIGMA_REL_FLOOR,
                        "ratio_scale_abs_floor": RATIO_SCALE_ABS_FLOOR,
                        "massbal_mask": "massbal_scope", "stripping_ratio_hard": True,
                        "hard_bands": False},
    )
    _log.info("stage 3: %s (G %d × %d)", run.run_id, *run.G.shape)
    return run.run_id


def run_stage4(run_root: PathLike, const_run_id: Optional[str] = None) -> str:
    """Stage 4: one Clarabel QP per year from the cold prior, at the ADR 0039
    tolerances. Returns the opt run id."""
    tree = _tree(run_root)
    const_run_id = const_run_id or _latest(ConstraintRunRegistry(str(tree["const"])))
    solver = ClarabelSolver(
        const_dir           = str(tree["const"]),
        opt_dir             = str(tree["opt"]),
        acc_dir             = str(tree["acc"]),
        tol_feas            = TOL_FEAS,
        tol_gap_rel         = TOL_GAP_REL,
        tol_gap_abs         = TOL_GAP_ABS,
        reduced_tol_feas    = REDUCED_TOL_FEAS,
        reduced_tol_gap_rel = REDUCED_TOL_GAP_REL,
        reduced_tol_gap_abs = REDUCED_TOL_GAP_ABS,
        max_iter            = MAX_ITER,
        verbose_first_solve = False,
        prior_mode          = "cold",
        qp_variables        = QP_VARIABLES,
    )
    results = solver.solve_all(const_run_id, feasibility_check="first")
    run_id  = solver.persist(
        const_run_id, results, const_run_id=const_run_id, label=LABEL,
        stage_config={"prior_mode": "cold", "max_iter": MAX_ITER, "tol_feas": TOL_FEAS,
                      "tol_gap_rel": TOL_GAP_REL, "tol_gap_abs": TOL_GAP_ABS,
                      "reduced_tol_feas": REDUCED_TOL_FEAS,
                      "reduced_tol_gap_rel": REDUCED_TOL_GAP_REL,
                      "reduced_tol_gap_abs": REDUCED_TOL_GAP_ABS,
                      "feasibility_check": "first"})
    _log.info("stage 4: %s (%s)", run_id,
              ", ".join(f"{t} {r.solver_status}" for t, r in results.items()))
    return run_id


def run_stage5(run_root: PathLike, opt_run_id: Optional[str] = None) -> str:
    """Stage 5: the posterior σ and the (1 − α) interval of every edge at the QP's
    active set (``KalmanUncProp``, α = 0.05). Returns the uncprop run id."""
    tree = _tree(run_root)
    opt_run_id = opt_run_id or _latest(OptimRunRegistry(str(tree["opt"])))
    prop = KalmanUncProp(opt_dir=str(tree["opt"]), const_dir=str(tree["const"]),
                         uncprop_dir=str(tree["uncprop"]), bind_tol=BIND_TOL,
                         rank_tol=RANK_TOL)
    result = prop.compute(opt_run_id, alpha=ALPHA)
    run_id = prop.persist(result, opt_run_id, label=LABEL)
    _log.info("stage 5: %s", run_id)
    return run_id


def run_stage6(run_root: PathLike, opt_run_id: Optional[str] = None,
               png: bool = True) -> str:
    """Stage 6: the tidy PSUT of the opt run, persisted; its market-share PIOT up to
    ``L`` (derived, never persisted); and the supply-use Sankey of the latest year in the
    content valuation, written as ``sankey_supply_use_<year>_<CONTENT>.png`` beside the
    PSUT run (Plotly, kaleido). Returns the psut run id."""
    tree = _tree(run_root)
    opt_run_id = opt_run_id or _latest(OptimRunRegistry(str(tree["opt"])))
    psut = PSUT.from_opt_run(opt_run_id, primary=CONTENT, opt_dir=str(tree["opt"]),
                             const_dir=str(tree["const"]), acc_dir=str(tree["acc"]))
    run_id = PsutRunRegistry(str(tree["psut"])).register(
        psut,
        parents = {"inputs":   {"acc": psut.acc_ref},
                   "upstream": {"opt_run_id": opt_run_id,
                                "const_run_id": psut.const_run_id}},
        label   = LABEL,
    )
    period = psut.latest()
    piot   = psut.to_piot("market_share", level="L")
    _log.info("stage 6: %s; PIOT %s — Z %s, A %s, L %s", run_id, period,
              piot.Z(period).shape, piot.A(period).shape, piot.L(period).shape)
    if png:
        path = tree["psut"] / run_id / f"sankey_supply_use_{period}_{CONTENT}.png"
        try:
            psut.sankey(period).write_image(str(path), width=1400, height=800,
                                            scale=1.5)
            _log.info("stage 6: Sankey → %s", path)
        except Exception as exc:        # kaleido missing or no browser: the PSUT stands
            _log.warning("stage 6: PNG skipped (%s: %s)", type(exc).__name__,
                         str(exc)[:120])
    return run_id


# =============================================================================
# Scoring against the truth
# =============================================================================

def _global(per_t: dict, attr: str, n_edges: int) -> np.ndarray:
    """Scatter a year-local attribute of every period onto the global edge ids."""
    out = np.full(n_edges, np.nan)
    for res in per_t.values():
        out[np.asarray(res.cols_t)] = np.asarray(getattr(res, attr), dtype=float)
    return out


def soft_residuals(run_root: PathLike, opt_run_id: Optional[str] = None) -> pd.DataFrame:
    """Every soft term of the solved QP as a z-score, largest ``|z|`` first.

    An edge's prior term is :func:`pysuto.diag.truth.prior_z`, ``(a* − a_prior) / σ_a``
    with σ_a conditioned as the solver conditions it; a soft row's (``~is_hard``) is
    ``(G·a* − c) / σ_c``.

    Returns
    -------
    DataFrame
        Columns ``kind`` (``"edge"`` / ``"row"``), ``edge_id`` (nullable ``Int64``, an
        edge's), ``constraint_id`` (``str``, a row's), ``time``, ``z``.
    """
    tree   = _tree(run_root)
    opt    = OptimRunRegistry(str(tree["opt"]))
    opt_run_id = opt_run_id or _latest(opt)
    const  = ConstraintRunRegistry(str(tree["const"])).load(
        _upstream(opt.load_sidecar(opt_run_id), "const_run_id"))
    _, sigma_a = condition_prior(const.a_prior_global, const.sigma_a)
    soft   = ~const.row_meta["is_hard"].to_numpy().astype(bool)
    cids   = const.row_meta["constraint_id"].to_numpy()
    frames = []
    for t, res in opt.load(opt_run_id).items():
        cols, rows = np.asarray(res.cols_t), np.asarray(res.rows_t)
        frames.append(pd.DataFrame({
            "kind": "edge", "edge_id": pd.array(cols, dtype="Int64"),
            "constraint_id": None, "time": t,
            "z": prior_z(res.F_opt, res.a_prior, sigma_a[cols])}))
        sc = condition_sigma_c(np.asarray(const.timestep_data[t]["sigma_c"])[rows])
        keep = soft[rows]
        frames.append(pd.DataFrame({
            "kind": "row", "edge_id": pd.array([pd.NA] * int(keep.sum()), dtype="Int64"),
            "constraint_id": cids[rows][keep], "time": t,
            "z": (res.residuals / sc)[keep]}))
    out = pd.concat(frames, ignore_index=True)
    return out.iloc[np.argsort(-np.abs(out["z"].to_numpy()), kind="stable")
                    ].reset_index(drop=True)


def _edge_id(acc: AccountingStructure, from_sector: str, from_region: str,
             to_sector: str, to_region: str, time: str, valuation: str) -> int:
    em  = acc.edge_meta
    hit = em[(em["from_sector"] == from_sector) & (em["from_region"] == from_region)
             & (em["to_sector"] == to_sector) & (em["to_region"] == to_region)
             & (em["time"] == time) & (em["valuation"] == valuation)]
    if len(hit) != 1:
        raise KeyError(f"no single edge {from_sector}[{from_region}] → {to_sector}"
                       f"[{to_region}] {time} {valuation} ({len(hit)} matches)")
    return int(hit["edge_id"].iloc[0])


def _recorded_seed(tree: dict, imp_run_id: str) -> int:
    """The one seed the feed runs upstream of ``imp_run_id`` were drawn with."""
    feeds = DataFeedRegistry(odir=str(tree["feed"]))
    feeds.load_index()
    upstream = ImputationRunRegistry(str(tree["imp"])).load_sidecar(imp_run_id)
    seeds = {feeds.load_sidecar(rid).get("stage_config", {}).get("seed")
             for rid in _upstream(upstream, "feed_run_ids")}
    if len(seeds) != 1 or None in seeds:
        raise ValueError(f"the feed runs upstream of {imp_run_id} record seeds {seeds}, "
                         f"not one seed")
    return int(seeds.pop())


def scoreboard(run_root: PathLike, run_ids: dict, seed: Optional[int] = None
               ) -> TruthDiag:
    """Every stage's estimate of the edges scored against the ground truth.

    ``run_ids`` names the ``imp``, ``opt`` and ``uncprop`` runs. The truth is drawn with
    the seed the feed runs recorded; an explicit ``seed`` that differs raises. The
    estimates are ``feed`` (the feeds' root values), ``aggregate`` (the aggregation,
    before imputation), ``prior`` (after imputation) and ``reconciled`` (``a*``); the
    posterior σ comes from the uncprop run and the prior σ is the one the QP used.
    """
    tree     = _tree(run_root)
    recorded = _recorded_seed(tree, run_ids["imp"])
    if seed is not None and seed != recorded:
        raise ValueError(f"scoreboard: seed={seed}, but the run tree's feeds were drawn "
                         f"with seed {recorded}")
    acc   = _load_acc(tree)
    agg   = ImputationRunRegistry(str(tree["imp"])).load(run_ids["imp"])
    opt   = OptimRunRegistry(str(tree["opt"]))
    const = ConstraintRunRegistry(str(tree["const"])).load(
        _upstream(opt.load_sidecar(run_ids["opt"]), "const_run_id"))
    post  = UncPropRunRegistry(str(tree["uncprop"])).load(run_ids["uncprop"])
    a_prior, sigma_prior = condition_prior(const.a_prior_global, const.sigma_a)

    observed = np.isfinite(agg.F)
    feed     = np.where(observed.any(axis=0),
                        np.nansum(np.where(observed, agg.F, 0.0), axis=0)
                        / np.maximum(observed.sum(axis=0), 1), np.nan)
    estimates = {
        "feed":       feed,
        "aggregate":  np.where(agg.coverage, agg.f_agg, np.nan),
        "prior":      a_prior,
        "reconciled": _global(opt.load(run_ids["opt"]), "F_opt", acc.n_edges),
    }
    return TruthDiag(
        estimates,
        truth_vector(truth(recorded), acc),
        _global(post.per_timestep, "sigma_a_star", acc.n_edges),
        acc.edge_meta,
        sigma_prior = sigma_prior,
        planted     = [_edge_id(acc, *(_PLANTED[k] for k in _PLANTED_EDGE))],
        alpha       = ALPHA,
        units       = UNITS,
        label       = LABEL,
    )


# =============================================================================
# Exercise hooks (#60): check_<chapter>_<n> and solution("<chapter>.<n>")
# =============================================================================
# A check computes the answer from the run tree, raises AssertionError on a wrong one
# and returns a tick line for the notebook to show. ``solution`` returns the source of
# the reference function, for the notebook to print.

def _close(given, expected, what: str, rtol: float = 1e-6, atol: float = 1e-9) -> None:
    if not np.allclose(np.asarray(given, dtype=float), np.asarray(expected, dtype=float),
                       rtol=rtol, atol=atol):
        raise AssertionError(f"{what}: expected {expected!r}, got {given!r}")


def _ref_1_1(run_root, region="Portua", year="2003") -> float:
    """CV of the Bureau's refined-output edge, by hand: two series pooled on one edge."""
    acc   = AccountingStructure.load(ACC_ID, storage_dir=str(Path(run_root) / "acc"))
    feeds = DataFeedRegistry(odir=str(Path(run_root) / "feed"))
    feeds.load_index()
    W     = feeds.load_feed("bureau", WeightMatrix, acc).weight_matrix
    F_src = np.load(Path(feeds.resolve("bureau").run_dir) / "F_src.npy")
    cv    = feeds.load_metadata("bureau").to_pedigree_scores().total_cv()
    edge  = _edge_id(acc, "Smelting", region, "Metal", region, year, CONTENT)
    part  = np.abs(W[:, edge].toarray().ravel() * F_src)    # each series' contribution
    omega = part[part > 0] / part.sum()                     # its value share
    return float(cv * (omega ** 1.3).sum() ** (1 / 1.3))   # Eq. (1), p = 1.3


def check_1_1(run_root: PathLike, cv: float, region: str = "Portua",
              year: str = "2003") -> str:
    """Exercise 1.1: the CV of the refined-output edge of ``region`` in ``year``."""
    acc   = _load_acc(_tree(run_root))
    feeds = DataFeedRegistry(odir=str(Path(run_root) / "feed"))
    feeds.load_index()
    edge     = _edge_id(acc, "Smelting", region, "Metal", region, year, CONTENT)
    expected = float(feeds.load_result("bureau")["cv_root"][edge])
    _close(cv, expected, f"CV of Smelting → Metal, {region} {year}")
    return f"✓ 1.1  the fan-in CV is {expected:.4f}, below the Bureau's own CV"


def _ref_2_1(run_root) -> set:
    """The region-years the time imputer filled, read off the imputation log."""
    acc = AccountingStructure.load(ACC_ID, storage_dir=str(Path(run_root) / "acc"))
    imp = ImputationRunRegistry(storage_dir=str(Path(run_root) / "imp"))
    log = imp.load(_latest(imp)).imputation_log.as_dataframe()
    filled = log.loc[log["imputer_id"] == "time_interp", "edge_id"].astype(int)
    em = acc.edge_meta.set_index("edge_id").loc[filled]
    return set(zip(em["from_region"], em["time"]))


def check_2_1(run_root: PathLike, region_years) -> str:
    """Exercise 2.1: the ``(region, year)`` pairs the time imputer filled."""
    expected = _ref_2_1(run_root)
    given    = {(str(r), str(t)) for r, t in region_years}
    if given != expected:
        raise AssertionError(f"time-imputed region-years: expected {sorted(expected)}, "
                             f"got {sorted(given)}")
    return f"✓ 2.1  the time imputer filled {sorted(expected)}"


def _ref_3_1(run_root, year="2003") -> tuple:
    """Hard and soft rows of one year, counted from row_meta."""
    const = ConstraintRunRegistry(str(Path(run_root) / "const"))
    rm    = const.load(_latest(const)).row_meta
    hard  = rm.loc[rm["time"].astype(str) == year, "is_hard"].astype(bool)
    return int(hard.sum()), int((~hard).sum())


def check_3_1(run_root: PathLike, n_hard: int, n_soft: int, year: str = "2003") -> str:
    """Exercise 3.1: how many hard and how many soft rows ``year`` has."""
    expected = _ref_3_1(run_root, year)
    if (int(n_hard), int(n_soft)) != expected:
        raise AssertionError(f"rows of {year} (hard, soft): expected {expected}, got "
                             f"{(n_hard, n_soft)}")
    return f"✓ 3.1  {year} has {expected[0]} hard and {expected[1]} soft rows"


def _ref_4_1(run_root, constraint_id) -> float:
    """One soft row's z, recomputed: (G·a* − c) / σ_c in its own year."""
    const_reg = ConstraintRunRegistry(str(Path(run_root) / "const"))
    opt_reg   = OptimRunRegistry(str(Path(run_root) / "opt"))
    const  = const_reg.load(_latest(const_reg))
    ids    = const.row_meta["constraint_id"].to_numpy()
    row    = int(np.flatnonzero(ids == constraint_id)[0])
    year   = str(const.row_meta["time"].iloc[row])
    res    = opt_reg.load(_latest(opt_reg))[year]
    a_star = np.zeros(const.G.shape[1])
    a_star[res.cols_t] = res.F_opt
    c, sigma_c = (const.timestep_data[year][k][row] for k in ("c", "sigma_c"))
    return float((const.G[row] @ a_star - c)[0] / sigma_c)


def check_4_1(run_root: PathLike, constraint_id: str, z: float) -> str:
    """Exercise 4.1: the z-score of the soft row ``constraint_id``."""
    rows = soft_residuals(run_root)
    hit  = rows[rows["constraint_id"] == constraint_id]
    if len(hit) != 1:
        raise AssertionError(f"{constraint_id!r} is not one soft row of the opt run")
    expected = float(hit["z"].iloc[0])
    _close(z, expected, f"z of {constraint_id}", rtol=1e-6, atol=1e-6)
    return f"✓ 4.1  z = {expected:+.3f}"


def _ref_5_1(run_root) -> float:
    """The largest posterior-over-prior σ ratio of any edge (it never exceeds 1)."""
    const = ConstraintRunRegistry(str(Path(run_root) / "const"))
    post  = UncPropRunRegistry(str(Path(run_root) / "uncprop"))
    run   = const.load(_latest(const))
    _, sigma_a = condition_prior(run.a_prior_global, run.sigma_a)
    ratios = [ts.sigma_a_star / sigma_a[ts.cols_t]
              for ts in post.load(_latest(post)).per_timestep.values()]
    return float(np.max(np.concatenate(ratios)))


def check_5_1(run_root: PathLike, max_ratio: float) -> str:
    """Exercise 5.1: the largest ``σ_a* / σ_a`` over every edge and year."""
    expected = _ref_5_1(run_root)
    _close(max_ratio, expected, "max σ_a*/σ_a")
    if expected > 1.0 + 1e-9:
        raise AssertionError(f"a posterior σ exceeds its prior σ (ratio {expected:.6f})")
    return f"✓ 5.1  max σ_a*/σ_a = {expected:.4f} ≤ 1"


def _ref_6_1(run_root, region="Portua", sector="Metal", year="2005",
             valuation=CONTENT) -> tuple:
    """Σ supply into and Σ use out of one commodity node, from the PSUT table."""
    reg   = PsutRunRegistry(str(Path(run_root) / "psut"))
    table = reg.load(_latest(reg)).table
    rows  = table[(table["period"] == year) & (table["valuation"] == valuation)]
    into  = (rows["to_region"] == region) & (rows["to_sector"] == sector)
    outof = (rows["from_region"] == region) & (rows["from_sector"] == sector)
    return float(rows.loc[into, "val"].sum()), float(rows.loc[outof, "val"].sum())


def check_6_1(run_root: PathLike, supply: float, use: float, region: str = "Portua",
              sector: str = "Metal", year: str = "2005",
              valuation: str = CONTENT) -> str:
    """Exercise 6.1: Σ supply into and Σ use out of a commodity (``Flow``) node."""
    if _ENTITY.get(sector) != "Flow":
        raise ValueError(f"{sector!r} is not a commodity (Flow) sector of the toy")
    expected = _ref_6_1(run_root, region, sector, year, valuation)
    _close((supply, use), expected, f"(supply, use) of {sector}[{region}] {year}")
    _close(expected[0], expected[1], "supply and use of one node", rtol=0.0, atol=1e-8)
    return (f"✓ 6.1  {sector}[{region}] {year}: supply {expected[0]:.4f} = use "
            f"{expected[1]:.4f} {UNITS[valuation]}")


_REFERENCES = {"1.1": _ref_1_1, "2.1": _ref_2_1, "3.1": _ref_3_1,
               "4.1": _ref_4_1, "5.1": _ref_5_1, "6.1": _ref_6_1}


def solution(key: str) -> str:
    """The source of exercise ``key``'s reference function (``"1.1"`` … ``"6.1"``), for
    the notebook to print."""
    if key not in _REFERENCES:
        raise KeyError(f"no exercise {key!r}; known: {sorted(_REFERENCES)}")
    return inspect.getsource(_REFERENCES[key])
