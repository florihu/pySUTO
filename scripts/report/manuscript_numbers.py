"""
manuscript_numbers.py
=====================
Re-derives the chain-dependent numbers of the manuscript's Method
(``docs/manuscript/method.md``) and Supporting Information (``si.md``) from one
optimisation run, so the text can be refreshed onto a new reference chain.

    python scripts/report/manuscript_numbers.py <opt_run_id> [--json OUT]
        [--data-dir data] [--base OPT] [--base-retol OPT] [--numbers PATH]

The JSON goes to ``docs/manuscript/figs/manuscript_numbers.json`` by default and is
committed with the manuscript, so a refresh onto a later chain diffs the new file
against git HEAD key by key and edits the text at each changed key's location. A run
on any other opt run overwrites it: pass ``--json <path>`` for an exploratory run,
or restore the file with ``git checkout``.

Run it from the repository root of the checkout that holds ``data/``. It walks
the chain opt → const → imp → feed runs → input versions through the run
sidecars (``parents.upstream`` / ``parents.inputs``), finds the PSUT and
posterior runs built from the opt run, and reads the pinned ``vN/`` input
folders directly. It never initialises the input registry, never runs a stage
driver and writes nothing but the ``--json`` file.

Every number carries a key naming its manuscript location —
``method:L133.n_bands`` (line of method.md), ``si:S4.T_S4.band_conc`` (section
and table of si.md), ``results:L60.…`` (line of results.md) — and a short label.
Groups, after the 2026-10-19 inventories of method/results and SI:

  1  sidecar walk: run ids, input versions, pedigree, knobs, solver facts
  2  accounting structure: nodes, sectors, pairs, edge counts
  3  Stage 2: feed coverage (``agg_result.npz``) and fills (imputation log)
  4  Stage 3: row families from ``row_meta`` (``type``, ``is_hard``,
     ``cone_type``, ``variant`` mapped to its parameter table — never a parsed
     ``constraint_id``), zero pins from ``l``/``u``
  5  Stage 4: the all-years edge frame and the soft-row residuals (in memory,
     no per-year CSVs): waste rock, smallest bounded flow, prior constant,
     leg and anchor CVs, the extraction chain outside the ore-grade years
  6  forced deletion: ``max(0, P − c_hi·X − U)`` per mine region-year with X
     from the netted BACI ``values.csv`` and U from the ICSG ``values.csv``
  7  common set: the run's scale-total Σz² on the rows it shares with the two
     base runs (their ``diag/constraints_*.csv``, the only surviving copy)
  8  input tables: params, feed concordances, BACI ledgers
  9  Results numbers that ``results_figs.py`` does not write to numbers.json:
     |z| > 3 shares by year, China's secondary share by year, the ROW band in
     HHI, and ratios of numbers.json keys when that file belongs to this run
 10  Stage 5: the default posterior built from the opt run (the #87 scale-total
     ablation beside it): active set per year, Birge ratio, calibration of the
     shifts, contraction by prior class with and without the scale totals
 11  limitations (method.md §3.2.8): the mine chains kept where ICSG reports no
     mine output, stock legs beyond |z| 5, the world producer stock change
     against ICSG, the regions that carry the scale-total objective

Historical measurements (REP/ADR figures) and the tables ``results_figs.py``
generates are deliberately absent; so are repository statistics (lines,
tests, ADR count), which depend on the code tree, not the chain.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "src")

from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.diag.uncprop import UncPropDiag             # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# The base run of the measurement series and its re-solve at the ADR 0039
# tolerance (SI S13.1): fixed by choice, not by the chain.
BASE_RUN       = "opt_20260916_124721_c4d1d944"
BASE_RETOL_RUN = "opt_20260930_093914_523d57d7"
NUMBERS_JSON   = "docs/manuscript/figs/numbers.json"
NUMBERS_OUT    = "docs/manuscript/figs/manuscript_numbers.json"   # committed: the refresh's diff base
Z_BIG          = 5.0                         # a stock leg beyond |z| 5 is a limitation (§3.2.8)
PSC_FIRST_YEAR = 1996                        # ICSG reports no producer stock change for 1995
EXCESS_MIN_KT  = 5.0                         # BACI exports above ICSG production (§3.2.8, #111)
PARAM_TABLES   = ["conc", "eff", "stripping_ratio", "ore_grade", "stock_bounds",
                  "scrap_share"]
SUM_VARIANTS   = {"supplytotout": "sum_supply", "usetotout": "sum_use",
                  "smelterscrap": "anchor_smelter", "directrefscrap": "anchor_directref"}
SCRAP_DEFAULT  = (0.0, 0.25)                 # scrap-share band without an own series
A2_SUMS        = 0.20                        # results_figs THRESH["A2_sums"]
HHI_STEPS      = {"mining":   ("Mining", "Ore", "region_from"),
                  "smelting": ("Smelting", "Matte", "region_from"),
                  "refining": ("CP", "Cathode", "region_from"),
                  "use":      ("Cathode", "FU", "region_to")}
CHILE_CODE     = 152                         # BACI partner codes
PNG_CODE       = 598
PNG_YEARS      = (2019, 2022)
TAIWAN_CODE    = 490
TABLE1_LINE    = {"icsg": 268, "baci": 269, "crude_ore": 270}   # method.md Table 1 rows
SEMI_USE       = ["Semi", "USC", "MESC"]     # ICSG Cathode Use processes (usage)
# =============================================================================


# =============================================================================
# collector
# =============================================================================

def _plain(v):
    """numpy scalars / arrays / containers → JSON-native Python values."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    return v


@dataclass
class Item:
    key:   str
    label: str
    value: object
    unit:  str = ""


@dataclass
class Numbers:
    items: dict = field(default_factory=dict)

    def add(self, key: str, label: str, value, unit: str = "") -> None:
        if key in self.items:
            raise KeyError(f"duplicate key {key}")
        self.items[key] = Item(key, label, _plain(value), unit)


# =============================================================================
# 1  chain walk
# =============================================================================

@dataclass
class Chain:
    opt:      str
    const:    str
    imp:      str
    feeds:    dict                  # feed id → feed run id
    psut:     list
    uncprop:  list
    meta:     dict                  # "opt" / "const" / "imp" / feed id → sidecar
    versions: dict                  # input id → version
    years:    list


def _sidecar(path: Path) -> dict:
    with open(path / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _built_from(stage_dir: Path, opt: str) -> list:
    """Run ids under ``stage_dir`` whose sidecar names ``opt`` as upstream."""
    out = []
    for d in sorted(stage_dir.glob("*_*")):
        if d.is_dir() and (d / "metadata.json").exists():
            up = _sidecar(d).get("parents", {}).get("upstream", {})
            if up.get("opt_run_id") == opt:
                out.append(d.name)
    return out


def walk_chain(proc: Path, opt: str) -> Chain:
    """opt → const → imp → feed runs, with every input version on the way."""
    om = _sidecar(proc / "opt" / opt)
    const = om["parents"]["upstream"]["const_run_id"]
    cm = _sidecar(proc / "const" / const)
    imp = cm["parents"]["upstream"]["imp_run_id"]
    im = _sidecar(proc / "imp" / imp)
    meta = {"opt": om, "const": cm, "imp": im}
    feeds, versions = {}, {"acc": cm["parents"]["inputs"]["acc"]["version"]}
    for p in cm["parents"]["inputs"]["params"]:
        versions[p["id"]] = p["version"]
    for fr in im["parents"]["upstream"]["feed_run_ids"]:
        fm = _sidecar(proc / "feed" / "registry" / fr)
        fid = fm["parents"]["inputs"]["feed"]["id"]
        feeds[fid], meta[fid] = fr, fm
        versions[fid] = fm["parents"]["inputs"]["feed"]["version"]
        if "trade_shares" in fm["parents"]["inputs"]:
            versions["trade_shares"] = fm["parents"]["inputs"]["trade_shares"]["version"]
    return Chain(opt=opt, const=const, imp=imp, feeds=feeds,
                 psut=_built_from(proc / "psut", opt),
                 uncprop=_built_from(proc / "uncprop", opt),
                 meta=meta, versions=versions,
                 years=[str(y) for y in om["stage_summary"]["timesteps"]])


def sidecar_numbers(N: Numbers, ch: Chain) -> None:
    """Group 1: ids, versions, pedigree, knobs and solver facts from sidecars."""
    v = ch.versions
    N.add("method:L10.opt_run_id",   "reference opt run (also SI Table S17)", ch.opt)
    N.add("method:L10.const_run_id", "its constraint run (also results.md:12, S17)", ch.const)
    N.add("method:L11.imp_run_id",   "its imputation run (also results.md:13, S17)", ch.imp)
    for line, fid in ((12, "icsg"), (13, "baci"), (14, "crude_ore")):
        N.add(f"method:L{line}.feed_{fid}_run_id", f"{fid} feed run (also S17)",
              ch.feeds.get(fid))
        N.add(f"method:L{line}.{fid}_version", f"{fid} feed version (also Table 1, S2)",
              v.get(fid))
    N.add("method:L12.trade_shares_version", "trade_shares version (ICSG feed sidecar; "
          "also Table 1, S11)", v.get("trade_shares"))
    N.add("method:L15.acc_version", "acc copper version (also S8)", v["acc"])
    for pid in PARAM_TABLES:
        N.add(f"method:L15.version.{pid}", f"{pid} version (const sidecar; also Table 1, "
              "S12 caption)", v.get(pid))
    N.add("method:L188.n_feeds", "data feeds (also SI Table S1 N_f)", len(ch.feeds))
    N.add("method:L188.n_param_tables", "parameter tables", len(PARAM_TABLES))

    # ---- feed pedigree (Stage 1) ---------------------------------------------
    for fid in ("icsg", "baci", "crude_ore"):
        fm = ch.meta.get(fid)
        if fm is None:
            continue
        md, rp = fm["feed_metadata"], fm["stage_summary"]["row_penalty"]
        ped = tuple(int(md[k]) for k in ("reliability", "completeness", "temporal_correlation",
                                         "geographical_correlation", "other_correlation"))
        N.add(f"method:L119.{fid}.pedigree", "(R,C,T,G,O) scores (also Table 1, S2)",
              "(" + ",".join(map(str, ped)) + ")")
        N.add(f"method:L119.{fid}.cv", "feed CV (also Table 1, S2)", md["cv_total"])
        N.add(f"method:L119.{fid}.cv_row", "ROW-penalised CV (also Table 1, S2)", rp["cv_pen"])
        N.add(f"method:L119.{fid}.n_row_edges", "Stage-1 edges the ROW CV applies to "
              "(also Table 1, S2)", rp["n_penalized"])
        N.add(f"si:S2.T_S2.{fid}.row_value_share_pct", "ROW edges' share of the feed's value",
              100 * rp["value_share"], "%")
        N.add(f"si:S2.T_S2.{fid}.date_of_assessment", "pedigree assessment date",
              md["date_of_assessment"])
    sens = [ch.meta[f]["feed_metadata"][k] for f in ch.feeds
            for k in ("completeness_sensitivity", "temporal_sensitivity",
                      "geographical_sensitivity", "other_sensitivity")]
    N.add("si:S2.all_sensitivities_medium", "every feed indicator is medium-sensitive",
          all(s == "medium_sensitive" for s in sens))

    # ---- Stage-2 knobs and summary --------------------------------------------
    ic, isum = ch.meta["imp"]["stage_config"], ch.meta["imp"]["stage_summary"]
    N.add("method:L125.leg_sigma_gamma", "leg σ damping exponent γ (also Table 2, S3.2)",
          ic.get("leg_sigma_gamma"))
    N.add("method:L127.stock_seed_cv", "USC/MESC seed σ as share of the cathode flow "
          "(ADR 0040; also S3 Table S3)", ic.get("stock_seed_cv"))
    N.add("method:L127.stock_seed_abs_floor_kt", "USC/MESC seed σ floor (also S3 Table S3)",
          ic.get("stock_seed_abs_floor"), "kt")
    N.add("method:L127.flat_sigma", "flat-prior σ (also S3 Table S3)",
          ic.get("flat_sigma_high"))
    N.add("method:L127.cascade_passes", "cascade passes run (also S3.3)",
          isum.get("cascade_passes_run"))
    N.add("si:S3.leg_sigma_cv_cap", "leg σ CV cap", ic.get("leg_sigma_cv_cap"))
    N.add("si:S3.leg_sigma_a_ref", "pinned leg reference sizes a_ref",
          ic.get("leg_sigma_a_ref"), "kt")
    N.add("si:S3.outlier_leg_k", "outlier-leg screen K", ic.get("outlier_leg_k"))
    N.add("si:S3.outlier_leg_cv", "outlier-leg σ multiple", ic.get("outlier_leg_cv"))
    N.add("si:S3.unclosed_k", "contradicted-node detector factor", ic.get("unclosed_k"))
    N.add("si:S3.cascade_max_iter", "cascade sweep cap", ic.get("cascade_max_iter"))
    t = ic.get("time", {})
    N.add("si:S3.T_S3.time_window", "time-interpolation window", t.get("window"), "yr")
    N.add("si:S3.T_S3.time_gap_inflation", "time-interpolation σ inflation per gap year",
          t.get("gap_inflation"))
    imps = ch.meta["imp"].get("imputers", [])
    net = [x.get("config", {}) for x in imps if x["imputer_id"] == "network"]
    N.add("si:S3.T_S3.network_cv_cap", "network imputer σ cap (× |f|)",
          net[0].get("cv_cap") if net else None)
    N.add("si:S3.T_S3.has_conc_formula", "the formula (cathode final-use split) imputer ran",
          any(x["imputer_id"] == "conc_formula" for x in imps))
    N.add("si:S12.T_S15.n_imputed", "fills in the imputation log (caption)",
          isum.get("n_imputed"))
    N.add("si:S12.T_S15.n_steps", "imputer steps (caption)", isum.get("n_steps"))

    # ---- Stage-3 knobs ------------------------------------------------------
    cc = ch.meta["const"]["stage_config"]
    N.add("method:L133.fabricated_anchor_cv", "scale-total σ_c CV (also S4 Table S4)",
          cc.get("fabricated_anchor_cv"))
    N.add("method:L133.nodata_anchor_cv", "no-data anchor CV of the sibling median "
          "(also Table 2, S4.2)", cc.get("nodata_anchor_cv"))
    N.add("method:L133.grade_region_years", "extraction-chain exempt region-years "
          "(const stage_config; also S4.2)", cc.get("nodata_anchor_grade_region_years"))
    N.add("si:S4.T_S4.feed_share_observed", "feed-observed value share for the 0.3 kt floor",
          cc.get("feed_share_observed"))
    N.add("si:S4.T_S4.ratio_sigma_rel_floor", "ratio-row σ_r floor",
          cc.get("ratio_sigma_rel_floor"))
    N.add("si:S4.T_S4.ratio_scale_abs_floor_kt", "ratio-row scale floor",
          cc.get("ratio_scale_abs_floor"), "kt")
    N.add("si:S4.T_S4.stock_bounds_scale_abs_floor_kt", "net-stock row scale floor",
          cc.get("stock_bounds_scale_abs_floor"), "kt")
    for k in ("sigma_a_cv_min", "sigma_a_abs_floor", "sigma_c_cv_min", "sigma_c_abs_floor"):
        N.add(f"si:S4.{k}", f"conditioning floor {k}", cc.get(k))

    # ---- Stage-4 solver -----------------------------------------------------
    oc, osum = ch.meta["opt"]["stage_config"], ch.meta["opt"]["stage_summary"]
    per = osum["per_timestep"]
    st = [per[y]["solver_status"] for y in ch.years]
    it = [per[y]["iterations"] for y in ch.years]
    ts = [per[y]["solve_time_s"] for y in ch.years]
    N.add("method:L154.prior_mode", "prior mode (also S14 Table S17)", oc.get("prior_mode"))
    N.add("method:L154.tol_gap_rel", "relative gap (also S5 Table S5)", oc.get("tol_gap_rel"))
    N.add("method:L154.reduced_tol_gap_rel", "reduced relative gap (also Table S5)",
          oc.get("reduced_tol_gap_rel"))
    N.add("method:L154.n_years", "years solved (also results.md:50)", len(st))
    N.add("method:L154.n_solved", "years Solved (also results.md:50, S5 Table S6)",
          st.count("Solved"))
    N.add("method:L154.n_almost_solved", "years AlmostSolved (also results.md:50, Table S6)",
          st.count("AlmostSolved"))
    N.add("method:L154.iterations_min", "fewest iterations of a year (also S5)", min(it))
    N.add("method:L154.iterations_max", "most iterations of a year (also S5 caption)", max(it))
    N.add("method:L154.solve_time_min_s", "fastest year", min(ts), "s")
    N.add("method:L154.solve_time_max_s", "slowest year", max(ts), "s")
    N.add("method:L154.solve_time_sum_s", "Σ per-year solve time (also S6)", sum(ts), "s")
    N.add("method:L154.solve_wall_s", "wall clock of the concurrent solve (#71; absent "
          "before it)", osum.get("solve_wall_s"), "s")
    for k in ("tol_feas", "reduced_tol_feas", "tol_gap_abs", "reduced_tol_gap_abs",
              "max_iter"):
        N.add(f"si:S5.T_S5.{k}", f"solver setting {k}", oc.get(k))
    for y in ch.years:
        p = per[y]
        N.add(f"si:S5.T_S6.{y}.iterations", f"{y} Clarabel iterations", p["iterations"])
        N.add(f"si:S5.T_S6.{y}.solve_time_s", f"{y} solve time", p["solve_time_s"], "s")
        N.add(f"si:S5.T_S6.{y}.gap_rel_e6", f"{y} relative duality gap × 1e6",
              1e6 * p["convergence"]["gap_rel"])

    # ---- Stage 5 / Stage 6 and the provenance table -------------------------------
    N.add("si:S6.uncprop_runs", "Stage-5 runs built from this opt run (also S17)", ch.uncprop)
    N.add("si:S14.T_S17.psut_run_id", "Stage-6 runs built from this opt run (also "
          "results.md:9)", ch.psut)
    N.add("si:S14.T_S17.feed_acc_versions", "acc version each feed run was built on",
          {f: ch.meta[f]["parents"]["inputs"]["acc"]["version"] for f in ch.feeds})
    imp_in = ch.meta["imp"]["parents"]["inputs"]
    N.add("si:S14.T_S17.imp_acc_version", "acc version of the imputation run",
          imp_in["acc"]["version"])
    N.add("si:S14.T_S17.imp_params", "params the imputation run read",
          {p["id"]: p["version"] for p in imp_in.get("params", [])})
    N.add("si:S14.T_S17.const_extra_params", "params the constraint run adds",
          sorted(set(PARAM_TABLES) - {p["id"] for p in imp_in.get("params", [])}))


# =============================================================================
# 2  accounting structure
# =============================================================================

def acc_numbers(N: Numbers, acc, ei: pd.DataFrame, years: list) -> None:
    """Group 2: node, sector, pair and edge counts of the account."""
    ni = acc.node_index
    sec = ni.drop_duplicates("sector")
    N.add("method:L57.n_nodes", "nodes of the graph", len(ni))
    N.add("method:L57.n_sectors", "sectors (also L194, S8)", ni.sector.nunique())
    N.add("method:L57.n_regions", "regions (also L194, S8)", ni.region.nunique())
    N.add("method:L57.n_valuations", "valuations", ei.valuation.nunique())
    N.add("method:L57.n_years", "years", len(years))
    per_vy = ei.groupby(["valuation", "time"]).size()
    N.add("method:L57.edges_per_valuation_year", "edges per valuation and year (also L254, "
          "S8)", int(per_vy.iloc[0]) if per_vy.nunique() == 1 else per_vy.to_dict())
    N.add("method:L57.n_edges", "edges in all (also L254, SI Table S1 E, S8)", len(ei))
    for ent, k in (("Process", "n_processes"), ("Flow", "n_products"), ("SB", "n_boundary")):
        N.add(f"method:L194.{k}", f"sectors of entity {ent}", int((sec.entity == ent).sum()))
    keys = [ei.sector_from, ei.sector_to]
    traded = (ei.region_from != ei.region_to).groupby(keys).any()
    N.add("method:L194.n_sector_pairs", "sector pairs (also Fig. 2, S7, S8)", len(traded))
    N.add("method:L194.n_traded_pairs", "pairs that exist between regions", int(traded.sum()))
    N.add("method:L194.n_domestic_pairs", "pairs within a region only", int((~traded).sum()))
    N.add("method:L194.n_traded_products", "products traded",
          traded[traded].index.get_level_values("sector_from").nunique())
    N.add("method:L194.n_named_regions", "regions other than ROW",
          ni.region[ni.region != "ROW"].nunique())
    N.add("method:L194.period", "period", f"{years[0]}–{years[-1]}")
    size = ei.groupby(keys).size()
    N.add("si:S8.T_S10.edges_per_traded_pair", "edges per traded pair, 30 years × 2 "
          "valuations", sorted(set(size[traded].tolist())))
    N.add("si:S8.T_S10.edges_per_domestic_pair", "edges per domestic pair",
          sorted(set(size[~traded].tolist())))
    cross = ei.region_from != ei.region_to
    N.add("si:S8.cross_region_edges", "cross-region edges", int(cross.sum()))
    N.add("si:S8.within_region_edges", "within-region edges", int((~cross).sum()))
    l = np.asarray(acc.bounds.l, dtype=float)
    N.add("si:S8.T_S10.signed_lower_bound", "lower bound of the signed stock edges",
          float(l.min()), "kt")
    N.add("si:S8.T_S10.n_signed_pairs", "sector pairs with a negative lower bound",
          ei.loc[l < 0].groupby(["sector_from", "sector_to"]).ngroups)


# =============================================================================
# 3  Stage 2
# =============================================================================

def coverage_numbers(N: Numbers, proc: Path, ch: Chain, ei: pd.DataFrame) -> None:
    """Group 3: feed coverage and imputer fills by valuation and region relation."""
    agg = np.load(proc / "imp" / ch.imp / "agg_result.npz", allow_pickle=True)
    cov = np.asarray(agg["coverage"], dtype=bool)
    F, fids = agg["F"], [str(f) for f in agg["feed_ids"]]
    val = ei.valuation.to_numpy()
    cross = (ei.region_from != ei.region_to).to_numpy()
    E = len(ei)
    N.add("method:L125.n_covered", "feed-observed edges (also S12 Table S14)", int(cov.sum()))
    N.add("method:L125.covered_pct", "feed-observed share of all edges", 100 * cov.mean(), "%")
    for k, fid in enumerate(fids):
        obs = np.isfinite(F[k])
        vals = sorted(set(val[obs]))
        t = ei.time.to_numpy()[obs]
        N.add(f"method:L125.{fid}_edges", f"edges {fid} observes (also Table 1, S14)",
              int(obs.sum()))
        N.add(f"method:L125.{fid}_valuation", f"valuation of the {fid} edges", vals)
        N.add(f"method:L{TABLE1_LINE[fid]}.{fid}_years", f"years {fid} observes (Table 1)",
              f"{min(t)}–{max(t)}" if obs.any() else None)
    N.add("method:L125.edges_in_two_feeds", "edges more than one feed observes",
          int((np.isfinite(F).sum(axis=0) > 1).sum()))
    for v in ("CC", "Mass"):
        N.add(f"si:S12.T_S14.covered_{v.lower()}", f"feed-observed {v} edges",
              int((cov & (val == v)).sum()))
        N.add(f"si:S12.T_S14.all_{v.lower()}", f"all {v} edges", int((val == v).sum()))
    N.add("si:S12.T_S14.covered_cross", "feed-observed cross-region edges",
          int((cov & cross).sum()))
    N.add("si:S12.T_S14.covered_within", "feed-observed within-region edges",
          int((cov & ~cross).sum()))

    log = pd.read_parquet(proc / "imp" / ch.imp / "imputation_log.parquet",
                          columns=["edge_id", "imputer_id", "pass"])
    log = log.merge(ei[["edge_id", "valuation", "region_from", "region_to", "sector_from",
                        "sector_to"]], on="edge_id", how="left", validate="one_to_one")
    log["cross"] = log.region_from != log.region_to
    flat = log.imputer_id == "flat_prior"
    n_flat, n_inf = int(flat.sum()), int((~flat).sum())
    N.add("method:L127.n_informative", "informatively imputed edges (also S14)", n_inf)
    N.add("method:L127.informative_pct", "informative share of all edges", 100 * n_inf / E,
          "%")
    N.add("method:L127.n_flat", "flat-prior edges (also S14, S15)", n_flat)
    N.add("method:L127.flat_pct", "flat-prior share of all edges", 100 * n_flat / E, "%")
    top = log[~flat].imputer_id.value_counts()
    N.add("method:L127.top_informative_imputers", "the two largest informative imputers",
          top.index[:2].tolist())
    for v in ("CC", "Mass"):
        N.add(f"si:S12.T_S14.informative_{v.lower()}", f"informative {v} fills",
              int((~flat & (log.valuation == v)).sum()))
        N.add(f"si:S12.T_S14.flat_{v.lower()}", f"flat-prior {v} edges",
              int((flat & (log.valuation == v)).sum()))
    ran = list(dict.fromkeys(x["imputer_id"] for x in ch.meta["imp"].get("imputers", [])))
    for imp_id in ran + sorted(set(log.imputer_id) - set(ran)):
        g = log[log.imputer_id == imp_id]
        N.add(f"si:S12.T_S15.{imp_id}", f"fills of {imp_id} by valuation and relation",
              {"CC":    int((g.valuation == "CC").sum()),
               "Mass":  int((g.valuation == "Mass").sum()),
               "cross": int(g.cross.sum()), "within": int((~g.cross).sum()), "total": len(g)})
    passes = log["pass"].value_counts()
    N.add("si:S12.T_S15.cascade_pass1", "cascade fills in pass 1", int(passes.get(1.0, 0)))
    N.add("si:S12.T_S15.cascade_pass2", "cascade fills in pass 2", int(passes.get(2.0, 0)))
    wr = log[(log.sector_from == "Mining") & (log.sector_to == "WR")
             & (log.valuation == "Mass")]
    N.add("si:S12.T_S15.wr_mass_edges_by_imputer", "Mining→WR mass edges by imputer (caption)",
          wr.imputer_id.value_counts().to_dict())


# =============================================================================
# 4  Stage 3
# =============================================================================

def _params(inp: Path, pid: str, version: int) -> pd.DataFrame:
    return pd.read_csv(inp / "params" / pid / f"v{version}" / "params.csv",
                       sep=None, engine="python", dtype=str)


def row_numbers(N: Numbers, rm: pd.DataFrame, P: dict, l: np.ndarray, u: np.ndarray,
                edges_per_year: int) -> None:
    """Group 4: row families by type, cone and parameter table."""
    table_of = {pid: t for t, p in P.items() for pid in p.param_id}
    var = rm.variant.astype(str)
    typ = rm.type.astype(str)
    hard = rm.is_hard.to_numpy(dtype=bool)
    cone = rm.cone_type.astype(str)
    ratio = (typ == "ratio").to_numpy()
    tab = var.map(table_of)
    unmapped = sorted(set(var[ratio & tab.isna().to_numpy()]))
    if unmapped:
        raise ValueError(f"{len(unmapped)} ratio variants in no pinned params table: "
                         f"{unmapped[:5]}")
    unknown = sorted(set(var[(typ == "sum").to_numpy()]) - set(SUM_VARIANTS))
    if unknown:
        raise ValueError(f"sum-row variants without a family: {unknown}")

    n = len(rm)
    per_year = rm.time.astype(str).value_counts()
    hard_y = rm.time[hard].astype(str).value_counts()
    soft_y = rm.time[~hard].astype(str).value_counts()
    bound_y = rm.time[(typ == "bound").to_numpy()].astype(str).value_counts()
    N.add("method:L133.n_rows", "constraint rows (also SI Table S1 M, S4 caption)", n)
    N.add("method:L133.rows_per_year_min", "fewest rows of a year (also S4 caption, S4.2)",
          per_year.min())
    N.add("method:L133.rows_per_year_max", "most rows of a year (also S4 caption, S4.2)",
          per_year.max())
    N.add("method:L133.n_hard", "hard rows (also S4 caption)", int(hard.sum()))
    N.add("method:L133.n_massbal", "mass balances (also S4 Table S4)",
          int((typ == "massbal").sum()))
    strip = ratio & (tab == "stripping_ratio").to_numpy()
    N.add("method:L133.n_stripping", "stripping relations (also Table S4)", int(strip.sum()))
    bands = ratio & hard & (cone == "nonneg").to_numpy()
    N.add("method:L133.n_bands", "hard one-sided ratio bands (also L170)", int(bands.sum()))
    N.add("method:L133.n_bounds", "non-negativity rows (also Table S4)",
          int((typ == "bound").sum()))
    N.add("method:L133.n_soft", "soft rows (also S4 caption)", int((~hard).sum()))
    N.add("method:L133.n_sum", "scale totals and scrap anchors", int((typ == "sum").sum()))
    soft_ratio = ratio & ~hard
    N.add("method:L133.n_soft_ratio", "soft ratio rows (net-stock rows included)",
          int(soft_ratio.sum()))
    N.add("method:L170.hard_rows_per_year_max", "most hard (zero-variance) rows of a year "
          "(also S6.2, S12)", hard_y.max())
    N.add("method:L170.edges_per_year", "edges of one year (also SI Table S1 E_t, S4.2, S6.2)",
          edges_per_year)
    N.add("method:L170.hard_rows_exceed_edges", "every year has more hard rows than edges",
          bool(hard_y.min() > edges_per_year))
    N.add("si:S6.hard_rows_per_year_min", "fewest hard rows of a year (also S12)",
          hard_y.min())
    N.add("si:S12.bound_rows_per_year", "non-negativity rows per year",
          sorted(set(bound_y.tolist())))
    N.add("si:S12.soft_rows_per_year_min", "fewest soft rows of a year", soft_y.min())
    N.add("si:S12.soft_rows_per_year_max", "most soft rows of a year", soft_y.max())
    N.add("si:S4.n_blocks", "yearly blocks of G", per_year.size)

    for v, k in SUM_VARIANTS.items():
        N.add(f"si:S4.T_S4.{k}", f"sum rows of variant {v}", int((var == v).sum()))
    for t, k in (("conc", "ratio_conc"), ("ore_grade", "ratio_oregrade"), ("eff", "ratio_eff"),
                 ("stock_bounds", "netstock")):
        N.add(f"si:S4.T_S4.{k}", f"soft ratio rows of table {t}",
              int((soft_ratio & (tab == t).to_numpy()).sum()))
    stock_of = dict(zip(P["stock_bounds"].param_id, P["stock_bounds"].b_sector_to))
    ns = var[soft_ratio & (tab == "stock_bounds").to_numpy()].map(stock_of)
    N.add("si:S4.T_S4.netstock_by_stock", "net-stock rows by stock-change node",
          ns.value_counts().to_dict())
    for t, k in (("conc", "band_conc"), ("ore_grade", "band_oregrade"), ("eff", "band_eff"),
                 ("scrap_share", "band_scrapshare")):
        N.add(f"si:S4.T_S4.{k}", f"hard band rows of table {t}",
              int((bands & (tab == t).to_numpy()).sum()))
    pin = (l == 0) & (u == 0)
    N.add("si:S4.T_S4.zero_pinned_edges", "edges with the flat-prior pin l = u = 0",
          int(pin.sum()))
    N.add("si:S4.T_S4.zero_pinned_pct", "pinned share of all edges", 100 * pin.mean(), "%")


# =============================================================================
# 5  Stage 4
# =============================================================================

def edge_frame(proc: Path, ch: Chain, ei: pd.DataFrame) -> pd.DataFrame:
    """Every edge of every year: prior, posterior and its decoded address."""
    parts = []
    for y in ch.years:
        o = np.load(proc / "opt" / ch.opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        parts.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=np.int64),
                                   "prior":   np.asarray(o["a_prior"], dtype=float),
                                   "post":    np.asarray(o["F_opt"], dtype=float)}))
    e = pd.concat(parts, ignore_index=True)
    e = e.merge(ei[["edge_id", "sector_from", "entity_from", "region_from", "sector_to",
                    "region_to", "valuation", "time"]], on="edge_id", how="left",
                validate="one_to_one")
    e["year"] = e.time.astype(int)
    return e


def soft_rows(proc: Path, ch: Chain, rm: pd.DataFrame) -> pd.DataFrame:
    """Every soft row of every year: c, σ_c and z = (c* − c)/σ_c, year-local."""
    time = rm.time.astype(str).to_numpy()
    soft = ~rm.is_hard.to_numpy(dtype=bool)
    cid, typ = rm.constraint_id.astype(str).to_numpy(), rm.type.astype(str).to_numpy()
    var, sec = rm.variant.astype(str).to_numpy(), rm.sector.astype(str).to_numpy()
    reg, val = rm.region.astype(str).to_numpy(), rm.valuation.astype(str).to_numpy()
    parts = []
    for y in ch.years:
        idx = np.flatnonzero(soft & (time == y))
        ct = np.load(proc / "const" / ch.const / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        o = np.load(proc / "opt" / ch.opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        cstar = np.full(len(rm), np.nan)
        cstar[np.asarray(o["rows_t"], dtype=np.int64)] = np.asarray(o["c_star"], dtype=float)
        c  = np.asarray(ct["c"], dtype=float)[idx]
        sc = np.asarray(ct["sigma_c"], dtype=float)[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sc > 0, (cstar[idx] - c) / sc, np.nan)
        parts.append(pd.DataFrame({"constraint_id": cid[idx], "type": typ[idx], "year": y,
                                   "variant": var[idx], "sector": sec[idx], "region": reg[idx],
                                   "valuation": val[idx], "c": c, "sigma_c": sc, "z": z}))
    return pd.concat(parts, ignore_index=True)


def _ret(e: pd.DataFrame, m) -> tuple:
    p, q = float(e.prior[m].sum()), float(e.post[m].sum())
    return p, q, (q / p if p else np.nan)


def stage4_numbers(N: Numbers, e: pd.DataFrame, s: pd.DataFrame, rm: pd.DataFrame, G,
                   a0: np.ndarray, sigma_a: np.ndarray, P: dict, strip: dict) -> None:
    """Group 5: quantities read off the edge frame and the soft-row residuals."""
    def pair(a, b, v):
        return (e.sector_from == a) & (e.sector_to == b) & (e.valuation == v)

    # smallest posterior on an edge that carries a non-negativity row
    rows = np.flatnonzero((rm.type.astype(str) == "bound").to_numpy())
    sub = G[rows]
    sub.eliminate_zeros()
    cols = np.unique(sub.indices)
    bounded = e.edge_id.isin(cols)
    N.add("method:L154.min_flow_bounded_kt", "smallest posterior on a non-negativity edge "
          "(also results.md:50, S13.2)", float(e.post[bounded].min()), "kt")
    N.add("method:L154.n_bounded_edges", "edges carrying a non-negativity row",
          int(bounded.sum()))

    # the prior-only constant against the misfit
    sa = sigma_a[e.edge_id.to_numpy()]
    with np.errstate(divide="ignore", invalid="ignore"):
        ze = (e.post.to_numpy() - e.prior.to_numpy()) / sa
        prior_const = float(np.nansum((e.prior.to_numpy() / sa) ** 2))
    zsq_e = float(np.nansum(ze ** 2))
    zs = s.z.to_numpy()
    fam = {"edges": zsq_e,
           "sum_rows": float(np.nansum(zs[(s.type == "sum").to_numpy()] ** 2)),
           "soft_ratios": float(np.nansum(zs[(s.type == "ratio").to_numpy()] ** 2))}
    total = sum(fam.values())
    N.add("method:L154.prior_constant", "Σ (a0/σa)² over 30 years", prior_const)
    N.add("method:L154.zsq_total", "misfit Σz² over 30 years (= numbers.json zsq_total_30y)",
          total)
    N.add("method:L154.zsq_by_family", "Σz² by family (= numbers.json zsq_by_family_30y)", fam)
    N.add("method:L154.prior_over_misfit", "prior constant ÷ misfit", prior_const / total)

    # waste rock (satellite flow fixed by the stripping relation)
    _, wr, _ = _ret(e, pair("Mining", "WR", "Mass"))
    _, ore, _ = _ret(e, pair("Mining", "Ore", "Mass"))
    N.add("method:L331.wr_posterior_kt", "world waste rock posterior, 30 years (also Table 2, "
          "S12 caption, S13.2)", wr, "kt")
    N.add("si:S10.T_S12.wr_over_ore", "world WR ÷ world ore posterior", wr / ore)
    N.add("si:S10.T_S12.wr_band_kt", "world WR at the stripping range of the chain's version "
          f"(k {strip['lo']}–{strip['hi']})", [ore * strip["lo"], ore * strip["hi"]], "kt")

    # mine output, world and Chile
    mf = pair("MF", "Concentrate", "CC")
    p, q, r = _ret(e, mf)
    N.add("si:S13.world_mf_conc_prior_kt", "world MF→Concentrate CC prior, 30 years", p,
          "kt CC")
    N.add("si:S13.world_mf_conc_post_kt", "world MF→Concentrate CC posterior", q, "kt CC")
    N.add("method:L353.world_conc_retention", "world MF→Concentrate CC retention, 1995–2024 "
          "(also S13.2)", r)
    N.add("method:L353.mine_deleted_kt", "mine output deleted, prior − posterior (also S13.2)",
          p - q, "kt CC")
    p, q, r = _ret(e, mf & (e.region_from == "Chile"))
    N.add("si:S13.chile_mf_conc_prior_kt", "Chile MF→Concentrate CC prior", p, "kt CC")
    N.add("si:S13.chile_mf_conc_post_kt", "Chile MF→Concentrate CC posterior", q, "kt CC")
    N.add("si:S13.chile_mf_conc_retention", "Chile MF→Concentrate CC retention, 1995–2024", r)

    # extraction chain outside the ore-grade region-years
    og = P["ore_grade"]
    co = og[(og.param_type == "concentration") & (og.a_sector_from == "CO")]
    ry = set(zip(co.a_region_from, co.a_time.astype(str)))
    m = e[mf]
    out = ~np.fromiter((k in ry for k in zip(m.region_from, m.time)), dtype=bool, count=len(m))
    by = m[out].groupby("region_from").post.sum().sort_values(ascending=False)
    N.add("method:L347.mf_conc_outside_grade_kt", "MF→Concentrate CC posterior outside the "
          "ore-grade region-years, 30 years", float(by.sum()), "kt CC")
    N.add("method:L347.mf_conc_outside_grade_top", "its five largest regions",
          by.head(5).round(1).to_dict(), "kt CC")

    # σ of a trade leg against σ of an anchor
    ex = e[(e.sector_from == "Concentrate") & (e.entity_from == "Flow") & (e.valuation == "CC")
           & (e.region_from != e.region_to)].edge_id.to_numpy()
    ok = a0[ex] > 0
    N.add("method:L347.median_leg_cv", "median σa/a0 of cross-region concentrate CC legs",
          float(np.median(sigma_a[ex][ok] / a0[ex][ok])))
    ss = s[(s.type == "sum") & (s.c != 0)]
    N.add("method:L347.median_anchor_cv", "median σc/|c| of the scale-total rows",
          float(np.median(ss.sigma_c / ss.c.abs())))
    pos = sigma_a[np.isfinite(sigma_a) & (sigma_a > 0)]
    N.add("method:L347.sigma_a_range", "smallest and largest prior σa", [pos.min(), pos.max()])
    N.add("method:L347.sigma_a_orders", "decades σa spans (the weights 1/σ² span twice that)",
          float(np.log10(pos.max() / pos.min())))
    sc = s.sigma_c[s.sigma_c > 0]
    N.add("method:L347.sigma_c_range", "smallest and largest soft-row σc",
          [float(sc.min()), float(sc.max())])


# =============================================================================
# 6  forced deletion
# =============================================================================

def _icsg(inp: Path, version: int) -> pd.DataFrame:
    d = inp / "feed" / "icsg" / f"v{version}"
    raw = pd.read_csv(d / "values.csv")
    rmap = pd.read_excel(d / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw.Region.map(dict(zip(rmap.src_region, rmap.root_region)))
    return raw


def _baci_mapped(d: Path) -> pd.DataFrame:
    raw = pd.read_csv(d / "values.csv")
    smap = pd.read_excel(d / "conc.xlsx", sheet_name="sector_map")
    rmap = pd.read_excel(d / "conc.xlsx", sheet_name="region_map")
    r_of = dict(zip(rmap.src_region.astype(int), rmap.root_region))
    raw["sector"] = raw.Sector.astype(int).map(dict(zip(smap.src_sector.astype(int),
                                                        smap.root_sector)))
    raw["region_from"] = raw.Region_from.astype(int).map(r_of)
    raw["region_to"] = raw.Region_to.astype(int).map(r_of)
    raw["time"] = raw.Year.astype(str)
    return raw


def forced_numbers(N: Numbers, inp: Path, ch: Chain, e: pd.DataFrame, band_hi: float) -> None:
    """Group 6: ``cap = c_hi·X_mass + U`` per mine region-year (SI Table S16).

    X_mass is the netted BACI cross-region concentrate export mass the chain's
    feed read, U the ICSG smelter concentrate intake of the chain's ICSG
    version; both read from their ``vN/`` folders.
    """
    raw = _baci_mapped(inp / "feed" / "baci" / f"v{ch.versions['baci']}")
    x = raw[(raw.sector == "Concentrate") & (raw.region_from != raw.region_to)]
    X = x.groupby(["region_from", "time"]).Value.sum().rename("X")
    ic = _icsg(inp, ch.versions["icsg"])
    ic = ic[ic.reg.notna() & (ic.Process == "Smelting") & (ic.Type == "Concentrate")
            & (ic.Supply_Use == "Use")]
    U = ic.assign(time=ic.Year.astype(str)).groupby(["reg", "time"]).Value.sum().rename("U")
    m = e[(e.sector_from == "MF") & (e.sector_to == "Concentrate") & (e.valuation == "CC")]
    d = m.groupby(["region_from", "time"])[["prior", "post"]].sum()
    d = d.join(X).join(U.rename_axis(["region_from", "time"])).fillna({"X": 0.0, "U": 0.0})
    d["cap"] = band_hi * d.X + d.U
    forced = float((d.prior - d.cap).clip(lower=0.0).sum())
    deleted = float((d.prior - d.post).sum())
    gross = float((d.prior - d.post).clip(lower=0.0).sum())
    N.add("method:L353.forced_kt", "forced deletion Σ max(0, P − c_hi·X − U) (also S13.2)",
          forced, "kt CC")
    N.add("method:L353.forced_pct", "forced ÷ net mine deletion (the text's share; above 100 "
          "when gains elsewhere offset the deletion; also S13.2)", 100 * forced / deleted, "%")
    N.add("method:L353.mine_deleted_gross_kt", "Σ max(0, prior − post) over mine region-years",
          gross, "kt CC")
    N.add("method:L353.forced_pct_of_gross", "forced ÷ gross mine deletion",
          100 * forced / gross, "%")
    N.add("method:L353.hard_capped_region_years", "region-years whose prior exceeds the cap "
          "(also S13.2)", int((d.prior > d.cap).sum()))


# =============================================================================
# 7  common set
# =============================================================================

def _csv_sums(proc: Path, opt: str):
    files = sorted((proc / "opt" / opt / "diag").glob("constraints_*.csv"))
    if not files:
        return None
    s = pd.concat([pd.read_csv(f, usecols=["constraint_id", "type", "z"]) for f in files],
                  ignore_index=True)
    return s[s.type == "sum"][["constraint_id", "z"]], len(files)


def common_set_numbers(N: Numbers, proc: Path, s: pd.DataFrame, base: str,
                       retol: str) -> None:
    """Group 7: Σz² over the scale-total rows present in all three runs."""
    new = s[s.type == "sum"][["constraint_id", "z"]]
    runs = {"base": _csv_sums(proc, base), "retol": _csv_sums(proc, retol)}
    missing = [k for k, v in runs.items() if v is None or v[1] != len(set(s.year))]
    if missing:
        print(f"   common set skipped: per-year CSVs missing or partial for {missing}")
        return
    S = {"new": new, "base": runs["base"][0], "retol": runs["retol"][0]}
    common = set.intersection(*(set(x.constraint_id) for x in S.values()))
    z = {k: float(np.nansum(x[x.constraint_id.isin(common)].z.to_numpy() ** 2))
         for k, x in S.items()}
    N.add("method:L353.common_set_rows", "scale-total rows in the run and both base runs",
          len(common))
    N.add("method:L353.common_set_zsq", "common-set Σz² of this run (also S13.2)", z["new"])
    N.add("si:S13.base_retol_common_set_zsq", f"common-set Σz² of {retol}", z["retol"])
    N.add("si:S13.base_common_set_zsq", f"common-set Σz² of {base}", z["base"])
    N.add("method:L353.vs_base_retol_pct", f"Σz² vs {retol} (also S13.2)",
          100 * (z["new"] / z["retol"] - 1), "%")
    N.add("method:L353.vs_base_pct", f"Σz² vs {base} (also S13.2)",
          100 * (z["new"] / z["base"] - 1), "%")


# =============================================================================
# 8  input tables
# =============================================================================

def _sigma(r) -> float:
    """σ of a parameter row: its sd, else range/4."""
    if str(r.unc_type) == "sd" and pd.notna(r.value_sd):
        return float(r.value_sd)
    return (float(r.value_max) - float(r.value_min)) / 4


def _gross_version(inp: Path, netted: int):
    """The BACI parse version a netted version was built from: the newest older
    version with the same ``quantity_recon.csv`` and no re-export ledger."""
    with open(inp / "feed" / "registry.json", encoding="utf-8") as f:
        vs = json.load(f)["baci"]["versions"]
    h = {v["version"]: v["hashes"] for v in vs}
    q = h[netted].get("quantity_recon.csv")
    cand = [v for v in h if v < netted and "reexport_ledger.csv" not in h[v]
            and h[v].get("quantity_recon.csv") == q]
    return max(cand) if cand else None


def params_numbers(N: Numbers, P: dict) -> None:
    """Group 8a: the parameter tables of Table 1 and SI Table S12."""
    conc = P["conc"].set_index("param_id")
    for pid, k in (("concconcentratesupply", "concentrate"), ("concmattesupply", "matte"),
                   ("concanodesupply", "anode"), ("conccathodesupply", "cathode"),
                   ("concscrapsupply", "scrap"), ("concslaguse", "slag"),
                   ("concpscuse", "psc")):
        if pid not in conc.index:
            N.add(f"method:L314.c_{k}", f"{pid} nominal (absent in this version)", None)
            continue
        r = conc.loc[pid]
        N.add(f"method:L314.c_{k}", f"{pid} nominal (also Fig. 2, S12)",
              float(r.value_nominal))
        N.add(f"method:L314.c_{k}_band", f"{pid} hard band (also S12)",
              [float(r.value_min), float(r.value_max)])
        N.add(f"si:S10.T_S12.c_{k}_sigma", f"{pid} σ (sd, or range/4)", _sigma(r))
    same = conc.loc[["conccersupply", "conccsxewsupply", "conccathodesupply"],
                    ["value_nominal", "value_min", "value_max"]].drop_duplicates()
    N.add("method:L314.cer_csxew_cathode_equal", "CER, CSXEW and cathode share one row",
          len(same) == 1)
    r = conc.loc["concconcentratesupply"]
    N.add("method:L314.conc_sd_pct", "concentrate σ in ratio units ×100 (the 3.75 %)",
          100 * float(r.value_sd), "%")
    N.add("method:L314.conc_sd_over_nominal_pct", "concentrate σ ÷ nominal (the 13.6 %)",
          100 * float(r.value_sd) / float(r.value_nominal), "%")
    N.add("method:L276.conc_rows", "concentration rows (Table 1)", len(conc))

    eff = P["eff"].set_index("param_id")
    N.add("method:L276.eff_rows", "recovery rows (Table 1)", len(eff))
    for pid, k in (("effmfflotation", "mf"), ("effsxew", "sxew"), ("effsmelting", "smelting"),
                   ("effcfr", "cfr"), ("effer", "er")):
        r = eff.loc[pid]
        N.add(f"method:L314.eta_{k}", f"{pid} nominal (also Fig. 2, S12)",
              float(r.value_nominal))
        N.add(f"method:L314.eta_{k}_band", f"{pid} band (also S12)",
              [float(r.value_min), float(r.value_max)])
        N.add(f"si:S10.T_S12.eta_{k}_sigma", f"{pid} σ (range/4)", _sigma(r))
        N.add(f"si:S10.T_S12.eta_{k}_pooled", f"{pid} pools all inputs (agg)",
              str(r["agg"]).lower() == "true")

    s = P["stripping_ratio"].iloc[0]
    N.add("method:L277.stripping_k", "stripping ratio k (also L194, Fig. 2, S4, S12)",
          float(s.value_nominal))
    N.add("method:L277.stripping_range", "its range (also L314, S12)",
          [float(s.value_min), float(s.value_max)])
    N.add("method:L277.stripping_sigma", "its σ = range/4 (also S12)", _sigma(s))

    og = P["ore_grade"]
    co = og[(og.param_type == "concentration") & (og.a_sector_from == "CO")]
    g = co.value_nominal.astype(float)
    N.add("method:L308.ore_grade_region_years", "region-years with an in-situ grade "
          "(also Table 1, S9.2)", co.groupby(["a_region_from", "a_time"]).ngroups)
    N.add("method:L308.ore_grade_regions", "mining regions with a grade (also Table 1, S9.2)",
          co.a_region_from.nunique())
    N.add("method:L308.ore_grade_median_pct", "median grade (also S9.2; S12 as a fraction)",
          100 * g.median(), "%")
    N.add("method:L308.ore_grade_p10_pct", "10th-percentile grade (also S9.2)",
          100 * g.quantile(0.1), "%")
    N.add("method:L308.ore_grade_p90_pct", "90th-percentile grade (also S9.2)",
          100 * g.quantile(0.9), "%")
    src = co.source.astype(str)
    N.add("si:S9.ore_grade_factor_fallback", "grade region-years from the factor file",
          int(src.str.contains("factor", case=False).sum()))
    N.add("si:S9.ore_grade_carried", "grade region-years carried forward past the series",
          int(src.str.contains("carried forward", case=False).sum()))
    rw = co[(co.a_region_from == "ROW") & co.a_time.astype(int).between(2004, 2006)]
    N.add("si:S9.row_grade_2004_06_pct", "ROW grade range in 2004–06 (issue #30)",
          [100 * rw.value_nominal.astype(float).min(),
           100 * rw.value_nominal.astype(float).max()] if len(rw) else None, "%")
    for route in ("MF", "SXEW"):
        t = og[(og.a_sector_from == route) & (og.a_sector_to == "Tailings")]
        N.add(f"si:S10.T_S12.tailings_{route.lower()}_median",
              f"tailings grade, {route}, median", t.value_nominal.astype(float).median())

    ss = P["scrap_share"]
    lo, hi = ss.value_min.astype(float), ss.value_max.astype(float)
    own = ~((lo == SCRAP_DEFAULT[0]) & (hi == SCRAP_DEFAULT[1]))
    N.add("method:L274.scrap_share_rows", "hard scrap-share bands (Table 1)", len(ss))
    N.add("method:L274.scrap_share_own_series", "bands from an own ICSG series (also S12)",
          int(own.sum()))
    for reg in ("China", "Austria", "Belgium"):
        r = ss[(ss.a_sector_to == "Smelting") & (ss.a_region_to == reg)]
        N.add(f"si:S10.T_S12.scrap_share_smelting_{reg.lower()}", f"{reg} smelting band",
              [float(r.value_min.iloc[0]), float(r.value_max.iloc[0])] if len(r) else None)

    sb = P["stock_bounds"]
    N.add("method:L275.stock_bounds_rows", "net-stock band rows (Table 1)", len(sb))
    N.add("method:L275.stock_bounds_rows_by_stock", "rows per stock-change node",
          sb.b_sector_to.value_counts().to_dict())
    band = sb.groupby("b_sector_to").value_max.apply(lambda x: sorted(set(x.astype(float))))
    N.add("method:L314.netstock_bound_pct", "net-stock band |v_max| by node (PSC 4.9 % in the "
          "text; also S12)", {k: [100 * v for v in vs] for k, vs in band.items()}, "%")


def feed_input_numbers(N: Numbers, inp: Path, ch: Chain) -> None:
    """Group 8b: feed concordances, BACI reconstruction and netting ledgers."""
    ic = _icsg(inp, ch.versions["icsg"])
    rm = pd.read_excel(inp / "feed" / "icsg" / f"v{ch.versions['icsg']}" / "conc.xlsx",
                       sheet_name="region_map")
    N.add("method:L194.icsg_labels_to_row", "ICSG labels folded into ROW (also S9)",
          int((rm.root_region == "ROW").sum()))
    N.add("si:S8.T_S9.icsg_labels", "ICSG source labels", rm.src_region.nunique())
    N.add("si:S8.T_S9.icsg_roots", "regions they map to", rm.root_region.nunique())
    N.add("method:L262.icsg_editions", "yearbook editions read",
          sorted(str(int(x)) for x in ic.Year_loaded_from_file.dropna().unique()))
    tw = ic[(ic.reg == "Taiwan") & (ic.Type == "Cathode") & (ic.Supply_Use == "Use")
            & ic.Process.isin(SEMI_USE)]
    N.add("method:L347.taiwan_usage_kt", "Taiwan's ICSG usage (Semi + USC + MESC), 30 years "
          "(also S11)", float(tw.Value.sum()), "kt CC")

    bv = ch.versions["baci"]
    bd = inp / "feed" / "baci" / f"v{bv}"
    rmb = pd.read_excel(bd / "conc.xlsx", sheet_name="region_map")
    smb = pd.read_excel(bd / "conc.xlsx", sheet_name="sector_map")
    N.add("method:L194.baci_codes_to_row", "BACI partner codes folded into ROW (also S9)",
          int((rmb.root_region == "ROW").sum()))
    N.add("method:L194.baci_codes_to_taiwan", "BACI codes mapped to Taiwan",
          int((rmb.root_region == "Taiwan").sum()))
    N.add("si:S8.T_S9.baci_codes", "BACI partner codes", rmb.src_region.nunique())
    N.add("si:S8.T_S9.baci_roots", "regions they map to", rmb.root_region.nunique())
    c490 = rmb[rmb.src_region.astype(int) == TAIWAN_CODE].root_region
    N.add("si:S8.T_S9.baci_code_490_root", "region of BACI 490 “Other Asia, nes”",
          c490.iloc[0] if len(c490) else None)
    N.add("method:L262.baci_hs_codes", "HS6 codes read, by root sector",
          smb.groupby("root_sector").src_sector.apply(lambda x: sorted(str(int(c)) for c in x))
             .to_dict())
    rmc = pd.read_excel(inp / "feed" / "crude_ore" / f"v{ch.versions['crude_ore']}"
                        / "conc.xlsx", sheet_name="region_map")
    N.add("si:S8.T_S9.crude_ore_series", "crude-ore country series", rmc.src_region.nunique())
    N.add("si:S8.T_S9.crude_ore_roots", "regions they map to", rmc.root_region.nunique())
    N.add("si:S8.T_S9.crude_ore_to_row", "series folded into ROW",
          int((rmc.root_region == "ROW").sum()))
    cdir = inp / "feed" / "crude_ore" / f"v{ch.versions['crude_ore']}"
    co = pd.read_csv(cdir / "values.csv", usecols=["Year"])
    N.add("method:L262.crude_ore_last_year", "last year of the crude-ore series",
          str(int(co.Year.max())))

    q = pd.read_csv(bd / "quantity_recon.csv")
    side = q.side if "side" in q.columns else pd.Series("above", index=q.index)
    a, b = q[side == "above"], q[side == "below"]
    gv = _gross_version(inp, bv)
    gross = pd.read_csv(inp / "feed" / "baci" / f"v{gv}" / "values.csv") if gv else None
    N.add("method:L291.gross_version", "BACI parse version the netting read", gv)
    N.add("method:L291.recon_above_legs", "concentrate legs rebuilt upwards (Table 1, S11)",
          len(a))
    N.add("method:L291.gross_conc_legs", "concentrate legs of the parse (also S11)",
          int((gross.Sector == 260300).sum()) if gross is not None else None)
    N.add("method:L291.recon_above_exporters", "exporters rebuilt upwards (also S11)",
          a.Region_from.nunique())
    N.add("method:L291.recon_above_kt", "export mass added upwards, 30 years (also S11)",
          a.recovered.sum() / 1e3, "kt")
    N.add("method:L291.recon_above_chile_kt", "of it Chile (also S11)",
          a[a.Region_from == CHILE_CODE].recovered.sum() / 1e3, "kt")
    has_below = "side" in q.columns
    N.add("method:L291.recon_below_legs", "legs rebuilt downwards (also Table 1, S11)",
          len(b) if has_below else None)
    N.add("method:L291.recon_below_q_obs_kt", "their recorded mass (also S11)",
          b.q_obs.sum() / 1e3 if has_below else None, "kt")
    N.add("method:L291.recon_below_q_new_kt", "their rebuilt mass (also S11)",
          b.q_new.sum() / 1e3 if has_below else None, "kt")
    png = b[(b.Region_from == PNG_CODE) & b.Year.between(*PNG_YEARS)]
    N.add("method:L291.recon_below_png_kt", "PNG 2019–22 recorded mass rebuilt downwards",
          png.q_obs.sum() / 1e3 if has_below else None, "kt")

    L = pd.read_csv(bd / "reexport_ledger.csv")
    N.add("method:L302.reexport_T_kt", "re-export volume moved, Σ T_applied (also Table 2, "
          "L336, S11)", L.T_applied.sum(), "kt CC")
    N.add("si:S9.T_S11.reexport_new_legs", "new origin–destination legs", L.n_new_legs.sum())
    N.add("si:S9.T_S11.reexport_self_dropped_kt", "re-imports dropped", L.self_dropped.sum(),
          "kt CC")
    N.add("si:S9.T_S11.reexport_origins_screened", "screened origin-years",
          L.origins_screened.sum())
    N.add("si:S9.T_S11.reexport_resid_kt", "arrivals left in the residual ledger",
          L.resid.sum(), "kt CC")
    if gross is not None:
        g = _baci_mapped(inp / "feed" / "baci" / f"v{gv}")
        same = int((g.region_from == g.region_to).sum())
        N.add("si:S9.T_S11.self_trade_rows", "parse rows that map to self-trade", same)
        N.add("si:S9.T_S11.parse_rows", "parse rows", len(g))
        N.add("si:S9.T_S11.self_trade_pct", "self-trade share of the parse rows",
              100 * same / len(g), "%")

    tv = ch.versions.get("trade_shares")
    ts = pd.read_csv(inp / "params" / "trade_shares" / f"v{tv}" / "params.csv")
    N.add("method:L272.trade_shares_rows", "trade-share weights (also S11)", len(ts))
    N.add("method:L347.taiwan_trade_share_weights", "positive trade-share weights into or out "
          "of Taiwan (0 = its usage stays undistributed)",
          int(((ts.root_region_to == "Taiwan") | (ts.root_region_from == "Taiwan"))
              [ts.weight > 0].sum()))


# =============================================================================
# 9  Results numbers without a numbers.json key
# =============================================================================

def results_numbers(N: Numbers, e: pd.DataFrame, s: pd.DataFrame, numbers_path: Path,
                    opt: str) -> None:
    """Group 9: the keyless Results numbers, and ratios of numbers.json keys."""
    sums = s[(s.type == "sum") & np.isfinite(s.z)]
    sh = sums.assign(gt3=sums.z.abs() > 3).groupby("year").gt3.mean()
    N.add("results:L60.share_gt3_sum_rows", "share of scale totals with |z| > 3, by year "
          "(= si_s15_decomposition.csv)", sh.round(4).to_dict())
    yr = sh.index.astype(int)
    N.add("results:L60.share_gt3_mean_to_2016", "its mean over 1995–2016",
          sh[yr <= 2016].mean())
    N.add("results:L60.share_gt3_mean_from_2017", "its mean over 2017–2024",
          sh[yr > 2016].mean())
    N.add("results:L60.a2_failing_years", f"years at or below {A2_SUMS} (also SI Table S18b "
          "A2)", sh[sh <= A2_SUMS].round(3).to_dict())

    cc = e[e.valuation == "CC"]
    inp = cc[(cc.sector_from == "Scrap")
             | ((cc.sector_from == "Concentrate") & (cc.sector_to == "Smelting"))
             | ((cc.sector_from == "SXEW") & (cc.sector_to == "CSXEW"))]
    cn = inp[inp.region_to == "China"]
    sec = cn[cn.sector_from == "Scrap"].groupby("year").post.sum()
    share = (sec / cn.groupby("year").post.sum()).reindex(sorted(set(e.year)))
    N.add("results:L95.china_secondary_post", "China's input-side secondary share, posterior, "
          "by year", share.round(4).to_dict())

    row = []                                  # ROW's share per step and year, posterior
    for a, b, col in HHI_STEPS.values():
        x = cc[(cc.sector_from == a) & (cc.sector_to == b)].groupby(["year", col]).post.sum()
        for y in sorted(set(e.year)):
            v = x.loc[y]
            row.append((y, float(v.get("ROW", 0.0) / v.sum())))
    ends = [r for y, r in row if y in (min(e.year), max(e.year))]
    row = [r for _, r in row]
    N.add("results:L97.hhi_row_band_max", "largest ROW band in HHI (upper − lower = ROW "
          "share²), any step and year, posterior", max(row) ** 2)
    N.add("results:L97.hhi_row_band_max_ends", "the same over the first and last year only "
          "(the years si_s15_hhi_1995_2024.csv holds)", max(ends) ** 2)
    N.add("si:S15.T_S18.row_share_range", "ROW's share across steps and years, posterior",
          [min(row), max(row)])
    N.add("si:S15.T_S18.row_share_range_ends", "the same over the first and last year only",
          [min(ends), max(ends)])

    if not numbers_path.exists():
        return
    with open(numbers_path, encoding="utf-8") as f:
        nj = json.load(f)
    if nj.get("opt_run_id") != opt:
        print(f"   numbers.json belongs to {nj.get('opt_run_id')}: its ratios skipped")
        return
    sp_ = nj["secondary_pooled_2020_24"]
    N.add("results:L95.eu14_over_world", "EU-14 ÷ world secondary share, pooled 2020–24",
          sp_["EU-14"]["post"] / sp_["World"]["post"])
    lift = {g: v["post"] - v["prior"] for g, v in sp_.items() if g != "World"}
    top = max(lift, key=lift.get)
    N.add("results:L95.largest_lift", "group with the largest post − prior share; the lift",
          [top, lift[top]])
    h = nj["hhi_2024"]
    d = {k: h[f"{k}_post"]["upper"] - h[f"{k}_prior"]["upper"] for k in HHI_STEPS}
    N.add("results:L97.dhhi_2024", "HHI post − prior in 2024 by step (also SI S18b B2)", d)
    N.add("results:L97.dhhi_other_max", "largest |ΔHHI| of the steps other than use",
          max(abs(v) for k, v in d.items() if k != "use"))
    N.add("results:L99.china_footprint_share", "China's footprint ÷ world extraction",
          nj["china_footprint_kt"] / nj["world_extraction_kt"])
    ret = nj["retention_30y"]
    N.add("results:L64.copper_stages_within_001", "|1 − retention| < 0.01 for CP→Cathode and "
          "MF→Concentrate (also S15.1)",
          all(abs(1 - ret[k]) < 0.01 for k in ("CP→Cathode CC", "MF→Concentrate CC")))
    fam = nj["zsq_by_family_30y"]
    N.add("si:S16.soft_ratio_share_pct", "soft-ratio share of the objective (S26 caption)",
          100 * fam["soft_ratios"] / nj["zsq_total_30y"], "%")


# =============================================================================
# 10  Stage 5
# =============================================================================

def stage5_runs(sidecars: dict) -> tuple:
    """``(default, ablation)``: the latest uncprop run without and the latest with
    ``exclude_scale_totals`` among ``sidecars`` (run id → sidecar); None where absent."""
    abl = {r for r, m in sidecars.items()
           if m.get("stage_config", {}).get("exclude_scale_totals", False)}
    default = sorted(set(sidecars) - abl)
    return (default[-1] if default else None, max(abl) if abl else None)


def _kappa(proc: Path, run: str, acc) -> pd.DataFrame:
    """Median and mass-weighted posterior contraction per valuation × prior class."""
    d = UncPropDiag.from_uncprop_id(run, uncprop_dir=str(proc / "uncprop"),
                                    opt_dir=str(proc / "opt"), const_dir=str(proc / "const"),
                                    imp_dir=str(proc / "imp"), acc=acc)
    f = d.stats.edge_frame()
    f = f[np.isfinite(f.kappa)]
    g = f.groupby(["valuation", "prior_class"], observed=True)
    return pd.DataFrame({"median":   g.kappa.median(),
                         "weighted": g.apply(lambda x: np.sum(x.kappa * x.weight) / np.sum(x.weight))})


def stage5_numbers(N: Numbers, proc: Path, ch: Chain, acc) -> None:
    """Group 10: the default posterior's active set, Birge ratio and calibration (#87),
    and the contraction with and without the scale totals."""
    runs = {r: _sidecar(proc / "uncprop" / r) for r in ch.uncprop}
    run, ablation = stage5_runs(runs)
    if run is None:
        return
    per = runs[run]["stage_summary"]["per_timestep"]

    def span(k):
        return [min(v[k] for v in per.values()), max(v[k] for v in per.values())]

    N.add("method:L175.uncprop_run_id", "the default Stage-5 run (also header, S6.2, S17)", run)
    N.add("method:L175.pinned_per_year", "edges pinned at a binding lower bound, per year (also S6.2)",
          span("n_bound_active"))
    N.add("method:L175.bands_binding_per_year", "bands binding, per year (also S6.2)",
          span("n_bands_binding"))
    N.add("method:L175.dependent_hard_per_year", "dependent hard rows dropped, per year (also S6.2)",
          span("n_hard_dependent"))
    N.add("si:S6.2.rows_kept_per_year", "rows kept, per year", span("n_rows_kept"))
    N.add("si:S6.2.pinned_edge_years", "edge-years pinned, all years",
          sum(v["n_bound_active"] for v in per.values()))
    N.add("si:S6.2.max_resid_hard_kt", "largest hard residual at a*",
          max(v["max_resid_hard"] for v in per.values()), "kt")
    trace = [1 - v["trace_post"] / v["trace_prior"] for v in per.values()]
    N.add("si:S6.2.trace_contraction", "smallest and largest yearly trace contraction "
          "1 − Σσ_a*²/Σσ_a²", [min(trace), max(trace)])

    diag = proc / "uncprop" / run / "diag"
    b = pd.read_csv(diag / "birge.csv")
    b["year"] = b.year.astype(str)
    allr, yr = b[b.year == "all"].iloc[0], b[b.year != "all"]
    N.add("method:L175.birge_ratio", "run Birge ratio Σz²/dof (#87; also results.md:68, S6.2)",
          allr.birge_ratio)
    N.add("si:S6.2.birge_zsq_dof", "the run's Σz² and dof", [allr.zsq_total, allr.dof])
    N.add("si:S6.2.birge_ratio_by_year", "smallest and largest yearly Birge ratio",
          [yr.birge_ratio.min(), yr.birge_ratio.max()])
    c = pd.read_csv(diag / "calibration.csv").set_index(["valuation", "prior_class"])
    for v in ("CC", "Mass"):
        for cls in ("from a feed", "imputed"):
            r = c.loc[(v, cls)]
            N.add(f"method:L175.calibration.{v}.{cls}", f"|z/√κ| > 3 share, the same Birge-scaled, "
                  f"median |z/√κ|: {v} {cls} (#87; also results.md:68, S6.2)",
                  [r.share_gt3, r.share_gt3_birge, r.q50_abs])

    k = _kappa(proc, run, acc)
    for v in ("CC", "Mass"):
        for cls in ("from a feed", "imputed"):
            N.add(f"si:S6.2.kappa.{v}.{cls}", f"median and mass-weighted contraction: {v} {cls} "
                  "(also results.md:68)", [k.loc[(v, cls), "median"], k.loc[(v, cls), "weighted"]])
    if ablation is not None:
        ka = _kappa(proc, ablation, acc)
        N.add("method:L175.uncprop_ablation_run_id", "the #87 scale-total ablation (also S6.2)",
              ablation)
        for v in ("CC", "Mass"):
            N.add(f"method:L175.kappa_weighted_ablation.{v}", f"mass-weighted contraction of "
                  f"feed-observed {v} edges, default → without scale totals (also S6.2)",
                  [k.loc[(v, "from a feed"), "weighted"], ka.loc[(v, "from a feed"), "weighted"]])


# =============================================================================
# 11  limitations
# =============================================================================

def mine_region_years(ic: pd.DataFrame) -> set:
    """(root region, year) pairs with positive ICSG concentrate output."""
    m = ic[ic.reg.notna() & (ic.Process == "Mining") & (ic.Type == "Concentrate")
           & (ic.Supply_Use == "Supply")]
    g = m.groupby(["reg", m.Year.astype(str)]).Value.sum()
    return {(r, y) for (r, y), v in g.items() if v > 0}


def phantom_mine_chain(e: pd.DataFrame, mine: set) -> dict:
    """The CC mine chain the solver keeps in region-years without ICSG mine output."""
    cc = e[e.valuation == "CC"]
    no_mine = ~pd.MultiIndex.from_arrays([cc.region_from, cc.time.astype(str)]).isin(list(mine))
    dom = (cc.region_from == cc.region_to).to_numpy()

    def step(sf, st):
        return cc[dom & no_mine & (cc.sector_from == sf).to_numpy() & (cc.sector_to == st).to_numpy()]

    mf = step("MF", "Concentrate")
    out = cc[no_mine & (cc.sector_from == "Concentrate").to_numpy()]
    by = mf.groupby("region_from").post.sum().sort_values(ascending=False)
    return {"mf_conc_post_kt":   float(mf.post.sum()),
            "mf_conc_prior_kt":  float(mf.prior.sum()),
            "co_mining_post_kt": float(step("CO", "Mining").post.sum()),
            "export_share":      float(out[out.region_from != out.region_to].post.sum()
                                       / out.post.sum()),
            "by_region":         {r: float(x) for r, x in by[by > 1.0].items()}}


def top_year_shares(s: pd.DataFrame, regions: list, n: int) -> dict:
    """Per region, its ``n`` years with the largest share of its scale-total Σz²."""
    x = s[(s.type == "sum") & s.region.isin(regions)]
    z2 = (x.z ** 2).groupby([x.region, x.year]).sum()
    out = {}
    for r in regions:
        y = z2.loc[r].sort_values(ascending=False)
        out[r] = {str(k): float(v / y.sum()) for k, v in y.head(n).items()}
    return out


def sum_shares_by_region(s: pd.DataFrame) -> dict:
    """Each region's share of the scale-total Σz², largest first."""
    x = s[s.type == "sum"]
    z2 = (x.z ** 2).groupby(x.region).sum().sort_values(ascending=False)
    return {r: float(v / z2.sum()) for r, v in z2.items()}


def exports_beyond_production(led: pd.DataFrame, sector: str, regions: list,
                              min_kt: float = EXCESS_MIN_KT) -> dict:
    """Per region and year, the netting ledger's BACI exports ``X`` and ICSG production
    ``P`` (``None`` where ICSG has no row) of ``sector`` where ``X − P > min_kt`` —
    #111's contradictions between the two sources (kt CC)."""
    x = led[(led.sector == sector) & led.region.isin(regions)]
    x = x[x.X - x.P.fillna(0.0) > min_kt]
    out: dict = {}
    for r in x.itertuples(index=False):
        out.setdefault(r.region, {})[str(r.time)] = [float(r.X),
                                                     None if pd.isna(r.P) else float(r.P)]
    return out


def withheld_usage(led: pd.DataFrame) -> dict:
    """The ICSG usage ADR 0027's ``imports_exhausted`` withhold leaves undistributed:
    the groups, their usage ``U`` and its split by region, largest first (kt CC)."""
    w = led[led.imports_exhausted.astype(bool)]
    by = w.groupby("region").U.sum().sort_values(ascending=False)
    return {"groups": int(len(w)), "usage_kt": float(w.U.sum()),
            "by_region": {r: float(v) for r, v in by.items()}}


def limitation_numbers(N: Numbers, inp: Path, ch: Chain, e: pd.DataFrame, s: pd.DataFrame,
                       sigma_a: np.ndarray) -> None:
    """Group 11: the numbers of method.md §3.2.8."""
    ic = _icsg(inp, ch.versions["icsg"])
    p = phantom_mine_chain(e, mine_region_years(ic))
    N.add("method:L352.phantom_mf_conc_kt", "MF→Concentrate CC posterior where ICSG reports no "
          "mine output, 30 years (#31)", p["mf_conc_post_kt"], "kt CC")
    N.add("method:L352.phantom_regions", "its regions above 1 kt CC, largest first", p["by_region"],
          "kt CC")
    N.add("method:L352.phantom_export_share", "the share of that concentrate exported",
          p["export_share"])
    N.add("method:L352.phantom_co_mining_kt", "CO→Mining CC posterior there", p["co_mining_post_kt"],
          "kt CC")

    fam = (((e.sector_from == "Cathode") & e.sector_to.isin(["MESC", "USC"]))
           | ((e.sector_from == "CP") & (e.sector_to == "PSC")))
    st = e[fam & (e.valuation == "CC")]
    z = (st.post - st.prior) / sigma_a[st.edge_id.to_numpy()]
    big = st[np.abs(z.to_numpy()) > Z_BIG]
    N.add("method:L352.stock_legs_beyond_z5", "CC stock legs with |z| > 5: count and Σ posterior "
          "(#95)", [len(big), big.post.sum()], "kt CC")
    N.add("method:L352.stock_legs_beyond_z5_by_region", "their Σ posterior by region (Kazakhstan: "
          "the text's figure)", big.groupby("region_from").post.sum().sort_values().to_dict(), "kt CC")

    years = [int(y) for y in ch.years if int(y) >= PSC_FIRST_YEAR]
    rep = ic[ic.Type == "PSC"]
    icsg = rep.groupby(rep.Year.astype(int)).Value.sum().reindex(years).fillna(0.0)
    psc = (e[(e.sector_from == "CP") & (e.sector_to == "PSC") & (e.valuation == "CC")]
           .groupby("year").post.sum().reindex(years).fillna(0.0))
    N.add("method:L352.world_psc_gap_kt", f"Σ|world CP→PSC posterior − ICSG's reported PSC|, "
          f"{years[0]}–{years[-1]} (#98)", float((psc - icsg).abs().sum()), "kt CC")

    sh = sum_shares_by_region(s)
    top = dict(list(sh.items())[:6])
    N.add("method:L352.sum_share_top_regions", "regions' shares of the scale-total Σz², largest "
          "six (#111)", top)
    N.add("method:L352.sum_share_austria", "Austria's share (#63)", sh.get("Austria"))
    N.add("method:L352.sum_top_years", "each of the four largest regions' four years with the "
          "largest share of its scale-total Σz² (Uzbekistan: the text's 1995–1998)",
          top_year_shares(s, list(top)[:4], n=4))
    an = s[(s.type == "sum") & (s.variant == "usetotout") & (s.sector == "Anode")
           & (s.valuation == "CC")]
    worst = {}
    for r in list(top)[:4]:
        x = an[an.region == r]
        if len(x):
            i = x.z.abs().idxmax()
            worst[r] = [x.year[i], x.c[i], x.c[i] + x.z[i] * x.sigma_c[i]]
    N.add("method:L352.anode_use_total_worst", "largest-|z| CC anode use total of the four regions "
          "with the largest shares: year, c, c* (#111)", worst, "kt CC")

    led = pd.read_csv(inp / "feed" / "baci" / f"v{ch.versions['baci']}" / "reexport_ledger.csv")
    for sector in ("Anode", "Cathode"):
        N.add(f"method:L352.{sector.lower()}_exports_beyond_production",
              f"BACI {sector.lower()} exports X against ICSG production P (None: no ICSG row) "
              f"of the four regions, year by year where X − P > {EXCESS_MIN_KT:g} kt (#111)",
              exports_beyond_production(led, sector, list(top)[:4]), "kt CC")
    N.add("method:L352.withheld_icsg_usage", "ICSG usage the netting's imports_exhausted withhold "
          "leaves undistributed: groups, kt CC, by region (ADR 0027 am., #111)",
          withheld_usage(led), "kt CC")


# =============================================================================
# output
# =============================================================================

def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        a = abs(v)
        if a == 0:
            s = "0"
        elif a >= 1000:
            s = f"{v:,.1f}"
        elif a >= 1e-3:
            s = f"{v:.4g}"
        else:
            s = f"{v:.3g}"
        return s
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def _section_sort(sec: str) -> tuple:
    doc, loc = sec.split(":", 1)
    m = re.match(r"[LS](\d+)", loc)
    return ({"method": 0, "si": 1, "results": 2}[doc], int(m.group(1)) if m else 0, loc)


def render(N: Numbers, ch: Chain) -> str:
    groups: dict = {}
    for k, it in N.items.items():
        groups.setdefault(k.split(".", 1)[0], []).append(it)
    w = max(len(k) for k in N.items) + 2
    out = [f"manuscript numbers of {ch.opt}  ({len(N.items)} numbers)",
           f"chain: {ch.const} ← {ch.imp} ← " + ", ".join(ch.feeds.values())]
    doc = None
    for sec in sorted(groups, key=_section_sort):
        if sec.split(":")[0] != doc:
            doc = sec.split(":")[0]
            n_doc = sum(len(v) for s_, v in groups.items() if s_.split(":")[0] == doc)
            out.append(f"\n== {doc}.md  ({n_doc} numbers) " + "=" * 50)
        out.append(f"-- {sec}")
        for it in groups[sec]:
            val = _fmt(it.value) + (f" {it.unit}" if it.unit and it.value is not None else "")
            out.append(f"   {it.key:<{w}} {val:<28} {it.label}")
    return "\n".join(out)


def main(opt: str, out_json, data_dir: str, base: str, retol: str, numbers: str) -> None:
    proc, inp = Path(data_dir) / "proc", Path(data_dir) / "input"
    ch = walk_chain(proc, opt)
    acc_id = ch.meta["const"]["parents"]["inputs"]["acc"]["id"]
    acc = AccountingStructure.load(id=acc_id, storage_dir=str(proc / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    ei["time"] = ei.time.astype(str)
    P = {pid: _params(inp, pid, ch.versions[pid]) for pid in PARAM_TABLES}

    N = Numbers()
    sidecar_numbers(N, ch)
    acc_numbers(N, acc, ei, ch.years)
    coverage_numbers(N, proc, ch, ei)

    cdir = proc / "const" / ch.const
    rm = pd.read_parquet(cdir / "row_meta.parquet",
                         columns=["constraint_id", "type", "variant", "is_hard", "cone_type",
                                  "time", "sector", "region", "valuation"])
    l, u = np.load(cdir / "l.npy"), np.load(cdir / "u.npy")
    edges_per_year = int(ei.groupby("time").size().max())
    row_numbers(N, rm, P, l, u, edges_per_year)

    e = edge_frame(proc, ch, ei)
    s = soft_rows(proc, ch, rm)
    strip = P["stripping_ratio"].iloc[0]
    sigma_a = np.load(cdir / "sigma_a.npy")
    stage4_numbers(N, e, s, rm, sp.load_npz(cdir / "G.npz").tocsr(),
                   np.load(cdir / "a0.npy"), sigma_a, P,
                   {"lo": float(strip.value_min), "hi": float(strip.value_max)})
    band_hi = float(P["conc"].set_index("param_id").loc["concconcentratesupply", "value_max"])
    N.add("si:S13.T_S16.conc_band_hi", "concentrate grade cap c_hi in the forced deletion",
          band_hi)
    forced_numbers(N, inp, ch, e, band_hi)
    common_set_numbers(N, proc, s, base, retol)
    params_numbers(N, P)
    feed_input_numbers(N, inp, ch)
    results_numbers(N, e, s, Path(numbers), opt)
    stage5_numbers(N, proc, ch, acc)
    limitation_numbers(N, inp, ch, e, s, sigma_a)

    print(render(N, ch))
    if out_json:
        payload = {"opt_run_id": opt, "generated": date.today().isoformat(),
                   "chain": {"const": ch.const, "imp": ch.imp, "feeds": ch.feeds,
                             "psut": ch.psut, "uncprop": ch.uncprop, "versions": ch.versions},
                   "base_runs": {"base": base, "base_retol": retol},
                   "numbers": {k: {"label": it.label, "value": it.value, "unit": it.unit}
                               for k, it in N.items.items()}}
        Path(out_json).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\nwritten {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("opt_run_id")
    ap.add_argument("--json", default=NUMBERS_OUT,
                    help="write the numbers to this JSON file (default: the committed diff base)")
    ap.add_argument("--data-dir", default="data", help="the checkout's data/ directory")
    ap.add_argument("--base", default=BASE_RUN, help="base run of the common set")
    ap.add_argument("--base-retol", default=BASE_RETOL_RUN,
                    help="the base re-solved at the reference tolerance")
    ap.add_argument("--numbers", default=NUMBERS_JSON,
                    help="results_figs numbers.json to take ratios from, if it is this run's")
    a = ap.parse_args()
    main(a.opt_run_id, a.json, a.data_dir, a.base, a.base_retol, a.numbers)
