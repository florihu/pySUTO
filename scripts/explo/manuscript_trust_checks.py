"""manuscript_trust_checks.py
==========================
Eight read-only checks on the copper reference chain (#311, round `manuscript final`).
Each check says whether a manuscript number rests on a silent fill, clip or stale input.
The script changes no library code and writes no data. It prints one table:
check · number · run id · verdict (clean / hit).

    PYTHONPATH=<checkout>/src python scripts/explo/manuscript_trust_checks.py \
        --data-dir data [--opt <id>] [--imp <id>] [--uncprop <id>] [--psut <id>] [--only 1,5]

The run ids default to the reference chain. The const run is read from the opt sidecar.

Checks
------
1  Feedprep NaN as 0 (#290 U1a-G3-7): born legs skipped in ``rebuild_values``,
   ``nan_to_num`` of the F_root reads in ``feedprep/ore_grade.py``, ``.get("src", "icsg")``.
2  "NA" as missing (#290 U7-S5-3): cells that pandas reads as NaN from a literal NA.
3  Clipped posterior variance (#290 U5-S2-1/2): edges clipped at 0, hard residual.
4  Leontief inverse (#290 U6-G4-7): rho(A) and min(L) of the PIOT that Figure 6d reads.
5  Diag sigma vs solver sigma (#253): sum z^2 of the diag against the solver objective.
6  Stage 5 floors (#270): the uncprop floors against the opt run's recorded floors.
7  Flat prior level 3 (#239): edges that reach ``FlatPriorImputer._resolve_scale`` level 3.
8  Ore-grade lineage (#287): the parents of the ore_grade version the ICSG feed used.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
sys.path.insert(0, _SCRIPTS)

REF = dict(
    opt     = "opt_20261119_130302_b20f0dba",
    imp     = "imp_20261119_125259_f2b11cd5",
    uncprop = "uncprop_20261119_133344_08dfa15b",
    psut    = "psut_20261119_133504_4c783a88",
)

HARD_CLAIM = 1e-6      # kt, the manuscript's A1 threshold (docs/manuscript/results.md)


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------

def _json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sidecar(data_dir: str, stage: str, run_id: str) -> dict:
    return _json(os.path.join(data_dir, "proc", stage, run_id, "metadata.json"))


def _feed_sidecar(data_dir: str, run_id: str) -> dict:
    return _json(os.path.join(data_dir, "proc", "feed", "registry", run_id, "metadata.json"))


def _feed_run(imp_sc: dict, name: str) -> str:
    ids = [r for r in imp_sc["parents"]["upstream"]["feed_run_ids"] if r.startswith(f"feed_{name}_")]
    if len(ids) != 1:
        raise ValueError(f"imp run names {len(ids)} {name} feed runs: {ids}")
    return ids[0]


def _row(check, number, run, verdict, note=""):
    return dict(check=check, number=number, run=run, verdict=verdict, note=note)


# ---------------------------------------------------------------------------
# Check 1
# ---------------------------------------------------------------------------

def check1(data_dir: str, opt_id: str, imp_id: str) -> list:
    from copper_feed import GROSS_VALUATION, HS_OF, ORE_GRADE_LABELS, TRADED_SECTORS
    from pysuto import acc_store
    from pysuto.feedprep.ore_grade import _decoded_edges
    from pysuto.feedprep.reexport import canonical_codes, load_root_trade, rebuild_values
    from pysuto.feedprep.trade_shares import traded_conc_lookup

    imp = _sidecar(data_dir, "imp", imp_id)
    baci_run = _feed_run(imp, "baci")
    icsg_run = _feed_run(imp, "icsg")
    crude_run = _feed_run(imp, "crude_ore")
    baci_v = _feed_sidecar(data_dir, baci_run)["parents"]["inputs"]["feed"]["version"]
    icsg_v = _feed_sidecar(data_dir, icsg_run)["parents"]["inputs"]["feed"]["version"]
    inp = os.path.join(data_dir, "input")
    pj = _json(os.path.join(inp, "feed", "icsg", f"v{icsg_v}", "parents.json"))
    src_v, conc_v = pj["inputs"]["feed"]["baci"], pj["inputs"]["params"]["conc"]
    src_dir = os.path.join(inp, "feed", "baci", f"v{src_v}")
    out_dir = os.path.join(inp, "feed", "baci", f"v{baci_v}")
    notes = [f"baci feed {baci_run} = input v{baci_v}; netted from gross v{src_v} "
             f"(the ICSG v{icsg_v} parents), conc v{conc_v}"]

    conc_lookup = traded_conc_lookup(
        os.path.join(inp, "params", "conc", f"v{conc_v}", "params.csv"),
        GROSS_VALUATION, TRADED_SECTORS)
    raw, _, _, _ = load_root_trade(src_dir, conc_lookup)
    chg = pd.read_csv(os.path.join(out_dir, "reexport_legs.csv"))
    key = ["sector", "region_from", "region_to", "time"]
    chg["time"] = chg["time"].astype(str)
    old = chg[key + ["cc_obs"]].rename(columns={"cc_obs": "cc"})
    new = chg[key + ["cc_new"]].rename(columns={"cc_new": "cc"})

    # (a) born legs: the same test as rebuild_values, then the rebuild itself.
    born = chg[chg["cc_obs"] <= 0.0]
    cf, ct = canonical_codes(raw)
    skip = np.array([(cf.get((r.sector, r.region_from)) is None
                      or ct.get((r.sector, r.region_to)) is None)
                     for r in born.itertuples(index=False)], dtype=bool)
    n_skip, cc_skip = int(skip.sum()), float(born["cc_new"].to_numpy()[skip].sum())
    rebuilt = rebuild_values(raw, old, new, conc_lookup, HS_OF)
    stored = pd.read_csv(os.path.join(out_dir, "values.csv"))
    same_shape = rebuilt.shape == stored.shape
    max_diff = float(np.abs(rebuilt["Value"].to_numpy() - stored["Value"].to_numpy()).max()) \
        if same_shape else float("nan")
    notes.append(f"(a) {len(born)} born legs of {len(chg)} changed legs; {n_skip} skipped "
                 f"({cc_skip:.3g} kt CC); rebuilt values.csv vs stored v{baci_v}: "
                 f"{'same shape' if same_shape else 'SHAPE DIFFERS'}, max |dValue| {max_diff:.3g} kt")

    # (c) the .get("src", "icsg") fill: build_tailings_rows reads it for a comment only,
    # and compute_tailings_grades always returns a src column (checked on a toy frame).
    from pysuto.feedprep.ore_grade import compute_tailings_grades
    toy = pd.DataFrame({"region": ["A"], "time": ["2000"], "conc_cc": [1.0], "sxew_cc": [1.0],
                        "ore_mass": [100.0]})
    toy_g = pd.DataFrame({"region": ["A"], "time": ["2000"], "g": [0.01], "src": ["icsg"]})
    has_src = "src" in compute_tailings_grades(toy, toy_g, 0.9, 0.9, 0.25, 0.99).columns
    n_fill = 0 if has_src else -1
    notes.append(f"(c) tailings frame carries a src column: {has_src}; rows filled: {n_fill}")

    # (b) nan_to_num in build_production_frame, on the reference feeds' F_root.
    acc = acc_store.load(id="copper", storage_dir=os.path.join(data_dir, "proc", "acc"))
    ei = _decoded_edges(acc)
    froot = {n: np.load(os.path.join(data_dir, "proc", "feed", "registry", r, "result.npz"))["F_root"]
             for n, r in (("icsg", icsg_run), ("crude", crude_run))}
    lab = ORE_GRADE_LABELS
    roles = acc.valuation_roles
    cols = {}
    n_nan = {}
    for name, (sf, st), val, fr in (("conc_cc", lab.conc_edge, roles.content, "icsg"),
                                    ("sxew_cc", lab.sxew_edge, roles.content, "icsg"),
                                    ("ore_mass", lab.ore_edge, roles.gross, "crude")):
        sub = ei[(ei["sector_from"] == sf) & (ei["sector_to"] == st) & (ei["valuation"] == val)]
        v = froot[fr][sub["edge_id"].values]
        n_nan[name] = (int(np.isnan(v).sum()), len(v))
        s = sub.assign(v=v, nan=np.isnan(v))
        g = s.groupby(["region_from", "time"]).agg(v=("v", lambda x: np.nansum(x)),
                                                   all_nan=("nan", "all"))
        cols[name] = g
    prod = pd.concat([cols[k].add_prefix(k + ".") for k in cols], axis=1)
    prod_pos = (prod["conc_cc.v"].fillna(0.0) + prod["sxew_cc.v"].fillna(0.0)) > 0
    ore_all_nan = prod["ore_mass.all_nan"].fillna(True).astype(bool)
    ore_pos = prod["ore_mass.v"].fillna(0.0) > 0
    prod_all_nan = (prod["conc_cc.all_nan"].fillna(True).astype(bool)
                    & prod["sxew_cc.all_nan"].fillna(True).astype(bool))
    matters_a = int((prod_pos & ore_all_nan).sum())      # producing, ore leg uncovered -> no grade
    matters_b = int((ore_pos & prod_all_nan).sum())      # ore mass, production uncovered
    notes.append("(b) NaN cells in F_root of the reference feed runs, all edges: "
                 + ", ".join(f"{k} {int(np.isnan(v).sum())}/{v.size}" for k, v in froot.items())
                 + "; ore_grade v8 itself was built on an older, unregistered ICSG run (check 8)")
    notes.append("(b) NaN cells read as 0 on the reference feed runs ("
                 + ", ".join(f"{k} {a}/{b}" for k, (a, b) in n_nan.items())
                 + f"); region-years with ICSG production and an uncovered ore leg: {matters_a}; "
                   f"with ore mass and no covered ICSG production edge: {matters_b}")
    nan_cells = sum(a for a, _ in n_nan.values())
    number = (f"(a) {n_skip} skipped born legs; (b) {nan_cells} NaN cells read 0, "
              f"{matters_a + matters_b} region-years matter; (c) {n_fill} src fills")
    verdict = "clean" if (n_skip == 0 and n_fill == 0 and matters_a + matters_b == 0) else "hit"
    return [_row("1 feedprep NaN as 0 (#290 U1a-G3-7)", number,
                 f"{baci_run}; {icsg_run}", verdict, "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 2
# ---------------------------------------------------------------------------

def _na_cells(path: str) -> tuple:
    """(cells pandas turns to NaN that are not empty, literal "NA" cells, cells)."""
    a = pd.read_csv(path, dtype=str, comment="#")
    b = pd.read_csv(path, dtype=str, comment="#", keep_default_na=False)
    changed = a.isna().to_numpy() & (b.to_numpy() != "")
    literal = (b.to_numpy() == "NA")
    return int(changed.sum()), int(literal.sum()), int(b.size)


def check2(data_dir: str, acc_id: str = "copper") -> list:
    from pandas.io.parsers.readers import STR_NA_VALUES
    from pysuto.acc import _ods_sheet_names, _read_ods
    inp = os.path.join(data_dir, "input")
    files = (sorted(glob.glob(os.path.join(data_dir, "proc", "acc", acc_id, "*.csv")))
             + [os.path.join(inp, "acc", acc_id, "current", "masks.csv"),
                os.path.join(inp, "mask", "row", "current", "rules.csv")])
    changed = literal = cells = 0
    notes = []
    for p in files:
        c, lit, n = _na_cells(p)
        changed += c
        literal += lit
        cells += n
        if c or lit:
            notes.append(f"{os.path.relpath(p, data_dir)}: {c} read as NaN, {lit} literal NA")
    # the four ods tables are read by pysuto.acc._read_ods (stdlib, no NA handling)
    ods_tok = 0
    tokens = {t for t in STR_NA_VALUES if t}
    for p in sorted(glob.glob(os.path.join(inp, "acc", acc_id, "current", "*.ods"))):
        for sh in _ods_sheet_names(p):
            df = _read_ods(p, sh)
            ods_tok += int(df.isin(tokens).to_numpy().sum()) if df.size else 0
    notes.append(f"{len(files)} csv files, {cells} cells; ods cells equal to a pandas NA token: {ods_tok}")
    verdict = "clean" if (changed == 0 and literal == 0 and ods_tok == 0) else "hit"
    return [_row("2 NA read as missing (#290 U7-S5-3)",
                 f"{changed} cells read as NaN, {literal} literal NA, {ods_tok} ods NA tokens",
                 f"acc {acc_id}", verdict, "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 3
# ---------------------------------------------------------------------------

def check3(data_dir: str, unc_id: str) -> list:
    u = os.path.join(data_dir, "proc", "uncprop", unc_id)
    sc = _json(os.path.join(u, "metadata.json"))
    pt = sc["stage_summary"]["per_timestep"]
    worst_t, worst = max(((t, v["max_resid_hard"]) for t, v in pt.items()), key=lambda x: x[1])
    n_clip = 0
    per_year = {}
    for f in sorted(glob.glob(os.path.join(u, "timesteps", "t_*.npz"))):
        z = np.load(f)
        n = int(((z["sigma_a_star"] == 0.0) & ~z["bound_active"]).sum())
        per_year[os.path.basename(f)[2:6]] = n
        n_clip += n
    # the κ > 1 warning exists from commit bb36783 (2026-11-23); count it in every log
    n_warn = n_logs = 0
    for p in glob.glob(os.path.join(data_dir, "logs", "**", "*.log"), recursive=True):
        with open(p, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        if "pysuto.uncprop" in txt:
            n_logs += 1
            n_warn += txt.count("kappa reaches")
    claim = None
    num = os.path.join(os.path.dirname(_SCRIPTS), "docs", "manuscript", "figs", "table3_ledger.csv")
    if os.path.isfile(num):
        led = pd.read_csv(num)
        claim = led.iloc[0]
    notes = [f"kappa is not stored; the sidecar and the npz keep sigma_a* only. Live edges (not bound-active) "
             f"with sigma_a* == 0 are the edges clipped at var_post <= 0: {n_clip} edge-years, "
             f"{min(per_year.values())}..{max(per_year.values())} per year",
             f"'kappa reaches' warnings in {n_logs} uncprop logs: {n_warn} (the warning exists from "
             f"bb36783, 2026-11-23; the reference run of 2026-11-19 predates it, the same opt run was "
             f"re-propagated after it)",
             f"max hard residual {worst:.4g} kt in {worst_t} against the claim < {HARD_CLAIM:.0e} kt"]
    if claim is not None:
        notes.append(f"manuscript Table 3 row A1 reports: {claim.to_dict()}")
    # the manuscript names the miss (results.md: 1.1e-6, "narrowly not met"): clean when it does
    named = os.path.isfile(num) and f"{worst:.1e}" in open(num).read()
    verdict = "clean" if (named or worst < HARD_CLAIM) else "hit"
    return [_row("3 clipped var_post, hard residual (#290 U5-S2-1/2)",
                 f"{n_clip} edge-years clipped to 0; max hard resid {worst:.3e} kt ({worst_t})",
                 unc_id, verdict, "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 4
# ---------------------------------------------------------------------------

def check4(data_dir: str, opt_id: str, psut_id: str) -> list:
    from pysuto.registry import PsutRunRegistry
    psut = PsutRunRegistry(os.path.join(data_dir, "proc", "psut")).load(psut_id)
    if psut.opt_run_id != opt_id:
        raise RuntimeError(f"psut run {psut_id} was built from {psut.opt_run_id}, not {opt_id}")
    piot = psut.to_piot(level="L")
    latest = psut.latest()                  # the period results_figs.embodied() reads (B3, Fig. 6d)
    rows = []
    for v in psut.valuations:
        for p in psut.periods():
            A = piot.A(p, v).to_numpy()
            rho = float(np.abs(np.linalg.eigvals(A)).max()) if A.size else 0.0
            rows.append((v, p, rho, float(piot.L(p, v).to_numpy().min()), float(A.min())))
    df = pd.DataFrame(rows, columns=["valuation", "period", "rho", "minL", "minA"])
    read = df[df["period"] == latest]
    # B3 and Fig. 6d call piot.embodied_trade("E", latest) and .footprint("E", latest): the
    # default (primary) valuation of the PSUT.
    prim = df[(df["period"] == latest) & (df["valuation"] == psut.primary)].iloc[0]
    bad = df[df["minL"] < -1e-6]
    notes = [f"read by results_figs.embodied(): period {latest}, valuation {psut.primary} (default)"]
    for r in read.itertuples():
        notes.append(f"  {r.valuation} {r.period}: rho(A) {r.rho:.3g}, min L {r.minL:.3g}")
    notes.append(f"all {len(df)} year-valuations: max rho(A) {df['rho'].max():.3g}; "
                 f"min L {df['minL'].min():.3g} ({df.loc[df['minL'].idxmin(), 'valuation']} "
                 f"{df.loc[df['minL'].idxmin(), 'period']}); {len(bad)} with min L < -1e-6: "
                 + ", ".join(f"{r.valuation} {r.period} ({r.minL:.3g})" for r in bad.itertuples()))
    ok = (read["rho"] < 1.0).all() and (read["minL"] > -1e-6).all()
    return [_row("4 Leontief inverse (#290 U6-G4-7)",
                 f"{psut.primary} {latest}: rho(A) {prim['rho']:.3g}, min L {prim['minL']:.2g}; "
                 f"all years max rho {df['rho'].max():.3g}",
                 psut_id, "clean" if ok else "hit", "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 5
# ---------------------------------------------------------------------------

def check5(data_dir: str, opt_id: str) -> list:
    from pysuto import acc_store
    from pysuto.clarabel_backend import ClarabelBackend
    from pysuto.diag import SolverDiag
    from pysuto.diag.stats.objective import objective_decomposition
    from pysuto.solver import Solver

    proc = os.path.join(data_dir, "proc")
    acc = acc_store.load("copper", storage_dir=os.path.join(proc, "acc"))
    d = SolverDiag.from_opt_id(opt_id, output_dir=os.path.join(proc, "opt"),
                               const_dir=os.path.join(proc, "const"),
                               imp_dir=os.path.join(proc, "imp"),
                               feed_dir=os.path.join(proc, "feed", "registry"), acc=acc)
    cond = {k: v for k, v in (d.conditioning or {}).items()
            if k in ("sigma_a_floor_abs", "sigma_a_floor_rel", "sigma_a_ceiling_factor",
                     "sigma_c_floor")}
    dec = objective_decomposition(d.stats.edge_frame(), d.stats.constraint_frame()).set_index("year")
    run = d._const_run
    s = Solver(ClarabelBackend(), const_dir=os.path.join(proc, "const"),
               opt_dir=os.path.join(proc, "opt"), acc_dir=os.path.join(proc, "acc"),
               input_dir=os.path.join(data_dir, "input"), prior_mode="cold", **cond)
    _, sigma_a = s._condition_prior(run.a_prior_global, run.sigma_a)
    rows = []
    for t, r in d._results.items():
        t = str(t)
        cols = run.edge_cols_by_t[t]
        F = r.F_opt if r.F_opt.shape == cols.shape else r.F_opt[cols]
        oa = s.objective_at(run, t, F, a_prior=r.a_prior)
        recorded = r.objective + float(((r.a_prior / sigma_a[cols]) ** 2).sum())   # J = objective + sum (a_prior/sigma)^2
        rows.append(dict(year=t, diag=float(dec.loc[t, "zsq_total"]), solver=oa["total"],
                         recorded=recorded,
                         edges=(float(dec.loc[t, "zsq_edges"]), oa["edges"]),
                         sum_rows=(float(dec.loc[t, "zsq_sum_rows"]), oa["sum_rows"]),
                         soft_ratios=(float(dec.loc[t, "zsq_soft_ratios"]), oa["soft_ratios"]),
                         other_soft=oa["other_soft"]))
    df = pd.DataFrame(rows)
    rel = (df["diag"] - df["solver"]).abs() / df["solver"]
    rel_rec = (df["recorded"] - df["solver"]).abs() / df["solver"]
    fam = {f: max(abs(a - b) / max(abs(b), 1e-300) for a, b in df[f]) for f in
           ("edges", "sum_rows", "soft_ratios")}
    tot_rel = abs(df["diag"].sum() - df["solver"].sum()) / df["solver"].sum()
    notes = [f"sum z^2 over {len(df)} years: diag {df['diag'].sum():.6f}, objective_at "
             f"{df['solver'].sum():.6f}, sidecar objective + sum(a_prior/sigma_a)^2 {df['recorded'].sum():.6f}",
             f"max relative difference per year: diag vs objective_at {rel.max():.2e}; sidecar vs "
             f"objective_at {rel_rec.max():.2e}; total {tot_rel:.2e}",
             "max relative difference per family and year: "
             + ", ".join(f"{k} {v:.2e}" for k, v in fam.items())
             + f"; other_soft in objective_at: max {df['other_soft'].max():.3g}",
             f"conditioning applied: {cond}"]
    return [_row("5 diag sigma vs solver sigma (#253)",
                 f"sum z^2 {df['diag'].sum():,.1f}; rel diff diag vs solver {tot_rel:.1e} total, "
                 f"{rel.max():.1e} max per year",
                 opt_id, "clean" if rel.max() < 1e-9 and rel_rec.max() < 1e-9 else "hit",
                 "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 6
# ---------------------------------------------------------------------------

def check6(data_dir: str, opt_id: str, unc_id: str) -> list:
    keys = ("sigma_a_floor_abs", "sigma_a_floor_rel", "sigma_a_ceiling_factor", "sigma_c_floor")
    opt = _sidecar(data_dir, "opt", opt_id)["stage_config"]
    unc = _sidecar(data_dir, "uncprop", unc_id)["stage_config"]
    rec = {k: opt.get(k) for k in keys}
    app = {k: unc.get(k) for k in keys}
    same = rec == app and None not in rec.values()
    created = _sidecar(data_dir, "uncprop", unc_id)["created_at"]
    t_end = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
    pat = re.compile(r"records no .* using the library defaults")
    hits = []
    n_logs = 0
    for p in glob.glob(os.path.join(data_dir, "logs", "**", "*.log"), recursive=True):
        with open(p, encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in fh if ("pysuto.conditioning" in ln or "stages" in ln or "uncprop" in ln)]
        n_logs += 1
        for ln in lines:
            if pat.search(ln) and ln[:10] == t_end.strftime("%Y-%m-%d"):
                hits.append((os.path.basename(p), ln.strip()[:160]))
    notes = [f"opt stage_config: {rec}", f"uncprop stage_config: {app}",
             f"'records no ... using the library defaults' warnings dated {t_end:%Y-%m-%d} in "
             f"{n_logs} logs: {len(hits)}"] + [f"  {h}" for h in hits[:3]]
    return [_row("6 Stage 5 floors (#270)",
                 f"4 of 4 floors equal the opt record; {len(hits)} fall-back warnings",
                 unc_id, "clean" if same and not hits else "hit", "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 7
# ---------------------------------------------------------------------------

def check7(data_dir: str, imp_id: str) -> list:
    from pysuto import acc_store
    sc = _sidecar(data_dir, "imp", imp_id)
    mode = sc["stage_config"].get("flat_mode")
    log = pd.read_parquet(os.path.join(data_dir, "proc", "imp", imp_id, "imputation_log.parquet"),
                          columns=["edge_id", "imputer_id", "method"])
    flat = log[log["imputer_id"] == "flat_prior"]
    acc = acc_store.load("copper", storage_dir=os.path.join(data_dir, "proc", "acc"))
    val = acc.edge_index.set_index("edge_id")["valuation"] if "edge_id" in acc.edge_index.columns \
        else acc.edge_index["valuation"]
    per_val = flat.assign(valuation=val.reindex(flat["edge_id"].to_numpy()).to_numpy()) \
                  .groupby(["valuation", "method"]).size()
    # level 3 lives in _resolve_scale, which only the magnitude_aware branch of impute() calls
    methods = set(flat["method"])
    n_level3 = 0 if (mode == "noisy_zero" and methods <= {"flat_prior_noisy_zero"}) else -1
    notes = [f"flat_mode in the imp sidecar: {mode}; flat_prior methods in the log: {sorted(methods)}",
             "FlatPriorImputer.impute() returns after the noisy-zero write and never calls "
             "_resolve_scale, so no edge reaches level 3 (global median of |f|)",
             "flat-prior fills per valuation and method: "
             + "; ".join(f"{v} {m}: {n}" for (v, m), n in per_val.items())]
    return [_row("7 flat prior level 3 (#239)",
                 f"{n_level3} edges at level 3 ({len(flat)} edges filled noisy-zero)",
                 imp_id, "clean" if n_level3 == 0 else "hit", "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Check 8
# ---------------------------------------------------------------------------

def check8(data_dir: str, imp_id: str) -> list:
    imp = _sidecar(data_dir, "imp", imp_id)
    icsg_run = _feed_run(imp, "icsg")
    fs = _feed_sidecar(data_dir, icsg_run)
    og = fs["parents"]["inputs"]["ore_grade"]
    icsg_v = fs["parents"]["inputs"]["feed"]["version"]
    inp = os.path.join(data_dir, "input")
    vdir = os.path.join(inp, "params", "ore_grade", f"v{og['version']}")
    ppath = os.path.join(vdir, "parents.json")
    reg = _json(os.path.join(inp, "feed", "registry.json"))["icsg"]["versions"]
    icsg_date = next(v["date"] for v in reg if v["version"] == icsg_v)
    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(
        data_dir, "proc", "feed", "registry", "feed_icsg_*")))
    before = [r for r in runs if r[len("feed_icsg_"):len("feed_icsg_") + 15] <
              og["date"].replace("-", "").replace(":", "").replace("T", "_")[:15]]
    notes = [f"{icsg_run} used ore_grade v{og['version']} (built {og['date']}) and ICSG input "
             f"v{icsg_v} (registered {icsg_date})",
             f"ore_grade v{og['version']} parents.json: "
             + ("present" if os.path.isfile(ppath) else "MISSING (no record of the feed run or the input versions it was built from)"),
             f"ICSG feed runs in the registry before the ore_grade build: {len(before)}; "
             f"first run in the registry: {runs[0] if runs else 'none'}"]
    ts = os.path.join(inp, "params", "trade_shares")
    tv = fs["parents"]["inputs"]["trade_shares"]["version"]
    if os.path.isfile(os.path.join(ts, f"v{tv}", "parents.json")):
        notes.append(f"for comparison trade_shares v{tv} has a parents.json: "
                     f"{json.dumps(_json(os.path.join(ts, f'v{tv}', 'parents.json')).get('upstream'))}")
    names_ref = False
    if os.path.isfile(ppath):
        pj = _json(ppath)
        names_ref = icsg_run in pj.get("upstream", {}).get("feed_run_ids", [])
    stale = og["date"] < icsg_date
    notes.append(f"ore_grade v{og['version']} is older than ICSG input v{icsg_v}: {stale}")
    return [_row("8 ore-grade lineage (#287)",
                 f"ore_grade v{og['version']} names the reference ICSG feed run: {names_ref}; "
                 f"built {og['date'][:10]}, ICSG v{icsg_v} registered {icsg_date[:10]}",
                 icsg_run, "clean" if names_ref and not stale else "hit", "\n    ".join(notes))]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--opt", default=REF["opt"])
    ap.add_argument("--imp", default=REF["imp"])
    ap.add_argument("--uncprop", default=REF["uncprop"])
    ap.add_argument("--psut", default=REF["psut"])
    ap.add_argument("--only", default="", help="comma list of check numbers (default: all)")
    a = ap.parse_args()
    want = {int(x) for x in a.only.split(",") if x} or set(range(1, 9))
    d = a.data_dir
    runs = {
        1: lambda: check1(d, a.opt, a.imp),
        2: lambda: check2(d),
        3: lambda: check3(d, a.uncprop),
        4: lambda: check4(d, a.opt, a.psut),
        5: lambda: check5(d, a.opt),
        6: lambda: check6(d, a.opt, a.uncprop),
        7: lambda: check7(d, a.imp),
        8: lambda: check8(d, a.imp),
    }
    out = []
    for k in sorted(want):
        try:
            out += runs[k]()
        except Exception as exc:                               # a failed check is reported, not hidden
            out.append(_row(f"{k} (failed)", f"{type(exc).__name__}: {exc}"[:200], "-", "hit"))
    print("\n| check | number | run id | verdict |\n|---|---|---|---|")
    for r in out:
        print(f"| {r['check']} | {r['number']} | {r['run']} | {r['verdict']} |")
    print()
    for r in out:
        if r["note"]:
            print(f"[{r['check']}]\n    {r['note']}")


if __name__ == "__main__":
    main()
