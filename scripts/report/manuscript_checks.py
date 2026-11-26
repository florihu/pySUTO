"""
manuscript_checks.py
====================
The measurements behind the ``check:`` comments of ``docs/manuscript/discussion.md`` and
the committee's Q3. Read-only: it writes nothing but the ``--json`` file.

    python scripts/report/manuscript_checks.py --opt <opt_run_id> --uncprop <uncprop_run_id>
        --uncprop-ablation <uncprop_run_id> --old-opt <opt_run_id> --json OUT

Run it from the repository root of the checkout that holds ``data/``, with
``PYTHONPATH=<worktree>/src`` when it runs from another checkout. The const and imp
runs are read through the opt and Stage-5 runs' sidecars. ``--old-opt`` needs only its
``diag/constraints_*.csv`` (its const run was pruned).

Every number has a key ``discussion:L<line>.<name>`` (line of discussion.md at the time
of writing, 2026-11-24), a label and a unit, in the layout of ``manuscript_numbers.py``,
so a refresh onto a new reference chain re-reads the file and diffs it key by key.

  L81  tail edges (beyond 3 Birge-scaled units) with an endpoint in the five regions of
       the scale-total objective, the four regions whose BACI exports exceed ICSG production, without Italy
  L83  per-year A2 share: scale totals with |z| > 3, on the old and the new chain
  L88  A3: the share of the objective on cross-region legs into the non-processing
       regions, by traded good; their gross-mass imports kept by region, and the
       cathode imports kept for the trade hubs (ISIE 2027 abstract)
  L104 Scrap→CFR CC: contraction by prior class, with and without the scale totals
       and the ICSG scrap level anchors (Stage-5 ablation), and the rows that touch it
  L110 the QQ sample's pinned, flat and zero-seed edges
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # results_figs, copper_metabolism_insights

import copper_metabolism_insights as CMI                    # noqa: E402
from results_figs import CALIB_CLASSES, merchant_hubs        # noqa: E402
from pysuto.diag import SolverDiag                           # noqa: E402
from pysuto.diag.uncprop import UncPropDiag                  # noqa: E402

# =============================================================================
# USER CONFIGURATION
# =============================================================================
N_OBJECTIVE_REGIONS = 5            # regions that carry 41.4 % of the scale-total Σz² (§5.2)
ISSUE_201_REGION    = "Italy"      # the one of them with no BACI-over-ICSG contradiction
Z_TAIL              = 3.0          # a tail edge: |z/√κ| beyond Z_TAIL·√R
FRINGE_KT           = 1e-4         # an edge whose a* is this close to 0 is near-pinned
FLAT_Z              = 1e-9         # |z| below this: a* equals the prior
A2_SUMS             = 0.20         # the A2 threshold (results_figs THRESH["A2_sums"])
CFR_PAIR            = ("Scrap", "CFR", "CC")
HUB_LOW_RET         = 0.8          # L88: a year whose import retention lies below this
TRADE_HUBS          = ("Switzerland", "Netherlands")   # L88: the trade hubs of the ISIE abstract
# =============================================================================


def _add(out: dict, key: str, label: str, value, unit: str = "") -> None:
    if key in out:
        raise KeyError(f"duplicate key {key}")
    out[key] = {"label": label, "value": value, "unit": unit}


def _sidecar(dir_: Path) -> dict:
    with open(dir_ / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _check_uncprop(proc: Path, uncprop: str, opt: str) -> None:
    up = _sidecar(proc / "uncprop" / uncprop)["parents"]["upstream"]
    if up.get("opt_run_id") != opt:
        raise ValueError(f"{uncprop} propagates {up.get('opt_run_id')}, not {opt}")


# =============================================================================
# L83  A2 per year
# =============================================================================

def a2_by_year(proc: Path, opt: str) -> pd.Series:
    """Share of the scale totals with |z| > 3 per year, from the run's ``diag`` CSVs."""
    files = sorted((proc / "opt" / opt / "diag").glob("constraints_*.csv*"))
    if not files:
        raise FileNotFoundError(f"{opt} has no diag/constraints_<year>.csv")
    out = {}
    for f in files:
        year = f.name.split("_")[1].split(".")[0]
        c = pd.read_csv(f, usecols=["type", "z"])
        z = c.z[(c.type == "sum")].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        out[year] = float((np.abs(z) > 3.0).mean())
    return pd.Series(out).sort_index()


def a2_numbers(out: dict, proc: Path, opt: str, old: str) -> None:
    sh = {"old": a2_by_year(proc, old), "new": a2_by_year(proc, opt)}
    for k, run in (("old", old), ("new", opt)):
        s = sh[k]
        yr = s.index.astype(int)
        _add(out, f"discussion:L83.a2_share_by_year_{k}",
             f"share of scale totals with |z| > 3 by year, {run}", s.round(4).to_dict())
        _add(out, f"discussion:L83.a2_mean_to_2018_{k}", f"mean 1995–2018, {run}",
             float(s[yr <= 2018].mean()))
        _add(out, f"discussion:L83.a2_mean_from_2019_{k}", f"mean 2019–2024, {run}",
             float(s[yr >= 2019].mean()))
        _add(out, f"discussion:L83.a2_range_{k}", f"smallest and largest yearly share, {run}",
             [float(s.min()), float(s.max())])
        _add(out, f"discussion:L83.a2_years_above_{k}", f"years with a share above "
             f"{A2_SUMS} (A2 needs the share above it), {run}", int((s > A2_SUMS).sum()))
    _add(out, "discussion:L83.a2_years_to_2018_old_above",
         "years 1995–2018 above the threshold, old chain",
         [int(y) for y in sh["old"].index[(sh["old"] > A2_SUMS).to_numpy()] if int(y) <= 2018])


# =============================================================================
# L81  tail edges in the objective regions
# =============================================================================

def objective_regions(cf: pd.DataFrame) -> tuple:
    """The ``N_OBJECTIVE_REGIONS`` regions with the largest share of the scale-total Σz²,
    their summed share, and the same without ``ISSUE_201_REGION``."""
    x = cf[cf.type == "sum"]
    z2 = (np.nan_to_num(x.z.to_numpy(dtype=float)) ** 2)
    by = pd.Series(z2, index=x.region.astype(str).to_numpy()).groupby(level=0).sum()
    sh = (by / by.sum()).sort_values(ascending=False)
    five = list(sh.index[:N_OBJECTIVE_REGIONS])
    return five, sh, [r for r in five if r != ISSUE_201_REGION]


def tail_numbers(out: dict, f: pd.DataFrame, R: float, five: list, sh: pd.Series,
                 four: list) -> None:
    _add(out, "discussion:L81.objective_regions", "regions with the largest shares of the "
         "scale-total Σz² and their shares", {r: float(sh[r]) for r in five})
    _add(out, "discussion:L81.objective_regions_share", "their summed share",
         float(sh[five].sum()))
    c = f[f.calibrated.to_numpy(dtype=bool) & (f.valuation == "CC")
          & (f.prior_class == CALIB_CLASSES[0])]
    tail = np.abs(c.std_shift.to_numpy(dtype=float)) > Z_TAIL * np.sqrt(R)
    _add(out, "discussion:L81.tail_edge_years", "feed-observed CC edge-years in the QQ "
         "sample, and those beyond 3 Birge-scaled units", [int(len(c)), int(tail.sum())])
    _add(out, "discussion:L81.tail_share", "their share (§4.2)", float(tail.mean()))
    fr, to = c.from_region.astype(str), c.to_region.astype(str)
    sets = {"five": five, "four": four, "italy": [ISSUE_201_REGION]}
    for name, regs in sets.items():
        for end, m in (("either", fr.isin(regs) | to.isin(regs)), ("origin", fr.isin(regs)),
                       ("destination", to.isin(regs))):
            m = m.to_numpy()
            _add(out, f"discussion:L81.tail_in_{name}_{end}",
                 f"tail edge-years with the {end} in the {name} regions {regs}: count, share "
                 "of the tail, share of all sample edge-years, ratio of the two",
                 [int((tail & m).sum()), float((tail & m).sum() / tail.sum()),
                  float(m.mean()), float((tail & m).sum() / tail.sum() / m.mean())])
    by = c[tail].assign(r=fr[tail]).groupby("r").size().sort_values(ascending=False)
    _add(out, "discussion:L81.tail_by_origin_top", "tail edge-years by origin region, top 8",
         by.head(8).to_dict())


# =============================================================================
# L88  A3 by traded good
# =============================================================================

def a3_numbers(out: dict, ef: pd.DataFrame) -> None:
    hubs = merchant_hubs(CMI.load_icsg(), sorted(ef.to_region.unique()))
    z2 = np.nan_to_num(ef.z.to_numpy(dtype=float)) ** 2
    ef = ef.assign(z2=z2, cross=(ef.from_region != ef.to_region).to_numpy())
    hub = ef.cross & ef.to_region.isin(hubs)
    tot = z2.sum()
    _add(out, "discussion:L88.hub_regions", "the non-processing regions (no ICSG smelter, "
         "refinery or SX-EW row in any year)", hubs)
    _add(out, "discussion:L88.hub_share", "share of the edge Σz² on cross-region legs into "
         "them (A3's 70%)", float(ef.z2[hub].sum() / tot))
    good = {"Concentrate": "concentrate", "Matte": "matte", "Anode": "anode",
            "Cathode": "cathode"}
    g = ef[hub].assign(good=ef.from_sector[hub].map(good).fillna("other"))
    rows = {}
    for (gd, val), x in g.groupby(["good", "valuation"]):
        rows.setdefault(gd, {})[val] = float(x.z2.sum() / tot)
    _add(out, "discussion:L88.hub_share_by_good", "the same, by the leg's origin sector "
         "(the traded good) and valuation, as a share of the whole edge Σz²", rows)
    by = {gd: float(sum(v.values())) for gd, v in rows.items()}
    _add(out, "discussion:L88.hub_share_by_good_total", "by good, both valuations", by)
    non_cath = sum(v for k, v in by.items() if k in ("concentrate", "matte", "anode"))
    _add(out, "discussion:L88.hub_share_conc_matte_anode_of_hub",
         "concentrate, matte and anode together, as a share of the hub legs' Σz²",
         float(non_cath / ef.z2[hub].sum() * tot))
    _add(out, "discussion:L88.hub_share_cathode_of_hub", "cathode, as a share of the hub "
         "legs' Σz²", float(by.get("cathode", 0.0) / ef.z2[hub].sum() * tot))
    top = (g.groupby(["from_sector", "to_sector", "valuation"]).z2.sum() / tot
           ).sort_values(ascending=False).head(6)
    _add(out, "discussion:L88.hub_top_pairs", "the six largest sector pairs into the hubs, "
         "share of the whole edge Σz²", {" ".join(k): float(v) for k, v in top.items()})
    cc = ef[hub & (ef.valuation == "CC")]
    d = cc.assign(good=cc.from_sector.map(good).fillna("other")).groupby("good")
    _add(out, "discussion:L88.hub_cc_prior_post_kt", "CC on the hub legs by good, 30 years: "
         "prior, posterior, retention", {k: [float(x.a_prior.sum()), float(x.a_post.sum()),
                                             float(x.a_post.sum() / x.a_prior.sum())]
                                         for k, x in d if x.a_prior.sum() > 0}, "kt CC")
    ms  = ef[hub & (ef.valuation == "Mass")]
    yr  = ms.groupby(["to_region", "year"])[["a_prior", "a_post"]].sum()
    low = (yr.a_post < HUB_LOW_RET * yr.a_prior).groupby(level="to_region").sum()
    tot_r = ms.groupby("to_region")[["a_prior", "a_post"]].sum()
    _add(out, "discussion:L88.hub_mass_imports_by_region", "gross-mass imports on the "
         "cross-region legs into each non-processing region, all goods, 30 years: prior, "
         f"posterior, retention, years with retention below {HUB_LOW_RET}",
         {r: [float(x.a_prior), float(x.a_post), float(x.a_post / x.a_prior), int(low[r])]
          for r, x in tot_r.iterrows() if x.a_prior > 0}, "kt")
    cath = ms[(ms.from_sector == "Cathode") & ms.to_region.isin(TRADE_HUBS)]
    _add(out, "discussion:L88.hub_cathode_mass_trade_hubs", "gross-mass cathode imports "
         f"into {' and '.join(TRADE_HUBS)}, 30 years: prior, posterior, retention",
         [float(cath.a_prior.sum()), float(cath.a_post.sum()),
          float(cath.a_post.sum() / cath.a_prior.sum())], "kt")


# =============================================================================
# L104  Scrap→CFR contraction
# =============================================================================

def _pair(f: pd.DataFrame) -> pd.DataFrame:
    a, b, v = CFR_PAIR
    return f[(f.from_sector == a) & (f.to_sector == b) & (f.valuation == v)]


def contraction_numbers(out: dict, d: UncPropDiag, fa: pd.DataFrame) -> None:
    f = d.stats.edge_frame()
    for tag, fr in (("default", f), ("ablation", fa)):
        g = _pair(fr)
        rows = {}
        for cls, x in g.groupby("prior_class", observed=True):
            rows[cls] = {"edge_years": int(len(x)), "weight_kt": float(x.weight.sum()),
                         "weight_share": float(x.weight.sum() / g.weight.sum()),
                         "kappa_weighted": float((x.kappa * x.weight).sum() / x.weight.sum()),
                         "kappa_median": float(x.kappa.median())}
        rows["all"] = {"edge_years": int(len(g)), "weight_kt": float(g.weight.sum()),
                       "weight_share": 1.0,
                       "kappa_weighted": float((g.kappa * g.weight).sum() / g.weight.sum()),
                       "kappa_median": float(g.kappa.median())}
        _add(out, f"discussion:L104.scrap_cfr_kappa_{tag}", f"Scrap→CFR CC contraction by "
             f"prior class, Stage-5 run {'with' if tag == 'default' else 'without'} the scale "
             "totals and the ICSG scrap level anchors", rows)
    cr, ids = d._cr, _pair(f).edge_id.unique()
    G = cr.G.tocsc()[:, ids].tocoo()
    rm = cr.row_meta
    rows = pd.DataFrame({"type": rm.type.astype(str).to_numpy()[G.row],
                         "variant": rm.variant.astype(str).to_numpy()[G.row],
                         "hard": rm.is_hard.to_numpy(dtype=bool)[G.row],
                         "row": G.row}).drop_duplicates("row")
    rows["variant"] = np.where(rows.variant.str.startswith("scrapshare"), "scrapshare*",
                               rows.variant)
    cnt = rows.groupby(["type", "variant", "hard"]).size().sort_values(ascending=False)
    _add(out, "discussion:L104.scrap_cfr_rows_touching", "constraint rows (30 years) with a "
         "nonzero entry on a Scrap→CFR CC edge, by type, variant and hardness",
         {" ".join(str(i) for i in k): int(v) for k, v in cnt.items()})


# =============================================================================
# L110  the QQ sample's pinned and flat edges
# =============================================================================

def qq_numbers(out: dict, f: pd.DataFrame) -> None:
    c = f[f.calibrated.to_numpy(dtype=bool) & (f.valuation == "CC")]
    c = c[c.prior_class.isin(CALIB_CLASSES)]
    core_cut = float(0.6744897501960817)                    # median of |N(0, 1)|
    res = {}
    for cls, x in c.groupby("prior_class", observed=True):
        s = np.abs(x.std_shift.to_numpy(dtype=float))
        core = s < core_cut
        marks = {"a_star_equals_prior": (np.abs(x.z.to_numpy(dtype=float)) < FLAT_Z),
                 "a_star_near_zero": (np.abs(x.a_post.to_numpy(dtype=float)) <= FRINGE_KT),
                 "zero_prior": (x.a_prior.to_numpy(dtype=float) == 0.0)}
        marks["pinned_or_flat"] = marks["a_star_equals_prior"] | marks["a_star_near_zero"]
        r = {"n": int(len(x)), "median_abs": float(np.median(s)),
             "share_in_core": float(core.mean())}
        for k, m in marks.items():
            r[k] = {"share_of_sample": float(m.mean()),
                    "share_of_core": float((m & core).sum() / core.sum()),
                    "median_abs_without": float(np.median(s[~m]))}
        res[cls] = r
    _add(out, "discussion:L110.qq_pinned_flat", "Figure 5b's sample (CC, calibrated: not "
         "bound-active, κ ≥ 0.01, classes from a feed / imputed), per prior class: size, "
         "median |z/√κ|, and for each kind of edge (a* equals the prior, a* within 1e-4 kt "
         "of 0, prior 0) its share of the sample, its share of the core (|z/√κ| below the "
         "N(0, 1) median 0.674) and the class median without those edges", res)


# =============================================================================

def main(opt: str, uncprop: str, ablation: str, old_opt: str, out_json: Path) -> None:
    proc = Path("data/proc")
    _check_uncprop(proc, uncprop, opt)
    _check_uncprop(proc, ablation, opt)
    abl_meta = _sidecar(proc / "uncprop" / ablation).get("stage_config", {})
    if not abl_meta.get("exclude_scale_totals", False):
        raise ValueError(f"{ablation} is not an exclude_scale_totals run")
    out: dict = {}
    sd = SolverDiag.from_opt_id(opt)
    ef, cf = sd.stats.edge_frame(), sd.stats.constraint_frame()
    d = UncPropDiag.from_uncprop_id(uncprop)
    f, R = d.stats.edge_frame(), d.stats.birge_ratio()
    fa = UncPropDiag.from_uncprop_id(ablation).stats.edge_frame()

    _add(out, "checks.runs", "runs read", {"opt": opt, "uncprop": uncprop,
                                           "uncprop_ablation": ablation, "old_opt": old_opt})
    _add(out, "checks.birge_ratio", "the Stage-5 run's Birge ratio R", float(R))
    five, sh, four = objective_regions(cf)
    tail_numbers(out, f, R, five, sh, four)
    a2_numbers(out, proc, opt, old_opt)
    a3_numbers(out, ef)
    contraction_numbers(out, d, fa)
    qq_numbers(out, f)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"{len(out)} keys → {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--opt", required=True)
    ap.add_argument("--uncprop", required=True, help="the default Stage-5 run of --opt")
    ap.add_argument("--uncprop-ablation", required=True,
                    help="the exclude_scale_totals Stage-5 run of --opt")
    ap.add_argument("--old-opt", default="opt_20260930_092111_e3e0677d",
                    help="the chain A2's expectation was written on (diag CSVs only)")
    ap.add_argument("--json", required=True, type=Path)
    a = ap.parse_args()
    main(a.opt, a.uncprop, a.uncprop_ablation, a.old_opt, a.json)
