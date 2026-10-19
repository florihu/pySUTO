"""
results_figs.py
===============
Manuscript Results (Section 4) from one reference run: the three key-tile figures
(Fig. 3 the problem, Fig. 4 reconciliation, Fig. 5 copper system), the
pre-registered hypothesis ledger (Table 3) and every number the prose cites, so
that a later run swaps in by re-running this script.

    python scripts/report/results_figs.py <opt_run_id> --psut-run-id <psut_run_id>
        [--tol-pair <opt_old> <opt_new>] [--out docs/manuscript/figs]

Reads persisted run artifacts and the registered feeds only; nothing upstream is
modified. Writes ``fig3_problem.png``, ``fig4_reconciliation.png``,
``fig5_copper_system.png``, ``table3_ledger.csv``, ``numbers.json`` and the SI
tables (``si_*.csv`` and one ``si_tables.md`` with the same tables rendered) to
``--out``. The thresholds of the ledger are frozen in the USER CONFIGURATION
block (SI S15) and reported whichever way they fall.

Fig. 3 is the problem one Stage-4 period is handed (``PROBLEM_YEAR``): edge
provenance, the chain profile, prior breach and active set per constraint family,
and ``PROBLEM_REGION``'s slice of the QP in standard form, all from
``ProblemStatsDiagnostic`` (the opt run's lineage to its const and imp runs).
Part A (RQ-A, compatibility) draws from the Stage-4 frames of ``SolverDiag``
(``objective_decomposition``, ``family_frame``, ``share_by_key``); retention and
every Part B share come from the per-edge prior / posterior of the run
(``copper_metabolism_insights.load_run``); the embodied-extraction matrix from
the persisted PSUT run through the market-share PIOT at level L.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.colors import LogNorm                 # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Patch, Rectangle       # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
sys.path.insert(0, "scripts/report")

import copper_metabolism_insights as CMI                              # noqa: E402
from pysuto.diag.stats import ProblemStatsDiagnostic                 # noqa: E402
from pysuto.diag.stats.objective import (family_frame, objective_decomposition,  # noqa: E402
                                         rank_identities, share_by_key)
from pysuto.diag.stats.problem import NO_PRIOR_OF_ITS_OWN, PROVENANCE_ORDER  # noqa: E402
from pysuto.registry.psut import PsutRunRegistry                      # noqa: E402

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
OUT_DIR  = Path("docs/manuscript/figs")
YEARS    = list(range(1995, 2025))
POOL     = list(range(2020, 2025))          # the "recent" pool used for shares
ROCKET_YEAR = 2020
PROBLEM_YEAR   = "2020"                     # Fig. 3: the period whose QP is shown
PROBLEM_REGION = "China"                    # Fig. 3e: the region sliced out of it

# Retention stages (from-sector, to-sector, valuation): the categories of
# Section 3.3 / SI Table S16. The first seven are drawn in Fig. 4c.
STAGES = [("CO", "Mining", "Mass"), ("MF", "Concentrate", "CC"),
          ("Concentrate", "Smelting", "CC"), ("Anode", "ER", "CC"),
          ("CP", "Cathode", "CC"), ("Cathode", "FU", "CC"), ("Scrap", "Smelting", "CC"),
          ("Smelting", "Matte", "CC"), ("Scrap", "CFR", "CC")]
A4_STAGES = {"cathode": ("CP", "Cathode", "CC"), "concentrate": ("MF", "Concentrate", "CC"),
             "ore": ("CO", "Mining", "Mass")}

# Concentration steps: one commodity per step, producer (or consumer) region.
HHI_STEPS = {"mining":   ("Mining", "Ore", "region_from"),
             "smelting": ("Smelting", "Matte", "region_from"),
             "refining": ("CP", "Cathode", "region_from"),
             "use":      ("Cathode", "FU", "region_to")}
HHI_BANDS = (0.15, 0.25)                    # unconcentrated < 0.15 < moderate < 0.25 < high

TRADED        = ["Concentrate", "Matte", "Anode", "Cathode"]
EXPORT_GROUPS = ["China", "JP/KR/TW", "EU-14", "United States", "India", "Rest"]
TRADE_PANELS  = [("Chile", "export"), ("Peru", "export"), ("China", "import")]
EMBODIED_TOP  = 6

# Pre-registered thresholds (SI S15). Reported whichever way they fall.
THRESH = {
    "A1_max_resid":  1e-6,     # kt, every year
    "A2_edges":      0.05,     # share |z|>3, edges, every year (below)
    "A2_sums":       0.20,     # share |z|>3, scale totals, every year (above)
    "A3_hub":        0.50,     # share of edge Σz² on legs into merchant hubs
    "A4_step":       0.02,     # retention gap per step
    "A4_years":      25,       # of 30
    "A5_delta":      0.02,     # |Δ retention| per stage, tolerance pair
    "B1_fidelity":   0.02,     # |world posterior − ICSG-implied|, pooled
    "B1_eu_factor":  2.0,      # EU-14 ≥ factor × world
    "B2_high":       HHI_BANDS[1],
    "B2_low":        HHI_BANDS[0],
    "B2_invariance": 0.01,     # |HHI_post − HHI_prior| every step, 2024
    "B3_ratio":      1.10,     # embodied / direct, China ← Chile+Peru, 2024
    "B3_chile_1995": 0.10,     # Chile export share to China, below in 1995
    "B3_chile_2024": 0.50,     # ... above in 2024
}

# dataviz reference palette (validated; fixed order, never cycled)
PALETTE = list(CMI.PALETTE[:7])
GREY, INK, INK2, GRID, SURFACE = CMI.GREY, CMI.INK, CMI.INK2, CMI.GRID, CMI.SURFACE
SEQ_CMAP = "Blues"
FAMILY_COLOR = {"edges": PALETTE[0], "sum_rows": PALETTE[1], "soft_ratios": PALETTE[2]}
FAMILY_LABEL = {"edges": "edge defects", "sum_rows": "scale-total residuals",
                "soft_ratios": "soft-ratio residuals"}
# Fig. 3: provenance in palette order; the categories that give an edge no
# prior of its own (0 kt) share the neutral grey, any other imputer is grey too.
PROV_COLOR = {"feed": PALETTE[0], "network": PALETTE[1], "trade": PALETTE[2], "conc": PALETTE[3],
              "ore_grade": PALETTE[4], "time_interp": PALETTE[5],
              **{c: GREY for c in NO_PRIOR_OF_ITS_OWN}}
PROV_LABEL = {"feed": "from a feed (ICSG)", "network": "network", "trade": "trade seed (BACI × grade)",
              "conc": "concentration", "ore_grade": "ore grade", "eff": "efficiency",
              "time_interp": "time interpolation", "net_stock_share": "net-stock share",
              "flat_prior": "flat prior"}
CONE_COLOR = {"zero": INK, "nonneg": INK2, "free": GREY}
CONE_LABEL = {"zero": "hard equality (zero cone)", "nonneg": "hard inequality (nonneg cone)",
              "free": "soft (free cone)"}
ACTIVE_COLOR = PALETTE[6]                   # binding on flow, in (d) and the ticks of (e)
SLICE_LABEL = {"mass balance": "mass balances", "conc band": "conc bands", "eff band": "eff",
               "scrap-share band": "scrap share", "lower bound": "lower bounds",
               "soft ratio": "soft ratios", "scale total & scrap anchor": "scale totals"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlesize": 8,
                     "legend.fontsize": 6.5, "axes.labelsize": 7.5})
# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

PLOT_STAGES = STAGES[:7]
STAGE_LABEL = {s: f"{s[0]}→{s[1]} {s[2]}" for s in STAGES}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_frames(opt: str):
    """Stage-4 all-years frames (edges, constraint rows) of one opt run."""
    from pysuto.diag import SolverDiag                      # heavy: only when needed
    d = SolverDiag.from_opt_id(opt)
    return d.stats.edge_frame(), d.stats.constraint_frame()


def merchant_hubs(icsg: pd.DataFrame, regions) -> list:
    """Model regions with no ICSG smelter, refinery or SX-EW production in any
    year — trade into them is transit or merchanting, not processing."""
    prod = icsg[icsg.Process.isin(["Smelting", "Refining", "SXEW"])].groupby("reg")["Value"].sum()
    producing = set(prod[prod > 0].index)
    return sorted(r for r in regions if r not in producing)


# ---------------------------------------------------------------------------
# Part A — compatibility, localisation, magnitude
# ---------------------------------------------------------------------------

def hub_share(fe: pd.DataFrame, hubs) -> dict:
    """Share of the edge family's Σz² on cross-region legs, and on those whose
    destination is a merchant hub."""
    z2 = np.nan_to_num(fe["z"].to_numpy(dtype=float)) ** 2
    cross = fe["from_region"].astype(str).to_numpy() != fe["to_region"].astype(str).to_numpy()
    hub = cross & fe["to_region"].astype(str).isin(hubs).to_numpy()
    return {"cross_region": float(z2[cross].sum() / z2.sum()),
            "hub": float(z2[hub].sum() / z2.sum())}


def retention_by_stage(edges: pd.DataFrame) -> pd.DataFrame:
    """Σpost/Σprior per stage and year, plus the 30-year pool (``year = 'all'``)."""
    rows = []
    for a, b, v in STAGES:
        m = (edges.sector_from == a) & (edges.sector_to == b) & (edges.valuation == v)
        g = edges[m].groupby("year")[["prior", "post"]].sum().reindex(YEARS)
        for y, r in g.iterrows():
            rows.append(dict(stage=STAGE_LABEL[(a, b, v)], year=str(y), prior=r.prior, post=r.post))
        rows.append(dict(stage=STAGE_LABEL[(a, b, v)], year="all", prior=g.prior.sum(), post=g.post.sum()))
    out = pd.DataFrame(rows)
    out["retention"] = out.post / out.prior
    return out


def a4_gradient(edges: pd.DataFrame) -> pd.DataFrame:
    r = retention_by_stage(edges)
    r = r[r.year != "all"].pivot(index="year", columns="stage", values="retention")
    g = pd.DataFrame({k: r[STAGE_LABEL[s]] for k, s in A4_STAGES.items()})
    g["cathode_minus_concentrate"] = g.cathode - g.concentrate
    g["concentrate_minus_ore"] = g.concentrate - g.ore
    g["both"] = (g.cathode_minus_concentrate >= THRESH["A4_step"]) & (g.concentrate_minus_ore >= THRESH["A4_step"])
    return g


def chile_export_retention(edges: pd.DataFrame, years=(2010, 2019)) -> float:
    m = ((edges.sector_from == "Concentrate") & (edges.sector_to == "Smelting") & (edges.valuation == "CC")
         & (edges.region_from == "Chile") & (edges.region_to != "Chile") & edges.year.between(*years))
    return float(edges[m].post.sum() / edges[m].prior.sum())


# ---------------------------------------------------------------------------
# Part B — secondary share, concentration, trade
# ---------------------------------------------------------------------------

def icsg_input_share(icsg: pd.DataFrame) -> pd.Series:
    """ICSG-implied secondary share on the model's input basis: scrap into
    smelters and refineries over scrap + concentrate intake + SX-EW cathode,
    all from the feed's own rows (kt Cu, input divisors applied)."""
    def ser(process, typ):
        x = icsg[(icsg.Process == process) & (icsg.Type == typ)]
        return x.groupby("Year")["Value"].sum().reindex(YEARS).fillna(0.0)
    sec = ser("Smelting", "Scrap") + ser("Refining", "Scrap")
    prim = ser("Smelting", "Concentrate") + ser("SXEW", "Cathode")
    return sec / (sec + prim)


def icsg_published_share(icsg: pd.DataFrame, eff: dict) -> pd.Series:
    """ICSG's published secondary share of refined output (secondary refined
    over total refined incl. SX-EW), reconstructed from the nested feed rows."""
    w = CMI.icsg_world(icsg, eff)
    k = eff["cfr"] * eff["er"]
    return (w.scrap_in_smelting * eff["sm"] * k + w.scrap_in_refining * k) / (w.total_refined + w.sxew)


def secondary_shares(cc: pd.DataFrame) -> tuple:
    """Input-side secondary share per group and year (prior, post), the world
    series, and the pooled 2020–24 values."""
    inp = cc[(cc.sector_from == "Scrap")
             | ((cc.sector_from == "Concentrate") & (cc.sector_to == "Smelting"))
             | ((cc.sector_from == "SXEW") & (cc.sector_to == "CSXEW"))].copy()
    inp["group"] = inp.region_to.map(CMI.group_of)
    inp["sec"] = inp.sector_from == "Scrap"

    def share(g):
        a = g.groupby("sec")[["prior", "post"]].sum()
        return pd.Series({"prior": a.loc[True, "prior"] / a["prior"].sum(),
                          "post":  a.loc[True, "post"] / a["post"].sum()})
    by_group = inp.groupby(["group", "year"])[["sec", "prior", "post"]].apply(share).reset_index()
    world = inp.groupby("year")[["sec", "prior", "post"]].apply(share).reindex(YEARS)
    pool = inp[inp.year.isin(POOL)]
    pooled = pool.groupby("group")[["sec", "prior", "post"]].apply(share)
    pooled.loc["World"] = share(pool)
    return by_group, world, pooled


def hhi_by_step(cc: pd.DataFrame) -> pd.DataFrame:
    """HHI per step, year and prior/post with the ROW band: ``hhi_upper`` counts
    ROW as one region, ``hhi_lower`` lets it contribute nothing."""
    rows = []
    for step, (a, b, col) in HHI_STEPS.items():
        x = cc[(cc.sector_from == a) & (cc.sector_to == b)]
        g = x.groupby(["year", col])[["prior", "post"]].sum()
        for y in YEARS:
            s = g.loc[y]
            for v in ("prior", "post"):
                sh = s[v] / s[v].sum()
                top = sh.drop("ROW", errors="ignore").idxmax()
                rows.append(dict(step=step, year=str(y), val=v,
                                 hhi_upper=float((sh ** 2).sum()),
                                 hhi_lower=float((sh.drop("ROW", errors="ignore") ** 2).sum()),
                                 row_share=float(sh.get("ROW", 0.0)),
                                 top_region=top, top_share=float(sh[top])))
    return pd.DataFrame(rows)


def partner_shares(cc: pd.DataFrame, country: str, direction: str, partner_of) -> pd.DataFrame:
    """Posterior partner shares per year of a country's cross-region legs in the
    traded commodities (``export``: destinations; ``import``: origins)."""
    x = cc[cc.sector_from.isin(TRADED) & (cc.region_from != cc.region_to)]
    if direction == "export":
        x = x[x.region_from == country].copy(); x["partner"] = x.region_to.map(partner_of)
    else:
        x = x[x.region_to == country].copy(); x["partner"] = x.region_from.map(partner_of)
    g = x.groupby(["year", "partner"])["post"].sum().unstack("partner").reindex(YEARS).fillna(0.0)
    tot = g.sum(axis=1)
    out = g.div(tot, axis=0)
    out.attrs["total_kt"] = tot
    return out


def chile_share_to_china(cc: pd.DataFrame, year: int) -> float:
    x = cc[cc.sector_from.isin(TRADED) & (cc.region_from == "Chile") & (cc.region_to != "Chile") & (cc.year == year)]
    s = x.groupby("region_to")["post"].sum()
    return float(s.get("China", 0.0) / s.sum())


def import_origins(cc: pd.DataFrame, country: str, n: int = 5) -> list:
    x = cc[cc.sector_from.isin(TRADED) & (cc.region_to == country) & (cc.region_from != country) & cc.year.isin(POOL)]
    s = x.groupby("region_from")["post"].sum().drop("ROW", errors="ignore").sort_values(ascending=False)
    return s.head(n).index.tolist()


def embodied(psut, period: str) -> tuple:
    """Embodied-extraction matrix (mining region × consuming region, kt CC) and
    the top-``EMBODIED_TOP`` + Rest fold of it."""
    piot = psut.to_piot(level="L")
    M = piot.embodied_trade("E", period)
    fp = piot.footprint("E", period, by="region")
    if not np.allclose(M.sum(0).reindex(fp.index).to_numpy(), fp.to_numpy(), rtol=1e-8):
        raise RuntimeError("embodied_trade column sums differ from the regional footprint")
    rows = M.sum(1).drop("ROW", errors="ignore").sort_values(ascending=False).head(EMBODIED_TOP).index.tolist()
    cols = M.sum(0).drop("ROW", errors="ignore").sort_values(ascending=False).head(EMBODIED_TOP).index.tolist()
    F = M.reindex(index=rows + [r for r in M.index if r not in rows],
                  columns=cols + [c for c in M.columns if c not in cols])
    folded = F.iloc[:len(rows), :len(cols)].copy()
    folded["Rest"] = F.iloc[:len(rows), len(cols):].sum(1)
    rest_row = pd.concat([F.iloc[len(rows):, :len(cols)].sum(0), pd.Series({"Rest": F.iloc[len(rows):, len(cols):].to_numpy().sum()})])
    folded.loc["Rest"] = rest_row
    return M, folded


def direct_conc_imports(cc: pd.DataFrame, to: str, origins, year: int) -> float:
    m = ((cc.sector_from == "Concentrate") & (cc.sector_to == "Smelting") & (cc.year == year)
         & (cc.region_to == to) & cc.region_from.isin(origins))
    return float(cc[m].post.sum())


def leg_table(cc: pd.DataFrame, countries, years=(2000, 2024), top: int = 5) -> pd.DataFrame:
    """SI S17: the largest cross-region legs (by prior) of each country per
    commodity and year, prior / posterior / retention."""
    rows = []
    x = cc[cc.sector_from.isin(TRADED) & (cc.region_from != cc.region_to) & cc.year.isin(years)]
    for c in countries:
        sub = x[(x.region_from == c) | (x.region_to == c)]
        g = sub.groupby(["year", "sector_from", "region_from", "region_to"])[["prior", "post"]].sum().reset_index()
        g = g[g.prior >= 1.0].sort_values(["year", "sector_from", "prior"], ascending=[True, True, False])
        g = g.groupby(["year", "sector_from"]).head(top)
        g.insert(0, "country", c)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["retention"] = out.post / out.prior
    return out


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def build_ledger(A: dict, B: dict) -> pd.DataFrame:
    T = THRESH
    rows = []

    def add(hid, expectation, metric, threshold, value, ok):
        rows.append(dict(id=hid, expectation=expectation, metric=metric, threshold=threshold,
                         value=value, outcome="PASS" if ok else "FAIL"))

    dec = A["dec"]
    v = float(dec.max_abs_resid_massbal.max())
    add("A1", "every hard identity closes", "max |G·a* − c| over hard mass balances, all years",
        f"< {T['A1_max_resid']:g} kt", f"{v:.1e} kt", v < T["A1_max_resid"])

    e_ok = int((dec.share_gt3_edges < T["A2_edges"]).sum()); s_ok = int((dec.share_gt3_sum_rows > T["A2_sums"]).sum())
    add("A2", "compatibility is family-specific",
        "share of rows with |z| > 3 per year: edges below, scale totals above",
        f"edges < {T['A2_edges']:.2f} and scale totals > {T['A2_sums']:.2f}, every year",
        f"edges {dec.share_gt3_edges.min():.3f}–{dec.share_gt3_edges.max():.3f} ({e_ok}/30 yr); "
        f"scale totals {dec.share_gt3_sum_rows.min():.3f}–{dec.share_gt3_sum_rows.max():.3f} ({s_ok}/30 yr)",
        e_ok == 30 and s_ok == 30)

    add("A3", "the adjustment localises on transit trade",
        "share of edge Σz² on cross-region legs into merchant hubs (no ICSG processing)",
        f"≥ {T['A3_hub']:.2f}", f"{A['hub']['hub']:.3f} (all cross-region legs {A['hub']['cross_region']:.3f})",
        A["hub"]["hub"] >= T["A3_hub"])

    g = A["a4"]; n_both = int(g.both.sum())
    add("A4", "the adjustment grows upstream",
        "retention Cathode CC > Concentrate CC > Ore Mass, gap per step, years passing",
        f"≥ {T['A4_step']:.2f} per step in ≥ {T['A4_years']} of 30 years",
        f"{n_both}/30 years both steps (cathode–concentrate {int((g.cathode_minus_concentrate >= T['A4_step']).sum())}, "
        f"concentrate–ore {int((g.concentrate_minus_ore >= T['A4_step']).sum())})",
        n_both >= T["A4_years"])

    if A.get("pair") is not None:
        d = A["pair"]; worst = d.loc[d.abs_delta.idxmax()]
        n_fail = int((d.abs_delta >= T["A5_delta"]).sum())
        add("A5", "termination accuracy does not move the answer",
            "|Δ retention| per stage, base run at gap_rel 1e-3 vs 1e-6, 30-year pool",
            f"< {T['A5_delta']:.2f} every stage",
            f"{len(d) - n_fail}/{len(d)} stages within; max {worst.abs_delta:.3f} ({worst.stage})",
            n_fail == 0)

    p = B["sec_pooled"]; w_post, w_icsg = float(p.loc["World", "post"]), float(B["icsg_pool"])
    order_ok = p.loc["EU-14", "post"] > p.loc["China", "post"] > p.loc["United States", "post"]
    eu_ok = p.loc["EU-14", "post"] >= T["B1_eu_factor"] * w_post
    add("B1", "secondary share: fidelity and regional order",
        "input-side share, pooled 2020–24: |world posterior − ICSG-implied|; EU-14 > China > US; EU-14 / world",
        f"< {T['B1_fidelity']:.2f}; order holds; ≥ {T['B1_eu_factor']:.0f}×",
        f"|{w_post:.3f} − {w_icsg:.3f}| = {abs(w_post - w_icsg):.3f}; EU-14 {p.loc['EU-14', 'post']:.3f} > China "
        f"{p.loc['China', 'post']:.3f} > US {p.loc['United States', 'post']:.3f}; {p.loc['EU-14', 'post'] / w_post:.2f}×",
        abs(w_post - w_icsg) < T["B1_fidelity"] and bool(order_ok) and bool(eu_ok))

    h = B["hhi"]; h24 = h[h.year == "2024"].set_index(["step", "val"])
    post = {s: float(h24.loc[(s, "post"), "hhi_upper"]) for s in HHI_STEPS}
    inv = {s: abs(float(h24.loc[(s, "post"), "hhi_upper"]) - float(h24.loc[(s, "prior"), "hhi_upper"])) for s in HHI_STEPS}
    down_ok = post["refining"] > post["mining"] and post["use"] > post["mining"]
    high_ok = post["refining"] > T["B2_high"] and post["use"] > T["B2_high"]
    low_ok = post["mining"] < T["B2_low"]
    inv_ok = all(x < T["B2_invariance"] for x in inv.values())
    add("B2", "concentration rises downstream and is reconciliation-invariant",
        "HHI (ROW as one region) 2024 posterior per step; |HHI_post − HHI_prior| per step",
        f"refining, use > mining and > {T['B2_high']:.2f}; mining < {T['B2_low']:.2f}; |Δ| < {T['B2_invariance']:.2f}",
        "HHI " + ", ".join(f"{s} {post[s]:.3f}" for s in HHI_STEPS) + "; |Δ| max "
        f"{max(inv.values()):.3f} ({max(inv, key=inv.get)})",
        down_ok and high_ok and low_ok and inv_ok)

    r = B["embodied_ratio"]; c95, c24 = B["chile_china_1995"], B["chile_china_2024"]
    add("B3", "embodied exceeds direct; Chile's exports turned to China",
        "China's embodied mined Cu from Chile+Peru / direct concentrate imports from them, 2024; "
        "Chile's export share to China 1995 and 2024",
        f"≥ {T['B3_ratio']:.2f}; < {T['B3_chile_1995']:.2f} → > {T['B3_chile_2024']:.2f}",
        f"{B['embodied_kt']:,.0f} / {B['direct_kt']:,.0f} kt CC = {r:.2f}; {c95:.3f} → {c24:.3f}",
        r >= T["B3_ratio"] and c95 < T["B3_chile_1995"] and c24 > T["B3_chile_2024"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def _tag(ax, letter, title, y=1.06):
    ax.text(0.0, y, f"({letter}) {title}", transform=ax.transAxes, fontsize=8,
            fontweight="bold", color=INK, va="bottom", ha="left")


def _year_axis(ax):
    ax.set_xlim(YEARS[0], YEARS[-1])
    ax.set_xticks([1995, 2005, 2015, 2024])


def _block_labels(names: np.ndarray, min_rows: int = 30) -> list:
    """(first row, end row, label) per run of equal names; a run shorter than
    ``min_rows`` joins the preceding labelled run and its label."""
    edges = np.flatnonzero(np.r_[True, names[1:] != names[:-1], True])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < min_rows and out:
            a0, _, lab = out[-1]
            out[-1] = (a0, b, f"{lab} · {names[a]}")
        else:
            out.append((a, b, str(names[a])))
    return out


def fig3(P, out: Path) -> Path:
    tot = P.provenance_totals("CC")
    tot = tot[tot["n_edges"] > 0]                    # an imputer with no fills has no bar
    cp, fam, blk = P.chain_profile("CC"), P.family_summary(), P.block_sizes()
    sl = P.region_slice(PROBLEM_REGION)
    year = P.year
    fig = plt.figure(figsize=(7.2, 10.2))
    g1 = fig.add_gridspec(1, 2, left=0.03, right=0.97, top=0.955, bottom=0.70, wspace=0.42)
    g2 = fig.add_gridspec(1, 2, left=0.20, right=0.97, top=0.605, bottom=0.40, wspace=0.12)
    g3 = fig.add_gridspec(1, 2, left=0.02, right=0.97, top=0.295, bottom=0.035, wspace=0.06,
                          width_ratios=[1.3, 1])

    # (a) edges and kt per provenance category ----------------------------------
    ga = GridSpecFromSubplotSpec(1, 3, subplot_spec=g1[0, 0], wspace=0.04, width_ratios=[1, 1.0, 1])
    al, am, ar = (fig.add_subplot(ga[k]) for k in range(3))
    cats = tot.index.tolist()
    y = np.arange(len(cats))[::-1]
    colors = [PROV_COLOR.get(c, GREY) for c in cats]
    al.barh(y, tot["n_edges"], color=colors, height=0.62)
    al.set_xscale("log"); al.set_xlim(6e4, 1)
    ar.barh(y, tot["kt"].where(tot["kt"] > 0), color=colors, height=0.62)
    ar.set_xscale("log"); ar.set_xlim(1, 2e6)
    for yi, n, kt in zip(y, tot["n_edges"], tot["kt"]):
        al.text(n * 1.25, yi, f"{n:,}", ha="right", va="center", fontsize=6, color=INK2)
        ar.text(kt * 1.25 if kt > 0 else 1.25, yi, f"{kt:,.0f}", ha="left", va="center", fontsize=6, color=INK2)
    for yi, c in zip(y, cats):
        am.text(0.5, yi, PROV_LABEL.get(c, c).replace(" (", "\n("), ha="center", va="center",
                fontsize=6.3, color=INK, linespacing=1.0)
    for ax in (al, am, ar):
        ax.set_ylim(-0.6, len(cats) - 0.4); ax.set_yticks([])
    am.axis("off")
    al.set_xlabel("edges (log)"); ar.set_xlabel("kt CC (log)")
    _style(al, grid_axis="x"); _style(ar, grid_axis="x")
    al.spines["left"].set_visible(False); al.spines["right"].set_visible(True)
    _tag(al, "a", f"What fills the {year} prior, CC", y=1.03)

    # (b) chain profile ------------------------------------------------------------
    ab = fig.add_subplot(g1[0, 1])
    stack = [c for c in PROVENANCE_ORDER if c in cp.columns and cp[c].abs().sum() > 0]
    pos, k = [], 0.0
    for i, kind in enumerate(cp["kind"]):
        if i and kind != cp["kind"].iloc[i - 1]:
            k += 0.9                                       # gap before the traded commodities
        pos.append(-k); k += 1.0
    pos, left = np.array(pos), np.zeros(len(cp))
    for c in stack:
        v = cp[c].to_numpy(dtype=float)
        ab.barh(pos, v, left=left, height=0.74, color=PROV_COLOR.get(c, GREY),
                edgecolor=SURFACE, linewidth=0.4, label=PROV_LABEL.get(c, c))
        left += v
    ab.set_yticks(pos); ab.set_yticklabels(cp.index, fontsize=5.6)
    ab.set_ylim(pos.min() - 0.7, 0.7)
    ab.set_xlabel(f"kt CC, world {year}")
    ab.legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(-0.42, -0.13), ncol=3)
    _style(ab, grid_axis="x")
    _tag(ab, "b", "Where the feeds sit along the chain", y=1.03)

    # (c) rows per family ------------------------------------------------------------
    ac = fig.add_subplot(g2[0, 0])
    yf = np.arange(len(fam))[::-1]
    ac.barh(yf, fam["rows"], color=[CONE_COLOR[c] for c in fam["cone"]], height=0.62)
    for yi, n in zip(yf, fam["rows"]):
        ac.text(n * 1.2, yi, f"{n:,}", va="center", fontsize=6, color=INK2)
    ac.set_xscale("log"); ac.set_xlim(10, 4e5)
    ac.set_yticks(yf); ac.set_yticklabels(fam.index, fontsize=6.5)
    ac.set_xlabel(f"rows, {year} (log)")
    ac.legend([Patch(color=CONE_COLOR[c]) for c in ("zero", "nonneg", "free")],
              [CONE_LABEL[c] for c in ("zero", "nonneg", "free")],
              frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    _style(ac, grid_axis="x")
    _tag(ac, "c", "What the QP constrains", y=1.04)

    # (d) prior breach and active set ---------------------------------------------------
    ad = fig.add_subplot(g2[0, 1], sharey=ac)
    status = [("prior_breach", "prior breach", INK, 1.0),
              ("binding_on_flow", "binding on flow", ACTIVE_COLOR, 1.0),
              ("binding_on_zero_flows", "binding on zero flows", ACTIVE_COLOR, 0.35)]
    h = 0.28
    smax = float(np.nanmax([(fam[c].astype(float) / fam["rows"] * 100.0).max() for c, *_ in status]))
    for j, (col, lab, colr, alpha) in enumerate(status):
        share = fam[col].astype(float) / fam["rows"] * 100.0
        ok = share.notna().to_numpy()
        yy = yf + (1 - j) * h
        ad.barh(yy[ok], share[ok], height=h * 0.9, color=colr, alpha=alpha, label=lab)
        for yi, s in zip(yy[ok], share[ok]):
            txt = "0" if s == 0 else (f"{s:.1f} %" if s < 10 else f"{s:.0f} %")
            ad.text(s + 0.8, yi, txt, va="center", fontsize=5.2, color=INK2)
    for yi, cone in zip(yf, fam["cone"]):
        if cone == "free":
            ad.text(0.8, yi, "soft rows: their pull is z², Fig. 4a", va="center", fontsize=5.8, color=INK2)
    ad.set_xlim(0, 10 * np.ceil(smax / 10) + 8); ad.tick_params(labelleft=False)
    ad.set_xlabel("% of the family's rows")
    ad.legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    _style(ad, grid_axis="x")
    _tag(ad, "d", "Prior breach and active set at a*", y=1.04)

    # (e) the region's slice in standard form + the full block ------------------------------
    ae, ax = fig.add_subplot(g3[0, 0]), fig.add_subplot(g3[0, 1])
    A = sl.A.tocoo()
    n_r, n_c = A.shape
    ae.scatter(A.col, A.row, s=0.35, marker="s", linewidths=0, color=INK2, rasterized=True)
    ae.set_xlim(-0.5, n_c - 0.5); ae.set_ylim(n_r - 0.5, -0.5)
    ae.set_aspect("equal"); ae.set_anchor("E")
    n_mass = int((sl.valuation == "Mass").sum())
    groups = [(0, n_mass, "Mass"), (n_mass, sl.n_edges, "CC"), (sl.n_edges, n_c, "ε")]
    for a, _, _ in groups[1:]:
        ae.axvline(a - 0.5, color=INK2, lw=0.6)
    for a, b, _ in groups[:2]:                               # domestic | legs inside each valuation
        legs = np.flatnonzero(sl.is_leg[a:b])
        if legs.size:
            ae.axvline(a + legs[0] - 0.5, color=GREY, lw=0.4, ls="--")
    ae.set_xticks([(a + b - 1) / 2 for a, b, _ in groups]); ae.set_xticklabels([g[2] for g in groups])
    ae.xaxis.tick_top(); ae.tick_params(axis="x", length=0, labelsize=6.5)
    cone_edges = np.flatnonzero(np.r_[True, sl.cone[1:] != sl.cone[:-1]])
    fam_edges = np.flatnonzero(np.r_[False, sl.family[1:] != sl.family[:-1]])
    for i in fam_edges:
        ae.axhline(i - 0.5, color=GRID, lw=0.5)
    for i in cone_edges[1:]:
        ae.axhline(i - 0.5, color=INK2, lw=0.6)
    blocks = _block_labels(np.array([SLICE_LABEL.get(f, f) for f in sl.family], dtype=object))
    ae.set_yticks([(a + b - 1) / 2 for a, b, _ in blocks])
    ae.set_yticklabels([lab for _, _, lab in blocks], fontsize=5.8)
    ae.tick_params(axis="y", length=0, pad=7)
    bracket = blended_transform_factory(ae.transAxes, ae.transData)
    for a, b, _ in blocks:                                   # family brackets on the row axis
        ae.plot([-0.012, -0.03, -0.03, -0.012], [a - 0.2, a - 0.2, b - 0.8, b - 0.8],
                transform=bracket, color=INK2, lw=0.6, clip_on=False)
    for a, b in zip(cone_edges, np.r_[cone_edges[1:], n_r]):
        ae.annotate({"free": "soft", "zero": "hard =", "nonneg": "hard ≤"}[sl.cone[a]],
                    xy=(0, (a + b - 1) / 2), xycoords=("axes fraction", "data"),
                    xytext=(-90, 0), textcoords="offset points", rotation=90,
                    ha="center", va="center", fontsize=6.3, color=INK)
    act = np.flatnonzero(sl.binding_on_flow)
    ae.hlines(act, n_c + 6, n_c + 40, colors=ACTIVE_COLOR, lw=0.5, clip_on=False)
    if act.size:
        ae.text(n_c + 52, act.mean(), "binding on flow", rotation=90, ha="left", va="center",
                fontsize=5.6, color=INK2)
    for s in ae.spines.values():
        s.set_color(INK2); s.set_linewidth(0.6)
    nnz = int(sl.A[:, :sl.n_edges].nnz)
    ae.text(1.0, -0.025, f"{sl.n_edges} edges ({int(sl.is_leg.sum())} trade legs) + {sl.n_slack} slack ε\n"
            f"{n_r:,} rows · {nnz:,} nonzeros · {sl.n_legs_omitted} legs with no prior of "
            "their own left out",
            transform=ae.transAxes, ha="right", va="top", fontsize=5.8, color=INK2)
    _tag(ae, "e", f"{PROBLEM_REGION}'s slice of the {year} QP in standard form", y=1.06)

    # the full block as a to-scale schematic (the zero cone drawn at a minimum height)
    rc, ne, ns = blk["rows_by_cone"], blk["edges"], blk["rows_by_cone"]["free"]
    wE = ne / (ne + ns)
    hts = {c: max(rc[c] / blk["rows"], 0.05) for c in ("free", "zero", "nonneg")}
    scale = 0.86 / sum(hts.values())
    top, x0 = 0.92, 0.04
    ax.text(x0 + wE * 0.86 / 2, top + 0.015, f"{ne:,} edges a", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.text(x0 + (wE + (1 - wE) / 2) * 0.86, top + 0.015, f"{ns:,} slack ε", ha="center", va="bottom",
            fontsize=6, color=INK2)
    text = {"free": f"G_free\n{rc['free']:,} soft rows", "zero": f"G_zero  {rc['zero']:,} rows",
            "nonneg": f"G_nonneg\n{rc['nonneg']:,} rows"}
    for cone in ("free", "zero", "nonneg"):
        hh = hts[cone] * scale
        ax.add_patch(Rectangle((x0, top - hh), wE * 0.86, hh, facecolor=GRID, edgecolor=INK2, lw=0.6))
        ax.text(x0 + wE * 0.43, top - hh / 2, text[cone], ha="center", va="center", fontsize=6, color=INK)
        ax.add_patch(Rectangle((x0 + wE * 0.86, top - hh), (1 - wE) * 0.86, hh, facecolor=SURFACE,
                               edgecolor=INK2, lw=0.6))
        xs = x0 + wE * 0.86
        if cone == "free":
            ax.plot([xs, xs + (1 - wE) * 0.86], [top, top - hh], color=INK2, lw=0.8)
            ax.text(xs + (1 - wE) * 0.62, top - hh * 0.3, "−I", fontsize=6.5, color=INK)
        else:
            ax.text(xs + (1 - wE) * 0.43, top - hh / 2, "0", ha="center", va="center", fontsize=6.5, color=INK)
        ax.text(x0 + 0.88 + 0.01, top - hh / 2, cone, ha="left", va="center", fontsize=5.8,
                color=INK2)
        top -= hh
    ax.text(x0, top - 0.04, f"the full {year} block: {blk['rows']:,} rows × {ne:,} edges, "
            f"{blk['nnz']:,} nonzeros\n(density {blk['nnz'] / (blk['rows'] * ne):.1e}); "
            f"Stage 4 solves {len(YEARS)} such QPs, one per year",
            ha="left", va="top", fontsize=5.8, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    path = out / "fig3_problem.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def problem_numbers(P) -> dict:
    """The Fig. 3 numbers the prose cites (``numbers.json["problem"]``)."""
    tot, fam = P.provenance_totals("CC"), P.family_summary()
    feed, sl = P.feed_edges_by_year("CC"), P.region_slice(PROBLEM_REGION)
    cp = P.chain_profile("CC")
    na = lambda v: None if pd.isna(v) else int(v)                    # noqa: E731
    return {
        "year": P.year, "region": PROBLEM_REGION,
        "provenance_cc": {c: {"n_edges": int(r.n_edges), "kt": round(float(r.kt), 1)} for c, r in tot.iterrows()},
        "feed_cc_edges_per_year": {"min": int(feed.min()), "max": int(feed.max())},
        "chain_profile_cc_kt": {b: {c: round(float(v), 1) for c, v in r.drop("kind").items()
                                    if round(float(v), 1)} for b, r in cp.iterrows()},
        "families": {f: {"cone": r.cone, "rows": int(r.rows), "prior_breach": na(r.prior_breach),
                         "binding": na(r.binding), "binding_on_flow": na(r.binding_on_flow),
                         "binding_on_zero_flows": na(r.binding_on_zero_flows)} for f, r in fam.iterrows()},
        "block": P.block_sizes(),
        "region_slice": {"edges": sl.n_edges, "legs": int(sl.is_leg.sum()), "legs_left_out": sl.n_legs_omitted,
                         "rows": int(len(sl.row_ids)), "complete": int(sl.complete.sum()),
                         "partial": int((~sl.complete).sum()), "free_rows": sl.n_slack,
                         "nnz": int(sl.A[:, :sl.n_edges].nnz),
                         "binding_on_flow": int(sl.binding_on_flow.sum())},
    }


def fig4(A: dict, out: Path) -> Path:
    dec, x = A["dec"], np.array(YEARS)
    fig = plt.figure(figsize=(7.2, 7.2))
    outer = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32, left=0.08, right=0.92, top=0.92, bottom=0.07)

    # (a) compatibility over time -------------------------------------------
    ga = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.12, height_ratios=[1.1, 1])
    a1, a2 = fig.add_subplot(ga[0]), fig.add_subplot(ga[1], sharex=None)
    for fam in ("edges", "sum_rows", "soft_ratios"):
        a1.plot(x, dec[f"zsq_{fam}"], color=FAMILY_COLOR[fam], lw=1.6, label=FAMILY_LABEL[fam])
        a2.plot(x, dec[f"share_gt3_{fam}"], color=FAMILY_COLOR[fam], lw=1.6)
    a1.set_yscale("log"); a1.set_ylabel("Σz² per family"); a1.tick_params(labelbottom=False)
    a1.legend(frameon=False, loc="center left", ncol=1)   # lower right ran under (b)'s tick labels
    a2.set_ylabel("share of rows |z| > 3"); a2.set_ylim(0, 0.4); a2.set_xlabel("year")
    for ax in (a1, a2):
        _style(ax); _year_axis(ax)
    _tag(a1, "a", "Compatibility with the stated uncertainty")

    # (b) localisation ---------------------------------------------------------
    gb = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.9)
    for j, (fam, tbl) in enumerate((("edges", A["share_e"]), ("sum_rows", A["share_s"]))):
        ax = fig.add_subplot(gb[j])
        tbl = tbl.iloc[::-1]
        ax.barh(np.arange(len(tbl)), tbl["share"], color=FAMILY_COLOR[fam], height=0.55)
        ax.set_yticks(np.arange(len(tbl))); ax.set_yticklabels(tbl["sector_key"], fontsize=6.5)
        ax.set_xscale("log"); ax.set_xlim(1e-3, 1.0)
        ax.set_xlabel("objective share (log)")
        ax.set_title(FAMILY_LABEL[fam], fontsize=7, loc="left", color=INK2)
        _style(ax, grid_axis="x")
        if j == 0:
            _tag(ax, "b", "Where the adjustment sits", y=1.09)

    # (c) retention by stage -----------------------------------------------------
    ax = fig.add_subplot(outer[1, 0])
    ret = A["retention"]
    for i, s in enumerate(PLOT_STAGES):
        r = ret[(ret.stage == STAGE_LABEL[s]) & (ret.year != "all")].set_index("year").reindex([str(y) for y in YEARS])
        ax.plot(x, r["retention"], color=PALETTE[i], lw=1.5, label=STAGE_LABEL[s])
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(0.6, 1.15); ax.set_ylabel("retention Σa* / Σa_prior"); ax.set_xlabel("year")
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=6)
    _style(ax); _year_axis(ax)
    _tag(ax, "c", "How much copper moves, by stage")

    # (d) rocket ---------------------------------------------------------------
    gd = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 1], wspace=0.35)
    norm = LogNorm(vmin=1e-2, vmax=1e2, clip=True)
    specs = [("scale totals", A["fs"], "lhs_prior_t", "lhs_post", "log₁₀ |G·a_prior|", "log₁₀ |c*|"),
             ("edges", A["fe"], "a_prior", "a_post", "log₁₀ |a_prior|", "log₁₀ |a*|")]
    sc = None
    for j, (name, fam, xc, yc, xl, yl) in enumerate(specs):
        ax = fig.add_subplot(gd[j])
        sub = fam[fam["year"] == str(ROCKET_YEAR)]
        xs = np.log10(np.maximum(np.abs(sub[xc].to_numpy(dtype=float)), 1e-6))
        ys = np.log10(np.maximum(np.abs(sub[yc].to_numpy(dtype=float)), 1e-6))
        zz = np.abs(sub["z"].to_numpy(dtype=float)); ok = np.isfinite(zz)
        lo, hi = -6.3, 6.3
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--")
        if (~ok).any():
            ax.scatter(xs[~ok], ys[~ok], color=GREY, s=3, alpha=0.35, edgecolors="none")
        sc = ax.scatter(xs[ok], ys[ok], c=zz[ok], cmap=SEQ_CMAP, norm=norm, s=3, alpha=0.7, edgecolors="none")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xticks([-6, -3, 0, 3, 6]); ax.set_yticks([-6, -3, 0, 3, 6])
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{name} · n = {len(sub):,}", fontsize=7, loc="left", color=INK2)
        _style(ax, grid_axis="both")
        if j == 0:
            _tag(ax, "d", f"Prior against posterior, {ROCKET_YEAR}, coloured by |z|", y=1.09)
    cb = fig.colorbar(sc, ax=fig.axes[-2:], fraction=0.04, pad=0.02)
    cb.set_label("|z|"); cb.outline.set_visible(False)

    path = out / "fig4_reconciliation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig5(B: dict, out: Path) -> Path:
    x = np.array(YEARS)
    fig = plt.figure(figsize=(7.2, 9.4))
    outer = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.3, left=0.1, right=0.93, top=0.965, bottom=0.05,
                             height_ratios=[1, 0.9, 1.05])

    # (a) secondary share ---------------------------------------------------------
    ax = fig.add_subplot(outer[0, 0])
    by_group, world, icsg = B["sec_group"], B["sec_world"], B["icsg_input"]
    groups = [g for g in CMI.GROUPS if g != "Chile+Peru"]
    for i, g in enumerate(groups):
        s = by_group[by_group.group == g].set_index("year").reindex(YEARS)
        ax.plot(x, s["post"], color=PALETTE[i], lw=1.5, label=g)
        ax.plot(x, s["prior"], color=PALETTE[i], lw=0.8, ls=":", alpha=0.8)
    ax.plot(x, world["post"], color=INK, lw=1.6, label="World (posterior)")
    ax.plot(x, icsg.reindex(YEARS), color=INK2, lw=1.0, ls="--", label="World, ICSG-implied")
    ax.set_ylim(0, 0.62); ax.set_ylabel("secondary share of refinery-side input"); ax.set_xlabel("year")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color=INK2, lw=0.8, ls=":")]; labels += ["prior (dotted)"]
    ax.legend(handles, labels, frameon=False, ncol=3, loc="upper left", fontsize=5.8)
    _style(ax); _year_axis(ax)
    _tag(ax, "a", "Secondary share by region group")

    # (b) HHI per step ------------------------------------------------------------
    ax = fig.add_subplot(outer[0, 1])
    h = B["hhi"]
    for i, step in enumerate(HHI_STEPS):
        p = h[(h.step == step) & (h.val == "post")].set_index("year").reindex([str(y) for y in YEARS])
        q = h[(h.step == step) & (h.val == "prior")].set_index("year").reindex([str(y) for y in YEARS])
        ax.fill_between(x, p.hhi_lower, p.hhi_upper, color=PALETTE[i], alpha=0.25, lw=0)
        ax.plot(x, p.hhi_upper, color=PALETTE[i], lw=1.5, label=step)
        ax.plot(x, q.hhi_upper, color=PALETTE[i], lw=0.8, ls=":", alpha=0.8)
    for yv, lab in zip(HHI_BANDS, ("moderate", "high")):
        ax.axhline(yv, color=INK2, lw=0.7, ls="--"); ax.text(1995.3, yv + 0.006, lab, fontsize=6, color=INK2)
    ax.set_ylim(0, 0.4); ax.set_ylabel("Herfindahl–Hirschman index (CC)"); ax.set_xlabel("year")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color=INK2, lw=0.8, ls=":")]; labels += ["prior (dotted)"]
    ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=6)
    _style(ax); _year_axis(ax)
    _tag(ax, "b", "Concentration along the chain")

    # (c) partner shares ---------------------------------------------------------------
    gc = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, :], wspace=0.25)
    for j, (country, direction) in enumerate(TRADE_PANELS):
        ax = fig.add_subplot(gc[j])
        sh = B["partners"][(country, direction)]
        cols = [c for c in B["partner_order"][direction] if c in sh.columns]
        colors = [GREY if c == "Rest" else PALETTE[k] for k, c in enumerate(cols)]
        ax.stackplot(x, [sh[c].to_numpy() for c in cols], labels=cols, colors=colors, alpha=0.9, lw=0)
        ax.set_ylim(0, 1); ax.set_xlabel("year")
        tot = sh.attrs["total_kt"]
        ax.set_title(f"{country} {'exports' if direction == 'export' else 'imports'} · "
                     f"{tot.loc[1995]:,.0f} → {tot.loc[2024]:,.0f} kt CC", fontsize=7, loc="left", color=INK2)
        if j == 0:
            ax.set_ylabel("share of traded copper")
            _tag(ax, "c", "Trade partners, posterior", y=1.16)
        ax.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.26))
        _style(ax, grid_axis="y"); _year_axis(ax)

    # (d) embodied extraction matrix -----------------------------------------------------
    ax = fig.add_subplot(outer[2, :])
    F = B["embodied_folded"]
    im = ax.imshow(F.to_numpy(), cmap=SEQ_CMAP, aspect="auto", vmin=0)
    ax.set_xticks(range(F.shape[1])); ax.set_xticklabels(F.columns, rotation=30, ha="right", fontsize=6.5)
    ax.set_yticks(range(F.shape[0])); ax.set_yticklabels(F.index, fontsize=6.5)
    ax.set_xlabel("region of final use"); ax.set_ylabel("mining region")
    vmax = float(F.to_numpy().max())
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            v = float(F.iat[i, j])
            if v >= 100:
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("kt CC embodied, 2024"); cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    _tag(ax, "d", f"Where the copper in final use was mined ({B['embodied_period']})")

    path = out / "fig5_copper_system.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def _fmt_col(col: pd.Series):
    """Per-column number format: integers with thousands separators when the
    column reaches 100, scientific below 1e-3, three decimals otherwise."""
    v = pd.to_numeric(col, errors="coerce")
    if v.notna().sum() == 0:
        return lambda x: str(x)
    amax = float(v.abs().max())
    if amax >= 100:
        return lambda x: f"{x:,.0f}"
    if 0 < amax < 1e-3:
        return lambda x: f"{x:.1e}"
    return lambda x: f"{x:.3f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmts = {c: _fmt_col(df[c]) for c in cols}
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else fmts[c](float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(out: Path, ledger: pd.DataFrame, numbers: dict, tables: dict) -> None:
    ledger.to_csv(out / "table3_ledger.csv", index=False)
    with open(out / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    md = ["<!-- generated by scripts/report/results_figs.py — do not edit by hand -->", ""]
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        md += [f"### {name}", "", _md(df), ""]
    (out / "si_tables.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(opt: str, psut_id: str, tol_pair=None, out: Path = OUT_DIR) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"reference run {opt} · psut {psut_id} · tolerance pair {tol_pair}")

    # ---- Fig. 3: the problem ------------------------------------------------------------
    P = ProblemStatsDiagnostic.from_opt_id(opt, PROBLEM_YEAR)

    # ---- Part A -------------------------------------------------------------------
    ef, cf = load_frames(opt)
    dec = objective_decomposition(ef, cf)
    fe, fs = family_frame(ef, cf, "edges"), family_frame(ef, cf, "sum_rows")
    fr = family_frame(ef, cf, "soft_ratios")
    edges = CMI.load_run(opt)
    icsg, eff = CMI.load_icsg(), CMI.load_eff()
    hubs = merchant_hubs(icsg, sorted(edges.region_to.unique()))
    A = dict(dec=dec, fe=fe, fs=fs, share_e=share_by_key(fe, top=8), share_s=share_by_key(fs, top=8),
             share_r=share_by_key(fr, top=8), hub=hub_share(fe, hubs), hubs=hubs,
             retention=retention_by_stage(edges), a4=a4_gradient(edges), pair=None,
             rank_e=rank_identities(fe, top=25), rank_s=rank_identities(fs, top=25),
             rank_r=rank_identities(fr, top=25))
    if tol_pair:
        old, new = (retention_by_stage(CMI.load_run(r)) for r in tol_pair)
        old, new = old[old.year == "all"].set_index("stage"), new[new.year == "all"].set_index("stage")
        pair = pd.DataFrame({"old": old.retention, "new": new.retention})
        pair["delta"] = pair.new - pair.old; pair["abs_delta"] = pair.delta.abs()
        pair.index.name = "stage"; A["pair"] = pair.reset_index()
        A["pair_chile"] = tuple(chile_export_retention(CMI.load_run(r)) for r in tol_pair)

    # ---- Part B -------------------------------------------------------------------
    cc = edges[edges.valuation == "CC"]
    psut = PsutRunRegistry("data/proc/psut").load(psut_id)
    if psut.opt_run_id != opt:
        raise RuntimeError(f"psut run {psut_id} was built from {psut.opt_run_id}, not {opt}")
    period = psut.latest()
    by_group, world, pooled = secondary_shares(cc)
    icsg_in = icsg_input_share(icsg)
    origins = import_origins(cc, "China", 5)
    partner_order = {"export": EXPORT_GROUPS, "import": origins + ["Rest"]}
    export_of = lambda r: CMI.group_of(r) if CMI.group_of(r) in EXPORT_GROUPS else "Rest"   # noqa: E731
    import_of = lambda r: r if r in origins else "Rest"                                      # noqa: E731
    partners = {(c, d): partner_shares(cc, c, d, export_of if d == "export" else import_of)
                for c, d in TRADE_PANELS}
    M, folded = embodied(psut, period)
    emb_kt = float(M.loc[["Chile", "Peru"], "China"].sum())
    direct_kt = direct_conc_imports(cc, "China", ["Chile", "Peru"], int(period))
    B = dict(sec_group=by_group, sec_world=world, sec_pooled=pooled, icsg_input=icsg_in,
             icsg_pool=float(icsg_in.loc[POOL].mean()), icsg_published=icsg_published_share(icsg, eff),
             hhi=hhi_by_step(cc), partners=partners, partner_order=partner_order,
             embodied=M, embodied_folded=folded, embodied_period=period,
             embodied_kt=emb_kt, direct_kt=direct_kt, embodied_ratio=emb_kt / direct_kt,
             chile_china_1995=chile_share_to_china(cc, 1995), chile_china_2024=chile_share_to_china(cc, 2024))

    # ---- ledger, figures, outputs -----------------------------------------------------
    ledger = build_ledger(A, B)
    print(ledger[["id", "value", "outcome"]].to_string(index=False))
    p3, p4, p5 = fig3(P, out), fig4(A, out), fig5(B, out)
    print("figures:", p3, p4, p5)

    ret_all = A["retention"][A["retention"].year == "all"].set_index("stage")
    h24 = B["hhi"][B["hhi"].year == "2024"]
    numbers = {
        "opt_run_id": opt, "psut_run_id": psut_id, "tolerance_pair": list(tol_pair or []),
        "n_rows": int(len(cf)), "n_edges": int(len(ef) // len(YEARS)) if len(ef) else None,
        "zsq_total_30y": float(dec.zsq_total.sum()),
        "zsq_by_family_30y": {f: float(dec[f"zsq_{f}"].sum()) for f in ("edges", "sum_rows", "soft_ratios")},
        "max_hard_residual_kt": float(dec.max_abs_resid_massbal.max()),
        "share_gt3": {f: {"min": float(dec[f"share_gt3_{f}"].min()), "max": float(dec[f"share_gt3_{f}"].max()),
                          "mean": float(dec[f"share_gt3_{f}"].mean())} for f in ("edges", "sum_rows", "soft_ratios")},
        "edge_share_top": A["share_e"].set_index("sector_key")["share"].round(6).to_dict(),
        "sum_share_top": A["share_s"].set_index("sector_key")["share"].round(6).to_dict(),
        "hub_regions": hubs, "hub_share": A["hub"],
        "retention_30y": ret_all["retention"].round(6).to_dict(),
        "chile_export_retention_2010_19": chile_export_retention(edges),
        "a4_years_both": int(A["a4"].both.sum()),
        "a4_years_cathode_concentrate": int((A["a4"].cathode_minus_concentrate >= THRESH["A4_step"]).sum()),
        "a4_years_concentrate_ore": int((A["a4"].concentrate_minus_ore >= THRESH["A4_step"]).sum()),
        "tolerance_pair_retention": (A["pair"].set_index("stage").round(6).to_dict("index") if A["pair"] is not None else None),
        "tolerance_pair_chile_export": A.get("pair_chile"),
        "secondary_pooled_2020_24": pooled.round(6).to_dict("index"),
        "icsg_input_share_pooled": B["icsg_pool"],
        "icsg_published_share": B["icsg_published"].loc[[1995, 2010, 2020, 2024]].round(6).to_dict(),
        "world_secondary_2024": {"prior": float(world.loc[2024, "prior"]), "post": float(world.loc[2024, "post"])},
        "hhi_2024": {f"{r.step}_{r.val}": {"upper": round(r.hhi_upper, 6), "lower": round(r.hhi_lower, 6),
                                          "row_share": round(r.row_share, 6), "top": r.top_region,
                                          "top_share": round(r.top_share, 6)} for r in h24.itertuples()},
        "hhi_1995_post": {r.step: round(r.hhi_upper, 6) for r in B["hhi"][(B["hhi"].year == "1995") & (B["hhi"].val == "post")].itertuples()},
        "partners_2024": {f"{c}_{d}": partners[(c, d)].loc[2024].round(6).to_dict() for c, d in TRADE_PANELS},
        "partners_1995": {f"{c}_{d}": partners[(c, d)].loc[1995].round(6).to_dict() for c, d in TRADE_PANELS},
        "trade_totals_kt": {f"{c}_{d}": {"1995": float(partners[(c, d)].attrs["total_kt"].loc[1995]),
                                        "2024": float(partners[(c, d)].attrs["total_kt"].loc[2024])} for c, d in TRADE_PANELS},
        "embodied_period": period, "embodied_china_from_chile_peru_kt": emb_kt, "direct_conc_china_from_chile_peru_kt": direct_kt,
        "embodied_ratio": emb_kt / direct_kt,
        "china_footprint_kt": float(M["China"].sum()), "china_domestic_extraction_kt": float(M.loc["China"].sum()),
        "china_footprint_origins": M["China"].sort_values(ascending=False).head(8).round(1).to_dict(),
        "world_extraction_kt": float(M.to_numpy().sum()),
        "chile_share_to_china": {"1995": B["chile_china_1995"], "2024": B["chile_china_2024"]},
        "ledger": ledger.to_dict("records"),
        "problem": problem_numbers(P),
    }
    hhi_tbl = B["hhi"][B["hhi"].year.isin(["1995", "2024"])].copy()
    # Stored at six decimals: the SI tables print three (or whole kt) and the prose is
    # read off numbers.json, so a value rounds once, not twice.
    tables = {
        "si_s15_ledger": ledger,
        "si_s15_decomposition": dec.round(6),
        "si_s15_retention_by_stage": A["retention"][A["retention"].year == "all"][["stage", "prior", "post", "retention"]].round(6),
        "si_s15_a4_gradient": A["a4"].reset_index().round(6),
        "si_s15_secondary_pooled": pooled.reset_index().rename(columns={"index": "group"}).round(6),
        "si_s15_hhi_1995_2024": hhi_tbl.round(6),
        "si_s15_embodied_folded": folded.round(6).rename_axis("mining region").reset_index(),
        "si_s16_top_identities_edges": A["rank_e"], "si_s16_top_identities_sum_rows": A["rank_s"],
        "si_s16_top_identities_soft_ratios": A["rank_r"], "si_s16_share_soft_ratios": A["share_r"],
        "si_s17_legs": leg_table(cc, ["Chile", "Peru", "China"]).round(6),
    }
    if A["pair"] is not None:
        tables["si_s15_tolerance_pair"] = A["pair"].round(6)
    write_outputs(out, ledger, numbers, tables)
    print("written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("opt_run_id")
    ap.add_argument("--psut-run-id", required=True)
    ap.add_argument("--tol-pair", nargs=2, metavar=("OPT_OLD", "OPT_NEW"), default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    main(a.opt_run_id, a.psut_run_id, a.tol_pair, Path(a.out))
