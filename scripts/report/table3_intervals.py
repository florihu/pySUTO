"""
table3_intervals.py
===================
#81: posterior intervals beside Table 3 and SI S18b. Stage 5, run in memory with the
persisted run's knobs, gives the exact posterior covariance of the linear totals
behind each row (:class:`pysuto.uncprop.AggregateSpec`):

    retention   Σa* per stage and year (S18b, A4, Fig. 4c), over its fixed Σa_prior
    B1          scrap and primary inputs per region group, pooled 2020–2024
    B2          each step's region totals, 2024
    B3          Chile's traded legs to China and in all (1995, 2024); China's direct
                concentrate imports from Chile and Peru (2024)

The delta method carries that covariance to the ratios, shares and HHIs; B3's embodied
ratio runs through the Leontief inverse of the market-share PIOT, so its delta-method σ
comes from posterior draws by finite differences. Every σ is Birge-scaled by √R of the
persisted Stage-5 run (#87). With SENSITIVITY, #99's degenerate family (cross-region
Cathode→FU legs whose prior sits on their lower bound) is pinned, then freed, and both
σ go beside the headline. The aggregates encode results_figs.py's own definitions: the
script raises where a point value differs from results_figs.py's.

    python scripts/report/table3_intervals.py <opt_run_id> <uncprop_run_id> \\
        [--out CSV] [--no-sensitivity]

Runs Stage 5 once (≈ 25 min on copper), three times with SENSITIVITY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
sys.path.insert(0, "scripts/report")

import copper_metabolism_insights as CMI                         # noqa: E402
import results_figs as RF                                        # noqa: E402
from pysuto.acc import AccountingStructure                       # noqa: E402
from pysuto.config import load_config                            # noqa: E402
from pysuto.psut import PSUT                                     # noqa: E402
from pysuto.registry import ConstraintRunRegistry, UncPropRunRegistry  # noqa: E402
from pysuto.uncprop import AggregateSpec, KalmanUncProp          # noqa: E402

# =============================================================================
# USER CONFIGURATION
# =============================================================================
OUT_CSV     = Path("docs/manuscript/figs/table3_intervals.csv")
ALPHA       = 0.05          # two-sided interval level
LEVEL_YEAR  = 2024          # B2's HHI and B3's embodied and direct imports
CHILE_YEARS = (1995, 2024)  # B3's export share to China
N_DRAWS     = 200           # posterior draws for B3's embodied ratio (σ to ≈ 1/√(2N) = 5 %)
SEED        = 81
FD_STEP     = 1e-6          # a draw enters at a* + FD_STEP·δ: the linear regime
SENSITIVITY = True          # #99's degenerate family pinned and freed
CONSTANTS   = ("sigma_a_floor_abs", "sigma_a_floor_rel", "sigma_a_ceiling_factor",
               "sigma_c_floor", "bind_tol", "rank_tol")


# =============================================================================
# Aggregates: the linear totals behind Table 3
# =============================================================================

def aggregate_columns(edges: pd.DataFrame) -> list:
    """``(name, mask over edges' rows)`` for every total a Table 3 row reads."""
    cc   = edges.valuation == "CC"
    cols = []
    for a, b, v in RF.STAGES:
        stage = (edges.sector_from == a) & (edges.sector_to == b) & (edges.valuation == v)
        for y in RF.YEARS:
            cols.append((f"stage|{RF.STAGE_LABEL[(a, b, v)]}|{y}", stage & (edges.year == y)))

    inp = cc & edges.year.isin(RF.POOL) & (
        (edges.sector_from == "Scrap")
        | ((edges.sector_from == "Concentrate") & (edges.sector_to == "Smelting"))
        | ((edges.sector_from == "SXEW") & (edges.sector_to == "CSXEW")))
    sec   = edges.sector_from == "Scrap"
    group = edges.region_to.map(CMI.group_of)
    for g in CMI.GROUPS:
        cols.append((f"b1|sec|{g}", inp & sec & (group == g)))
        cols.append((f"b1|prim|{g}", inp & ~sec & (group == g)))

    for step, (a, b, col) in RF.HHI_STEPS.items():
        m = cc & (edges.sector_from == a) & (edges.sector_to == b) & (edges.year == LEVEL_YEAR)
        for r in sorted(edges.loc[m, col].unique()):
            cols.append((f"b2|{step}|{r}", m & (edges[col] == r)))

    traded = (cc & edges.sector_from.isin(RF.TRADED) & (edges.region_from == "Chile")
              & (edges.region_to != "Chile"))
    for y in CHILE_YEARS:
        cols.append((f"b3|chile_to_china|{y}", traded & (edges.year == y) & (edges.region_to == "China")))
        cols.append((f"b3|chile_exports|{y}", traded & (edges.year == y)))
    cols.append(("b3|direct", direct_mask(edges)))
    return [(name, m.to_numpy()) for name, m in cols]


def direct_mask(edges: pd.DataFrame) -> pd.Series:
    """China's direct concentrate imports from Chile and Peru (results_figs.direct_conc_imports)."""
    return ((edges.valuation == "CC") & (edges.sector_from == "Concentrate")
            & (edges.sector_to == "Smelting") & (edges.year == LEVEL_YEAR)
            & (edges.region_to == "China") & edges.region_from.isin(["Chile", "Peru"]))


def build_spec(edges: pd.DataFrame, cols: list, n_edges: int) -> AggregateSpec:
    """The (n_edges × K) 0/1 weights of ``cols`` over the global edge index."""
    ids  = edges.edge_id.to_numpy(dtype=np.int64)
    rows = [ids[m] for _, m in cols]
    k    = [np.full(r.size, j) for j, r in enumerate(rows)]
    r    = np.concatenate(rows)
    W    = sp.csr_matrix((np.ones(r.size), (r, np.concatenate(k))), shape=(n_edges, len(cols)))
    return AggregateSpec(names=[n for n, _ in cols], W=W)


# =============================================================================
# Quantities: functions of the totals
# =============================================================================

def hhi(x: np.ndarray) -> float:
    """Σ s² over the shares of ``x`` (ROW counted as one region: B2's upper value)."""
    s = x / x.sum()
    return float((s ** 2).sum())


def quantities(edges: pd.DataFrame, cols: list, icsg_pool: float) -> list:
    """Every reported quantity: ``(row, name, inputs, f, linear)``; ``f`` maps
    the inputs' values (in order) to the quantity. Priors enter as constants.
    Table 3's |Δ| tests are reported signed: an interval around |Δ| would be
    symmetric about a folded value."""
    prior = {name: float(edges.prior.to_numpy()[m].sum()) for name, m in cols}
    out = []
    for a, b, v in RF.STAGES:
        label = RF.STAGE_LABEL[(a, b, v)]
        names = [f"stage|{label}|{y}" for y in RF.YEARS]
        for name, y in zip(names, RF.YEARS):
            out.append(("retention", f"{label} {y}", [name],
                        lambda x, p=prior[name]: x[0] / p, True))
        total = sum(prior[n] for n in names)
        out.append(("retention", f"{label} 1995–2024", names,
                    lambda x, p=total: x.sum() / p, True))

    sec  = [f"b1|sec|{g}" for g in CMI.GROUPS]
    prim = [f"b1|prim|{g}" for g in CMI.GROUPS]
    k    = len(sec)
    world = lambda x: x[:k].sum() / x.sum()                               # noqa: E731
    out.append(("B1", "world secondary share, 2020–24", sec + prim, world, False))
    out.append(("B1", "world − ICSG-implied", sec + prim,
                lambda x: world(x) - icsg_pool, False))
    for g in ("EU-14", "China", "United States"):
        out.append(("B1", f"{g} secondary share, 2020–24", [f"b1|sec|{g}", f"b1|prim|{g}"],
                    lambda x: x[0] / (x[0] + x[1]), False))
    eu = ["b1|sec|EU-14", "b1|prim|EU-14"]
    out.append(("B1", "EU-14 / world", eu + sec + prim,
                lambda x: (x[0] / (x[0] + x[1])) / world(x[2:]), False))

    for step in RF.HHI_STEPS:
        names = [n for n, _ in cols if n.startswith(f"b2|{step}|")]
        h_prior = hhi(np.array([prior[n] for n in names]))
        out.append(("B2", f"HHI {step} {LEVEL_YEAR}", names, hhi, False))
        out.append(("B2", f"HHI_post − HHI_prior {step}", names,
                    lambda x, h=h_prior: hhi(x) - h, False))

    for y in CHILE_YEARS:
        out.append(("B3", f"Chile export share to China {y}",
                    [f"b3|chile_to_china|{y}", f"b3|chile_exports|{y}"],
                    lambda x: x[0] / x[1], False))
    out.append(("B3", f"direct concentrate, China ← Chile+Peru {LEVEL_YEAR} (kt CC)",
                ["b3|direct"], lambda x: x[0], True))
    return out


def delta_sigma(f, x: np.ndarray, C: np.ndarray, rel_step: float = 1e-6) -> float:
    """√(gᵀCg) with ``g`` the central-difference gradient of ``f`` at ``x``."""
    g = np.empty(x.size)
    for i in range(x.size):
        h = rel_step * max(abs(x[i]), 1e-6)
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2.0 * h)
    return float(np.sqrt(max(g @ C @ g, 0.0)))


def fd_sigma(f, draws: np.ndarray, step: float = FD_STEP) -> float:
    """Delta-method σ of ``f`` from posterior draws δ (rows): the draws' second
    moment of the directional derivative ``(f(step·δ) − f(0)) / step``."""
    f0 = f(np.zeros(draws.shape[1]))
    g  = np.array([(f(step * d) - f0) / step for d in draws])
    return float(np.sqrt(np.mean(g ** 2)))


def evaluate(qs: list, moments, scale: float) -> pd.DataFrame:
    """Value, σ and the √R-scaled interval of every quantity; ``sigma_quadrature``
    (linear rows only) adds the edges' σ_a* as if independent."""
    at   = {n: i for i, n in enumerate(moments.names)}
    z    = float(norm.ppf(1.0 - ALPHA / 2.0))
    rows = []
    for row, name, inputs, f, linear in qs:
        idx = [at[n] for n in inputs]
        x, C = moments.value[idx], moments.cov_post[np.ix_(idx, idx)]
        val, s = f(x), delta_sigma(f, x, C)
        quad = np.nan
        if linear:
            g = np.array([(f(x + np.eye(x.size)[i]) - f(x)) for i in range(x.size)])
            quad = float(np.sqrt(((g * moments.sigma_quadrature[idx]) ** 2).sum()))
        rows.append(dict(row=row, quantity=name, value=val, sigma=s, sigma_birge=s * scale,
                         lo_birge=val - z * s * scale, hi_birge=val + z * s * scale,
                         sigma_quadrature=quad))
    return pd.DataFrame(rows)


# =============================================================================
# B3's embodied ratio: posterior draws through the PIOT
# =============================================================================

def positions(cols_t: np.ndarray, edge_ids: np.ndarray) -> np.ndarray:
    """Each edge id's position in the year's ``cols_t``; raises on an edge outside it."""
    where = pd.Index(cols_t).get_indexer(edge_ids)
    if (where < 0).any():
        raise ValueError(f"{int((where < 0).sum())} edges are not in the year's cols_t")
    return where


def embodied_ratio_fn(opt: str, edges: pd.DataFrame, cols_t: np.ndarray, period: str):
    """``f(δ)`` = China's embodied mined Cu from Chile+Peru over its direct
    concentrate imports from them, at ``a* + δ`` (δ over ``cols_t`` of ``period``)."""
    psut = PSUT.from_opt_run(opt, timesteps=[period])
    tbl  = psut.table[psut.table.valuation == "CC"].reset_index(drop=True)
    pos  = positions(cols_t, tbl.edge_id.to_numpy())
    base = tbl.val.to_numpy(dtype=float)
    dpos = positions(cols_t, edges.loc[direct_mask(edges), "edge_id"].to_numpy())
    a_t  = (edges.set_index("edge_id").post.reindex(cols_t).to_numpy(dtype=float))

    def f(delta: np.ndarray) -> float:
        t = tbl.copy()
        t["val"] = base + delta[pos]
        p = PSUT(t, psut.nodes, valuations=["CC"], primary="CC", acc_id=psut.acc_id)
        M, _ = RF.embodied(p, period)
        return float(M.loc[["Chile", "Peru"], "China"].sum()) / float((a_t + delta)[dpos].sum())
    return f


# =============================================================================
# Checks and the sensitivity family
# =============================================================================

def check_definitions(edges: pd.DataFrame, cols: list, qs: list) -> None:
    """The totals must give results_figs.py's own numbers on the posterior."""
    post = {name: float(edges.post.to_numpy()[m].sum()) for name, m in cols}
    val  = {name: f(np.array([post[n] for n in inputs])) for _, name, inputs, f, _ in qs}
    cc   = edges[edges.valuation == "CC"]
    _, _, pooled = RF.secondary_shares(cc)
    h = RF.hhi_by_step(cc)
    h = h[(h.year == str(LEVEL_YEAR)) & (h.val == "post")].set_index("step")
    ret = RF.retention_by_stage(edges)
    ret = ret[ret.year == "all"].set_index("stage").retention
    ref = {"world secondary share, 2020–24": pooled.loc["World", "post"],
           "EU-14 secondary share, 2020–24": pooled.loc["EU-14", "post"],
           f"direct concentrate, China ← Chile+Peru {LEVEL_YEAR} (kt CC)":
               RF.direct_conc_imports(cc, "China", ["Chile", "Peru"], LEVEL_YEAR),
           **{f"HHI {s} {LEVEL_YEAR}": h.loc[s, "hhi_upper"] for s in RF.HHI_STEPS},
           **{f"Chile export share to China {y}": RF.chile_share_to_china(cc, y)
              for y in CHILE_YEARS},
           **{f"{s} 1995–2024": r for s, r in ret.items()}}
    bad = {k: (val[k], v) for k, v in ref.items() if not np.isclose(val[k], v, rtol=1e-9)}
    if bad:
        raise RuntimeError(f"aggregates differ from results_figs.py: {bad}")


def degenerate_family(edges: pd.DataFrame, const_run, n_edges: int) -> np.ndarray:
    """#99's family: cross-region Cathode→FU legs, either valuation, whose prior
    is 0 and which carry a lower-bound row (the slack rule sorts them by noise)."""
    rm  = const_run.row_meta
    lb  = np.flatnonzero(((rm["type"] == "bound") & (rm["variant"] == "lower")).to_numpy())
    has = np.zeros(n_edges, dtype=bool)
    has[const_run.G[lb].tocoo().col] = True
    m = ((edges.sector_from == "Cathode") & (edges.sector_to == "FU")
         & (edges.region_from != edges.region_to) & (edges.prior == 0.0))
    out = np.zeros(n_edges, dtype=bool)
    out[edges.loc[m, "edge_id"].to_numpy()] = True
    return out & has


# =============================================================================
# Main
# =============================================================================

def main(opt: str, unc: str, out: Path = OUT_CSV, sensitivity: bool = SENSITIVITY) -> None:
    from pysuto.diag import UncPropDiag                      # heavy: only when run
    reg     = UncPropRunRegistry("data/proc/uncprop")
    sidecar = reg.load_sidecar(unc)
    up      = sidecar["parents"]["upstream"]
    cfg     = sidecar["stage_config"]
    if up["opt_run_id"] != opt:
        raise ValueError(f"{unc} propagates {up['opt_run_id']}, not {opt}")
    if cfg.get("exclude_scale_totals"):
        raise ValueError(f"{unc} is a scale-total ablation: its σ_a* is not the fit's")
    R = UncPropDiag.from_uncprop_id(unc).stats.birge_ratio()

    acc   = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir="data/proc/acc")
    edges = CMI.load_run(opt)
    cols  = aggregate_columns(edges)
    icsg_pool = float(RF.icsg_input_share(CMI.load_icsg()).loc[RF.POOL].mean())
    qs    = quantities(edges, cols, icsg_pool)
    check_definitions(edges, cols, qs)
    spec  = build_spec(edges, cols, acc.n_edges)
    print(f"{opt} · {unc} · Birge R {R:.3f} · {len(cols)} totals · {len(qs)} quantities")

    prop   = KalmanUncProp(**{k: cfg[k] for k in CONSTANTS})
    period = str(LEVEL_YEAR)
    base   = prop.compute(opt, cfg["alpha"], aggregates=spec, draws={period: N_DRAWS},
                          rng=np.random.default_rng(SEED))
    stored = reg.load(unc).per_timestep[period].sigma_a_star
    if not np.allclose(base.per_timestep[period].sigma_a_star, stored, rtol=1e-8, atol=1e-12):
        raise RuntimeError(f"the in-memory Stage 5 does not reproduce {unc}'s σ_a* in {period}")
    table = evaluate(qs, base.aggregates, np.sqrt(R))

    cols_t = base.per_timestep[period].cols_t
    f_b3   = embodied_ratio_fn(opt, edges, cols_t, period)
    s_b3   = fd_sigma(f_b3, base.draws[period])
    v_b3   = f_b3(np.zeros(cols_t.size))
    z      = float(norm.ppf(1.0 - ALPHA / 2.0))
    table  = pd.concat([table, pd.DataFrame([dict(
        row="B3", quantity=f"embodied / direct, China ← Chile+Peru {LEVEL_YEAR}", value=v_b3,
        sigma=s_b3, sigma_birge=s_b3 * np.sqrt(R), lo_birge=v_b3 - z * s_b3 * np.sqrt(R),
        hi_birge=v_b3 + z * s_b3 * np.sqrt(R), sigma_quadrature=np.nan)])], ignore_index=True)
    # The draws' check: their σ of a linear total against the exact one.
    direct = table.quantity.str.startswith("direct concentrate")
    pos    = positions(cols_t, edges.loc[direct_mask(edges), "edge_id"].to_numpy())
    s_fd   = float(np.sqrt(np.mean(base.draws[period][:, pos].sum(1) ** 2)))
    print(f"   draws check, direct imports σ: exact {float(table.loc[direct, 'sigma'].iloc[0]):.3f}"
          f" kt CC, from {N_DRAWS} draws {s_fd:.3f}")

    if sensitivity:
        const = ConstraintRunRegistry("data/proc/const").load(up["const_run_id"])
        fam   = degenerate_family(edges, const, acc.n_edges)
        del const
        print(f"   degenerate family: {int(fam.sum()):,} edge-years")
        for tag, kw in (("pinned", {"force_pinned": fam}), ("free", {"force_free": fam})):
            alt = prop.compute(opt, cfg["alpha"], aggregates=spec, **kw)
            table[f"sigma_{tag}"] = list(evaluate(qs, alt.aggregates, 1.0).sigma) + [np.nan]

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    show = table[~table.quantity.str.match(r".* (19|20)\d\d$") | (table.row != "retention")]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(show.round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("opt_run_id")
    ap.add_argument("uncprop_run_id")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()
    main(a.opt_run_id, a.uncprop_run_id, a.out, sensitivity=not a.no_sensitivity)
