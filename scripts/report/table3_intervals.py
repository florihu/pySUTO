"""
table3_intervals.py
===================
#81: posterior intervals beside Table 3 and SI S18b. Stage 5, run in memory with the
persisted run's knobs, gives the exact posterior covariance of the linear totals
behind each row (:class:`pysuto.uncprop.AggregateSpec`):

    retention   Σa* per stage and year over its fixed Σa_prior (S18b, Fig. 4c), and
                A4's per-year gaps cathode − concentrate − ore
    B1          scrap and primary inputs per region group, pooled 2020–2024
    B2          each step's region totals, 2024
    B3          Chile's traded legs to China and in all (1995, 2024); China's direct
                concentrate imports from Chile and Peru (2024)

The delta method carries that covariance to the ratios, shares, HHIs and their
differences; B3's embodied ratio runs through the Leontief inverse of the market-share
PIOT, so its delta-method σ comes from posterior draws by finite differences. Every σ
is Birge-scaled by √R of the persisted Stage-5 run (#87). A1–A3 and A5 get none: they
are residuals, z-shares and a tolerance pair, not totals of a*.

With SENSITIVITY, #99's degenerate family is pinned, then freed: cross-region
Cathode→FU legs whose prior is 0 and whose a* lies within the FRINGE_KT of their
lower bound, where the slack rule sorts them by noise. The two Birge-scaled σ bracket
its split. The totals encode results_figs.py's own definitions: the script raises
where a total-based value differs from results_figs.py's (``check_definitions``).

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
from pysuto.diag import UncPropDiag                              # noqa: E402
from pysuto.psut import PSUT                                     # noqa: E402
from pysuto.registry import ConstraintRunRegistry, UncPropRunRegistry  # noqa: E402
from pysuto.uncprop import AggregateSpec, KalmanUncProp          # noqa: E402

# =============================================================================
# USER CONFIGURATION
# =============================================================================
OUT_CSV      = Path("docs/manuscript/figs/table3_intervals.csv")
ALPHA        = 0.05          # two-sided interval level
LEVEL_YEAR   = 2024          # B2's HHI and B3's embodied and direct imports
CHILE_YEARS  = (1995, 2024)  # B3's export share to China
N_DRAWS      = 200           # draws for B3's embodied ratio (σ to ≈ 1/√(2N) = 5 %)
SEED         = 81
FD_STEP      = 1e-6          # a draw enters at a* + FD_STEP·δ: the linear regime
SENSITIVITY  = True          # #99's degenerate family pinned and freed
FRINGE_KT    = 1e-4          # #99's fringe: a* this close to its bound is ambiguous
STAGE5_KNOBS = ("sigma_a_floor_abs", "sigma_a_floor_rel", "sigma_a_ceiling_factor",
                "sigma_c_floor", "bind_tol", "rank_tol")
Z_ALPHA      = float(norm.ppf(1.0 - ALPHA / 2.0))
PER_YEAR     = "(year)"      # row suffix of the per-year rows (CSV only, not printed)


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
            cols.append((stage_name(RF.STAGE_LABEL[(a, b, v)], y), stage & (edges.year == y)))

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
        m = (cc & (edges.sector_from == a) & (edges.sector_to == b)
             & (edges.year == LEVEL_YEAR))
        for r in sorted(edges.loc[m, col].unique()):
            cols.append((f"b2|{step}|{r}", m & (edges[col] == r)))

    traded = (cc & edges.sector_from.isin(RF.TRADED) & (edges.region_from == "Chile")
              & (edges.region_to != "Chile"))
    for y in CHILE_YEARS:
        ty = traded & (edges.year == y)
        cols.append((f"b3|chile_to_china|{y}", ty & (edges.region_to == "China")))
        cols.append((f"b3|chile_exports|{y}", ty))
    cols.append(("b3|direct", direct_mask(edges)))
    return [(name, m.to_numpy()) for name, m in cols]


def stage_name(label: str, year: int) -> str:
    return f"stage|{label}|{year}"


def direct_mask(edges: pd.DataFrame) -> pd.Series:
    """China's direct concentrate imports from Chile and Peru
    (results_figs.direct_conc_imports)."""
    return ((edges.valuation == "CC") & (edges.sector_from == "Concentrate")
            & (edges.sector_to == "Smelting") & (edges.year == LEVEL_YEAR)
            & (edges.region_to == "China") & edges.region_from.isin(["Chile", "Peru"]))


def build_spec(edges: pd.DataFrame, cols: list, n_edges: int) -> AggregateSpec:
    """The (n_edges × K) 0/1 weights of ``cols`` over the global edge index."""
    ids  = edges.edge_id.to_numpy(dtype=np.int64)
    rows = [ids[m] for _, m in cols]
    k    = [np.full(r.size, j) for j, r in enumerate(rows)]
    r    = np.concatenate(rows)
    W    = sp.csr_matrix((np.ones(r.size), (r, np.concatenate(k))),
                         shape=(n_edges, len(cols)))
    return AggregateSpec(names=[n for n, _ in cols], W=W)


# =============================================================================
# Quantities: functions of the totals
# =============================================================================

def hhi(x: np.ndarray) -> float:
    """Σ s² over the shares of ``x`` (ROW counted as one region: B2's upper value)."""
    s = x / x.sum()
    return float((s ** 2).sum())


def quantities(edges: pd.DataFrame, cols: list, icsg_pool: float) -> list:
    """Every reported quantity: ``(row, name, inputs, f, linear)``; ``f`` maps the
    inputs' values (in order) to the quantity. Priors enter as constants. Table 3's
    |Δ| tests and orderings are reported as signed differences: an interval around a
    folded |Δ| would be symmetric about the fold."""
    prior = {name: float(edges.prior.to_numpy()[m].sum()) for name, m in cols}
    out = []
    for a, b, v in RF.STAGES:
        label = RF.STAGE_LABEL[(a, b, v)]
        names = [stage_name(label, y) for y in RF.YEARS]
        for name, y in zip(names, RF.YEARS):
            out.append((f"retention {PER_YEAR}", f"{label} {y}", [name],
                        lambda x, p=prior[name]: x[0] / p, True))
        total = sum(prior[n] for n in names)
        out.append(("S18b", f"retention {label} 1995–2024", names,
                    lambda x, p=total: x.sum() / p, True))

    steps = [RF.STAGE_LABEL[RF.A4_STAGES[k]] for k in ("cathode", "concentrate", "ore")]
    for y in RF.YEARS:
        for hi_, lo_ in zip(steps, steps[1:]):
            n_hi, n_lo = stage_name(hi_, y), stage_name(lo_, y)
            out.append((f"A4 {PER_YEAR}", f"{hi_} − {lo_} {y}", [n_hi, n_lo],
                        lambda x, p=prior[n_hi], q=prior[n_lo]: x[0] / p - x[1] / q, True))

    sec   = [f"b1|sec|{g}" for g in CMI.GROUPS]
    prim  = [f"b1|prim|{g}" for g in CMI.GROUPS]
    k     = len(sec)
    world = lambda x: x[:k].sum() / x.sum()                       # noqa: E731
    share = lambda x: x[0] / (x[0] + x[1])                        # noqa: E731
    pair  = lambda g: [f"b1|sec|{g}", f"b1|prim|{g}"]             # noqa: E731
    out.append(("B1", "world secondary share, 2020–24", sec + prim, world, False))
    out.append(("B1", "world − ICSG-implied", sec + prim,
                lambda x: world(x) - icsg_pool, False))
    for g in ("EU-14", "China", "United States"):
        out.append(("B1", f"{g} secondary share, 2020–24", pair(g), share, False))
    for g, h in (("EU-14", "China"), ("China", "United States")):
        out.append(("B1", f"{g} − {h} secondary share", pair(g) + pair(h),
                    lambda x: share(x[:2]) - share(x[2:]), False))
    out.append(("B1", "EU-14 / world", pair("EU-14") + sec + prim,
                lambda x: share(x[:2]) / world(x[2:]), False))

    by_step = {s: [n for n, _ in cols if n.startswith(f"b2|{s}|")] for s in RF.HHI_STEPS}
    for step, names in by_step.items():
        h_prior = hhi(np.array([prior[n] for n in names]))
        out.append(("B2", f"HHI {step} {LEVEL_YEAR}", names, hhi, False))
        out.append(("B2", f"HHI_post − HHI_prior {step}", names,
                    lambda x, h=h_prior: hhi(x) - h, False))
    m = len(by_step["mining"])
    for step in ("refining", "use"):
        out.append(("B2", f"HHI {step} − mining {LEVEL_YEAR}",
                    by_step["mining"] + by_step[step],
                    lambda x, m=m: hhi(x[m:]) - hhi(x[:m]), False))

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


def interval_row(row: str, quantity: str, value: float, sigma: float, scale: float,
                 quadrature: float = np.nan) -> dict:
    """One output row: σ, the √R-scaled σ and its two-sided interval."""
    half = Z_ALPHA * sigma * scale
    return dict(row=row, quantity=quantity, value=value, sigma=sigma,
                sigma_birge=sigma * scale, lo_birge=value - half, hi_birge=value + half,
                sigma_quadrature=quadrature)


def evaluate(qs: list, moments, scale: float) -> pd.DataFrame:
    """Value, σ and the √R-scaled interval of every quantity; ``sigma_quadrature``
    (linear rows only) adds the edges' σ_a* as if independent."""
    at   = {n: i for i, n in enumerate(moments.names)}
    rows = []
    for row, name, inputs, f, linear in qs:
        idx = [at[n] for n in inputs]
        x, C = moments.value[idx], moments.cov_post[np.ix_(idx, idx)]
        quad = np.nan
        if linear:
            g = np.array([f(x + e) - f(x) for e in np.eye(x.size)])
            quad = float(np.sqrt(((g * moments.sigma_quadrature[idx]) ** 2).sum()))
        rows.append(interval_row(row, name, f(x), delta_sigma(f, x, C), scale, quad))
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
    a_t  = edges.set_index("edge_id").post.reindex(cols_t).to_numpy(dtype=float)

    def f(delta: np.ndarray) -> float:
        t = tbl.copy()
        t["val"] = base + delta[pos]
        p = PSUT(t, psut.nodes, valuations=["CC"], primary="CC", acc_id=psut.acc_id)
        M, _ = RF.embodied(p, period)
        china = float(M.loc[["Chile", "Peru"], "China"].sum())
        return china / float((a_t + delta)[dpos].sum())
    return f


def intervals(result, qs: list, f_b3, scale: float, period: str) -> pd.DataFrame:
    """Every quantity's row from one Stage-5 result, B3's embodied ratio from its draws."""
    table = evaluate(qs, result.aggregates, scale)
    b3 = interval_row("B3", f"embodied / direct, China ← Chile+Peru {LEVEL_YEAR}",
                      f_b3(np.zeros(result.draws[period].shape[1])),
                      fd_sigma(f_b3, result.draws[period]), scale)
    return pd.concat([table, pd.DataFrame([b3])], ignore_index=True)


# =============================================================================
# Checks and the sensitivity family
# =============================================================================

def check_definitions(edges: pd.DataFrame, cols: list, qs: list) -> None:
    """The totals must give results_figs.py's own numbers on the posterior: the
    pooled secondary shares (world, EU-14, China, US), the 2024 HHIs, Chile's export
    share to China, the direct imports, the 30-year retentions and A4's per-year gaps.
    Raises otherwise."""
    post = {name: float(edges.post.to_numpy()[m].sum()) for name, m in cols}
    val  = {name: f(np.array([post[n] for n in inputs])) for _, name, inputs, f, _ in qs}
    cc   = edges[edges.valuation == "CC"]
    _, _, pooled = RF.secondary_shares(cc)
    h = RF.hhi_by_step(cc)
    h = h[(h.year == str(LEVEL_YEAR)) & (h.val == "post")].set_index("step")
    ret = RF.retention_by_stage(edges)
    ret = ret[ret.year == "all"].set_index("stage").retention
    a4  = RF.a4_gradient(edges)
    lab = {k: RF.STAGE_LABEL[RF.A4_STAGES[k]] for k in ("cathode", "concentrate", "ore")}
    ref = {"world secondary share, 2020–24": pooled.loc["World", "post"],
           f"direct concentrate, China ← Chile+Peru {LEVEL_YEAR} (kt CC)":
               RF.direct_conc_imports(cc, "China", ["Chile", "Peru"], LEVEL_YEAR),
           **{f"{g} secondary share, 2020–24": pooled.loc[g, "post"]
              for g in ("EU-14", "China", "United States")},
           **{f"HHI {s} {LEVEL_YEAR}": h.loc[s, "hhi_upper"] for s in RF.HHI_STEPS},
           **{f"Chile export share to China {y}": RF.chile_share_to_china(cc, y)
              for y in CHILE_YEARS},
           **{f"retention {s} 1995–2024": r for s, r in ret.items()},
           **{f"{lab['cathode']} − {lab['concentrate']} {y}": r
              for y, r in a4.cathode_minus_concentrate.items()},
           **{f"{lab['concentrate']} − {lab['ore']} {y}": r
              for y, r in a4.concentrate_minus_ore.items()}}
    bad = {k: (val[k], v) for k, v in ref.items() if not np.isclose(val[k], v, rtol=1e-9)}
    if bad:
        raise RuntimeError(f"aggregates differ from results_figs.py: {bad}")


def degenerate_family(edges: pd.DataFrame, const_run, n_edges: int) -> np.ndarray:
    """#99's family: cross-region Cathode→FU legs, either valuation, with a lower-bound
    row, a prior of 0 and a* within ``FRINGE_KT`` of it: the slack rule pins those
    below ``bind_tol`` and frees the rest, by noise."""
    rm  = const_run.row_meta
    lb  = np.flatnonzero(((rm["type"] == "bound") & (rm["variant"] == "lower")).to_numpy())
    has = np.zeros(n_edges, dtype=bool)
    has[const_run.G[lb].tocoo().col] = True
    m = ((edges.sector_from == "Cathode") & (edges.sector_to == "FU")
         & (edges.region_from != edges.region_to) & (edges.prior == 0.0)
         & (edges.post.abs() <= FRINGE_KT))
    out = np.zeros(n_edges, dtype=bool)
    out[edges.loc[m, "edge_id"].to_numpy()] = True
    return out & has


# =============================================================================
# Main
# =============================================================================

def main(opt: str, uncprop_id: str, out: Path = OUT_CSV,
         sensitivity: bool = SENSITIVITY) -> None:
    reg     = UncPropRunRegistry("data/proc/uncprop")
    sidecar = reg.load_sidecar(uncprop_id)
    up      = sidecar["parents"]["upstream"]
    cfg     = sidecar["stage_config"]
    if up["opt_run_id"] != opt:
        raise ValueError(f"{uncprop_id} propagates {up['opt_run_id']}, not {opt}")
    if cfg.get("exclude_scale_totals"):
        raise ValueError(f"{uncprop_id} is a scale-total ablation: its σ_a* is not the fit's")
    R = UncPropDiag.from_uncprop_id(uncprop_id).stats.birge_ratio()

    acc   = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                     storage_dir="data/proc/acc")
    edges = CMI.load_run(opt)
    cols  = aggregate_columns(edges)
    icsg_pool = float(RF.icsg_input_share(CMI.load_icsg()).loc[RF.POOL].mean())
    qs    = quantities(edges, cols, icsg_pool)
    check_definitions(edges, cols, qs)
    spec  = build_spec(edges, cols, acc.n_edges)
    print(f"{opt} · {uncprop_id} · Birge R {R:.3f} · {len(cols)} totals · "
          f"{len(qs) + 1} quantities")

    prop   = KalmanUncProp(**{k: cfg[k] for k in STAGE5_KNOBS})
    period = str(LEVEL_YEAR)
    run    = lambda **kw: prop.compute(                        # noqa: E731
        opt, ALPHA, aggregates=spec, draws={period: N_DRAWS},
        rng=np.random.default_rng(SEED), **kw)
    base   = run()
    stored = reg.load(uncprop_id).per_timestep[period].sigma_a_star
    if not np.allclose(base.per_timestep[period].sigma_a_star, stored, rtol=1e-8, atol=1e-12):
        raise RuntimeError(f"the in-memory Stage 5 does not reproduce {uncprop_id}'s σ_a* "
                           f"in {period}")
    cols_t = base.per_timestep[period].cols_t
    f_b3   = embodied_ratio_fn(opt, edges, cols_t, period)
    table  = intervals(base, qs, f_b3, np.sqrt(R), period)

    # The draws' check: their σ of a linear total against the exact one.
    direct = table.quantity.str.startswith("direct concentrate")
    pos    = positions(cols_t, edges.loc[direct_mask(edges), "edge_id"].to_numpy())
    s_fd   = float(np.sqrt(np.mean(base.draws[period][:, pos].sum(1) ** 2)))
    print(f"   draws check, direct imports σ: exact "
          f"{float(table.loc[direct, 'sigma'].iloc[0]):.3f} kt CC, "
          f"from {N_DRAWS} draws {s_fd:.3f}")

    if sensitivity:
        const = ConstraintRunRegistry("data/proc/const").load(up["const_run_id"])
        fam   = degenerate_family(edges, const, acc.n_edges)
        del const
        print(f"   degenerate family: {int(fam.sum()):,} edge-years")
        for tag, kw in (("pinned", {"force_pinned": fam}), ("free", {"force_free": fam})):
            alt = intervals(run(**kw), qs, f_b3, np.sqrt(R), period)
            table[f"sigma_birge_{tag}"] = table.quantity.map(
                alt.set_index("quantity").sigma_birge)

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(table[~table.row.str.endswith(PER_YEAR)].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("opt_run_id")
    ap.add_argument("uncprop_run_id")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()
    main(a.opt_run_id, a.uncprop_run_id, a.out, sensitivity=not a.no_sensitivity)
