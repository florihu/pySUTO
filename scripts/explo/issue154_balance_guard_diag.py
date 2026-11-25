# Frozen record: last ran against commit fd58e9a; rerun by checking out that commit (see README.md).
"""
issue154_balance_guard_diag.py
==============================
#154 measurement of #153: a primary a relaxed close counted absent (ADR 0010) is never
re-derived, neither by a Flow balance nor by ``network_backward`` in a later cascade pass.
Reads persisted run artefacts only and prints #154's pre-registered items for <new>
against <base>:

    P   prior: the assumed-absent primaries and who fills them in each run; the edges
        whose f_agg / σ_agg move, split into those primaries, their Mass twins and the
        rest; network log rows per pass (item 1, P1, P2)
    Y   solve status per year (item 2, P4)
    E   the Round-E items (ADR 0025) (item 3, P4)
    C   common-set sum-row z² by year and region, z² by row family, attribution (item 4, P3)
    M   mines: CO→Mining CC prior/posterior by region, and the base's pass-2 network fills
        by sector pair — the phantom chains behind item 4

    python scripts/explo/issue154_balance_guard_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure                  # noqa: E402
from pysuto.config import load_config                       # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic   # noqa: E402
from pysuto.relation.param_relation import expand_edge_index  # noqa: E402
import issue69_aggregate_gap_diag as P69                    # noqa: E402

ROOT       = Path("data/proc")
IMP_TAGGED = "imp_20261112_154659_f4ec08ae"  # bit-identical to the base imp, #150's tags


def _sidecar(stage: str, run: str) -> dict:
    with open(ROOT / stage / run / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _imp_of(opt: str) -> str:
    const = _sidecar("opt", opt)["parents"]["upstream"]["const_run_id"]
    return _sidecar("const", const)["parents"]["upstream"]["imp_run_id"]


def years(opt: str) -> None:
    """Y: solve status per year."""
    ss = _sidecar("opt", opt)["stage_summary"]
    st = pd.Series({t: v["solver_status"] for t, v in ss["per_timestep"].items()})
    print(f"\n== Y {opt}: {st.value_counts().to_dict()} over {len(st)} years; "
          f"solve_wall_s {ss['solve_wall_s']:.0f}")


def _log(imp: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")


def _absent(log: pd.DataFrame) -> set:
    """Edge ids a relaxed close counted as 0."""
    return {int(i) for lst in log.loc[log.relaxed, "assumed_absent"] for i in lst}


def _twins(ei: pd.DataFrame, edges: set) -> set:
    """The Mass edge of each content edge in ``edges`` (same from, to and time)."""
    key = ["from_id", "to_id", "time"]
    cc  = ei[ei.edge_id.isin(edges)][key]
    m   = ei[ei.valuation == "Mass"].merge(cc, on=key)
    return set(m.edge_id.astype(int))


def prior(ei: pd.DataFrame, imp_base: str, imp_new: str) -> None:
    """P: who fills the assumed-absent primaries, and which edges the prior moves on."""
    print(f"\n== P prior, {imp_base} → {imp_new}")
    lt, lb, ln = _log(IMP_TAGGED), _log(imp_base), _log(imp_new)
    ab, an = _absent(lt), _absent(ln)
    print(f"   assumed-absent primaries: {len(ab)} ({IMP_TAGGED}), {len(an)} (new); "
          f"in both {len(ab & an)}, base only {len(ab - an)}, new only {len(an - ab)}")
    for name, lg, s in (("base", lt, ab), ("new", ln, an)):
        x = lg[lg.edge_id.isin(s)]
        g = (x.assign(nz=x.f_after != 0, pass_=x["pass"].astype(str))
             .groupby(["imputer_id", "method", "pass_"])
             .agg(n=("edge_id", "size"), nonzero=("nz", "sum"), kt=("f_after", "sum")))
        print(f"   {name}: who fills them ({len(set(x.edge_id))} of {len(s)} written)")
        print(g.round(2).to_string())
    a = np.load(ROOT / "imp" / imp_base / "agg_result.npz", allow_pickle=True)
    b = np.load(ROOT / "imp" / imp_new / "agg_result.npz", allow_pickle=True)
    print("   coverage: " + ("bit-identical" if np.array_equal(a["coverage"], b["coverage"])
                             else "DIFFERS"))
    tw = _twins(ei, ab)
    for k in ("f_agg", "sigma_agg"):
        x, y = a[k], b[k]
        moved = np.where(~((x == y) | (np.isnan(x) & np.isnan(y))))[0]
        s = set(moved.tolist())
        rest = s - ab - tw
        dx = np.nansum(np.abs(y[moved] - x[moved]))
        print(f"   {k}: {len(s):,} edges move — primaries {len(s & ab)}, their Mass twins "
              f"{len(s & tw)}, other {len(rest):,}; Σ|Δ| {dx:,.1f}")
    fb = a["f_agg"][sorted(ab)]
    print(f"   base value of the {len(ab)} primaries in f_agg: 0 on {int(np.sum(fb == 0))}, "
          f"nonzero on {int(np.sum(fb != 0))}")
    for name, lg in (("base", lb), ("new", ln)):
        n = lg[lg.imputer_id == "network"]
        per = n.groupby(n["pass"].astype(int)).agg(n=("edge_id", "size"),
                                                   nonzero=("f_after", lambda v: (v != 0).sum()))
        print(f"   network rows per pass, {name}: "
              + "  ".join(f"p{p} {r.n:,} ({r.nonzero:,} ≠ 0)" for p, r in per.iterrows()))
    oth = ln[~ln.edge_id.isin(set(lb.edge_id))]
    print(f"   edges the new log writes and the base log does not: {len(oth):,}, by imputer "
          f"{oth.imputer_id.value_counts().to_dict()}")
    gone = lb[~lb.edge_id.isin(set(ln.edge_id))]
    print(f"   edges the base log writes and the new log does not: {len(gone):,}, by method "
          f"{gone.method.value_counts().to_dict()}")


def _frame(opt: str, acc) -> pd.DataFrame:
    d  = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    cf = cf[~cf.is_hard.astype(bool)]
    fam = cf["type"].astype(str).where(cf["type"].astype(str) != "ratio",
                                       "ratio:" + cf["variant"].astype(str))
    return pd.DataFrame({"constraint_id": cf.constraint_id.astype(str).to_numpy(),
                         "year": cf.year.astype(str).to_numpy(), "family": fam.to_numpy(),
                         "z2": (cf.z ** 2).to_numpy()})


def families(acc, base: str, new: str) -> None:
    """C: common-set z² of every soft row, by row family."""
    k = ["constraint_id", "year"]
    j = _frame(base, acc).merge(_frame(new, acc)[k + ["z2"]], on=k,
                                suffixes=("_base", "_new"))
    g = j.groupby("family")[["z2_base", "z2_new"]].sum()
    g["d"] = g.z2_new - g.z2_base
    g["pct"] = 100 * g.d / g.z2_base
    print(f"\n== C z² of every soft row by family, {len(j):,} common rows "
          f"(% vs {base}):")
    print(g.round(3).to_string())


def _post(opt: str) -> pd.DataFrame:
    rows = []
    for y in range(1995, 2025):
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    return pd.concat(rows, ignore_index=True)


def mines(ei: pd.DataFrame, base: str, new: str) -> None:
    """M: mine output by region in both runs, and what the base's pass 2 wrote."""
    j = (_post(base).merge(_post(new), on="edge_id", suffixes=("_b", "_n"))
         .merge(ei, on="edge_id"))
    m = j[(j.valuation == "CC") & (j.sector_from == "CO") & (j.sector_to == "Mining")]
    g = m.groupby("region_to")[["prior_b", "post_b", "prior_n", "post_n"]].sum()
    g["d_post"] = g.post_n - g.post_b
    print("\n== M CO→Mining CC, Σ 1995–2024 (kt CC), regions whose posterior moves > 5 kt:")
    print(g[g.d_post.abs() > 5].sort_values("d_post").round(1).to_string())
    print("   world " + "  ".join(f"{k} {v:,.0f}" for k, v in g.sum().items()))
    lt = _log(IMP_TAGGED)
    p2 = lt[(lt.imputer_id == "network") & (lt["pass"] == 2) & (lt.f_after != 0)]
    p2 = p2.merge(ei[["edge_id", "sector_from", "sector_to", "valuation", "region_to"]],
                  on="edge_id")
    p2 = p2[p2.valuation == "CC"]
    print(f"   base pass-2 nonzero network CC fills ({IMP_TAGGED}), by sector pair:")
    print(p2.groupby(["sector_from", "sector_to"]).f_after.agg(["size", "sum"])
          .sort_values("sum").round(1).to_string())
    co = p2[(p2.sector_from == "CO") & (p2.sector_to == "Mining")]
    print(f"   of them CO→Mining: {len(co)} region-years, {co.f_after.sum():,.1f} kt CC, "
          f"{co.region_to.nunique()} regions")


def main(base: str, new: str) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    prior(ei, _imp_of(base), _imp_of(new))
    years(base)
    years(new)
    E, S, j = P69.load(base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    families(acc, base, new)
    P69.attribution(j)
    mines(ei, base, new)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
