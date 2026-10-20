"""
issue27_uncprop_active_set.py
=============================
The numbers of the #27 REP: the first copper Stage-5 run, propagated through
the QP's active set (ADR 0044). Read-only — it writes nothing.

1  per year: trace contraction, active-set counts (sidecar), max hard residual,
   and the invariant σ_a* ≤ σ_a on every edge (conditioned σ_a);
2  the pinned set's sensitivity to bind_tol: edges whose lower-bound row binds
   at tolerances 1e-8 … 1e-4 kt, and the kt they carry (Σ max(|a_prior|, |a*|));
3  posterior contraction κ by prior class × valuation (mass-weighted mean,
   mass share with κ ≥ 0.5) and the UncPropDiag quadrant summary;
4  the edges with the largest σ_a*, and the dependent hard rows by row type
   per year (re-derived with Stage 5's own reduction, read-only);
7  for LU_YEARS, κ recomputed as Stage 5 had it before the QR (an LU of S,
   d9dbc0a) against the run's persisted κ, and the kt of the edges they split on.

Usage (from the main checkout, which holds data/proc)::

    python scripts/explo/issue27_uncprop_active_set.py <uncprop_run_id>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pysuto._conditioning import condition_prior, condition_sigma_c
from pysuto.diag import UncPropDiag
from pysuto.uncprop import KalmanUncProp

TOLS     = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
LU_YEARS = ("1995", "2016")


def per_year(diag, sidecar) -> pd.DataFrame:
    f = diag.stats.edge_frame()
    tc = diag.stats.trace_contraction(f)
    rows = []
    for t, d in sidecar["stage_summary"]["per_timestep"].items():
        ft = f[f["year"] == t]
        ratio = ft["sigma_a_star"] / ft["sigma_a"]
        rows.append({
            "year": t, "trace_kappa": tc[t],
            "n_bound_active": d["n_bound_active"], "n_bands_binding": d["n_bands_binding"],
            "n_hard_dependent": d["n_hard_dependent"], "n_rows_kept": d["n_rows_kept"],
            "max_resid_hard": d["max_resid_hard"],
            "max_sigma_ratio": float(ratio.max()),
            "n_sigma_above_prior": int((ft["sigma_a_star"] > ft["sigma_a"] * (1 + 1e-12)).sum()),
        })
    return pd.DataFrame(rows)


def bind_tol_sensitivity(diag) -> pd.DataFrame:
    """Edges pinned at each tolerance, from the opt residuals of the lower-bound rows."""
    cr, f = diag.constraint_run, diag.stats.edge_frame()
    rm = cr.row_meta
    lb = KalmanUncProp._lower_bound_mask(rm)
    row_time = rm["time"].astype(str).to_numpy()
    rows = []
    for t, res in diag.opt_results.items():
        t = str(t)
        rows_t = np.flatnonzero(row_time == t)
        lb_t = lb[rows_t]
        G_lb = cr.G_blocks[t].tocsr()[np.flatnonzero(lb_t)].tocoo()
        resid = np.asarray(res.residuals, dtype=float)[np.flatnonzero(lb_t)][G_lb.row]
        cols = np.asarray(res.cols_t)[G_lb.col]
        w = f[f["year"] == t].set_index("edge_id")["weight"].reindex(cols).to_numpy()
        row = {"year": t}
        for tol in TOLS:
            m = resid >= -tol
            row[f"n@{tol:g}"] = int(m.sum())
            row[f"kt@{tol:g}"] = float(np.nansum(w[m]))
        rows.append(row)
    return pd.DataFrame(rows)


def kappa_by_class(diag) -> pd.DataFrame:
    f = diag.stats.edge_frame()
    f = f.assign(wk=f["weight"] * f["kappa"], informed=f["weight"] * (f["kappa"] >= 0.5))
    g = f.groupby(["valuation", "prior_class"], observed=True)
    out = g.agg(edge_years=("edge_id", "size"), weight_kt=("weight", "sum"),
                wk=("wk", "sum"), informed=("informed", "sum"),
                median_kappa=("kappa", "median"))
    out["kappa_w"] = out["wk"] / out["weight_kt"]
    out["share_informed"] = out["informed"] / out["weight_kt"]
    return out.drop(columns=["wk", "informed"]).reset_index()


def largest_sigma(diag, n: int = 8) -> pd.DataFrame:
    f = diag.stats.edge_frame()
    cols = ["year", "label", "prior_class", "a_prior", "a_post", "sigma_a", "sigma_a_star",
            "kappa"]
    return f.nlargest(n, "sigma_a_star")[cols]


def dependent_rows_by_type(diag, sidecar) -> pd.DataFrame:
    """Which rows Stage 5 found linearly dependent, per year and row type."""
    cfg = sidecar["stage_config"]
    prop = KalmanUncProp.__new__(KalmanUncProp)
    prop.bind_tol, prop.rank_tol = cfg["bind_tol"], cfg["rank_tol"]
    cr, f = diag.constraint_run, diag.stats.edge_frame()
    rm = cr.row_meta
    hard = rm["is_hard"].to_numpy().astype(bool)
    cone = rm["cone_type"].to_numpy()
    lb = KalmanUncProp._lower_bound_mask(rm)
    kind = (rm["type"].astype(str) + "/" + rm["cone_type"].astype(str)).to_numpy()
    row_time = rm["time"].astype(str).to_numpy()
    out = []
    for t, res in diag.opt_results.items():
        t = str(t)
        rows = np.flatnonzero(row_time == t)
        G_t = cr.G_blocks[t].tocsr()
        resid = np.asarray(res.residuals, dtype=float)
        binding = hard[rows] & (cone[rows] == "nonneg") & (resid >= -prop.bind_tol)
        pinned = f[f["year"] == t].set_index("edge_id")["bound_active"] \
                  .reindex(np.asarray(res.cols_t)).to_numpy()
        live = ~pinned
        keep = (hard[rows] & (cone[rows] == "zero")) | ~hard[rows] | (binding & ~lb[rows])
        keep &= np.asarray(abs(G_t) @ live.astype(float)).ravel() > 0.0
        dep = prop._dependent_hard_rows(G_t, live, np.flatnonzero(keep & hard[rows]))
        counts = pd.Series(kind[rows][dep]).value_counts().to_dict()
        out.append({"year": t, "n_dependent": int(dep.size),
                    "n_bound_active": int(pinned.sum()), **counts})
    return pd.DataFrame(out).fillna(0)


def lu_vs_qr(diag, sidecar, years=LU_YEARS, block: int = 2000) -> pd.DataFrame:
    """κ from an LU of S (Stage 5 at d9dbc0a) against the run's κ (QR, #27)."""
    from scipy.sparse.linalg import splu
    cfg = sidecar["stage_config"]
    prop = KalmanUncProp.__new__(KalmanUncProp)
    prop.bind_tol, prop.rank_tol = cfg["bind_tol"], cfg["rank_tol"]
    cr, f = diag.constraint_run, diag.stats.edge_frame()
    _, sa = condition_prior(cr.a0, cr.sigma_a, floor_abs=cfg["sigma_a_floor_abs"],
                            floor_rel=cfg["sigma_a_floor_rel"],
                            ceiling_factor=cfg["sigma_a_ceiling_factor"])
    rm = cr.row_meta
    hard = rm["is_hard"].to_numpy().astype(bool)
    cone = rm["cone_type"].to_numpy()
    lb = KalmanUncProp._lower_bound_mask(rm)
    row_time = rm["time"].astype(str).to_numpy()
    out = []
    for t in years:
        res, cols = diag.opt_results[t], cr.edge_cols_by_t[t]
        rows = np.flatnonzero(row_time == t)
        G_t = cr.G_blocks[t].tocsr()
        c_t = np.asarray(cr.timestep_data[t]["c"], dtype=float)[rows]
        sc = condition_sigma_c(np.asarray(cr.timestep_data[t]["sigma_c"], dtype=float)[rows],
                               floor=cfg["sigma_c_floor"], hard_mask=hard[rows])
        red = prop._reduce_at_active_set(G_t, c_t, sa[cols] ** 2, hard[rows], cone[rows],
                                         lb[rows], np.asarray(res.residuals, dtype=float),
                                         np.zeros(rows.size, dtype=bool))
        s2 = np.where(red.bound_active, 0.0, sa[cols] ** 2)
        live = np.flatnonzero(s2 > 0.0)
        G_k = G_t[red.keep].tocsc()
        G_l = G_k[:, live].tocsc()
        lu = splu(KalmanUncProp._build_S(G_k, s2, sc[red.keep] ** 2))
        k_lu = np.empty(live.size)
        for a in range(0, live.size, block):
            g = G_l[:, a:a + block].toarray()
            k_lu[a:a + block] = s2[live[a:a + block]] * np.einsum("mb,mb->b", g, lu.solve(g))
        ft = f[f["year"] == t].set_index("edge_id").reindex(cols[live])
        k_qr, w = ft["kappa"].to_numpy(), ft["weight"].to_numpy()
        err = np.abs(np.clip(k_lu, 0.0, 1.0) - k_qr)
        row = {"year": t, "n_live": int(live.size),
               "qr_kappa_min": float(k_qr.min()), "qr_kappa_max": float(k_qr.max()),
               "lu_kappa_lt0": int((k_lu < -1e-12).sum()),
               "lu_kappa_gt1": int((k_lu > 1.0 + 1e-12).sum()),
               "lu_kappa_min": float(k_lu.min())}
        for thr in (1e-6, 1e-3, 1e-1):
            m = err > thr
            row[f"n>{thr:g}"]  = int(m.sum())
            row[f"kt>{thr:g}"] = float(np.nansum(w[m]))
            row[f"max_kt>{thr:g}"] = float(np.nanmax(w[m])) if m.any() else 0.0
        out.append(row)
    return pd.DataFrame(out)


def main(uncprop_run_id: str) -> None:
    diag = UncPropDiag.from_uncprop_id(uncprop_run_id)
    from pysuto.registry import UncPropRunRegistry
    sidecar = UncPropRunRegistry("data/proc/uncprop").load_sidecar(uncprop_run_id)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    print("## 1 — per year\n")
    py = per_year(diag, sidecar)
    print(py.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n## 2 — bind_tol sensitivity of the pinned set\n")
    print(bind_tol_sensitivity(diag).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\n## 3 — κ by prior class × valuation (all years)\n")
    print(kappa_by_class(diag).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n## 4 — largest σ_a*\n")
    print(largest_sigma(diag).to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    print("\n## 5 — dependent hard rows by row type\n")
    dep = dependent_rows_by_type(diag, sidecar)
    print(dep.to_string(index=False))
    print(f"\ncorr(n_dependent, n_bound_active) = "
          f"{np.corrcoef(dep['n_dependent'], dep['n_bound_active'])[0, 1]:.3f}")
    print("\n## 6 — UncPropDiag summary\n")
    diag.summary()
    print("\n## 7 — κ from an LU of S (d9dbc0a) against this run's κ (QR)\n")
    print(lu_vs_qr(diag, sidecar).T.to_string(header=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("uncprop_run_id")
    main(ap.parse_args().uncprop_run_id)
