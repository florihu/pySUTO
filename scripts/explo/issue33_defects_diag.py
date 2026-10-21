"""
issue33_defects_diag.py
=======================
#33 measurement: the period QP solved for the defects d = a − a_prior at feas/gap 1e-8
(QP_VARIABLES = "defects", TOL_GAP_REL = TOL_GAP_ABS = 1e-8) against the flows
formulation at the ADR 0039 tolerances, on one constraint run — both opt runs solve the
same year-QPs, so their answers are scored on the same objective. Reuses #72's tables
(issue72_qdldl_diag: T per period, X cross-evaluation, E Round-E, C common set) and
adds, for <new> against <base>:

    B   Stage 5's bound-active edges per year (ADR 0044), when both opt runs have an
        uncprop run: the defects form's hard-row residuals (≤ 9e-7 kt on the probe) sit
        near Stage 5's BIND_TOL of 1e-6 kt
    V   #33's decision items, fixed before the run (2026-10-20, the maintainer accepted
        the residual floor and the tolerance change as one measured change):
          1  every period Solved (not AlmostSolved)
          2  Σ_t J(new) < Σ_t J(base), lower in every period — the base is certified
             off its optimum by 945–4,363 per probed year (probe of const_20261019_
             170151_13460e94, docs/rep/figs/issue33/)
          3  hard-row violations ≤ 1e-6 kt (zero cone and nonneg excess)
          4  Stage 4 faster
          5  every year keeps ≥ 90 % of the base's bound-active edges in Stage 5
        Round-E (ADR 0025) is reported, not a gate: it judges σ and repricing changes,
        and here the base itself is the inexact answer.

    python scripts/explo/issue33_defects_diag.py <opt_base> <opt_new> [<uncprop_base> <uncprop_new>]
"""
from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue72_qdldl_diag as Q                            # noqa: E402

VIOL_FLOOR = 1e-6                                         # kt
KEEP_BOUND = 0.90                                         # item 5


def bound_active(uncprop_base: str, uncprop_new: str) -> pd.DataFrame:
    """B: n_bound_active per year from both uncprop sidecars."""
    def per_year(u: str) -> pd.Series:
        with open(f"data/proc/uncprop/{u}/metadata.json", encoding="utf-8") as f:
            per = json.load(f)["stage_summary"]["per_timestep"]
        return pd.Series({t: v["n_bound_active"] for t, v in per.items()}, dtype=float)
    b = pd.DataFrame({"base": per_year(uncprop_base), "new": per_year(uncprop_new)})
    b["ratio"] = b["new"] / b["base"]
    print(f"\n== B Stage 5 bound-active edges per year, {uncprop_base} / {uncprop_new}")
    print(b.round(3).to_string())
    print(f"   Σ {b['base'].sum():,.0f} → {b['new'].sum():,.0f}; min ratio "
          f"{b['ratio'].min():.3f} ({b['ratio'].idxmin()})")
    return b


def verdict(base: str, new: str, phases: pd.DataFrame, x: pd.DataFrame,
            b: pd.DataFrame | None) -> None:
    """V: #33's pre-registered decision items."""
    per_new = Q.I71._periods(Q.I71._meta(new))
    items = [
        ("1 every period Solved", bool((per_new.solver_status == "Solved").all())),
        (f"2 Σ_t J lower ({x['ΔJ'].sum():+,.1f}), in every period "
         f"({int((x['ΔJ'] < 0).sum())}/{len(x)})", bool((x["ΔJ"] < 0).all())),
        (f"3 hard violations ≤ {VIOL_FLOOR:g} kt (zero {x['zero_v new'].max():.2e}, "
         f"nonneg {x['nn_v new'].max():.2e})",
         max(x["zero_v new"].max(), x["nn_v new"].max()) <= VIOL_FLOOR),
        (f"4 Stage 4 faster ({phases.loc[base, 'total']:.0f} → "
         f"{phases.loc[new, 'total']:.0f} s)",
         phases.loc[new, "total"] < phases.loc[base, "total"]),
    ]
    if b is not None:
        items.append((f"5 bound-active edges ≥ {KEEP_BOUND:.0%} of base in every year "
                      f"(min {b['ratio'].min():.3f})", bool((b["ratio"] >= KEEP_BOUND).all())))
    print(f"\n== V #33 decision items, {new} against {base}")
    for name, ok in items:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    if b is None:
        print("   (item 5 pending: pass both uncprop run ids)")


def main(base: str, new: str, ub: str | None, un: str | None) -> None:
    _, phases = Q.table(base, new)
    x = Q.crosseval(base, new)
    Q.round_e(base, new)
    b = bound_active(ub, un) if ub and un else None
    verdict(base, new, phases, x, b)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:5] if len(sys.argv) == 5 else (None, None)))
