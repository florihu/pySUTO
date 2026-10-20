"""
issue72_qdldl_diag.py
=====================
#72 measurement: Clarabel's linear solver for the period QPs, qdldl against faer, on one
constraint run — both opt runs solve the same year-QPs, so their answers can be scored on
the same objective. Prints, for <new> against <base>:

    T   per period: status, objective (and Δ), iterations, solve time; Stage 4's phases
    X   cross-evaluation (ADR 0039): both posteriors scored on the same year-QP,
        J = Σ((F − a₀)/σ_a)² + Σ_free z², its soft-row part Σz², and each posterior's
        hard-row violations (max |zero-cone residual|, max nonneg excess, kt)
    E   the Round-E items (ADR 0025) and z² by anchor band
    C   common-set sum-row z²
    V   #72's decision items, pre-registered before the run: every period Solved or
        AlmostSolved; Σ_t J(new) ≤ Σ_t J(base); hard-row violations no worse in
        practice (max ≤ max(10 × base's, 1e-6 kt)); Round-E (a) no world retention falls
        by more than 0.01, (b) the small-node median moves by at most 0.05, (c) no
        exporter retention moves by more than 0.01, (d) the |c| ≥ 100 kt bands' z² moves
        by less than 1 %; Stage 4 faster

    python scripts/explo/issue72_qdldl_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue71_parallel_resolve_check as I71              # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
import roundZ_png_kg_diag as Z                            # noqa: E402
from pysuto.solver import ClarabelSolver                  # noqa: E402

VIOL_FLOOR = 1e-6                                         # kt; "no worse in practice"


def table(base: str, new: str) -> tuple[bool, pd.DataFrame]:
    """T: per period status, objective, iterations, solve time; Stage 4's phases."""
    print("== T sidecars")
    for o in (base, new):
        I71.sidecar(o)
    pb, pn = I71._periods(I71._meta(base)), I71._periods(I71._meta(new))
    per = pd.DataFrame({
        "status":     pb.solver_status + " / " + pn.solver_status,
        "obj base":   pb.objective.astype(float),
        "obj new":    pn.objective.astype(float),
        "Δobj":       pn.objective.astype(float) - pb.objective.astype(float),
        "iterations": pb.iterations.astype(str) + " / " + pn.iterations.astype(str),
        "time base":  pb.solve_time_s.astype(float),
        "time new":   pn.solve_time_s.astype(float),
    }).sort_index()
    print(f"\n== T per period, {base} / {new}")
    print(per.round(1).to_string())
    d = per["Δobj"]
    print(f"   Δobj (new − base): Σ {d.sum():+,.1f}, min {d.min():+,.1f} ({d.idxmin()}), "
          f"max {d.max():+,.1f} ({d.idxmax()}), lower in {int((d < 0).sum())}/{len(d)} "
          f"periods; Σ solve_time_s {per['time base'].sum():,.0f} / "
          f"{per['time new'].sum():,.0f}")
    phases = pd.DataFrame({base: I71._phases(base), new: I71._phases(new)}).T
    print("\n== T Stage 4 phases, s (from the runs' solver logs)")
    print(phases.round(1).to_string())
    return bool(pn.solver_status.isin(I71.OK).all()), phases


def crosseval(base: str, new: str) -> pd.DataFrame:
    """X: both posteriors on the same year-QP, in the solver's conditioning."""
    const = L._chain(base)[0]
    if L._chain(new)[0] != const:
        raise SystemExit(f"{new} and {base} solve different constraint runs")
    solver = ClarabelSolver()
    cr = solver.load_run(const)
    rows = []
    for t in sorted(cr.edge_cols_by_t):
        qp = Z._qp_year(cr, solver, t)
        free = qp["ct"] == "free"
        for o in (base, new):
            z = np.load(Z.ROOT / "opt" / o / "timesteps" / f"t_{t}.npz", allow_pickle=True)
            if not np.array_equal(np.asarray(z["cols_t"], dtype=np.int64), qp["cols"]):
                raise SystemExit(f"{o} {t}: cols_t differs from the const run's")
            F = z["F_opt"]
            J, zero_v, nn_v = Z._J(qp, F)
            r = qp["G"] @ F - qp["c"]
            rows.append({"period": t, "run": o, "J": J,
                         "z2": float(((r[free] / qp["sc"][free]) ** 2).sum()),
                         "zero_v": zero_v, "nn_v": nn_v})
    x = pd.DataFrame(rows).pivot(index="period", columns="run")
    out = pd.DataFrame({
        "J base":       x[("J", base)],
        "ΔJ":           x[("J", new)] - x[("J", base)],
        "Σz² base":     x[("z2", base)],
        "ΔΣz²":         x[("z2", new)] - x[("z2", base)],
        "zero_v base":  x[("zero_v", base)],
        "zero_v new":   x[("zero_v", new)],
        "nn_v base":    x[("nn_v", base)],
        "nn_v new":     x[("nn_v", new)],
    })
    print(f"\n== X cross-evaluation on {const}: J and Σz² of {base}, Δ = {new} − {base}")
    with pd.option_context("display.float_format", "{:,.4g}".format):
        print(out.to_string())
    jb, dj = out["J base"].sum(), out["ΔJ"].sum()
    zb, dz = out["Σz² base"].sum(), out["ΔΣz²"].sum()
    print(f"   Σ_t J {jb:,.1f} → {jb + dj:,.1f} ({dj:+,.1f}, {100 * dj / jb:+.4f} % vs {base});"
          f" lower in {int((out['ΔJ'] < 0).sum())}/{len(out)} periods\n"
          f"   Σ_t Σz² (soft rows) {zb:,.1f} → {zb + dz:,.1f} ({100 * dz / zb:+.4f} % vs {base})"
          f"\n   max hard violation, kt: zero {out['zero_v base'].max():.2e} / "
          f"{out['zero_v new'].max():.2e}, nonneg {out['nn_v base'].max():.2e} / "
          f"{out['nn_v new'].max():.2e}")
    return out


def round_e(base: str, new: str) -> tuple[dict, pd.DataFrame]:
    """E + C: the Round-E items for both runs, and the common-set sum-row z²."""
    E, S, j = P69.load(base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    return {o: Z._round_e(E[o], S[o]) for o in (base, new)}, j


def verdict(base: str, new: str, solved: bool, phases: pd.DataFrame, x: pd.DataFrame,
            rete: dict, j: pd.DataFrame) -> None:
    """V: #72's pre-registered decision items."""
    wr_b, med_b, _, exp_b, _ = rete[base]
    wr_n, med_n, _, exp_n, _ = rete[new]
    big = j.c_base.abs() >= 100
    zb, zn = j.z2_base[big].sum(), j.z2_new[big].sum()
    viol = {k: max(x[f"{k} new"].max() - max(10 * x[f"{k} base"].max(), VIOL_FLOOR), 0.0)
            for k in ("zero_v", "nn_v")}
    items = [
        ("every period Solved/AlmostSolved", solved),
        ("Σ_t J(new) ≤ Σ_t J(base)", x["ΔJ"].sum() <= 0.0),
        ("hard violations no worse in practice", all(v == 0.0 for v in viol.values())),
        ("(a) no world retention falls > 0.01",
         all(wr_n[k] - wr_b[k] >= -0.01 for k in wr_b)),
        ("(b) small-node median moves ≤ 0.05", abs(med_n - med_b) <= 0.05),
        ("(c) exporter retention moves ≤ 0.01",
         all(abs(exp_n[k] - exp_b[k]) <= 0.01 for k in L.EXPORTERS)),
        (f"(d) |c| ≥ 100 kt z² moves < 1 % ({100 * (zn / zb - 1):+.3f} %)",
         abs(zn / zb - 1) < 0.01),
        (f"Stage 4 faster ({phases.loc[base, 'total']:.0f} → "
         f"{phases.loc[new, 'total']:.0f} s)",
         phases.loc[new, "total"] < phases.loc[base, "total"]),
    ]
    print(f"\n== V #72 decision items, {new} against {base}")
    for name, ok in items:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    print(f"   → {'adopt qdldl' if all(ok for _, ok in items) else 'rejected on measurement'}")


def main(base: str, new: str) -> None:
    solved, phases = table(base, new)
    x = crosseval(base, new)
    rete, j = round_e(base, new)
    verdict(base, new, solved, phases, x, rete, j)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
