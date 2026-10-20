"""
issue33_conditioning_probe.py
=============================
#33 prototype: does a reformulated period QP let Clarabel converge at 1e-8? Loads one
constraint run read-only and solves a handful of year-QPs under

    0   the incumbent: x = [a; ε], P = 2·diag(1/σ_a², 1/σ_c²), q = −2·a₀/σ_a²; Clarabel's
        cost is J − Σ(a₀/σ_a)², so the relative gap is measured against that constant
    A   centred:  x = [d; ε] with a = a₀ + d; same P, q = 0, b = c − G·a₀; cost = J
    B   whitened: x = [z; e] with a = a₀ + σ_a·z, ε = σ_c·e; P = 2·I, q = 0,
        A = [G·diag(σ_a) | −diag(σ_c); 0], b = c − G·a₀; cost = J

at the production tolerances (``scripts/run_solver.py``'s TOL_* block) and at the strict
target (feas, gap_abs and gap_rel 1e-8, the reduced tolerances unchanged). Formulation 0
at the production tolerances runs through ``ClarabelSolver._solve_period`` itself and must
give the stored posterior of the opt run bit for bit; every other solve uses this script's
builder, whose formulation-0 matrices are first checked equal to the ones
``_solve_period`` hands Clarabel (captured, not solved), with the settings of
``ClarabelSolver._period_settings``. Every posterior, including the stored one, is scored
outside the solver on the same year-QP (ADR 0039):

    J      = Σ((F − a₀)/σ_a)² + Σ_free((G·F − c)/σ_c)²
    zero   = max |G·F − c| on the zero-cone (hard equality) rows, kt
    band   = max (G·F − c)⁺ on the hard ratio-band rows, kt
    lb     = max (G·F − c)⁺ on the lower-bound rows (= max −F on bounded edges), kt
    dF     = max |F − F_stored|, kt
    LB     = the Lagrange dual of the year-QP at the solve's multipliers: a certified
             lower bound on min J (see :func:`dual_bound`); the table's J − LB* bounds
             how far a posterior is from the optimum, LB* the year's best bound

Writes ``probe.csv`` and one ``F_<year>_<form>_<tol>.npz`` (F, ε, multipliers) per solve
under ``--out``; nothing is registered and nothing is written under the data root.
``--report`` reprints the table of an earlier ``probe.csv``.

    PYTHONPATH=src python scripts/explo/issue33_conditioning_probe.py \\
        --data-root /home/flhuber/Projects/pySUTO/data/proc --out <dir> \\
        [--years 1995 2008 …] [--forms 0 A B] [--tols prod tight] [--threads 4]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import clarabel                                           # noqa: E402

import pysuto.solver as solver_mod                        # noqa: E402
import run_solver as RS                                   # noqa: E402
from pysuto.solver import ClarabelSolver                  # noqa: E402

CONST_RUN_ID = "const_20261019_170151_13460e94"
OPT_RUN_ID   = "opt_20261019_171746_3272682b"
YEARS        = ["1995", "2008", "2009", "2015", "2016", "2022"]
FORMS        = ["0", "A", "B"]
TOLS         = {
    "prod":  dict(tol_feas=RS.TOL_FEAS, tol_gap_rel=RS.TOL_GAP_REL,
                  tol_gap_abs=RS.TOL_GAP_ABS),
    "tight": dict(tol_feas=1e-8, tol_gap_rel=1e-8, tol_gap_abs=1e-8),
}


# =============================================================================
# The year-QP and its formulations
# =============================================================================

def year_qp(cr, solver: ClarabelSolver, t: str) -> dict:
    """One year's QP in the solver's conditioning, rows in their original order."""
    a0, sa = solver._condition_prior(cr.a_prior_global, cr.sigma_a)
    cols   = np.asarray(cr.edge_cols_by_t[t])
    rows   = np.where(cr.row_meta["time"].astype(str).to_numpy() == t)[0]
    rm     = cr.row_meta.iloc[rows]
    sc     = np.asarray(cr.timestep_data[t]["sigma_c"], dtype=float)[rows]
    sc     = np.maximum(np.where(np.isnan(sc), solver.sigma_c_floor, sc),
                        solver.sigma_c_floor)
    return dict(cols=cols, rows=rows, G=cr.G_blocks[t].tocsr(),
                ct=rm["cone_type"].to_numpy(), typ=rm["type"].to_numpy(),
                c=np.asarray(cr.timestep_data[t]["c"], dtype=float)[rows],
                sc=sc, a0=a0[cols], sa=sa[cols])


def build(qp: dict, form: str) -> dict:
    """Clarabel data (P, q, A, b, cones) of one formulation and its map back to (F, ε)."""
    ct = qp["ct"]
    idx_free, idx_zero, idx_nonneg = (np.where(ct == k)[0] for k in ("free", "zero", "nonneg"))
    perm   = np.concatenate([idx_free, idx_zero, idx_nonneg])
    n_free = idx_free.size
    G_p    = qp["G"][perm]
    n_a, m = G_p.shape[1], G_p.shape[0]
    S      = sp.vstack([-sp.eye(n_free, format="csr"),
                        sp.csr_matrix((m - n_free, n_free))], format="csr")
    sa, a0 = qp["sa"], qp["a0"]
    scf    = qp["sc"][perm[:n_free]]
    b      = qp["c"][perm]
    if form == "0":
        inv_sa = 1.0 / sa ** 2
        P      = sp.diags(2.0 * np.concatenate([inv_sa, 1.0 / scf ** 2]), format="csc")
        q      = np.concatenate([-2.0 * inv_sa * a0, np.zeros(n_free)])
        A      = sp.hstack([G_p, S], format="csc")
        unmap  = lambda x: (x[:n_a], x[n_a:])                         # noqa: E731
    elif form == "A":
        P      = sp.diags(2.0 * np.concatenate([1.0 / sa ** 2, 1.0 / scf ** 2]), format="csc")
        q      = np.zeros(n_a + n_free)
        A      = sp.hstack([G_p, S], format="csc")
        b      = b - G_p @ a0
        unmap  = lambda x: (a0 + x[:n_a], x[n_a:])                    # noqa: E731
    elif form == "B":
        P      = sp.diags(np.full(n_a + n_free, 2.0), format="csc")
        q      = np.zeros(n_a + n_free)
        A      = sp.hstack([G_p @ sp.diags(sa), S @ sp.diags(scf)], format="csc")
        b      = b - G_p @ a0
        unmap  = lambda x: (a0 + sa * x[:n_a], scf * x[n_a:])         # noqa: E731
    else:
        raise ValueError(f"unknown formulation {form!r}")
    cones = [k(n) for k, n in ((clarabel.ZeroConeT, n_free),
                               (clarabel.ZeroConeT, idx_zero.size),
                               (clarabel.NonnegativeConeT, idx_nonneg.size)) if n > 0]
    return dict(P=P, q=q, A=A, b=b, cones=cones, unmap=unmap)


def score(qp: dict, F: np.ndarray) -> dict:
    """J and the hard-row violations of a posterior, outside the solver."""
    r     = qp["G"] @ F - qp["c"]
    free  = qp["ct"] == "free"
    zero  = qp["ct"] == "zero"
    bound = (qp["ct"] == "nonneg") & (qp["typ"] == "bound")
    band  = (qp["ct"] == "nonneg") & (qp["typ"] != "bound")
    prior = float((((F - qp["a0"]) / qp["sa"]) ** 2).sum())
    soft  = float(((r[free] / qp["sc"][free]) ** 2).sum())
    return dict(J=prior + soft, J_prior=prior, J_soft=soft,
                zero=float(np.abs(r[zero]).max(initial=0.0)),
                band=float(r[band].max(initial=0.0)),
                lb=float(r[bound].max(initial=0.0)))


def dual_bound(qp: dict, z: np.ndarray) -> float:
    """A certified lower bound on the year-QP's optimal J from any multipliers ``z``.

    All three formulations share the rows (in the order of :func:`build`), so one
    Lagrange dual scores them all. In the centred form (P = 2·diag(1/σ²) over
    [σ_a; σ_c_free], q = 0, b = c − G·a₀) the dual function is

        g(z) = −¼ Σ_j σ_j² (Aᵀz)_j² − bᵀz,

    a lower bound on min J for every z with a non-negative nonneg-cone part, whatever
    the solver's residuals: z is taken from the solve, clipped into the dual cone,
    and the minimising x is recomputed from it rather than read from the solve.
    """
    d   = build(qp, "A")
    n_n = int((qp["ct"] == "nonneg").sum())
    z   = np.array(z, dtype=float, copy=True)
    if n_n:
        z[-n_n:] = np.maximum(z[-n_n:], 0.0)
    sig = np.concatenate([qp["sa"], qp["sc"][qp["ct"] == "free"]])
    v   = d["A"].T @ z
    return -0.25 * math.fsum((sig * v) ** 2) - math.fsum(d["b"] * z)


# =============================================================================
# The incumbent's own code path
# =============================================================================

class _Captured(Exception):
    def __init__(self, args):
        super().__init__("captured")
        self.qp_args = args


def capture_incumbent(solver: ClarabelSolver, cr, t: str, a0_g, inv_sa_g, row_time):
    """The (P, q, A, b, cones) ``_solve_period`` hands Clarabel, without solving."""
    real = solver_mod.clarabel

    def _fake(*args):
        raise _Captured(args)

    solver_mod.clarabel = types.SimpleNamespace(
        DefaultSolver=_fake, DefaultSettings=real.DefaultSettings,
        ZeroConeT=real.ZeroConeT, NonnegativeConeT=real.NonnegativeConeT,
    )
    try:
        cols = cr.edge_cols_by_t[t]
        solver._solve_period(cr, t, a0_g[cols].copy(), inv_sa_g[cols], row_time)
    except _Captured as e:
        return e.qp_args
    finally:
        solver_mod.clarabel = real
    raise RuntimeError("_solve_period did not construct a Clarabel solver")


def _same_csc(X, Y) -> bool:
    X, Y = sp.csc_matrix(X), sp.csc_matrix(Y)
    return (X.shape == Y.shape and np.array_equal(X.indptr, Y.indptr)
            and np.array_equal(X.indices, Y.indices) and np.array_equal(X.data, Y.data))


def check_builder(solver, cr, qp, t, a0_g, inv_sa_g, row_time) -> None:
    """Formulation 0 of :func:`build` must be ``_solve_period``'s QP exactly."""
    P, q, A, b, cones, _ = capture_incumbent(solver, cr, t, a0_g, inv_sa_g, row_time)
    mine = build(qp, "0")
    ok = (_same_csc(P, mine["P"]) and np.array_equal(q, mine["q"])
          and _same_csc(A, mine["A"]) and np.array_equal(b, mine["b"])
          and [repr(k) for k in cones] == [repr(k) for k in mine["cones"]])
    if not ok:
        raise SystemExit(f"{t}: the probe's formulation 0 differs from _solve_period's QP")


# =============================================================================
# Solves
# =============================================================================

def solve_built(qp: dict, form: str, settings):
    """One formulation's solve: (F, ε, z, convergence, wall seconds)."""
    d  = build(qp, form)
    t0 = time.perf_counter()
    s  = clarabel.DefaultSolver(d["P"], d["q"], d["A"], d["b"], d["cones"], settings)
    sol = s.solve()
    wall = time.perf_counter() - t0
    F, eps = d["unmap"](np.asarray(sol.x))
    return F, eps, np.asarray(sol.z), ClarabelSolver._info_to_dict(s.get_info()), wall


class _Recording:
    """A Clarabel solver that keeps its last solution, so ``_solve_period``'s
    multipliers can be scored; every other call goes to the wrapped instance."""

    def __init__(self, inner):
        self._inner, self.sol = inner, None

    def solve(self):
        self.sol = self._inner.solve()
        return self.sol

    def __getattr__(self, name):
        return getattr(self._inner, name)


def solve_incumbent(solver, cr, t, a0_g, inv_sa_g, row_time):
    """Formulation 0 through ``ClarabelSolver._solve_period`` itself."""
    cols = cr.edge_cols_by_t[t]
    t0 = time.perf_counter()
    res, rec = solver._solve_period(cr, t, a0_g[cols].copy(), inv_sa_g[cols], row_time)
    return (res.F_opt, res.eps_opt, np.asarray(rec.sol.z), res.convergence,
            time.perf_counter() - t0)


def main(data_root: Path, out: Path, years: list, forms: list, tols: list,
         threads: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"pysuto from {solver_mod.__file__}")
    solvers = {
        k: ClarabelSolver(
            const_dir=str(data_root / "const"), opt_dir=str(out / "opt_unused"),
            reduced_tol_feas=RS.REDUCED_TOL_FEAS, reduced_tol_gap_rel=RS.REDUCED_TOL_GAP_REL,
            reduced_tol_gap_abs=RS.REDUCED_TOL_GAP_ABS, max_iter=RS.MAX_ITER,
            prior_mode="cold", n_workers=1, direct_solve_method=RS.DIRECT_SOLVE_METHOD,
            verbose_first_solve=False, **TOLS[k],
        ) for k in tols
    }
    s0 = solvers[tols[0]]
    cr = s0.load_run(CONST_RUN_ID)
    a0_g, sa_g = s0._condition_prior(cr.a_prior_global, cr.sigma_a)
    inv_sa_g   = 1.0 / sa_g ** 2
    row_time   = cr.row_meta["time"].astype(str).to_numpy()

    qps, stored, rows = {}, {}, []
    for t in years:
        qps[t] = year_qp(cr, s0, t)
        check_builder(s0, cr, qps[t], t, a0_g, inv_sa_g, row_time)
        z = np.load(data_root / "opt" / OPT_RUN_ID / "timesteps" / f"t_{t}.npz",
                    allow_pickle=True)
        if not np.array_equal(np.asarray(z["cols_t"], dtype=np.int64), qps[t]["cols"]):
            raise SystemExit(f"{t}: the stored cols_t differs from the const run's")
        stored[t] = np.asarray(z["F_opt"])
        rows.append(dict(year=t, form="stored", tol="prod", status="Solved",
                         **score(qps[t], stored[t]), dF=0.0))
    print(f"builder formulation 0 == _solve_period's QP for {years}")

    jobs = [(t, f, k) for k in reversed(tols) for t in years for f in forms]

    def _run(job):
        t, f, k = job
        if f == "0" and k == "prod":
            return solve_incumbent(solvers[k], cr, t, a0_g, inv_sa_g, row_time)
        return solve_built(qps[t], f, solvers[k]._period_settings(verbose=False))

    real = solver_mod.clarabel
    solver_mod.clarabel = types.SimpleNamespace(
        DefaultSolver=lambda *a: _Recording(real.DefaultSolver(*a)),
        DefaultSettings=real.DefaultSettings,
        ZeroConeT=real.ZeroConeT, NonnegativeConeT=real.NonnegativeConeT,
    )
    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futs = {pool.submit(_run, j): j for j in jobs}
            for fut in as_completed(futs):
                _record(futs[fut], fut, qps, stored, rows, out)
    finally:
        solver_mod.clarabel = real

    df = pd.DataFrame(rows)
    df.to_csv(out / "probe.csv", index=False)
    report(df)


def _record(job, fut, qps, stored, rows, out) -> None:
    """Score one finished solve, print it and rewrite ``probe.csv``."""
    t, f, k = job
    try:
        F, eps, z, conv, wall = fut.result()
    except RuntimeError as e:                            # _solve_period's non-OK status
        rows.append(dict(year=t, form=f, tol=k, status=f"raised: {e}"))
        print(f"{t} {f} {k}: {e}", flush=True)
        return
    np.savez(out / f"F_{t}_{f}_{k}.npz", F=F, eps=eps, z=z)
    sc = score(qps[t], F)
    row = dict(year=t, form=f, tol=k, status=conv["status"],
               iters=conv["iterations"], wall_s=wall, **sc,
               LB=dual_bound(qps[t], z),
               dF=float(np.abs(F - stored[t]).max()),
               bitid=bool(np.array_equal(F, stored[t])),
               cost=conv["cost_primal"], gap_abs=conv["gap_abs"],
               gap_rel=conv["gap_rel"], res_primal=conv["res_primal"],
               res_dual=conv["res_dual"])
    rows.append(row)
    print(f"{t} {f:>1} {k:<5} {row['status']:<13} it {row['iters']:>4} "
          f"{wall:6.1f} s  J {sc['J']:,.3f}  LB {row['LB']:,.3f}  zero {sc['zero']:.1e} "
          f"band {sc['band']:.1e} lb {sc['lb']:.1e}  dF {row['dF']:.2e}  "
          f"gap {row['gap_abs']:.2e}/{row['gap_rel']:.1e}", flush=True)
    pd.DataFrame(rows).to_csv(out / "probe.csv", index=False)


def report(df: pd.DataFrame) -> None:
    """Per year: each solve's J against the incumbent's strict-tolerance J, and
    against the best certified lower bound of that year (``J − LB*`` bounds how far
    a posterior is from the optimum, up to its own hard-row violations)."""
    ref = df[(df.form == "0") & (df.tol == "tight")].set_index("year")["J"]
    lbs = df.groupby("year")["LB"].max()
    df  = df.assign(dJ_vs_0tight=df.J - df.year.map(ref),
                    J_minus_LBstar=df.J - df.year.map(lbs))
    cols = ["year", "form", "tol", "status", "iters", "wall_s", "J", "dJ_vs_0tight",
            "LB", "J_minus_LBstar", "zero", "band", "lb", "dF", "gap_abs", "gap_rel"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 220, "display.max_rows", 500,
                           "display.float_format", "{:,.4g}".format):
        print(df.sort_values(["year", "tol", "form"])[cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path("data/proc"))
    ap.add_argument("--out",       type=Path, required=True)
    ap.add_argument("--years",     nargs="+", default=YEARS)
    ap.add_argument("--forms",     nargs="+", default=FORMS, choices=["0", "A", "B"])
    ap.add_argument("--tols",      nargs="+", default=list(TOLS), choices=list(TOLS))
    ap.add_argument("--threads",   type=int, default=4)
    ap.add_argument("--report",    action="store_true",
                    help="print the table of an earlier run's probe.csv and exit")
    a = ap.parse_args()
    if a.report:
        report(pd.read_csv(a.out / "probe.csv", dtype={"year": str}))
    else:
        main(a.data_root, a.out, a.years, a.forms, a.tols, a.threads)
