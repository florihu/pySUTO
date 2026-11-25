# Frozen record: last ran against commit fb36cee; rerun by checking out that commit (see README.md).
"""
issue149_parametric_sigma_units_diag.py
=======================================
#149 measurement of #148: ``ParametricImputer`` adds a ratio's σ in flow units,
|f_target|·σ_r/|r|, instead of the bare ratio-unit σ_r. Reads persisted run artefacts
only and prints #149's pre-registered items for <new> against <base>:

    P   prior: f_agg / coverage bit-identity, σ_agg moves, the log's method counts, and
        the median CV of the conc and ore_grade fills by valuation (item 1, P1, P2)
    Y   solve status per year (item 2, P5)
    E   the Round-E items (ADR 0025) (item 3, P4)
    C   common-set sum-row z² by year and region, and the attribution (item 4, P3)

    python scripts/explo/issue149_parametric_sigma_units_diag.py <opt_base> <opt_new>
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
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402
import issue69_aggregate_gap_diag as P69                # noqa: E402

ROOT     = Path("data/proc")
IMP_BASE = "imp_20261102_094925_95744204"   # the final chain's imp, post-#113 log order


def _sidecar(stage: str, run: str) -> dict:
    with open(ROOT / stage / run / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _imp_of(opt: str) -> str:
    const = _sidecar("opt", opt)["parents"]["upstream"]["const_run_id"]
    return _sidecar("const", const)["parents"]["upstream"]["imp_run_id"]


def _ei():
    """Expanded edge index of the copper acc (valuation per edge id)."""
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    return expand_edge_index(acc.edge_index, acc.node_index)


def prior(imp_base: str, imp_new: str) -> None:
    """P: the prior's values must not move; its σ moves on the parametric fills."""
    print(f"\n== P prior, {imp_base} → {imp_new}")
    a = np.load(ROOT / "imp" / imp_base / "agg_result.npz", allow_pickle=True)
    b = np.load(ROOT / "imp" / imp_new / "agg_result.npz", allow_pickle=True)
    for k in ("f_agg", "coverage"):
        same = np.array_equal(a[k], b[k], equal_nan=(k == "f_agg"))
        print(f"   {k}: {'bit-identical' if same else 'DIFFERS'}")
    sa, sb = a["sigma_agg"], b["sigma_agg"]
    moved = ~((sa == sb) | (np.isnan(sa) & np.isnan(sb)))
    print(f"   sigma_agg: {int(moved.sum()):,} of {sa.size:,} edges move")
    la = pd.read_parquet(ROOT / "imp" / imp_base / "imputation_log.parquet")
    lb = pd.read_parquet(ROOT / "imp" / imp_new / "imputation_log.parquet")
    mc = pd.concat({"base": la.method.value_counts(), "new": lb.method.value_counts()},
                   axis=1).fillna(0).astype(int)
    mc["d"] = mc.new - mc.base
    print("   log method counts (rows that change):")
    print(mc[mc.d != 0].to_string() if (mc.d != 0).any() else "   none")
    val = _ei().set_index("edge_id")["valuation"]
    for imp_id in ("conc", "ore_grade"):
        rows = []
        for name, lg in (("base", la), ("new", lb)):
            x = lg[lg.imputer_id == imp_id]
            cv = (x.sigma_after / x.f_after.abs()).replace(np.inf, np.nan)
            v = val.reindex(x.edge_id.to_numpy()).to_numpy()
            for vv in sorted(set(v)):
                m = v == vv
                rows.append({"run": name, "valuation": vv, "n": int(m.sum()),
                             "median_cv": float(np.nanmedian(cv[m])),
                             "p90_cv": float(np.nanpercentile(cv[m], 90)),
                             "capped": int(x.method[m].str.endswith("_capped").sum())})
        print(f"   {imp_id} fills, CV σ/|f| by valuation:")
        print(pd.DataFrame(rows).pivot(index="valuation", columns="run",
                                       values=["n", "median_cv", "p90_cv", "capped"])
              .round(4).to_string())
    # σ moves downstream of the parametric fills (later imputers read σ of their sources)
    pm = set(lb.edge_id[lb.imputer_id.isin(["conc", "ore_grade"])])
    ids = np.where(moved)[0]
    print(f"   edges whose σ moves: {len(ids):,}; of them parametric fills "
          f"{len(pm.intersection(ids)):,}, downstream {len(set(ids) - pm):,}")


def years(opt: str) -> None:
    """Y: solve status per year."""
    ss  = _sidecar("opt", opt)["stage_summary"]
    st  = pd.Series({t: v["solver_status"] for t, v in ss["per_timestep"].items()})
    print(f"\n== Y {opt}: {st.value_counts().to_dict()} over {len(st)} years; "
          f"solve_wall_s {ss['solve_wall_s']:.0f}")


def main(base: str, new: str) -> None:
    prior(IMP_BASE, _imp_of(new))
    years(new)
    E, S, j = P69.load(base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    by = j.groupby(j.label.str.split("·").str[0].str.split(" ").str[-1]).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
