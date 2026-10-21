"""
issue97_observed_zero_diag.py
=============================
#97 / #31 / #96 measurement: a region-year ICSG covers that produces none of a product
says zero on its own output edge (ADR 0024 am. 2026-10-21). Reads persisted run
artefacts and each run's own constraint run (priors), and prints, for <new> against
<base>:

    M   the no-data mine chains, CC, domestic edges, prior / posterior per run, by region:
        CO→Mining and MF→Concentrate in region-years with no ICSG mine production;
        Ore→SXEW where neither mine nor SX-EW output is reported (#97); Ore→SXEW in
        mining region-years without ICSG SX-EW output (reported separately)
    E   the Round-E items (ADR 0025)
    C   common-set sum-row z² by region, sector and year

    python scripts/explo/issue97_observed_zero_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import build_baci_reexport as R                           # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
import roundZ_png_kg_diag as Z                            # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.solver import ClarabelSolver                  # noqa: E402


def _edges() -> pd.DataFrame:
    acc = AccountingStructure.load(id="copper", storage_dir="data/proc/acc")
    ei, ni = acc.edge_index, acc.node_index.reset_index(drop=True)
    return pd.DataFrame({
        "sf": ni["sector"].values[ei["from_id"].values],
        "st": ni["sector"].values[ei["to_id"].values],
        "rf": ni["region"].values[ei["from_id"].values],
        "rt": ni["region"].values[ei["to_id"].values],
        "val": ei["valuation"].values, "t": ei["time"].astype(str).values,
    })


def _prior_post(opt: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """a_prior from the run's own constraint run, F_opt from its period files."""
    s = ClarabelSolver()
    cr = s.load_run(L._chain(opt)[0])
    a0, F = np.full(n, np.nan), np.full(n, np.nan)
    for t in sorted(cr.edge_cols_by_t):
        qp = Z._qp_year(cr, s, t)
        a0[qp["cols"]] = qp["a0"]
        z = np.load(Z.ROOT / "opt" / opt / "timesteps" / f"t_{t}.npz", allow_pickle=True)
        F[np.asarray(z["cols_t"], dtype=np.int64)] = z["F_opt"]
    return a0, F


def mine_chains(base: str, new: str) -> None:
    """M: the no-data mine chains, prior / posterior per run."""
    E = _edges()
    _, prod, _ = R.icsg_intake_production(["Concentrate"])
    mine = {(r, t) for (_, r, t), v in prod.items() if v > 0}
    reg = R._input_registries().feed.resolve("icsg")
    raw = pd.read_csv(reg / "values.csv")
    rmap = pd.read_excel(reg / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw["Region"].map(dict(zip(rmap["src_region"], rmap["root_region"])))
    sx = raw[(raw["Process"] == "SXEW") & (raw["Supply_Use"] == "Supply") & (raw["Value"] > 0)]
    sxew = set(zip(sx["reg"], sx["Year"].astype(str)))
    for o in (base, new):
        E[f"a0 {o[-8:]}"], E[f"post {o[-8:]}"] = _prior_post(o, len(E))
    rt = list(zip(E["rf"], E["t"]))
    has_mine = np.array([k in mine for k in rt])
    has_sx = np.array([k in sxew for k in rt])
    dom = (E["rf"] == E["rt"]).to_numpy() & (E["val"] == "CC").to_numpy()
    fam = {
        "CO→Mining, no ICSG mine":           (E["sf"] == "CO") & (E["st"] == "Mining") & ~has_mine,
        "MF→Concentrate, no ICSG mine (#31)": (E["sf"] == "MF") & (E["st"] == "Concentrate") & ~has_mine,
        "Ore→SXEW, no mine and no SX-EW (#97)": (E["sf"] == "Ore") & (E["st"] == "SXEW") & ~has_mine & ~has_sx,
        "Ore→SXEW, mine but no ICSG SX-EW":  (E["sf"] == "Ore") & (E["st"] == "SXEW") & has_mine & ~has_sx,
    }
    cols = [c for c in E.columns if c.startswith(("a0 ", "post "))]
    for name, m in fam.items():
        d = E[dom & m.to_numpy()]
        tot = d[cols].sum()
        print(f"\n== M {name}, kt CC: " + "  ".join(f"{c} {v:,.0f}" for c, v in tot.items()))
        top = d.groupby("rf")[cols].sum()
        top = top.loc[top.abs().max(axis=1).sort_values(ascending=False).index[:8]]
        print(top.round(0).to_string())


def main(base: str, new: str) -> None:
    mine_chains(base, new)
    E, S, j = P69.load(base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    g = j.groupby("region")[["z2_base", "z2_new"]].sum()
    g["dz2"] = g["z2_new"] - g["z2_base"]
    print("\n== C regions, common-set Σz² base → new (largest |Δ|)")
    print(g.reindex(g["dz2"].abs().sort_values(ascending=False).index[:12]).round(0).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
