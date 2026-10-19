"""
issue94_relative_ratio_sigma_diag.py
====================================
#94 measurement of #29: a soft ratio row carries its stated σ relative to r,
σ_r = σ/|r|, before ``RATIO_SIGMA_REL_FLOOR``. Reads both constraint runs' per-year σ_c and
row_meta, the params versions the new constraint run cites (from their ``vN`` folders, so
nothing is registered, #93), and the runs' persisted artefacts; builds the sum-row and
ratio-row tables in memory (no per-year CSVs, #46). Prints #94's pass-rule items and the
numbers its predictions are scored on, for <new> against <base>:

    Q   σ_c: every row whose σ_c changes, by type and cone; each changed soft ratio row's
        factor σ_c(new)/σ_c(base) against max(σ/|r|, floor)/max(σ, floor) (item 1)
    P   ratio-row z² by family (concentrate, matte, slag, scrap, ore grade, tailings, eff …),
        the rows with relative σ above 100 %, and the concentrate grade posterior of Chile,
        Peru and the world (reported)
    E   the Round-E items (ADR 0025) (item 2)
    S   stock legs newly above |z| 5, their size, and the net stock posterior (item 3)
    C   common-set sum-row z² by region, sector and year; rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue94_relative_ratio_sigma_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import issue79_baci490_taiwan_diag as P79                 # noqa: E402
import issue86_congo_sacu_diag as P86                     # noqa: E402
import issue92_refined_scope_diag as P92                  # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.config import load_config                     # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic  # noqa: E402
from pysuto.param_relation import sigma_from_row          # noqa: E402

ROOT      = Path("data/proc")
PARAMS_IN = Path("data/input/params")
RATIO_PARAMS = ("conc", "eff", "ore_grade")              # the soft ratio builders' inputs
FAMILIES  = ("concconcentrate", "concmatte", "concslag", "concscrap", "concanode",
             "conccathode", "conccsxew", "conccer", "concpsc", "concwr",
             "oregradetailings", "oregradecosupply", "oregradeoreuse", "oregradeoresupply",
             "effsxew", "effsmelting", "effmfflotation", "effcfr", "effer", "netstockbound")
CONC_ROW  = "concconcentratesupply"                       # concentrate grade, Mining→MF side


def _floor() -> float:
    """``RATIO_SIGMA_REL_FLOOR`` as ``scripts/constraints.py`` states it."""
    s = Path("scripts/constraints.py").read_text()
    return float(re.search(r"^RATIO_SIGMA_REL_FLOOR\s*=\s*([0-9.eE+-]+)", s, re.M).group(1))


def _family(variant: pd.Series) -> pd.Series:
    out = pd.Series("other", index=variant.index)
    for f in FAMILIES[::-1]:
        out[variant.str.startswith(f)] = f
    return out


def _params(const: str) -> pd.DataFrame:
    """The soft ratio rows' params, from the versions the constraint run cites."""
    m = json.loads((ROOT / "const" / const / "metadata.json").read_text())
    refs = {p["id"]: p["version"] for p in m["parents"]["inputs"]["params"]}
    frames = []
    for pid in RATIO_PARAMS:
        p = pd.read_csv(PARAMS_IN / pid / f"v{refs[pid]}" / "params.csv", sep=None,
                        engine="python", dtype=str).fillna("*")
        frames.append(p.assign(source=f"{pid} v{refs[pid]}"))
    print(f"   params read: " + ", ".join(f"{pid} v{refs[pid]}" for pid in RATIO_PARAMS))
    return pd.concat(frames, ignore_index=True)


def sigma_check(base: str, new: str) -> None:
    """Q: which rows' σ_c change, and the soft ratio rows' factor against the rule."""
    cb, cn = L._chain(base)[0], L._chain(new)[0]
    print(f"\n== Q σ_c, {cb} → {cn} (item 1)")
    mb = pd.read_parquet(ROOT / "const" / cb / "row_meta.parquet")
    mn = pd.read_parquet(ROOT / "const" / cn / "row_meta.parquet")
    same = len(mb) == len(mn) and (mb.constraint_id.to_numpy() == mn.constraint_id.to_numpy()).all()
    print(f"   rows {len(mb):,} → {len(mn):,}; same rows in the same order: {same}")
    if not same:
        return
    sb = np.zeros(len(mn)); sn = np.zeros(len(mn))
    for f in sorted((ROOT / "const" / cn / "timesteps").glob("t_*.npz")):
        y = f.stem[2:]
        rows = (mn.time.astype(str) == y).to_numpy()
        sb[rows] = np.load(ROOT / "const" / cb / "timesteps" / f.name)["sigma_c"][rows]
        sn[rows] = np.load(f)["sigma_c"][rows]
    changed = ~np.isclose(sb, sn, rtol=1e-12, atol=0.0)
    ch = mn[changed].assign(factor=sn[changed] / sb[changed])
    print(f"   changed rows {changed.sum():,}; by type/cone/hard "
          f"{ch.groupby(['type', 'cone_type', 'is_hard']).size().to_dict()}")
    soft = ch[(ch.type == "ratio") & (ch.cone_type == "free")].copy()
    soft["family"] = _family(soft.variant.astype(str))
    p = _params(cn).set_index("param_id")
    floor = _floor()
    exp = {}
    for pid in soft.variant.astype(str).unique():
        if pid not in p.index:
            continue
        row = p.loc[pid]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        r = float(row["value_nominal"])
        try:
            s = sigma_from_row(row, r)
        except Exception:
            s = 0.0
        exp[pid] = max(s / abs(r), floor) / max(s, floor) if r != 0 else np.nan
    soft["expected"] = soft.variant.astype(str).map(exp)
    ok = np.isclose(soft.factor, soft.expected, rtol=1e-9)
    print(f"   soft ratio rows changed {len(soft):,}; factor matches max(σ/|r|, {floor})/max(σ, "
          f"{floor}) to rtol 1e-9: {int(ok.sum()):,}; mismatched {int((~ok).sum())}; "
          f"no params row {int(soft.expected.isna().sum())}")
    print("   factor by family (min–max): " + "  ".join(
        f"{k} {g.factor.min():.2f}–{g.factor.max():.2f} ({len(g)})"
        for k, g in soft.groupby("family")))
    soft["dev"] = soft.factor / soft.expected - 1.0
    print("   relative deviation factor/expected − 1 by family (min … max): " + "  ".join(
        f"{k} {g.dev.min():.1e}…{g.dev.max():.1e}" for k, g in soft.groupby("family")))
    print(f"   all changed soft ratio rows: max |deviation| {soft.dev.abs().max():.1e}")


def _ratio_frame(opt: str, acc) -> pd.DataFrame:
    d = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    r = cf[cf["type"].astype(str) == "ratio"]
    return pd.DataFrame({"constraint_id": r["constraint_id"].astype(str).to_numpy(),
                         "variant": r["variant"].astype(str).to_numpy()
                         if "variant" in r else r["constraint_id"].astype(str).str.split("_")
                         .str[1].to_numpy(),
                         "region": r["region"].astype(str).to_numpy(),
                         "z": r["z"].to_numpy(), "lhs_post": r["lhs_post"].to_numpy()})


def ratio_rows(acc, base: str, new: str) -> None:
    """P: ratio-row z² by family, the > 100 % rows, and the concentrate grade posterior."""
    print("\n== P ratio rows, Σz² by family (base → new) and |z| > 3")
    f = {o: _ratio_frame(o, acc) for o in (base, new)}
    for o in f:
        f[o]["family"] = _family(f[o].variant)
    g = pd.concat({o: f[o].groupby("family").agg(n=("z", "size"), z2=("z", lambda x: (x ** 2).sum()),
                                                 big=("z", lambda x: int((x.abs() > 3).sum())))
                   for o in f}, axis=1)
    for fam, r in g.iterrows():
        print(f"   {fam:18s} rows {int(r[(base, 'n')]):,}: Σz² {r[(base, 'z2')]:,.0f} → "
              f"{r[(new, 'z2')]:,.0f}; |z| > 3 {int(r[(base, 'big')])} → {int(r[(new, 'big')])}")
    print(f"   all ratio rows: Σz² {(f[base].z ** 2).sum():,.0f} → {(f[new].z ** 2).sum():,.0f}")


def concentrate_grade(E: dict, base: str, new: str) -> None:
    """P: posterior concentrate grade (MF→Concentrate CC / Mass) of Chile, Peru, world."""
    print("\n== P concentrate grade, Σ CC / Σ Mass of MF→Concentrate, 1995–2024 (prior → post)")
    for reg in ("Chile", "Peru", "world"):
        out = []
        for o in (base, new):
            e = E[o][(E[o].sector_from == "MF") & (E[o].sector_to == "Concentrate")]
            if reg != "world":
                e = e[e.region_to == reg]
            cc, ms = e[e.valuation == "CC"], e[e.valuation == "Mass"]
            out.append(f"{cc.prior.sum() / ms.prior.sum():.4f} → {cc.post.sum() / ms.post.sum():.4f}")
        print(f"   {reg:6s} base {out[0]}; new {out[1]}")


def main(base: str, new: str) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    sigma_check(base, new)
    E, S, j = P69.load(base, new)
    ratio_rows(acc, base, new)
    concentrate_grade(E, base, new)
    P69.round_e(E, S, j, base, new)
    P79.stock_z(E, base, new)
    P92.stock_sizes(E, base, new)
    P69.common_set(j, base)
    P86.by_year(j)
    by = j.groupby(P92._sector(j.label)).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
