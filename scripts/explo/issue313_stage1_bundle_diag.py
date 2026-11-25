# Frozen record: last ran against commit 9f49f50; rerun by checking out that commit (see README.md).
"""
issue313_stage1_bundle_diag.py
==============================
#313 measurement of the Stage 1 bundle: #233 (BACI pedigree cv from meta.xlsx), #240 (root-leg
cv_scale over all folded codes) and the ore_grade rebuild on the reference ICSG feed (#311,
check 8). Reads persisted run artefacts only and prints #313's pre-registered items for <new>
against <base>:

    R1  solve status per year, the two chains' input versions
    D1  the BACI feeds: F_root identical; σ_root ratio on the legs #240 leaves alone
    D2  the BACI feeds: σ_root ratio on the root legs whose cv_scale #240 changes
    D3  the ICSG feeds: ore_grade parents.json; every array that differs and on which edges,
        split into the hook's step (base → the hook's feed run) and the ore grade's step
    D4  the Round-E items (ADR 0025)
    D5  common-set sum-row z² by anchor band, year and region

    python scripts/explo/issue313_stage1_bundle_diag.py <opt_base> <opt_new> \
        [--icsg-hook <feed run built before the ore_grade rebuild>] \
        [--baci-cv-old <quantity_cv_scale.csv of the base parse>]

Run from the repository root of the checkout that holds ``data/proc``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
from pysuto import acc_store                                # noqa: E402
from pysuto.acc import AccountingStructure                  # noqa: E402
from pysuto.config import load_config                       # noqa: E402
from feedprep.trade_shares import decoded_edges      # noqa: E402
import issue69_aggregate_gap_diag as P69                    # noqa: E402

ROOT     = Path("data/proc")
FEED_DIR = ROOT / "feed" / "registry"
INPUT    = Path("data/input")
MIXED_TOL = 2e-3     # relative cv_scale change that separates #240 from #233


def _sidecar(stage: str, run: str) -> dict:
    with open(ROOT / stage / run / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _feeds_of(opt: str) -> dict:
    const = _sidecar("opt", opt)["parents"]["upstream"]["const_run_id"]
    imp   = _sidecar("const", const)["parents"]["upstream"]["imp_run_id"]
    ids   = _sidecar("imp", imp)["parents"]["upstream"]["feed_run_ids"]
    return {i.split("_20")[0].replace("feed_", ""): i for i in ids}


def _feed_meta(run: str) -> dict:
    with open(FEED_DIR / run / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _feed(run: str):
    return np.load(FEED_DIR / run / "result.npz", allow_pickle=True)


def solve(opt: str) -> None:
    """R1: solve status per year and the chain's input versions."""
    ss = _sidecar("opt", opt)["stage_summary"]
    st = pd.Series({t: v["solver_status"] for t, v in ss["per_timestep"].items()})
    const = _sidecar("opt", opt)["parents"]["upstream"]["const_run_id"]
    imp   = _sidecar("const", const)["parents"]["upstream"]["imp_run_id"]
    pin   = _sidecar("imp", imp)["parents"]["inputs"]
    ver   = {p["id"]: p["version"] for p in pin["params"]}
    print(f"\n== R1 {opt}: {st.value_counts().to_dict()} over {len(st)} years; "
          f"solve_wall_s {ss['solve_wall_s']:.0f}\n   const {const}  imp {imp}"
          f"\n   acc {pin['acc']['version']}, params {ver}")
    for k, run in _feeds_of(opt).items():
        print(f"   feed {k}: {run} inputs "
              f"{ {n: v['version'] for n, v in _feed_meta(run)['parents']['inputs'].items()} }")


def _same(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.dtype.kind == "f":
        return (x == y) | (np.isnan(x) & np.isnan(y))
    return x == y


def _cv_key(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index(["sector", "region_from", "region_to", "time"])


def baci(acc, base_run: str, new_run: str, cv_old: str, cv_new: str) -> None:
    """D1, D2: the BACI feeds, σ_root by whether #240 changes the edge's root leg."""
    a, b = _feed(base_run), _feed(new_run)
    print(f"\n== D1/D2 BACI feed {base_run} → {new_run}")
    for k in a.files:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"   {k}: shape {x.shape} → {y.shape}")
            continue
        n = int((~_same(x, y)).sum())
        print(f"   {k}: {n:,} entries differ")
    old, new = _cv_key(cv_old), _cv_key(cv_new)
    j = old.join(new, lsuffix="_old", rsuffix="_new", how="outer")
    # #233 moves every flagged leg by about 5e-4 relative; #240 moves a mixed leg by more
    d240 = ((j.cv_scale_new - j.cv_scale_old).abs() > MIXED_TOL * j.cv_scale_old).fillna(True)
    rel  = (j.cv_scale_new / j.cv_scale_old - 1.0)
    print(f"   cv_scale rel change over all {len(j)} legs: |rel| ≤ {MIXED_TOL:g} on "
          f"{int((rel.abs() <= MIXED_TOL).sum())}, max rise {rel.max():+.2e}, "
          f"above 1 + {MIXED_TOL:g} on {int((rel > MIXED_TOL).sum())}")
    print(f"   quantity_cv_scale: {len(old)} root-leg rows old, {len(new)} new; "
          f"cv_scale falls by more than MIXED_TOL on {int(d240.sum())} rows (#240's mixed legs); "
          f"new/old cv_scale on them: median "
          f"{(j.cv_scale_new / j.cv_scale_old)[d240].median():.3f}, min "
          f"{(j.cv_scale_new / j.cv_scale_old)[d240].min():.3f}, max "
          f"{(j.cv_scale_new / j.cv_scale_old)[d240].max():.3f}")
    ei = decoded_edges(acc)
    ch = np.where(~_same(a["sigma_root"], b["sigma_root"]))[0]
    e  = ei.set_index("edge_id").loc[ch]
    ratio = b["sigma_root"][ch] / a["sigma_root"][ch]
    key = list(zip(e.root_sector_from, e.root_region_from, e.root_region_to,
                   e.root_time.astype(str)))
    sec = pd.read_csv(cv_new).sector.unique().tolist()
    print(f"   σ_root differs on {len(ch):,} edges; root sector from on them "
          f"{pd.Series(e.root_sector_from).value_counts().to_dict()}; cv_scale sectors {sec}")
    changed = set(d240[d240].index.map(lambda k: (k[0], k[1], k[2], str(k[3]))))
    hit = np.array([k in changed for k in key])
    for name, m in (("D2: edges on a root leg #240 changes", hit),
                    ("D1: all other edges with σ_root changed", ~hit)):
        r = ratio[m]
        if not len(r):
            print(f"   {name}: 0 edges")
            continue
        print(f"   {name}: {len(r):,} edges; new/old σ_root median {np.median(r):.6f}, "
              f"min {r.min():.6f}, max {r.max():.6f}; share ≤ 1 {np.mean(r <= 1 + 1e-12):.3f}; "
              f"|rel| median {np.median(np.abs(r - 1)):.2e}, max {np.abs(r - 1).max():.2e}")
    print(f"   edges with σ_root changed, in #240's legs {int(hit.sum()):,}; legs touching ROW "
          f"{int(d240[[k[2] == 'ROW' for k in d240.index.to_list()] ].sum())}")


def icsg(acc, base_run: str, hook_run: str, new_run: str) -> None:
    """D3: the ICSG feeds. base → hook run (the build without the ore grade) → new."""
    pj = INPUT / "params" / "ore_grade" / "current" / "parents.json"
    print(f"\n== D3 ore_grade parents.json: {json.loads(pj.read_text())}")
    ei = decoded_edges(acc)
    for lab, r0, r1 in (("base → hook run", base_run, hook_run),
                        ("hook run → new (the ore grade's step)", hook_run, new_run),
                        ("base → new (the bundle)", base_run, new_run)):
        a, b = _feed(r0), _feed(r1)
        print(f"   -- ICSG {lab}: {r0} → {r1}")
        union = set()
        for k in a.files:
            x, y = a[k], b[k]
            if x.shape != y.shape:
                print(f"      {k}: shape {x.shape} → {y.shape}")
                continue
            if x.ndim == 0 or x.shape[0] != len(ei):
                print(f"      {k}: {int((~_same(x, y)).sum())} entries differ")
                continue
            idx = np.where(~_same(x, y))[0]
            union |= set(idx.tolist())
            print(f"      {k}: {len(idx):,} entries differ")
        if union:
            s = ei[ei.edge_id.isin(union)]
            print("      edges that differ in any array, by sector pair and valuation:")
            print(s.groupby(["root_sector_from", "root_sector_to", "valuation"]).size()
                  .to_string().replace("\n", "\n        "))
            on = (a["observed_zero"] != b["observed_zero"]).nonzero()[0]
            print(f"      observed_zero stamped {int((b['observed_zero'] & ~a['observed_zero']).sum())},"
                  f" unstamped {int((a['observed_zero'] & ~b['observed_zero']).sum())} "
                  f"(of {int(a['observed_zero'].sum())} stamped in {r0}); flips {len(on)}")
            fr = np.where(~_same(a["F_root"], b["F_root"]))[0]
            if len(fr):
                print(f"      F_root differs on {len(fr)} edges, Σ|Δ| "
                      f"{np.abs(a['F_root'][fr] - b['F_root'][fr]).sum():,.3f} kt")


def crude(base_run: str, new_run: str) -> None:
    a, b = _feed(base_run), _feed(new_run)
    n = {k: int((~_same(a[k], b[k])).sum()) for k in a.files if a[k].shape == b[k].shape}
    print(f"\n== crude-ore feed {base_run} → {new_run}: entries differing {n}")


def main(base: str, new: str, hook: str, cv_old: str) -> None:
    # the older explo helpers (issue69 ...) still call AccountingStructure.load, which #243
    # moved to acc_store.load
    AccountingStructure.load = staticmethod(acc_store.load)
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    solve(base)
    solve(new)
    fb, fn = _feeds_of(base), _feeds_of(new)
    cv_new = str(INPUT / "feed" / "baci" / "v0" / "quantity_cv_scale.csv")
    baci(acc, fb["baci"], fn["baci"], cv_old, cv_new)
    icsg(acc, fb["icsg"], hook, fn["icsg"])
    crude(fb["crude_ore"], fn["crude_ore"])
    E, S, j = P69.load(base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    P69.attribution(j)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--icsg-hook", required=True)
    ap.add_argument("--baci-cv-old", required=True)
    args = ap.parse_args()
    main(args.base, args.new, args.icsg_hook, args.baci_cv_old)
