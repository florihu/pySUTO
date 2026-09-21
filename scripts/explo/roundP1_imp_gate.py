"""Round P.1 Stage-2 gate — what did the pure-secondary recipe change move?

The imputer amendment (ADR 0010 amendment, `network.py:_do_recipe`) should
close 100 %-secondary process recipes and touch NOTHING else. Diff the new
`f_agg` against the old one and report every edge class that moved, so a wider
blast radius than intended is visible before any solve.

    python scripts/explo/roundP1_imp_gate.py <imp_old> <imp_new>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/proc")


def main(old: str, new: str) -> int:
    sys.path.insert(0, "src")
    from pysuto.acc import AccountingStructure
    from pysuto.config import load_config
    from pysuto.param_relation import expand_edge_index

    a = np.nan_to_num(np.asarray(
        np.load(ROOT / "imp" / old / "agg_result.npz", allow_pickle=True)["f_agg"], float))
    b = np.nan_to_num(np.asarray(
        np.load(ROOT / "imp" / new / "agg_result.npz", allow_pickle=True)["f_agg"], float))
    if a.shape != b.shape:
        print(f"*** FAIL shape {a.shape} vs {b.shape}")
        return 1
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index).copy()
    ei["old"], ei["new"] = a[ei.edge_id.values], b[ei.edge_id.values]
    ei["d"] = ei.new - ei.old
    moved = ei[ei.d.abs() > 1e-6].copy()
    print(f"Round P.1 Stage-2 gate — {old} → {new}")
    print(f"   edges moved: {len(moved):,} of {len(ei):,} "
          f"({len(moved) / len(ei) * 100:.3f} %)")
    if not len(moved):
        print("   nothing moved — the amendment did not fire")
        return 1
    cls = (moved.groupby(["valuation", "sector_from", "sector_to"])
           .agg(n=("d", "size"), net=("d", "sum")).sort_values("net", key=abs,
                                                               ascending=False))
    print("\n   edge classes that moved (net kt):")
    print(cls.head(16).round(1).to_string())
    regs = (moved.groupby("region_to").d.sum().abs()
            .sort_values(ascending=False).head(10))
    print(f"\n   regions touched: {moved.region_to.nunique()} — top: "
          + ", ".join(f"{k} {v:,.0f}" for k, v in regs.items()))
    print(f"\n   NOTE: only Slovakia (Smelting) and a few small CFR regions were "
          f"expected; anything else is a finding.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
