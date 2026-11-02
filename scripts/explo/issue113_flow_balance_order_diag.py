"""
issue113_flow_balance_order_diag.py
===================================
#113 measurement: ``NetworkImputer`` visits its flow balances in sorted order (0d49c45),
so no Stage 2 output follows the process's string-hash seed. Two imp runs from the
reference chain's feed runs, under ``PYTHONHASHSEED=1`` and ``PYTHONHASHSEED=2``, are
compared with the reference imp run and with each other.

The reference is the final chain's imp ``imp_20261022_111102_1b49a2b3`` (chain
``opt_20261022_112855_928da10c``), the base of #108's sweep, not the issue's
``imp_20261021_125954_b112b170``: #111 moved the reference chain after #113 was filed,
and the sweep compares its points with the new one. That run was written before the
fix, in its process's hash order.

Decision rule, fixed in this docstring before either run started:

    1  each new run reproduces every array of the reference bit for bit: every key of
       ``agg_result.npz`` by ``np.array_equal`` with NaN equal to NaN; its
       ``imputation_log.parquet`` holds the same rows as the reference's once both are
       sorted by every column;
    2  the two new runs' imputation logs are identical as stored, row order included
       (``DataFrame.equals``).

Item 1 failing on an array means the order of the balances can move a value, and where.
Item 2 failing means another unordered iteration is left. The number of log rows out of
place against the reference is printed as a diagnostic (the #106 gate counted 15,665
and 15,651 in its two reruns).

    python scripts/explo/issue113_flow_balance_order_diag.py <imp_seed1> <imp_seed2>
                                                              [--ref <imp_run_id>]

Read-only; run from the main checkout's root.
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import pandas as pd

REF  = "imp_20261022_111102_1b49a2b3"          # final chain opt_20261022_112855_928da10c
IMP  = os.path.join("data", "proc", "imp")
LOG  = "imputation_log.parquet"

# The #106 gate's comparison: every stored array, and the log as a multiset.
_spec = importlib.util.spec_from_file_location(
    "_gate106", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "issue106_driver_gate.py"))
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)


def item1(new: str, ref: str) -> bool:
    """Every array of ``new`` equals ``ref``'s; the log compared as a multiset."""
    rows = _gate.compare_dirs(os.path.join(IMP, new), os.path.join(IMP, ref))
    arrays = [r for r in rows if r[0] != LOG]
    bad = [r for r in arrays if not r[2]]
    same, moved = _gate.log_as_multiset(os.path.join(IMP, new), os.path.join(IMP, ref))
    print(f"   {new}: {len(arrays) - len(bad)}/{len(arrays)} arrays identical"
          + ("" if not bad else f"; DIFFER: {bad}"))
    print(f"   {new}: log {'identical' if same else 'DIFFERENT'} as a multiset; "
          f"{moved} rows out of place against {ref} (diagnostic)")
    return not bad and same


def item2(a: str, b: str) -> bool:
    """The two logs are equal as stored, row order included."""
    la, lb = (pd.read_parquet(os.path.join(IMP, r, LOG)) for r in (a, b))
    same = la.equals(lb)
    print(f"   {a} vs {b}: logs {'identical' if same else 'DIFFERENT'} as stored "
          f"({len(la):,} and {len(lb):,} rows)")
    return same


def main(seed1: str, seed2: str, ref: str) -> None:
    print(f"== item 1: each run against {ref}")
    one = item1(seed1, ref) & item1(seed2, ref)
    print("== item 2: the two runs' logs, row order included")
    two = item2(seed1, seed2)
    print(f"\nitem 1 {'HOLDS' if one else 'FAILS'} · item 2 {'HOLDS' if two else 'FAILS'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="#113: flow-balance order measurement.")
    parser.add_argument("seed1", help="imp run under PYTHONHASHSEED=1")
    parser.add_argument("seed2", help="imp run under PYTHONHASHSEED=2")
    parser.add_argument("--ref", default=REF, help=f"reference imp run (default {REF})")
    args = parser.parse_args()
    main(args.seed1, args.seed2, args.ref)
