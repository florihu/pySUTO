"""
init_copper_acc.py
==================
Initialise, report, and save the copper accounting structure.

Usage
-----
    python init_copper_acc.py \
        --root   path/to/root.ods   \
        --rules  path/to/rules.ods  \
        --bounds path/to/bounds.ods \   # optional
        --out    data/proc/acc          # optional, default shown

All three paths must be supplied by the user at the command line.
The bounds file is optional; omit --bounds to use defaults (l=0, u=1e9).

What this script does
---------------------
1.  Reads the root ODS (sectors, regions, valuations, time).
2.  Reads the rules ODS (Supply, Use, Trade, Accounting sheets) and
    builds the RuleSet.
3.  Constructs the AccountingStructure:
      - 4-D admissibility tensor A
      - Admissible edge_index
      - Full (unconstrained) edge_index_all
4.  Prints a detailed edge-space reduction report.
5.  Loads bounds exceptions (if provided) and prints a bounds summary.
6.  Saves everything to  <out>/<id>/  for later use by ConstraintBuilder.
7.  Writes a human-readable summary CSV of the admissible edges.
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

# ── Resolve the repo root (parent of the directory this script lives in) ──────
# This means you can run the script from ANY working directory and relative
# paths like "data/input/acc/root.ods" are always interpreted relative to the
# repo root, not wherever your shell happens to be.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)   # one level up from scripts/

def _resolve(path: str) -> str:
    """Return an absolute path.  Relative paths are resolved against the repo root."""
    if os.path.isabs(path):
        return path
    # Try as-is relative to cwd first (honours fully explicit invocations)
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    # Fall back to repo-root-relative
    return os.path.join(_REPO_ROOT, path)

# ── allow running from the repo root without installing the package ───────────
sys.path.insert(0, _SCRIPT_DIR)
from pysuto.acc import AccountingStructure


# =============================================================================
# Logging setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_copper_acc")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Initialise and save the copper accounting structure.",
        epilog=(
            "Relative paths are resolved first against the current working "
            "directory, then against the repo root (one level above this script). "
            "So you can run this script from any directory."
        ),
    )
    p.add_argument(
        "--root", required=True,
        help="Path to root.ods  (sheets: Sector_Entity, Region, Valuation, Time)"
    )
    p.add_argument(
        "--rules", required=True,
        help="Path to rules.ods  (sheets: Supply, Use, Trade, Accounting)"
    )
    p.add_argument(
        "--bounds", default=None,
        help="(optional) Path to bounds.ods  (sheet: Sheet1 or bounds)"
    )
    p.add_argument(
        "--out", default="data/proc/acc",
        help="Output directory, relative to repo root (default: data/proc/acc)"
    )
    p.add_argument(
        "--id", default="copper",
        help="Structure identifier used as sub-folder name (default: copper)"
    )
    args = p.parse_args()
    # Resolve all paths so the script works from any working directory
    args.root   = _resolve(args.root)
    args.rules  = _resolve(args.rules)
    args.bounds = _resolve(args.bounds) if args.bounds else None
    args.out    = _resolve(args.out)
    return args


# =============================================================================
# Edge summary helper
# =============================================================================

def build_edge_summary(acc: AccountingStructure) -> pd.DataFrame:
    """
    Return a readable DataFrame for the admissible edge_index, annotated
    with sector / entity / region labels for both from- and to-nodes.
    """
    ei  = acc.edge_index.copy()
    ni  = acc.node_index

    for side, col in (("from", "from_id"), ("to", "to_id")):
        node_info = ni.loc[ei[col].values].reset_index(drop=True)
        ei[f"{side}_sector"] = node_info["sector"].values
        ei[f"{side}_entity"] = node_info["entity"].values
        ei[f"{side}_region"] = node_info["region"].values

    col_order = [
        "edge_id",
        "from_sector", "from_entity", "from_region",
        "to_sector",   "to_entity",   "to_region",
        "time", "valuation",
    ]
    return ei[col_order]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Copper Accounting Structure — Initialisation")
    logger.info("=" * 60)
    logger.info(f"  root   : {args.root}")
    logger.info(f"  rules  : {args.rules}")
    logger.info(f"  bounds : {args.bounds or '(none — using defaults)'}")
    logger.info(f"  out    : {args.out}")
    logger.info(f"  id     : {args.id}")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # ── Build ─────────────────────────────────────────────────────────────────
    acc = AccountingStructure(
        id               = args.id,
        root_file_path   = args.root,
        rules_file_path  = args.rules,
        bounds_file_path = args.bounds,   # None is fine → defaults applied
        storage_dir      = args.out,
    )

    elapsed = time.perf_counter() - t0
    logger.info(f"Structure built in {elapsed:.1f}s")

    # ── Detailed stats ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  STRUCTURE STATISTICS")
    print("=" * 60)
    print(f"  Nodes              : {acc.n_nodes:>10,d}")
    print(f"  Time points        : {acc.n_timeps:>10,d}")
    print(f"  Valuations         : {acc.n_valtns:>10,d}")
    print(f"  Admissible edges   : {acc.n_edges:>10,d}")
    print(f"  All edges (no rule): {acc.n_edges_all:>10,d}")
    print(f"  Tensor shape       :  {acc.A.shape}")
    print(f"  Tensor nnz         : {acc.A.nnz:>10,d}")
    print()
    print(f"  Bounds summary     :  {acc.bounds.summary()}")

    # ── Per-sector-pair breakdown ─────────────────────────────────────────────
    edge_summary = build_edge_summary(acc)
    pair_counts  = (
        edge_summary
        .groupby(["from_sector", "to_sector"])
        .agg(
            n_edges          = ("edge_id", "count"),
            n_cross_region   = ("from_region",
                                lambda x: (x.values != edge_summary.loc[x.index, "to_region"].values).sum()),
            n_time_vals      = ("time",
                                lambda x: x.nunique() * edge_summary.loc[x.index, "valuation"].nunique()),
        )
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )

    print()
    print("=" * 60)
    print("  ADMISSIBLE EDGES BY SECTOR PAIR")
    print("=" * 60)
    print(pair_counts.to_string(index=False))

    # ── Tradable-flow cross-region counts ─────────────────────────────────────
    cross = edge_summary[edge_summary["from_region"] != edge_summary["to_region"]]
    if len(cross) > 0:
        print()
        print("=" * 60)
        print("  CROSS-REGION EDGES  (tradable flows only)")
        print("=" * 60)
        cr_counts = (
            cross
            .groupby(["from_sector", "to_sector"])["edge_id"]
            .count()
            .reset_index()
            .rename(columns={"edge_id": "n_cross_region_edges"})
            .sort_values("n_cross_region_edges", ascending=False)
        )
        print(cr_counts.to_string(index=False))
    else:
        print()
        print("  (no cross-region edges in admissible set)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print()
    logger.info("Saving structure...")
    acc.save()

    # Also save the annotated edge summary as a CSV for inspection
    out_dir = os.path.join(args.out, args.id)
    csv_path = os.path.join(out_dir, "edge_summary.csv")
    edge_summary.to_csv(csv_path, index=False)
    logger.info(f"Edge summary saved to: {csv_path}")

    # Save pair-count table
    pc_path = os.path.join(out_dir, "sector_pair_counts.csv")
    pair_counts.to_csv(pc_path, index=False)
    logger.info(f"Sector-pair counts saved to: {pc_path}")

    print()
    logger.info("=" * 60)
    logger.info(f"  Done.  All files saved to: {out_dir}")
    logger.info("=" * 60)

    return acc   # useful when importing / testing


if __name__ == "__main__":
    main()
