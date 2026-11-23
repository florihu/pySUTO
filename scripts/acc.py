"""
acc.py
======
Initialise, report, and save the accounting structure.

Reads the acc spec from the **input registry** (``data/input/acc/<id>/current/``):
one ``acc.json`` when the version holds it, else root.ods / rules.ods (+ bounds.ods,
masks.csv); a version holding both raises (ADR 0046).  At script start the input registry
is initialised — if the user has edited any of the load-bearing files
since the last registered version, a new ``vN/`` snapshot is materialised
under ``data/input/acc/<id>/`` and the version is recorded in
``data/input/acc/registry.json``.

Outputs to ``data/proc/acc/<id>/`` (acc is auxiliary to the six-stage pipeline, so
its output side does not use the harmonized ``{stage}_{ts}_{code}`` run-id pattern).
The statistics go to the log, ``<data dir>/logs/acc_<timestamp>.log`` (the data dir is
the parent of ``--input-root``).

Usage::

    python scripts/acc.py --id copper
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from pysuto.acc import AccountingStructure, read_acc_spec_dir, write_build_record
from pysuto.input_registry import InputRegistries
from pysuto.logging import get_logger, start_run_log


logger = get_logger("acc")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialise the accounting structure.")
    p.add_argument(
        "--id", default="copper",
        help="Accounting-structure id (subfolder of data/input/acc/ and "
             "data/proc/acc/). Default: copper",
    )
    p.add_argument(
        "--input-root", default=os.path.join(_REPO_ROOT, "data", "input"),
        help="Root of the input registries (default: data/input).",
    )
    p.add_argument(
        "--out", default=os.path.join(_REPO_ROOT, "data", "proc", "acc"),
        help="Where to write the built AccountingStructure (default: data/proc/acc).",
    )
    p.add_argument(
        "--label", default=None,
        help="Optional human-readable label for the input acc registry entry. "
             "Carried forward from the prior version if omitted.",
    )
    return p.parse_args()


# =============================================================================
# Build: pick the acc-spec reader from what the version holds
# =============================================================================

def build_acc(acc_path: Path, id: str, storage_dir: str) -> AccountingStructure:
    """Build from ``acc.json`` when *acc_path* holds it, else from root.ods + rules.ods
    (:func:`pysuto.acc.read_acc_spec_dir` picks the reader).

    Raises
    ------
    ValueError
        *acc_path* holds ``acc.json`` and any file of the ODS/CSV format: one of the two
        would be ignored.
    """
    return AccountingStructure.from_spec(read_acc_spec_dir(acc_path), id,
                                         storage_dir=storage_dir)


# =============================================================================
# Edge summary helper
# =============================================================================

def build_edge_summary(acc: AccountingStructure) -> pd.DataFrame:
    """One row per admissible edge, annotated with sector/entity/region labels."""
    ei = acc.edge_index.copy()
    ni = acc.node_index
    for side, col in (("from", "from_id"), ("to", "to_id")):
        node_info = ni.loc[ei[col].values].reset_index(drop=True)
        ei[f"{side}_sector"] = node_info["sector"].values
        ei[f"{side}_entity"] = node_info["entity"].values
        ei[f"{side}_region"] = node_info["region"].values
    return ei[[
        "edge_id",
        "from_sector", "from_entity", "from_region",
        "to_sector",   "to_entity",   "to_region",
        "time", "valuation",
    ]]


# =============================================================================
# Main
# =============================================================================

def main() -> AccountingStructure:
    args = parse_args()
    start_run_log(os.path.dirname(os.path.abspath(args.input_root)), "acc")

    # ── Initialise input registry (scan-and-bump every kind) ──────────────
    label_overrides = {"acc": {args.id: args.label}} if args.label else None
    inputs = InputRegistries.initialize(
        root            = args.input_root,
        label_overrides = label_overrides,
    )
    acc_entry = inputs.acc.entry(args.id)
    acc_path  = inputs.acc.resolve(args.id)

    logger.info("=" * 60)
    logger.info("  AccountingStructure — Initialisation")
    logger.info("=" * 60)
    logger.info("  id              : %s", args.id)
    logger.info("  input acc       : v%d (date=%s code=%s label=%r)",
                acc_entry["version"], acc_entry["date"],
                acc_entry["code"], acc_entry["label"])
    logger.info("  reading from    : %s", acc_path)
    logger.info("  output dir      : %s", args.out)
    logger.info("=" * 60)

    t0 = time.perf_counter()
    acc = build_acc(acc_path, args.id, args.out)
    logger.info("Structure built in %.1fs", time.perf_counter() - t0)

    logger.info("STRUCTURE STATISTICS")
    logger.info("  Nodes              : %10d", acc.n_nodes)
    logger.info("  Time points        : %10d", acc.n_timeps)
    logger.info("  Valuations         : %10d", acc.n_valtns)
    logger.info("  Admissible edges   : %10d", acc.n_edges)
    logger.info("  All edges (no rule): %10d", acc.n_edges_all)
    logger.info("  Tensor shape       : %s", acc.A.shape)
    logger.info("  Tensor nnz         : %10d", acc.A.nnz)
    logger.info("  Bounds summary     : %s", acc.bounds.summary())

    edge_summary = build_edge_summary(acc)
    pair_counts  = (
        edge_summary
        .groupby(["from_sector", "to_sector"])
        .agg(
            n_edges        = ("edge_id", "count"),
            n_cross_region = ("from_region",
                              lambda x: (x.values != edge_summary.loc[x.index, "to_region"].values).sum()),
            n_time_vals    = ("time",
                              lambda x: x.nunique() * edge_summary.loc[x.index, "valuation"].nunique()),
        )
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )
    logger.info("ADMISSIBLE EDGES BY SECTOR PAIR\n%s", pair_counts.to_string(index=False))

    cross = edge_summary[edge_summary["from_region"] != edge_summary["to_region"]]
    if len(cross) > 0:
        cr_counts = (
            cross.groupby(["from_sector", "to_sector"])["edge_id"]
                 .count().reset_index()
                 .rename(columns={"edge_id": "n_cross_region_edges"})
                 .sort_values("n_cross_region_edges", ascending=False)
        )
        logger.info("CROSS-REGION EDGES (tradable flows only)\n%s",
                    cr_counts.to_string(index=False))
    else:
        logger.info("(no cross-region edges in admissible set)")

    logger.info("Saving structure …")
    acc.save()
    write_build_record(args.out, acc_entry)     # the input entry it was built from (#53)
    out_dir = os.path.join(args.out, args.id)
    edge_summary.to_csv(os.path.join(out_dir, "edge_summary.csv"), index=False)
    pair_counts .to_csv(os.path.join(out_dir, "sector_pair_counts.csv"), index=False)
    logger.info("Edge summary + sector-pair counts saved under: %s", out_dir)
    logger.info("=" * 60)
    logger.info("  Done.  AccountingStructure saved to: %s", out_dir)
    logger.info("=" * 60)
    return acc


if __name__ == "__main__":
    main()
