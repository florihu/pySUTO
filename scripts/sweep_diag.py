"""
sweep_diag.py
=============
Score a sweep run's points against its base chain and rank its axes (#107):
writes ``points.csv``, ``categories.csv``, ``retention_by_year.csv``,
``sensitivity.csv`` and the SW-R-01…04 figures under ``data/proc/sweep/<id>/diag/``
and prints the ranking.

    python scripts/sweep_diag.py <sweep_run_id> [--outcome J_base_at_point]
                                 [--category "MF→Concentrate CC" ...]

Outcomes (columns of points.csv): ``J_base_at_point`` (each point's answer under the
base QP; ≥ 0 while it is feasible for the base's hard rows) and its families,
``mass_moved_total`` / ``mass_moved_<valuation>`` (kt over all years; gross mass is
dominated by the waste-rock satellite), ``J_own`` (comparable only across points whose
σ-defining knobs equal the base's), and ``mass_moved_<cat>`` / ``retention_<cat>`` for
each ``--category``.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.diag import SweepDiag, SweepStatsDiagnostic

ROOT = os.path.join("data", "proc")


def main(sweep_run_id: str, outcome: str, categories: list) -> None:
    stats = SweepStatsDiagnostic(sweep_run_id, root=ROOT, outcome_categories=categories)
    diag = SweepDiag(stats, outcome=outcome)
    odir = os.path.join(ROOT, "sweep", sweep_run_id, "diag")
    diag.plot(odir, show=False)
    diag.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score and rank a sweep run.")
    parser.add_argument("sweep_run_id")
    parser.add_argument("--outcome", default="J_base_at_point",
                        help="Outcome column SW-R-01/02 rank and plot.")
    parser.add_argument("--category", action="append", default=[], dest="categories",
                        help="A category promoted to outcome columns (repeatable).")
    args = parser.parse_args()
    main(args.sweep_run_id, args.outcome, args.categories)
