"""
sweep_diag.py
=============
Score a sweep run's points against its base chain and rank its axes (#107, #108):
writes ``points.csv``, ``categories.csv``, ``retention_by_year.csv``,
``sensitivity.csv`` and the SW-R-01…04 figures under ``data/proc/sweep/<id>/diag/``
and prints the ranking.

    python scripts/sweep_diag.py <sweep_run_id> [--outcome "d_retention[max stage]"]

Outcomes (columns of points.csv, GLOSSARY.md "Outcome"): ``J_base_at_point`` (each
point's answer under the base QP; ≥ 0 while it is feasible for the base's hard rows) and
its families, ``mass_moved_total`` / ``mass_moved_<valuation>`` (kt over all years; gross
mass is dominated by the waste-rock satellite, so ``mass_moved_CC`` is the scale-free
one), and the set declared below: per Table S16 stage the change in retention
(``d_retention[…]``) and the relative change of the posterior (``d_total[…]``), the
largest |change| over the nine stages (``…[max stage]``, the A6 measure of #108), and
the concentrate-export retention of Round-E (c)'s exporters.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.diag import SweepDiag, SweepStatsDiagnostic
from pysuto.diag.stats.sweep import MaxAbs, Outcome

ROOT = os.path.join("data", "proc")

# Table S16's retention stages (from-sector, to-sector, valuation), as in
# scripts/report/results_figs.py STAGES.
STAGES = [("CO", "Mining", "Mass"), ("MF", "Concentrate", "CC"),
          ("Concentrate", "Smelting", "CC"), ("Anode", "ER", "CC"),
          ("CP", "Cathode", "CC"), ("Cathode", "FU", "CC"), ("Scrap", "Smelting", "CC"),
          ("Smelting", "Matte", "CC"), ("Scrap", "CFR", "CC")]
# Round-E (c): concentrate exports of the three exporters, cross-region CC, 2010–2019.
EXPORTERS    = ("Chile", "Peru", "Indonesia")
EXPORT_YEARS = [str(y) for y in range(2010, 2020)]


def _exports(region: str):
    def select(e):
        return ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
                & (e.region_from == region) & (e.region_to != region)
                & (e.valuation == "CC") & e.time.isin(EXPORT_YEARS))
    return select


def outcomes() -> tuple:
    stage = {f"{a}→{b} {v}": {"sector_from": a, "sector_to": b, "valuation": v}
             for a, b, v in STAGES}
    ret = tuple(Outcome(f"d_retention[{k}]", "retention", sel) for k, sel in stage.items())
    tot = tuple(Outcome(f"d_total[{k}]", "total", sel) for k, sel in stage.items())
    exp = tuple(Outcome(f"d_export_retention[{x}]", "retention", _exports(x))
                for x in EXPORTERS)
    return (ret + tot + (MaxAbs("d_retention[max stage]", ret),
                         MaxAbs("d_total[max stage]", tot)) + exp)


def main(sweep_run_id: str, outcome: str) -> None:
    stats = SweepStatsDiagnostic(sweep_run_id, root=ROOT, outcomes=outcomes())
    diag = SweepDiag(stats, outcome=outcome)
    odir = os.path.join(ROOT, "sweep", sweep_run_id, "diag")
    diag.plot(odir, show=False)
    diag.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score and rank a sweep run.")
    parser.add_argument("sweep_run_id")
    parser.add_argument("--outcome", default="d_retention[max stage]",
                        help="Outcome column SW-R-01/02 rank and plot.")
    args = parser.parse_args()
    main(args.sweep_run_id, args.outcome)
