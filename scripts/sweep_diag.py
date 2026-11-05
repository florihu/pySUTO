"""
sweep_diag.py
=============
Score a sweep run's points against its base chain, rank and size its axes and judge its
criteria (#107, #108, #117): writes ``points.csv``, ``categories.csv``,
``retention_by_year.csv``, ``sensitivity.csv``, ``year_profiles.csv``,
``sensitivity_by_year.csv``, ``verdicts.csv`` and the SW-R-01, 02 and 04–07 figures
under ``data/proc/sweep/<id>/diag/``, and prints the verdicts and the ranking.

    python scripts/sweep_diag.py <sweep_run_id>

Outcomes (columns of points.csv, GLOSSARY.md "Outcome"): ``J_base_at_point`` (each
point's answer under the base QP; ≥ 0 while it is feasible for the base's hard rows) and
its families, ``mass_moved_total`` / ``mass_moved_<valuation>`` (kt over all years; gross
mass is dominated by the waste-rock satellite, so ``mass_moved_CC`` is the scale-free
one), and the set declared below: the change in pooled retention of every CC category
(``d_retention[<category> CC]``), per Table S16 stage the change in pooled retention and
the relative change of the posterior (``d_total[…]``), the largest |change| over the
nine stages (``…[max stage]``), and the concentrate-export retention of Round-E (c)'s
exporters over 2010–2019. The criteria are #108's, registered before the sweep was read.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.diag import SweepDiag, SweepStatsDiagnostic
from pysuto.diag.stats.sweep import Criterion, EveryCategory, MaxAbs, Outcome

ROOT = os.path.join("data", "proc")

# Table S16's retention stages (from-sector, to-sector, valuation), as in
# scripts/report/results_figs.py STAGES.
STAGES = [("CO", "Mining", "Mass"), ("MF", "Concentrate", "CC"),
          ("Concentrate", "Smelting", "CC"), ("Anode", "ER", "CC"),
          ("CP", "Cathode", "CC"), ("Cathode", "FU", "CC"), ("Scrap", "Smelting", "CC"),
          ("Smelting", "Matte", "CC"), ("Scrap", "CFR", "CC")]
# Round-E (c): concentrate exports of the three exporters, cross-region CC, 2010–2019.
EXPORTERS     = ("Chile", "Peru", "Indonesia")
EXPORT_WINDOW = ("2010", "2019")


def _exports(region: str):
    def select(e):
        return ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
                & (e.region_from == region) & (e.region_to != region)
                & (e.valuation == "CC"))
    return select


def declarations() -> tuple:
    """The outcomes and the criteria of #108."""
    # A stage's selector is the one EveryCategory gives its category, so a CC stage is
    # declared and scored once.
    stage = {f"{a}→{b} {v}": {"category": f"{a}→{b}", "valuation": v}
             for a, b, v in STAGES}
    ret = tuple(Outcome(f"d_retention[{k}]", "retention", sel) for k, sel in stage.items())
    tot = tuple(Outcome(f"d_total[{k}]", "total", sel) for k, sel in stage.items())
    exp = tuple(Outcome(f"d_export_retention[{x}]", "retention", _exports(x),
                        EXPORT_WINDOW) for x in EXPORTERS)
    outcomes = (ret + (EveryCategory("CC"),) + tot
                + (MaxAbs("d_retention[max stage]", ret), MaxAbs("d_total[max stage]", tot))
                + exp)
    criteria = (Criterion("A6", ret, 0.02),                 # every stage, every point
                Criterion("Round-E (c)", exp, 0.01))
    return outcomes, criteria


def main(sweep_run_id: str) -> None:
    outcomes, criteria = declarations()
    stats = SweepStatsDiagnostic(sweep_run_id, root=ROOT, outcomes=outcomes,
                                 criteria=criteria)
    diag = SweepDiag(stats, valuation="CC")
    odir = os.path.join(ROOT, "sweep", sweep_run_id, "diag")
    diag.plot(odir, show=False)
    diag.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score, rank and judge a sweep run.")
    parser.add_argument("sweep_run_id")
    args = parser.parse_args()
    main(args.sweep_run_id)
