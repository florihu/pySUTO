"""
uncprop_diag.py
===============
Stage 5-diag driver: render the UncPropDiag portfolio (UD-R-01 contraction–shift,
UD-R-02 prior vs posterior bands, UD-R-03 shift calibration,
``contraction_summary.csv``, ``calibration.csv``, ``birge.csv``) for a persisted
uncprop run.

The opt and const runs are resolved from the uncprop run's sidecar
(``parents.upstream``), the imputation run from the const run's ``agg_id``, so
only ``uncprop_run_id`` is required.

Usage::

    python scripts/uncprop_diag.py <uncprop_run_id> [--label LABEL] [--regions Chile Zambia]
                                   [--data-dir data]

Figures land in ``<data-dir>/proc/uncprop/<uncprop_run_id>/diag/`` (``--regions`` adds a
``__<region>`` suffix and restricts the figures and the per-edge CSVs; ``birge.csv``
is run-level and never filtered).
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

TIMESTEPS   = None              # None → all years pooled in UD-R-01; or e.g. ("2020",)

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.diag    import UncPropDiag
from pysuto.logging import get_logger, start_run_log

_log = get_logger("uncprop_diag")


def main(uncprop_run_id: str, label: str = "", regions=None, *,
         data_dir: str = "data") -> None:
    """Render the diag of ``<data_dir>/proc/uncprop/<uncprop_run_id>``; its opt,
    const, imp and acc runs are read from ``<data_dir>/proc``."""
    proc = os.path.join(data_dir, "proc")
    uncprop_dir = os.path.join(proc, "uncprop")
    start_run_log(data_dir, "uncprop_diag")

    _log.info("=== uncprop_diag START ===")
    _log.info("Loading UncPropDiag from uncprop_run_id='%s'", uncprop_run_id)

    diag = UncPropDiag.from_uncprop_id(
        uncprop_run_id,
        uncprop_dir = uncprop_dir,
        opt_dir     = os.path.join(proc, "opt"),
        const_dir   = os.path.join(proc, "const"),
        imp_dir     = os.path.join(proc, "imp"),
        acc_dir     = os.path.join(proc, "acc"),
        label       = label,
    )

    diag.summary()
    paths = diag.plot(regions=regions, timesteps=TIMESTEPS)
    diag_dir = os.path.join(uncprop_dir, uncprop_run_id, "diag")
    _log.info("Wrote %d files to %s", len(paths), diag_dir)

    _log.info("=== uncprop_diag DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render UncPropDiag portfolio for a persisted uncprop run.",
    )
    parser.add_argument(
        "uncprop_run_id",
        help="Uncprop run ID under <data-dir>/proc/uncprop/. The opt and const runs are "
             "resolved from this run's sidecar.",
    )
    parser.add_argument(
        "--label", default="",
        help="Optional display label for figures (defaults to uncprop_run_id).",
    )
    parser.add_argument(
        "--regions", nargs="*", default=None,
        help="Restrict both figures and the CSV to edges touching these regions "
             "(filenames get a __<region> suffix).",
    )
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ (default 'data').")
    args = parser.parse_args()
    main(args.uncprop_run_id, args.label, args.regions or None, data_dir=args.data_dir)
