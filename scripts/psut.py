"""
psut.py
=======
Stage 6 driver: materialise the solved edge vector a* of a persisted OptimRun
as a tidy physical supply-use table (PSUT) and persist it, one parquet per
period, under data/proc/psut/<psut_run_id>/.

Pipeline
--------
1  PSUT.from_opt_run(opt_run_id, ...) — resolves the AccountingStructure via
   provenance (opt sidecar → const sidecar → acc id), maps year-local F_opt to
   global edge ids through cols_t and classifies every edge into its block
   (S, U, Y, E, O, P; every valuation, ``primary`` = the default for views)
2  PsutRunRegistry.register(psut, parents=..., label=...) — writes
   timesteps/t_<period>.parquet + nodes.parquet + the harmonized sidecar with
   parents.upstream.opt_run_id
3  --sankey [PERIOD ...] — collapse to World (keep_crossings=True), build the
   market-share PIOT, and write both Sankeys per period — the supply-use
   Sankey of the PSUT and the process-network Sankey of the PIOT — as HTML
   (+ PNG when kaleido can render) under figs/psut/

Usage::

    python scripts/psut.py <opt_run_id> [--label LABEL] [--primary CC]
                           [--timesteps 2020 2021 ...] [--data-dir data]
                           [--sankey [2020 2024 ...]] [--figs-dir figs/psut] [--no-png]
                           [--run-id-file PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR  = "data"          # holds proc/opt, proc/const, proc/acc; output → proc/psut
PRIMARY   = None          # valuation the PSUT views default to (None: the acc's content
                          # valuation; copper's case names CC); every valuation is loaded
KNOBS     = ("PRIMARY",)  # the constants a case may set (CaseSpec.driver_knobs["psut"])
FIGS_DIR  = os.path.join("figs", "psut")   # where --sankey writes its figures
# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.case import CaseSpec, check_case
from pysuto.cases import add_case_arg, case_from_args
from pysuto.cases.copper import CASE
from pysuto.knobs import write_run_id
from pysuto.psut import PSUT
from pysuto.registry import OptimRunRegistry, PsutRunRegistry, stage_lock
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_psut")


def main(
    opt_run_id: str,
    label:      str = "",
    primary:    str | None = None,       # None: the case's PRIMARY, else the content valuation
    timesteps:  list | None = None,
    data_dir:   str = DATA_DIR,
    sankey:     list | None = None,      # periods to draw; [] → the latest
    figs_dir:   str = FIGS_DIR,
    png:        bool = True,             # also a static PNG per figure, when kaleido can render
    case:       CaseSpec = CASE,         # recorded in the sidecar; must have made the opt run
) -> str:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/init_psut_{datetime.now():%Y%m%d_%H%M%S}.log")

    _log.info("=== init_psut START ===")
    _log.info("opt_run_id=%s  primary=%s  timesteps=%s  data_dir=%s",
              opt_run_id, primary, timesteps or "all", data_dir)

    case_knobs = case.driver_knobs_for("psut", KNOBS, globals())
    if primary is None:
        primary = case_knobs.get("PRIMARY", PRIMARY)
    proc = os.path.join(data_dir, "proc")
    check_case(OptimRunRegistry(os.path.join(proc, "opt")).load_sidecar(opt_run_id), case)
    psut = PSUT.from_opt_run(
        opt_run_id,
        timesteps = timesteps,
        primary   = primary,
        opt_dir   = os.path.join(proc, "opt"),
        const_dir = os.path.join(proc, "const"),
        acc_dir   = os.path.join(proc, "acc"),
        input_dir = os.path.join(data_dir, "input"),
    )

    registry    = PsutRunRegistry(os.path.join(proc, "psut"))
    psut_run_id = registry.register(
        psut,
        parents = {
            "inputs":   {"acc": psut.acc_ref},
            "upstream": {"opt_run_id": opt_run_id, "const_run_id": psut.const_run_id},
        },
        stage_config = {"primary": psut.primary, "case_knobs": case_knobs,
                         "case": case.ref()},
        label = label,
    )

    _log.info("PsutRun '%s' persisted under %s  %r", psut_run_id, registry.storage_dir, psut)

    if sankey is not None:
        world = psut.aggregate(region="World", keep_crossings=True)
        piot  = world.to_piot("market_share", level="Z")
        os.makedirs(figs_dir, exist_ok=True)
        for period in (sankey or [world.latest()]):
            for name, view in (("supply_use", world), ("process_network", piot)):
                fig  = view.sankey(period)
                stem = os.path.join(figs_dir, f"sankey_{name}_World_{period}_{psut.primary}")
                fig.write_html(stem + ".html", include_plotlyjs="cdn")
                _log.info("Sankey → %s.html", stem)
                if png:
                    _write_png(fig, stem + ".png")
    _log.info("=== init_psut DONE ===")
    return psut_run_id


def _write_png(fig, path: str) -> None:
    """A static copy when kaleido (and its browser) can render; otherwise a note."""
    try:
        fig.write_image(path, width=1400, height=800, scale=1.5)
        _log.info("Sankey → %s", path)
    except Exception as exc:                    # kaleido missing or no browser: the HTML stands
        _log.warning("PNG skipped for %s (%s: %s)", path, type(exc).__name__, str(exc)[:120])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Materialise a persisted OptimRun as a tidy PSUT (one parquet per period)."
    )
    parser.add_argument("opt_run_id", help="OptimRun ID from data/proc/opt/.")
    parser.add_argument("--label", default="", help="Optional label for the psut run sidecar.")
    parser.add_argument("--primary", default=None,
                        help="Valuation the views default to; every valuation is loaded "
                             "(default: the case's, else the acc's content valuation).")
    parser.add_argument("--timesteps", nargs="+", default=None,
                        help="Subset of timestep labels (default: every timestep of the run).")
    add_case_arg(parser)
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Data root holding proc/opt, proc/const, proc/acc (default {DATA_DIR!r}).")
    parser.add_argument("--sankey", nargs="*", default=None, metavar="PERIOD",
                        help="Write the World supply-use and process-network Sankeys for these "
                             "periods (flag without values: the latest period).")
    parser.add_argument("--figs-dir", default=FIGS_DIR,
                        help=f"Directory for the Sankey HTML/PNG files (default {FIGS_DIR!r}).")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip the static PNG copies (needs kaleido and a browser).")
    parser.add_argument("--run-id-file", default=None, metavar="PATH",
                        help="Write the persisted run id to PATH (pysuto.pipeline).")
    args = parser.parse_args()
    case = case_from_args(parser, args)
    try:
        case.driver_knobs_for("psut", KNOBS, globals())     # a bad case value is a usage error
    except (KeyError, TypeError) as exc:
        parser.error(exc.args[0] if exc.args else str(exc))
    with stage_lock("psut", root=os.path.join(args.data_dir, "proc")):   # #101
        write_run_id(args.run_id_file,
                     main(args.opt_run_id, args.label, args.primary, args.timesteps,
                          args.data_dir, args.sankey, args.figs_dir, not args.no_png, case))
