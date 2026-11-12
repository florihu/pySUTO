"""
init_copper_constraints.py
==========================
Stage 3a driver: build and persist the constraint matrix for the copper model.

The constraint builders and their registration order are the case's
(pysuto.cases.copper.builders, #120; its docstring lists the constraint types):
hard mass balance, the supply/use scale totals and the scrap level anchors, the
CSV-fed soft ratios and hard ratio bounds, the net-stock and scrap-share bands, and
the hard lower bounds.

═══════════════════════════════════════════════════════════
CONFIGURATION  ←  every knob is a field of the case spec,
                  src/pysuto/cases/copper.py (edit it there)
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# Upstream wiring (`imp_run_id`) is passed via the `--imp-run-id` CLI flag. main()
# reads `acc_id` from `--config` (default config/config.yaml) and derives every path
# from `--data-dir` (default data): proc/acc, proc/imp, proc/const, input/. Every
# knob (P_PROP, σ floors, builder registrations) lives in the case spec
# (pysuto.cases.copper).

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

# make sure the package is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc import ALL_SPEC_KEYS, AccountingStructure, check_built_acc
from pysuto.case import CaseContext, CaseSpec
from pysuto.cases.copper import CASE
from pysuto.config import load_config
from pysuto.registry import ImputationRunRegistry, stage_lock
from pysuto.input_registry import InputRegistries
from pysuto.constraints import ConstraintEngine
from pysuto.logging import get_logger, add_run_handler
from pysuto.knobs import add_knob_args, overrides_from_cli, write_run_id

# The knobs a pipeline step may override with --set (#100, pysuto.knobs): the UPPER_CASE
# fields of the case's Stage 3 section. Paths and the acc id are not knobs.
KNOBS = tuple(CASE.const.knobs())

_log = get_logger("init_copper_constraints")


def main(imp_run_id: str, label: str, overrides: dict | None = None, *,
         case: CaseSpec = CASE,
         data_dir: str = "data", config: str = os.path.join("config", "config.yaml"),
         imp_dir: str | None = None, const_dir: str | None = None) -> str:
    """Build and persist the constraint run of ``imp_run_id``; return its id.

    The run builds from ``case``'s Stage 3 section (default the copper case; the
    command line passes it with its ``--set`` overrides applied); ``overrides`` is
    recorded in the sidecar. ``data_dir`` holds ``proc/`` and ``input/``; the imp run
    is read from ``imp_dir`` (default ``<data_dir>/proc/imp``) and the run written to
    ``const_dir`` (default ``<data_dir>/proc/const``).
    """
    from datetime import datetime
    spec = case.const
    add_run_handler(f"logs/init_copper_constraints_{datetime.now():%Y%m%d_%H%M%S}.log")

    acc_id  = load_config(config).pipeline.acc_id
    acc_dir = os.path.join(data_dir, "proc", "acc")
    if imp_dir is None:
        imp_dir = os.path.join(data_dir, "proc", "imp")
    if const_dir is None:
        const_dir = os.path.join(data_dir, "proc", "const")

    # ── 0  Initialise input registries (scan-and-bump) ────────────────────
    inputs    = InputRegistries.initialize(root=os.path.join(data_dir, "input"))
    acc_entry = inputs.acc.entry(acc_id)
    _log.info("0  Input acc: %s v%d (code=%s)",
              acc_entry["id"], acc_entry["version"], acc_entry["code"])

    # ── A. Load inputs ────────────────────────────────────────────────────
    # Stage 3 reads the built acc's bounds and masks too: every spec key but role.
    check_built_acc(acc_dir, acc_entry, inputs.acc, ALL_SPEC_KEYS)
    _log.info("Loading AccountingStructure '%s' from %s", acc_id, acc_dir)
    acc = AccountingStructure.load(acc_id, storage_dir=acc_dir)
    _log.info("  nodes=%d  edges=%d  time=%s–%s",
              acc.n_nodes, acc.n_edges, acc.time_range[0], acc.time_range[-1])

    _log.info("Loading imputation run from %s", imp_dir)
    imp_registry = ImputationRunRegistry(storage_dir=imp_dir)
    agg_result = imp_registry.load(imp_run_id)
    _log.info("  f_agg shape=%s  coverage=%d edges  imp_run_id=%s",
              agg_result.f_agg.shape, agg_result.coverage.sum(), imp_run_id)

    # ── A2. Cross-stage consistency: imp must reference same acc.id+version
    imp_sidecar = imp_registry.load_sidecar(imp_run_id)
    imp_acc_ref = (imp_sidecar.get("parents", {})
                              .get("inputs", {})
                              .get("acc"))
    if imp_acc_ref and (imp_acc_ref.get("id"), imp_acc_ref.get("version")) \
                       != (acc_entry["id"], acc_entry["version"]):
        raise RuntimeError(
            f"acc mismatch: imp run was built on "
            f"acc {imp_acc_ref!r}, but this const run is using "
            f"{acc_entry['id']} v{acc_entry['version']}."
        )

    ctx = CaseContext(case=case, acc=acc, inputs=inputs, data_dir=data_dir,
                      agg_result=agg_result, logger=_log)
    anchor_exempt, n_grade_ry = (spec.nodata_anchor_exempt(ctx)
                                 if spec.nodata_anchor_exempt is not None else (None, 0))

    # ── B. Assemble constraint engine ─────────────────────────────────────
    engine = ConstraintEngine(
        acc_struct        = acc,
        storage_dir       = const_dir,
        p                 = spec.P_PROP,
        sigma_a_cv_min    = spec.SIGMA_A_CV_MIN,
        sigma_a_abs_floor = spec.SIGMA_A_ABS_FLOOR,
        sigma_c_cv_min    = spec.SIGMA_C_CV_MIN,
        sigma_c_abs_floor = spec.SIGMA_C_ABS_FLOOR,
        prior_box_k       = spec.PRIOR_BOX_K,
        nodata_anchor_cv  = spec.NODATA_ANCHOR_CV,
        nodata_anchor_exclude_sectors = spec.NODATA_ANCHOR_EXCLUDE,
        nodata_anchor_exempt_mask     = anchor_exempt,
        pin_flat_prior_zeros = spec.PIN_FLAT_PRIOR_ZEROS,
    )

    for builder in spec.builders(ctx):
        engine.register(builder)

    # ── C. Run and persist ────────────────────────────────────────────────
    _log.info("Running ConstraintEngine over %d timesteps", len(acc.time_range))
    parents = {
        "inputs": {
            "acc":    acc_entry,
            "params": ctx.param_refs,
        },
        "upstream": {"imp_run_id": imp_run_id},
    }
    constraint_run = engine.run(
        agg_result   = agg_result,
        timesteps    = [str(t) for t in acc.time_range],
        persist      = True,
        diag_plot    = spec.DIAG_PLOT,
        label        = label,
        parents      = parents,
        stage_config = {
            "sigma_a_cv_min":    spec.SIGMA_A_CV_MIN,
            "sigma_a_abs_floor": spec.SIGMA_A_ABS_FLOOR,
            "sigma_c_cv_min":    spec.SIGMA_C_CV_MIN,
            "sigma_c_abs_floor": spec.SIGMA_C_ABS_FLOOR,
            "prior_box_k":       spec.PRIOR_BOX_K,
            "nodata_anchor_cv":  spec.NODATA_ANCHOR_CV,
            "nodata_anchor_exclude": list(spec.NODATA_ANCHOR_EXCLUDE),
            "nodata_anchor_data_aware": list(spec.NODATA_ANCHOR_DATA_AWARE),
            "nodata_anchor_grade_region_years": n_grade_ry,
            "pin_flat_prior_zeros": spec.PIN_FLAT_PRIOR_ZEROS,
            "ratio_sigma_rel_floor": spec.RATIO_SIGMA_REL_FLOOR,
            "ratio_scale_abs_floor": spec.RATIO_SCALE_ABS_FLOOR,
            "stock_bounds_scale_abs_floor": spec.STOCK_BOUNDS_SCALE_ABS_FLOOR,
            "ore_grade_basis": "in-situ (ADR 0015)",
            "fabricated_anchor_cv": spec.FABRICATED_ANCHOR_CV,
            "feed_share_observed":  spec.FEED_SHARE_OBSERVED,
            "stock_bounds_mode": spec.STOCK_BOUNDS_MODE,
            "stock_bounds_skip_reported": spec.STOCK_BOUNDS_SKIP_REPORTED,
            "scrap_share_mode":  spec.SCRAP_SHARE_MODE,
            "scrap_level_anchor": spec.SCRAP_LEVEL_ANCHOR,
            "massbal_mask":      spec.MASSBAL_MASK,
            "enforce_lower_bounds": spec.ENFORCE_LOWER_BOUNDS,
            "stripping_ratio_hard": spec.STRIPPING_RATIO_HARD,
            "overrides":         dict(overrides or {}),
            "case":              case.describe("const"),
        },
    )

    # ── D. Summary ────────────────────────────────────────────────────────
    _log.info("ConstraintRun saved: %s  label=%s", constraint_run.run_id, label)
    _log.info("G shape: %s  (rows × edges)", constraint_run.G.shape)
    _log.info("Constraint type breakdown:\n%s",
              constraint_run.row_meta["type"].value_counts().to_string())
    _log.info("Cone type breakdown:\n%s",
              constraint_run.row_meta["cone_type"].value_counts().to_string())
    return constraint_run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and persist constraint matrix.")
    parser.add_argument(
        "--imp-run-id", required=True,
        help="ImputationRun ID to load from --imp-dir.",
    )
    parser.add_argument("label", help="Label for this constraint run")
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ and input/ (default 'data').")
    parser.add_argument("--config", default=os.path.join("config", "config.yaml"),
                        help="Config naming the acc id (default config/config.yaml).")
    parser.add_argument("--imp-dir", default=None,
                        help="Where the imputation run is read from "
                             "(default <data-dir>/proc/imp).")
    parser.add_argument("--const-dir", default=None,
                        help="Where the run is written (default <data-dir>/proc/const); "
                             "a sweep's scratch store (#108).")
    add_knob_args(parser)
    args = parser.parse_args()
    overrides = overrides_from_cli(parser, args, KNOBS, CASE.const.knobs())
    case = CASE.with_knobs("const", overrides)
    # registries have no locking (#101)
    with stage_lock("constraints", root=os.path.join(args.data_dir, "proc")):
        write_run_id(args.run_id_file,
                     main(args.imp_run_id, args.label, overrides, case=case,
                          data_dir=args.data_dir, config=args.config,
                          imp_dir=args.imp_dir, const_dir=args.const_dir))
