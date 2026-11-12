"""
stages.py
=========
The bodies of the Stage 2 and Stage 3 drivers as library functions over a
:class:`pysuto.case.CaseSpec` (GLOSSARY.md "Case", #26 §3, #124).

* :func:`run_imp` — Stage 2: collect the feeds, aggregate, impute with the case's
  cascade, register the imp run, draw its diagnostics.
* :func:`run_const` — Stage 3: assemble the constraint engine with the case's
  builders, run it over the acc's time range, persist the const run.

A driver (``scripts/collect_aggregate_impute.py``, ``scripts/constraints.py``) keeps the
argument parsing, the stage lock, the log handler and the knob protocol, and passes the
case object it resolved; the library never imports ``pysuto.cases`` (README §5). A case
declares every value these functions read as a field of its section
(:class:`pysuto.case.ImpSection`, :class:`pysuto.case.ConstSection`).

The module is driver-level code, like the scripts: it imports ``diag`` at module level,
which the data layer never does (README §5). The guard Stages 4–6 apply to the run they
read, ``check_case``, lives in :mod:`pysuto.case`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pysuto.acc import ALL_SPEC_KEYS, AccountingStructure, check_built_acc, check_feed_accs
from pysuto.case import Cascade, CaseContext, CaseSpec, check_case
from pysuto.collector import FeedCollector
from pysuto.constraints import ConstraintEngine
from pysuto.diag import (
    AggDiagnostic, CoverageDiagnostic, ImpDiagnostic, StatsDiagnostic,
    ownership_violations,
)
from pysuto.impute import ImputationEngine
from pysuto.input_registry import InputRegistries, input_pins
from pysuto.logging import get_logger
from pysuto.registry import DataFeedRegistry, ImputationRunRegistry


def run_imp(case: CaseSpec, feed_ids: list, label: str,
            overrides: Optional[dict] = None, *, data_dir: str = "data",
            imp_dir: Optional[str] = None, pin_inputs_from: Optional[str] = None,
            logger: Optional[logging.Logger] = None) -> str:
    """Run Stage 2 of ``case`` on ``feed_ids``; return the imp run id.

    The run builds from ``case``'s Stage 2 section on the acc ``case.acc_id``;
    ``overrides`` (the ``--set`` knobs already applied to ``case``) is recorded in the
    sidecar. ``data_dir`` holds ``proc/`` and ``input/``; the run is written to
    ``imp_dir`` (default ``<data_dir>/proc/imp``). ``pin_inputs_from`` names an imp run
    in ``<data_dir>/proc/imp`` whose recorded input versions this run uses (#53).
    """
    spec   = case.imp
    if spec.validate is not None:
        spec.validate(spec)
    logger = logger or get_logger("stages")
    acc_id = case.acc_id
    proc         = os.path.join(data_dir, "proc")
    acc_dir      = os.path.join(proc, "acc")
    registry_dir = os.path.join(proc, "feed", "registry")
    if imp_dir is None:
        imp_dir = os.path.join(proc, "imp")
    feed_keys = list(feed_ids)

    # ── 0  Initialise input registries (scan-and-bump) ────────────────────────
    pins = None
    if pin_inputs_from is not None:
        ref_sidecar = (ImputationRunRegistry(storage_dir=os.path.join(proc, "imp"))
                       .load_sidecar(pin_inputs_from))
        pins = input_pins(ref_sidecar, acc_id)
        logger.info("0  Input versions pinned from %s: %s", pin_inputs_from, pins)
    inputs = InputRegistries.initialize(root=os.path.join(data_dir, "input"), pins=pins)
    acc_entry = inputs.acc.entry(acc_id)
    logger.info("0  Input acc: %s v%d (code=%s)",
                acc_entry["id"], acc_entry["version"], acc_entry["code"])

    # ── A  Load infrastructure ────────────────────────────────────────────────
    # A pinned acc version is verifiable only against a structure with a build record.
    check_built_acc(acc_dir, acc_entry, inputs.acc, ALL_SPEC_KEYS,
                    require_record=pins is not None)
    logger.info("A  Loading AccountingStructure '%s' from %s", acc_id, acc_dir)
    acc = AccountingStructure.load(id=acc_id, storage_dir=acc_dir)

    logger.info("A  Loading DataFeedRegistry from %s", registry_dir)
    registry = DataFeedRegistry(odir=registry_dir)
    registry.load_index()

    # ── B  Collect and aggregate ──────────────────────────────────────────────
    logger.info("B  Collecting feeds: %s (p=%.2f)", feed_keys, spec.P_PROP)
    collector  = FeedCollector(registry=registry, acc_struct=acc, min_flow_kt=0.01)
    collector.select(feed_keys)
    feed_run_ids = [s["feed_run_id"] for s in collector.selected]
    logger.info("B  Resolved feed_run_ids: %s", feed_run_ids)
    check_feed_accs({r: registry.load_sidecar(r) for r in feed_run_ids},
                    inputs.acc, acc_entry)
    agg_result = collector.aggregate(p=spec.P_PROP)
    logger.info(
        "B  Aggregation done: %d covered / %d total edges",
        agg_result.n_covered, agg_result.n_edges,
    )

    ctx = CaseContext(case=case, acc=acc, inputs=inputs, data_dir=data_dir,
                      agg_result=agg_result, logger=logger)
    if spec.pre_impute is not None:
        spec.pre_impute(ctx)

    # Snapshot post-aggregation distributions BEFORE imputation mutates them.
    # The distribution panels focus on one year (agg_root_by_year stays all-year).
    diag_year = spec.diag_year or str(acc.time_range[0])
    agg_diag = AggDiagnostic(agg_result, acc_struct=acc, label=label, year=diag_year)
    agg_diag.summary()

    # ── C  Imputation pipeline ────────────────────────────────────────────────
    # The case's cascade (spec.imputers): its fixpoint group in the group's order, then
    # each terminal imputer, registered in the order listed.
    logger.info("C  Building imputation engine")
    engine = ImputationEngine(acc_struct=acc)

    for item in spec.imputers(ctx):
        if isinstance(item, Cascade):
            engine.register_cascade(item.imputers, max_iter=item.max_iter)
        else:
            engine.register(item)

    logger.info("C  Running imputation engine")
    agg_result = engine.run(agg_result)
    _meta_df    = agg_result.imputation_metadata.as_dataframe()
    _passes     = _meta_df["pass"].dropna()
    passes_run  = int(_passes.max()) if not _passes.empty else 0
    logger.info("C  Imputation done: %d log entries, cascade ran %d pass(es)",
                len(agg_result.imputation_log), passes_run)

    ctx.agg_result = agg_result
    if spec.post_impute is not None:
        spec.post_impute(ctx)

    # Ownership-partition check (ADR 0010): trade owns cross-region edges, the
    # network owns same-region edges. A clean run has zero violations; warn loudly
    # if a guard ever regressed.
    _viol = ownership_violations(agg_result.imputation_log.as_dataframe(), acc.edge_meta)
    if len(_viol):
        logger.warning("C  ownership-partition VIOLATIONS: %d edge(s) filled by an "
                       "unintended owner\n%s", len(_viol),
                       _viol.head(20).to_string(index=False))

    # ── D  Register run ───────────────────────────────────────────────────────
    logger.info("D  Registering imputation run in %s", imp_dir)
    imp_registry = ImputationRunRegistry(storage_dir=imp_dir)
    parents = {
        "inputs": {
            "acc":    acc_entry,
            "params": ctx.param_refs,
        },
        "upstream": {"feed_run_ids": feed_run_ids},
    }
    stage_config = {"p": spec.P_PROP}
    if spec.stage_config is not None:
        stage_config.update(spec.stage_config(ctx))
    stage_config.update({
        "input_pins": (None if pins is None
                       else {"from": pin_inputs_from, "versions": pins}),
        "overrides":  dict(overrides or {}),
        "case":       case.describe("imp"),
    })
    run_id = imp_registry.register(
        agg_result,
        label         = label,
        parents       = parents,
        imputers      = engine.imputers,
        stage_summary = {"cascade_passes_run": passes_run},
        stage_config  = stage_config,
    )
    run_dir = os.path.join(imp_dir, run_id)
    logger.info("D  Registered run_id=%s  label=%s → %s", run_id, label, run_dir)

    # ── E  Diagnostics ────────────────────────────────────────────────────────
    logger.info("E  Running diagnostics")
    diag_dir = os.path.join(run_dir, "diag")
    os.makedirs(diag_dir, exist_ok=True)
    for name, report in ctx.reports.items():      # the case's screens
        report.to_csv(os.path.join(diag_dir, f"{name}.csv"), index=False)
    agg_diag.plot(odir=diag_dir, show=False)
    diag = ImpDiagnostic(agg_result, acc_struct=acc, label=run_id, year=diag_year)
    diag.plot(odir=diag_dir, show=False)
    diag.summary()
    CoverageDiagnostic(agg_result, acc_struct=acc, year=diag_year).write(diag_dir)
    StatsDiagnostic(agg_result, acc_struct=acc, year=diag_year).write(diag_dir)

    return run_id


def run_const(case: CaseSpec, imp_run_id: str, label: str,
              overrides: Optional[dict] = None, *, data_dir: str = "data",
              imp_dir: Optional[str] = None, const_dir: Optional[str] = None,
              logger: Optional[logging.Logger] = None) -> str:
    """Build and persist the constraint run of ``imp_run_id``; return its id.

    The run builds from ``case``'s Stage 3 section on the acc ``case.acc_id``;
    ``overrides`` (the ``--set`` knobs already applied to ``case``) is recorded in the
    sidecar. ``data_dir`` holds ``proc/`` and ``input/``; the imp run is read from
    ``imp_dir`` (default ``<data_dir>/proc/imp``) and the run written to ``const_dir``
    (default ``<data_dir>/proc/const``).
    """
    spec    = case.const
    _log    = logger or get_logger("stages")
    acc_id  = case.acc_id
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
    check_case(imp_sidecar, case)

    ctx = CaseContext(case=case, acc=acc, inputs=inputs, data_dir=data_dir,
                      agg_result=agg_result, logger=_log)
    anchor_exempt, ctx.n_anchor_exempt = (spec.nodata_anchor_exempt(ctx)
                                          if spec.nodata_anchor_exempt is not None
                                          else (None, 0))

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
    stage_config = dict(spec.stage_config(ctx)) if spec.stage_config is not None else {}
    stage_config.update({
        "overrides": dict(overrides or {}),
        "case":      case.describe("const"),
    })
    constraint_run = engine.run(
        agg_result   = agg_result,
        timesteps    = [str(t) for t in acc.time_range],
        persist      = True,
        diag_plot    = spec.DIAG_PLOT,
        diag_year    = spec.diag_year,
        label        = label,
        parents      = parents,
        stage_config = stage_config,
    )

    # ── D. Summary ────────────────────────────────────────────────────────
    _log.info("ConstraintRun saved: %s  label=%s", constraint_run.run_id, label)
    _log.info("G shape: %s  (rows × edges)", constraint_run.G.shape)
    _log.info("Constraint type breakdown:\n%s",
              constraint_run.row_meta["type"].value_counts().to_string())
    _log.info("Sense breakdown:\n%s",
              constraint_run.row_meta["sense"].value_counts().to_string())
    return constraint_run.run_id
