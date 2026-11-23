"""
stages.py
=========
The bodies of the Stage 2–6 drivers as library functions over a
:class:`pysuto.case.CaseSpec` (GLOSSARY.md "Case", #26 §3, #124, #186).

* :func:`run_imp` — Stage 2: collect the feeds, aggregate, impute with the case's
  cascade, register the imp run, draw its diagnostics.
* :func:`run_const` — Stage 3: assemble the constraint engine with the case's
  builders, run it over the acc's time range, persist the const run.
* :func:`run_opt` — Stage 4: solve the σ-weighted QP of a const run, persist the opt run,
  draw its diagnostics.
* :func:`run_uncprop` — Stage 5: the posterior σ and intervals of an opt run.
* :func:`run_psut` — Stage 6: the tidy PSUT of an opt run, optionally its Sankeys.

A driver (``scripts/collect_aggregate_impute.py`` … ``scripts/psut.py``) keeps the
argument parsing, the stage lock, the log handler and the knob protocol, and passes the
case object it resolved; each step of ``python -m pysuto run`` calls the same functions
in a child process of its own (:mod:`pysuto._step`, :mod:`pysuto.pipeline`, #186). The
library never imports ``pysuto.tutorial`` (README §5). A case
declares every value Stages 2 and 3 read as a field of its section
(:class:`pysuto.case.ImpSection`, :class:`pysuto.case.ConstSection`);
Stages 4–6 read the constants of the block below, and a case replaces them through
``CaseSpec.driver_knobs`` (ADR 0045: a knob is overridden at the driver boundary, by
``--set`` or by the ``overrides`` argument, which wins over the case).

The module is driver-level code, like the scripts: it imports ``diag`` at module level,
which the data layer never does (README §5). The guard Stages 4–6 apply to the run they
read, ``check_case``, lives in :mod:`pysuto.case`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pysuto.acc import (ALL_SPEC_KEYS, EDGE_UNIVERSE_ROLE_KEYS, AccountingStructure,
                        check_built_acc, check_feed_accs)
from pysuto.case import Cascade, CaseContext, CaseSpec, check_case
from pysuto.clarabel_backend import ClarabelBackend
from pysuto.collector import FeedCollector
from pysuto.constraints import ConstraintEngine
from pysuto.diag import (AggDiagnostic, CoverageDiagnostic, ImpDiagnostic, SolverDiag,
                         StatsDiagnostic, UncPropDiag)
from pysuto.impute import ImputationEngine
from pysuto.input_registry import InputRegistries, InputRegistry, input_pins
from pysuto.knobs import check_type
from pysuto.logging import get_logger
from pysuto.psut import PSUT
from pysuto.registry import (ConstraintRunRegistry, DataFeedRegistry,
                             ImputationRunRegistry, OptimRunRegistry, PsutRunRegistry,
                             stage_timer)
from pysuto.solver import Solver
from pysuto.uncprop import KalmanUncProp


@stage_timer()
def run_imp(case: CaseSpec, feed_ids: list, label: str,
            overrides: Optional[dict] = None, *, data_dir: str = "data",
            imp_dir: Optional[str] = None, pin_inputs_from: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            strict_scope: bool = False) -> str:
    """Run Stage 2 of ``case`` on ``feed_ids``; return the imp run id.

    The run builds from ``case``'s Stage 2 section on the acc ``case.acc_id``;
    ``overrides`` (the ``--set`` knobs already applied to ``case``) is recorded in the
    sidecar. ``data_dir`` holds ``proc/`` and ``input/``; the run is written to
    ``imp_dir`` (default ``<data_dir>/proc/imp``). ``pin_inputs_from`` names an imp run
    in ``<data_dir>/proc/imp`` whose recorded input versions this run uses (#53).
    ``strict_scope`` makes an imputer's write outside its writer scope raise
    :class:`~pysuto.impute.ScopeError`; by default it is logged and dropped.
    """
    spec   = case.imp
    spec.validate()
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
    collector  = FeedCollector(registry=registry, acc_struct=acc,
                               min_flow_kt=spec.MIN_FLOW_KT)
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
    # The case's cascade (spec.imputers): its entries as a fixpoint group in
    # CASCADE_ORDER, then each terminal imputer, registered in the order listed.
    logger.info("C  Building imputation engine")
    engine = ImputationEngine(acc_struct=acc, strict_scope=strict_scope)

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


@stage_timer()
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

    # ── A3. Cross-stage consistency: the Lp exponent of Stage 2 and Stage 3 ─
    imp_p = imp_sidecar.get("stage_config", {}).get("p")
    if imp_p is None:
        raise ValueError(f"imp run {imp_run_id} records no stage_config['p']; cannot "
                         f"check it against this const run's P_PROP={spec.P_PROP}.")
    if float(imp_p) != float(spec.P_PROP):
        raise ValueError(f"P_PROP mismatch: imp run {imp_run_id} was aggregated with "
                         f"p={imp_p}, but this const run has P_PROP={spec.P_PROP}.")

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
        pin_exclude_sectors  = spec.PIN_EXCLUDE_SECTORS,
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


# =============================================================================
# Stages 4–6: the drivers' tuning constants (moved here from scripts/run_solver.py,
# uncertainty_propagation.py and psut.py so that the library runs without them, #186)
# =============================================================================
# USER CONFIGURATION
# Every constant keeps the value it had in its driver. The drivers' ``--set`` and a
# pipeline step's knobs override the ``OPT_KNOBS`` and ``PSUT_KNOBS``; a case replaces
# some through ``CaseSpec.driver_knobs``. Paths are not knobs: ``data_dir`` and its
# ``proc/`` subdirectories are arguments.

TIMESTEPS  = None                    # None → all timesteps in the run
# Feasibility pre-check: which years the probe checks before any solve — "all" (a
# zero-objective QP per year over its hard rows and bounds, on the same N_WORKERS
# threads as the solves; ADR 0042 replaced the full-G first-year probe, ≈ 7 of Stage 4's
# ≈ 11 min) or "none" (#104: a sweep point). Each year's QP reports its own status
# anyway: a `failed` status raises with or without the probe, which only localises it.
FEASIBILITY_CHECK = "all"
# Clarabel iteration cap. The default (200) is too low for the pinned, soft-stock
# 1.37M-row phase-1 feasibility QP (converges but not within 200 iters → spurious
# MaxIterations reported as infeasible). max_iter is a CAP, so easy runs still stop
# early; raising it only helps the hard pinned scan. 2000 solves 1995 feasibility.
MAX_ITER = 2000
# Clarabel termination tolerances (measured in Round W.1, ADR 0039; the measurement is
# in the REP roundW1-lower-bounds, "Solver-tolerance re-solve"). The former production
# setting (gap_rel 1e-3 / feas 1e-7, reduced 1e-2 / 1e-5) stopped every run while the
# cost was still falling by thousands per iteration (const_20260929_120611_6ef43f64,
# year 2015), and the reported gap bounds nothing: |objective| is dominated by the
# constant Σ(a0/σ)², and the dual cost at the stop sits above the true optimum. With
# the lower-bound rows the same stop left W.1 far short of its optimum. These
# tolerances run each year to its numerical floor (InsufficientProgress after ~150–250 iterations, ≈ 30–60 s/year)
# and accept that floor as AlmostSolved through the reduced tolerances (gap_rel
# 1e-5 is not reached there, 1e-4 is).
# #33 (ADR 0039 am. 2026-10-21): solved for the defects, every year of the final chain
# reaches Solved at gap 1e-8 in 64–81 iterations within 0.2 of a certified lower bound on
# J
# (opt_20261020_112726_2f451f31 vs opt_20261019_171746_3272682b: Σ_t J −66,702).
TOL_FEAS            = 1e-8
TOL_GAP_REL         = 1e-8
TOL_GAP_ABS         = 1e-8
REDUCED_TOL_FEAS    = 1e-6
REDUCED_TOL_GAP_REL = 1e-4
REDUCED_TOL_GAP_ABS = 10.0
# Parallel solve (#70): cold periods are independent QPs, so they run on
# N_WORKERS threads, each with its own Clarabel instance; 1 = the serial loop,
# the only setting --prior-mode warm accepts (a warm run is a chain). Each
# period's linear solver is pinned to one thread: multi-threaded faer was slower
# than one thread (1995 of const_20261012_113828_12c562c7: 48.0 s vs 34.9 s) and
# gave the objective of opt_20261012_122124_40b1b370 bit for bit (#70).
N_WORKERS           = os.cpu_count() or 1
# Clarabel's linear solver for the period QPs. "faer" is what "auto" picked in
# opt_20261012_122124_40b1b370 (sidecar linsolver_name). "qdldl" was rejected on
# measurement (#72): on const_20261019_170151_13460e94 its period solves took 52 s vs
# 234 s, but its posteriors were indistinguishable from faer's on the same QPs (Σ_t J
# +1,337.5, +0.043 %, lower in 13/30 years; opt_20261020_090246_99da4a39 vs
# opt_20261019_171746_3272682b), and Stage 4 fell only 745 → 563 s because the
# feasibility probe dominates (ADR 0039 am.).
DIRECT_SOLVE_METHOD = "faer"
# The period QP's decision variables (#33): "flows" solves for a, so Clarabel's cost
# is J − Σ(a_prior/σ_a)², a constant ≈ 1e6 against J ≈ 1e5; "defects" solves the
# same QP for d = a − a_prior (q = 0, cost = J). Probed on
# const_20261019_170151_13460e94, years 1995/2008/2009/2015/2016/2022
# (scripts/explo/issue33_conditioning_probe.py): at feas/gap 1e-8 "defects" reached
# Solved in every year in 71–81 iterations (20–22 s vs 52–68 s), its J within 0.2 of
# a certified dual lower bound, while "flows" stopped 516–4,349 above that bound
# (opt_20261019_171746_3272682b 945–4,363 above it), hard-row residuals ≤ 9e-7 kt vs
# ≤ 1e-7 kt. The sidecar's objective stays J − Σ(a_prior/σ_a)² under both; gap_rel is
# relative to Clarabel's own cost, so TOL_GAP_REL is ~10× tighter in absolute terms
# under "defects". Measured on the full chain (#33, ADR 0039 am. 2026-10-21):
# opt_20261020_112726_2f451f31 lowers Σ_t J by 66,702 (−2.13 % vs
# opt_20261019_171746_3272682b) in 30/30 years; the feasibility probe has no σ-weighted
# cost
# and is unaffected.
QP_VARIABLES        = "defects"

# The knobs a pipeline step may override with --set (#100, pysuto.knobs).
OPT_KNOBS = (
    "TIMESTEPS", "FEASIBILITY_CHECK", "MAX_ITER",
    "TOL_FEAS", "TOL_GAP_REL", "TOL_GAP_ABS",
    "REDUCED_TOL_FEAS", "REDUCED_TOL_GAP_REL", "REDUCED_TOL_GAP_ABS",
    "N_WORKERS", "DIRECT_SOLVE_METHOD", "QP_VARIABLES",
)

# Stage 5 (no sweep or case knobs: the step takes no `--set`; the constants below are
# recorded in the sidecar).
ALPHA                = 0.05
BIND_TOL             = 1e-6    # kt — a hard inequality row binds when G·a* − c ≥ −BIND_TOL
                               # (#27)
RANK_TOL             = 1e-10   # pivot tolerance for hard rows dependent on the others
                               # (#27)
EXCLUDE_SCALE_TOTALS = False   # True leaves the scale totals out of S(t) — ablation (#87)

# Stage 6.
PRIMARY   = None          # valuation the PSUT views default to (None: the acc's content
                          # valuation; copper's case names CC); every valuation is loaded
PSUT_KNOBS = ("PRIMARY",)  # the constants a case may set (CaseSpec.driver_knobs["psut"])
FIGS_DIR  = os.path.join("figs", "psut")   # where --sankey writes its figures
# END USER CONFIGURATION


def opt_defaults() -> dict:
    """The Stage 4 constants as they stand, ``{KNOB: value}`` over :data:`OPT_KNOBS`."""
    return {k: globals()[k] for k in OPT_KNOBS}


@stage_timer()
def run_opt(case: CaseSpec, const_run_id: str, label: str = "",
            overrides: Optional[dict] = None, *, prior_mode: str = "cold",
            data_dir: str = "data", const_dir: Optional[str] = None,
            logger: Optional[logging.Logger] = None) -> str:
    """Solve the constraint run ``const_run_id``; return the opt run id.

    ``case`` is recorded in the sidecar and must be the case that made the constraint
    run. The knobs are the constants of this module, except those the case sets in
    ``driver_knobs["opt"]`` (checked by the knob protocol); ``overrides`` (``--set``)
    wins over both and is recorded in the sidecar, as are the applied case values
    (``case_knobs``).

    ``data_dir`` holds ``proc/``; the constraint run is read from ``const_dir``
    (default ``<data_dir>/proc/const``), the run written to ``<data_dir>/proc/opt``
    and the accounting structure read from ``<data_dir>/proc/acc`` (checked against the
    acc input registry under ``<data_dir>/input``).
    """
    _log = logger or get_logger("stages")
    defaults = opt_defaults()
    over = dict(overrides or {})
    for name, value in over.items():
        if name not in OPT_KNOBS:
            raise KeyError(f"{name!r} is not a knob of Stage 4; its knobs are "
                           f"{', '.join(OPT_KNOBS)}")
        check_type(name, value, defaults[name])
    case_knobs = case.driver_knobs_for("opt", OPT_KNOBS, defaults)
    cfg = dict(defaults)
    cfg.update({k: v for k, v in case_knobs.items() if k not in over})
    cfg.update(over)
    proc    = os.path.join(data_dir, "proc")
    opt_dir = os.path.join(proc, "opt")
    acc_dir = os.path.join(proc, "acc")
    input_dir = os.path.join(data_dir, "input")
    if const_dir is None:
        const_dir = os.path.join(proc, "const")

    _log.info("=== run_opt START ===")
    # The diagnostics below load the built acc: check its build record against the
    # const run's acc ref before the solve, so a refused run persists nothing (#53).
    const_sidecar = ConstraintRunRegistry(const_dir).load_sidecar(const_run_id)
    check_case(const_sidecar, case)
    const_parents = const_sidecar["parents"]
    check_built_acc(acc_dir, const_parents["inputs"].get("acc", {}),
                    InputRegistry("acc", root=input_dir), EDGE_UNIVERSE_ROLE_KEYS)
    _log.info("Loading Solver with ClarabelBackend (const_dir=%s)", const_dir)
    # Termination tolerances: see the constants above (Round W.1).
    backend = ClarabelBackend(tol_feas=cfg["TOL_FEAS"],
                              tol_gap_rel=cfg["TOL_GAP_REL"],
                              tol_gap_abs=cfg["TOL_GAP_ABS"],
                              reduced_tol_feas=cfg["REDUCED_TOL_FEAS"],
                              reduced_tol_gap_rel=cfg["REDUCED_TOL_GAP_REL"],
                              reduced_tol_gap_abs=cfg["REDUCED_TOL_GAP_ABS"],
                              max_iter=cfg["MAX_ITER"],
                              direct_solve_method=cfg["DIRECT_SOLVE_METHOD"])
    solver = Solver(backend, const_dir=const_dir, opt_dir=opt_dir, acc_dir=acc_dir,
                    input_dir=input_dir,
                    prior_mode=prior_mode,
                    n_workers=cfg["N_WORKERS"],
                    qp_variables=cfg["QP_VARIABLES"],)

    _log.info("Solving ConstraintRun '%s'", const_run_id)
    results = solver.solve_all(
        const_run_id,
        timesteps         = cfg["TIMESTEPS"],
        feasibility_check = cfg["FEASIBILITY_CHECK"],
    )

    optim_run_id = solver.persist(
        run_id       = const_run_id,
        results      = results,
        output_dir   = opt_dir,
        const_run_id = const_run_id,
        label        = label,
        stage_config = {
            "prior_mode":                   solver.prior_mode,
            "max_iter":                     cfg["MAX_ITER"],
            "tol_feas":                     cfg["TOL_FEAS"],
            "tol_gap_rel":                  cfg["TOL_GAP_REL"],
            "tol_gap_abs":                  cfg["TOL_GAP_ABS"],
            "reduced_tol_feas":             cfg["REDUCED_TOL_FEAS"],
            "reduced_tol_gap_rel":          cfg["REDUCED_TOL_GAP_REL"],
            "reduced_tol_gap_abs":          cfg["REDUCED_TOL_GAP_ABS"],
            "feasibility_check":            cfg["FEASIBILITY_CHECK"],
            "overrides":                    dict(overrides or {}),
            "case_knobs":                   case_knobs,
            "case":                         case.ref(),
        },
    )

    total_iter   = sum(r.iterations   for r in results.values())
    total_time   = sum(r.solve_time_s for r in results.values())
    max_residual = max(
        float(abs(r.residuals).max()) if r.residuals.size else 0.0
        for r in results.values()
    )
    _log.info(
        "Solved %d timesteps  total_iter=%d  total_time=%.2fs  max|residual|=%.3e",
        len(results), total_iter, total_time, max_residual,
    )
    _log.info("OptimRun saved as '%s' under %s", optim_run_id, opt_dir)

    const_run = solver.load_run(const_run_id)
    diag_dir  = os.path.join(opt_dir, optim_run_id, "diag")

    # One AccountingStructure for figures and ranking CSVs (loading it is the
    # expensive step); the diag's stats object owns the all-years frames both read from.
    acc  = AccountingStructure.load(const_run.acc_id, storage_dir=acc_dir)
    diag = SolverDiag(results, const_run, label=label or optim_run_id, acc=acc)
    diag.summary()
    diag.plot(odir=diag_dir)

    _log.info("=== run_opt DONE ===")
    return optim_run_id


@stage_timer()
def run_uncprop(case: CaseSpec, opt_run_id: str, label: str = "", *,
                data_dir: str = "data",
                logger: Optional[logging.Logger] = None) -> str:
    """Propagate opt run ``opt_run_id`` of ``<data_dir>/proc/opt``; the run is written
    to ``<data_dir>/proc/uncprop``. ``case`` is recorded in the sidecar and must be the
    case that made the opt run. Returns the uncprop run id."""
    _log = logger or get_logger("stages")
    proc        = os.path.join(data_dir, "proc")
    opt_dir     = os.path.join(proc, "opt")
    const_dir   = os.path.join(proc, "const")
    uncprop_dir = os.path.join(proc, "uncprop")

    _log.info("=== run_uncprop START ===")
    _log.info("opt_run_id=%s  alpha=%.3f  exclude_scale_totals=%s",
              opt_run_id, ALPHA, EXCLUDE_SCALE_TOTALS)

    check_case(OptimRunRegistry(opt_dir).load_sidecar(opt_run_id), case)
    prop = KalmanUncProp(
        opt_dir              = opt_dir,
        const_dir            = const_dir,
        uncprop_dir          = uncprop_dir,
        bind_tol             = BIND_TOL,
        rank_tol             = RANK_TOL,
        exclude_scale_totals = EXCLUDE_SCALE_TOTALS,
    )

    result = prop.compute(opt_run_id, alpha=ALPHA)
    uncprop_run_id = prop.persist(result, opt_run_id, label=label, case=case.ref())

    contractions = [
        1.0 - ts.diagnostics["trace_post"] / ts.diagnostics["trace_prior"]
        for ts in result.per_timestep.values()
        if ts.diagnostics["trace_prior"] > 0.0
    ]
    mean_contraction = sum(contractions) / len(contractions) if contractions else 0.0
    max_resid_hard = max(
        ts.diagnostics["max_resid_hard"] for ts in result.per_timestep.values()
    )

    _log.info(
        "UncPropRun '%s' persisted under %s  (n_t=%d, mean_trace_contraction=%.3f, "
        "max_resid_hard=%.3e)",
        uncprop_run_id, uncprop_dir, len(result.per_timestep),
        mean_contraction, max_resid_hard,
    )

    diag = UncPropDiag.from_uncprop_id(
        uncprop_run_id,
        uncprop_dir = uncprop_dir,
        opt_dir     = opt_dir,
        const_dir   = const_dir,
        imp_dir     = os.path.join(proc, "imp"),
        acc_dir     = os.path.join(proc, "acc"),
        label       = label,
    )
    diag.summary()
    paths = diag.plot()
    _log.info("Wrote %d diag files to %s", len(paths),
              os.path.join(uncprop_dir, uncprop_run_id, "diag"))
    _log.info("=== run_uncprop DONE ===")
    return uncprop_run_id


@stage_timer()
def run_psut(case: CaseSpec, opt_run_id: str, label: str = "",
             primary: Optional[str] = None, timesteps: Optional[list] = None, *,
             data_dir: str = "data", sankey: Optional[list] = None,
             figs_dir: str = FIGS_DIR, png: bool = True,
             logger: Optional[logging.Logger] = None) -> str:
    """Materialise opt run ``opt_run_id`` of ``<data_dir>/proc/opt`` as a tidy PSUT and
    persist it under ``<data_dir>/proc/psut``; return the psut run id.

    ``primary`` is the valuation the views default to (``None``: the case's ``PRIMARY``
    knob, else the acc's content valuation); ``case`` is recorded in the sidecar and
    must be the case that made the opt run. ``sankey`` lists the periods whose World
    supply-use and process-network Sankeys are written to ``figs_dir`` (``[]``: the
    latest period; ``None``: none), each also as a PNG when ``png`` and kaleido can
    render.
    """
    _log = logger or get_logger("stages")
    _log.info("=== run_psut START ===")
    _log.info("opt_run_id=%s  primary=%s  timesteps=%s  data_dir=%s",
              opt_run_id, primary, timesteps or "all", data_dir)

    case_knobs = case.driver_knobs_for("psut", PSUT_KNOBS, {"PRIMARY": PRIMARY})
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

    _log.info("PsutRun '%s' persisted under %s  %r", psut_run_id, registry.storage_dir,
              psut)

    if sankey is not None:
        world = psut.aggregate(region="World", keep_crossings=True)
        piot  = world.to_piot("market_share", level="Z")
        os.makedirs(figs_dir, exist_ok=True)
        for period in (sankey or [world.latest()]):
            for name, view in (("supply_use", world), ("process_network", piot)):
                fig  = view.sankey(period)
                stem = os.path.join(
                    figs_dir, f"sankey_{name}_World_{period}_{psut.primary}")
                fig.write_html(stem + ".html", include_plotlyjs="cdn")
                _log.info("Sankey → %s.html", stem)
                if png:
                    _write_png(fig, stem + ".png", _log)
    _log.info("=== run_psut DONE ===")
    return psut_run_id


def _write_png(fig, path: str, _log: logging.Logger) -> None:
    """A static copy when kaleido (and its browser) can render; otherwise a note."""
    try:
        fig.write_image(path, width=1400, height=800, scale=1.5)
        _log.info("Sankey → %s", path)
    except Exception as exc:          # kaleido missing or no browser: the HTML stands
        _log.warning("PNG skipped for %s (%s: %s)", path, type(exc).__name__,
                     str(exc)[:120])
