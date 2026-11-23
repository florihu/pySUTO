"""
toy_ext.py
==========
The toy case's **extras** (#61): the code behind the notebooks ``08_break_it`` and
``09_bridge``. Nothing here belongs to the core toy chain; :mod:`pysuto.tutorial.toy` is
imported, never edited.

Break it
--------
Each ``break_*`` takes the run tree ``run_root`` of a finished toy chain plus its knobs
as arguments, runs the stages it must into the same tree under a **registry label other
than** ``"toy"`` (``toy.latest_run_id`` selects by label, so an experiment never becomes "the
latest run" of the core chapters), and returns what it ran: the run ids, and a small
frame the notebook displays. The base chain's runs are reused wherever nothing changed.

:func:`break_contradiction`
    σ cannot repair a contradiction (ADR 0019, 0035). Copper's prior-anchored scale
    totals (``SumConstraintBuilder.from_mask_anchor``, 1 %, ADR 0006) restate the
    planted Customs leg's prior at 1 %; loosening the σ of the row it contradicts hides
    the residual and leaves the mass wrong; rebuilding the Customs feed without the
    planted error removes both.
:func:`break_phantom`
    A second Customs leg, larger than the exporter's whole output: Stage 4 refuses it.
:func:`break_tolerance`
    One year solved at two relative gaps (ADR 0039).

The bridge
----------
The three seams a user extends, each a few lines: :class:`ScrapShareSeed` (a
``BaseImputer``), :class:`NetStockRow` (a ``BaseConstraintBuilder``) and
:func:`fourth_feed` (a synthetic feed through the Stage 1 API). The posterior-propagator
seam is described by the notebook, not coded.

Your own case
-------------
The helpers of ``10_your_case`` (#197): :func:`params_from_template` (a params family
under the shipped template's header), :func:`register_feeds` (the acc built from its
registered spec, the feeds run through ``run_feed``) and :func:`run_case`
(``python -m pysuto run`` as a call).
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto._conditioning import condition_prior
from pysuto import acc_store
from pysuto.acc import AccountingStructure, read_acc_spec_dir
from pysuto.acc_store import write_build_record
from pysuto.tutorial import toy
from pysuto.clarabel_backend import ClarabelBackend
from pysuto.constraints import (BaseConstraintBuilder, ConstraintBlock, FlowBoundsBuilder,
                                anchors)
from pysuto.constraints.naming import ROW_META_COLS, build_constraint_id
from pysuto.impute import BaseImputer
from pysuto.input_registry import InputRegistries
from pysuto.feedpipe import FeedSpec, run_feed
from pysuto.pipeline import ALL_STEPS
from pysuto.registry import (
    ConstraintRunRegistry,
    DataFeedRegistry,
    ImputationRunRegistry,
    OptimRunRegistry,
    UncPropRunRegistry,
)
from pysuto.solver import Solver


# =============================================================================
# Plumbing: a labelled variant of the toy chain
# =============================================================================

@contextlib.contextmanager
def _variant(label: str, case=None):
    """Run the ``toy.run_stage*`` functions under ``label`` (and, optionally, another
    :class:`~pysuto.case.CaseSpec`): they read ``toy.LABEL`` and ``toy.CASE`` at call
    time, so the experiment reuses them instead of copying them. Not thread-safe."""
    if label == toy.LABEL:
        raise ValueError(f"an experiment must not use the core label {toy.LABEL!r}")
    saved = toy.LABEL, toy.CASE
    toy.LABEL, toy.CASE = label, (case if case is not None else toy.CASE)
    try:
        yield
    finally:
        toy.LABEL, toy.CASE = saved


def _with_builders(builders: Callable):
    """``toy.CASE`` with its Stage 3 builders replaced."""
    return dataclasses.replace(
        toy.CASE, const=dataclasses.replace(toy.CASE.const, builders=builders))


def base_ids(run_root) -> dict:
    """The latest ``"toy"``-labelled ``imp``, ``const``, ``opt`` and ``uncprop`` runs of
    the tree: the core chain the experiments start from."""
    tree = toy.run_tree(run_root)
    return {"imp":     toy.latest_run_id(ImputationRunRegistry(str(tree["imp"]))),
            "const":   toy.latest_run_id(ConstraintRunRegistry(str(tree["const"]))),
            "opt":     toy.latest_run_id(OptimRunRegistry(str(tree["opt"]))),
            "uncprop": toy.latest_run_id(UncPropRunRegistry(str(tree["uncprop"])))}


def _feed_run_ids(run_root, imp_run_id: str) -> "dict[str, str]":
    """The feed runs an imp run aggregated, by feed id (the label they were registered
    under)."""
    tree = toy.run_tree(run_root)
    feeds = DataFeedRegistry(odir=str(tree["feed"]))
    feeds.load_index()
    ids = toy.upstream_run_id(ImputationRunRegistry(str(tree["imp"])).load_sidecar(imp_run_id),
                        "feed_run_ids")
    return {feeds.load_sidecar(r)["label"]: r for r in ids}


def _register_feed(run_root, feed: toy.SyntheticFeed, label: str,
                   seed: int = toy.SEED) -> str:
    """Stage 1 for one synthetic feed, under ``label`` (a feed registered as
    ``"customs"`` would become the latest Customs feed of the core chapters)."""
    tree      = toy.run_tree(run_root)
    acc_entry = InputRegistries.initialize(root=str(tree["input"])).acc.entry(toy.ACC_ID)
    registry  = DataFeedRegistry(odir=str(tree["feed"]))
    registry.load_index()
    entry, _ = toy._register_synthetic_feed(feed, toy.load_acc(tree), acc_entry,
                                            registry, label, seed)
    return entry.run_id


def _const_of(run_root, opt_run_id: str):
    """The loaded const run an opt run solved (the opt sidecar's ``const_run_id``)."""
    tree = toy.run_tree(run_root)
    return ConstraintRunRegistry(str(tree["const"])).load(
        toy.upstream_run_id(OptimRunRegistry(str(tree["opt"])).load_sidecar(opt_run_id),
                      "const_run_id"))


def _run_stages(run_root, imp_run_id: str, uncprop: bool = False) -> dict:
    """Stages 3–4 (and 5 with ``uncprop``) from an imp run, under the current
    :func:`_variant`: ``{"imp", "const", "opt"[, "uncprop"]}``."""
    const = toy.run_stage3(run_root, imp_run_id)
    opt   = toy.run_stage4(run_root, const)
    ids   = {"imp": imp_run_id, "const": const, "opt": opt}
    if uncprop:
        ids["uncprop"] = toy.run_stage5(run_root, opt)
    return ids


def _row_z(run_root, opt_run_id: str, soft: pd.DataFrame, **match) -> float:
    """The signed z of the soft rows whose ``row_meta`` columns equal ``match`` (the
    largest ``|z|`` of them; ``row_meta``, never a parsed ``constraint_id``)."""
    meta = _const_of(run_root, opt_run_id).row_meta
    keep = np.ones(len(meta), dtype=bool)
    for col, value in match.items():
        keep &= (meta[col] == value).to_numpy()
    z = soft[soft["constraint_id"].isin(set(meta.loc[keep, "constraint_id"]))]["z"]
    return float(z.iloc[z.abs().argmax()]) if len(z) else float("nan")


def _score(run_root, ids: dict, seed: int = toy.SEED) -> dict:
    """What the experiments tabulate about one chain: the reconciled estimate against
    the truth (:func:`toy.scoreboard`) and the largest soft residual."""
    tree = toy.run_tree(run_root)
    diag = toy.scoreboard(run_root, ids, seed)
    acc  = toy.load_acc(tree)
    planted = toy.reveal()
    edge = toy._edge_id(acc, *(planted[k] for k in toy._PLANTED_EDGE))
    rel  = diag.error_by_stage().loc["reconciled"]
    soft = toy.soft_residuals(run_root, ids["opt"])
    rows = soft[soft["kind"] == "row"]
    return {"planted_row_z": _row_z(run_root, ids["opt"], soft, variant="usetotout",
                                    sector=planted["from_sector"],
                                    region=planted["from_region"], time=planted["time"]),
            "top_row_z":      float(rows["z"].abs().max()),
            "top_row":        str(rows["constraint_id"].iloc[0]),
            "coverage":       diag.coverage(),
            "planted_a_star": float(diag.estimates["reconciled"][edge]),
            "planted_truth":  float(diag.truth[edge]),
            "median_rel_err": float(rel["median"]),
            "p90_rel_err":    float(rel["p90"]),
            "imp": ids["imp"], "opt": ids["opt"], "uncprop": ids["uncprop"]}


# =============================================================================
# Break it 1: a contradiction σ cannot repair
# =============================================================================

def _anchored_builders(row: Optional[tuple], row_cv: float, anchor_cv: float):
    """The toy's builders plus copper's supply/use scale totals at ``anchor_cv``, the
    row ``row = (sector_from, region_from, time)`` (both valuations) at ``row_cv``."""
    def builders(ctx) -> list:
        def resolver(key_map, edge_ids):
            hit = (key_map["sector_from"], key_map["region_from"], key_map["time"]) == row
            return row_cv if hit else None
        return toy.builders(ctx) + anchors.scale_total_anchors(
            ctx.acc, ctx.agg_result.f_agg, resolver, anchor_cv=anchor_cv,
            p=ctx.case.const.P_PROP)
    return builders


def break_contradiction(run_root, row_cvs=(0.01, 0.1, 1.0, 10.0), anchor_cv: float = 0.01,
                        base: Optional[dict] = None, label: str = "toy_contradiction"
                        ) -> dict:
    """Restate the planted prior at ``anchor_cv`` and loosen the row it contradicts.

    The toy's builders gain copper's supply/use scale totals (``anchors.
    scale_total_anchors``, ``Σ edges = Σ prior`` at σ_c = ``anchor_cv·|c|``, ADR 0006),
    which restate the Customs leg planted by :func:`toy.reveal` (``Montara → Portua``
    concentrate, 2003, ×1000) at ``anchor_cv``. For each ``row_cvs`` the σ of the row it
    contradicts, the use total of Montara's Concentrate flow in 2003, is set to that cv
    and Stages 3–5 run. A last chain keeps the anchors at ``anchor_cv`` everywhere and
    rebuilds the Customs feed *without* the planted error (``label + "_fixed"``).

    Returns
    -------
    dict
        ``{"table": DataFrame, "runs": {name: {stage: run_id}}}``; ``table`` has one row
        per variant (``base`` = the core chain, ``cv=<x>``, ``fixed``): the contradicted
        row's cv, the largest soft-row ``|z|``, the share of edges whose posterior
        interval holds the truth, the planted edge's ``a*`` against its truth and the
        median and p90 relative error of ``a*`` against the truth over all edges.
    """
    base = base or base_ids(run_root)
    planted = toy.reveal()
    row  = (planted["from_sector"], planted["from_region"], planted["time"])
    rows, runs = [], {"base": dict(base)}
    rows.append({"variant": "base", "row_cv": np.nan, **_score(run_root, base)})

    def chain(name, imp_id, builders, tag):
        with _variant(f"{label}_{tag}", _with_builders(builders)):
            runs[name] = _run_stages(run_root, imp_id, uncprop=True)
        return runs[name]

    for cv in row_cvs:
        ids = chain(f"cv={cv:g}", base["imp"], _anchored_builders(row, cv, anchor_cv),
                    f"cv{cv:g}")
        rows.append({"variant": f"cv={cv:g}", "row_cv": cv, **_score(run_root, ids)})

    # The input fixed: the Customs feed without the planted error, aggregated with the
    # base chain's other two feeds.
    feeds = _feed_run_ids(run_root, base["imp"])
    saved = dict(toy._PLANTED)
    toy._PLANTED["factor"] = 1.0
    try:
        customs = toy.make_feeds(toy.truth(toy.SEED), toy.SEED)["customs"]
    finally:
        toy._PLANTED.clear()
        toy._PLANTED.update(saved)
    feeds["customs"] = _register_feed(run_root, customs, f"{label}_fixed_customs")
    with _variant(f"{label}_fixed", _with_builders(
            _anchored_builders(row, anchor_cv, anchor_cv))):
        runs["fixed"] = _run_stages(run_root, toy.run_stage2(run_root, feeds),
                                    uncprop=True)
    runs["fixed"]["feed"] = feeds
    rows.append({"variant": "fixed", "row_cv": anchor_cv,
                 **_score(run_root, runs["fixed"])})
    return {"table": pd.DataFrame(rows).set_index("variant"), "runs": runs}


def check_8_1(table: pd.DataFrame, factor: float) -> str:
    """Exercise 8.1: the factor by which the contradicted row's z fell from the tightest
    to the loosest row cv of :func:`break_contradiction`'s ``table``.

    The row's σ_c is ``cv·|c|``, so it grows by the ratio of the two cvs; the tick line
    states how much of the fall that ratio leaves to the residual itself.
    """
    rows     = table[table.index.str.startswith("cv=")].sort_values("row_cv")
    expected = float(rows["planted_row_z"].iloc[0] / rows["planted_row_z"].iloc[-1])
    if not np.isclose(factor, expected, rtol=1e-6):
        raise AssertionError(f"factor by which z fell: expected {expected!r}, "
                             f"got {factor!r}")
    grew = float(rows["row_cv"].iloc[-1] / rows["row_cv"].iloc[0])
    return (f"✓ 8.1  z fell {expected:.0f}× while the row's σ grew {grew:.0f}×: the "
            f"residual itself grew {grew / expected:.2f}×")


# =============================================================================
# Break it 2: a phantom leg, larger than its exporter's whole output
# =============================================================================

_PHANTOM_CODE = ("Isla", "Portua")      # the phantom leg's exporter and importer


def _edge_values(run_root, opt_run_id: str, edge_id: int
                 ) -> "tuple[float, float, float]":
    """``(a_prior, a*, σ_prior)`` of one edge: the prior the QP started from, its
    solution and the conditioned prior σ the solver used."""
    const = _const_of(run_root, opt_run_id)
    a_prior, sigma = condition_prior(const.a_prior_global, const.sigma_a)
    opt = OptimRunRegistry(str(toy.run_tree(run_root)["opt"]))
    a_star = toy._global(opt.load(opt_run_id), "F_opt", a_prior.size)
    return float(a_prior[edge_id]), float(a_star[edge_id]), float(sigma[edge_id])


def break_phantom(run_root, year: str = "2003", factor: float = 3.0,
                  base: Optional[dict] = None, label: str = "toy_phantom") -> dict:
    """Plant a second Customs leg, Isla → Portua concentrate, of ``factor`` times Isla's
    whole concentrate output in ``year``, and run Stages 2–4 on it.

    The leg is one more row of the Customs feed beside Isla's real one; Stage 2
    aggregates it into the prior, the mass balances of Isla's Concentrate flow and Isla's
    Milling process refuse it, and Stage 4 delivers the leg at a fraction of its prior.

    Returns
    -------
    dict
        ``{"table": one-row DataFrame, "runs": {stage: run_id}}``; ``table`` has the
        leg's ``output`` (Isla's concentrate output, truth), ``truth``, ``prior`` and
        ``a_star`` (kt, gross mass), ``posterior_over_prior`` and ``z``, the leg's
        edge-term z ``(a* − prior) / σ_prior`` (the conditioned prior σ the solver used).
    """
    base = base or base_ids(run_root)
    tree = toy.run_tree(run_root)
    acc  = toy.load_acc(tree)
    exporter, importer = _PHANTOM_CODE
    truth = toy.truth(toy.SEED)
    output = float(truth[(truth["time"] == year) & (truth["valuation"] == toy.GROSS)
                         & (truth["from_sector"] == "Milling")
                         & (truth["from_region"] == exporter)
                         & (truth["to_sector"] == "Concentrate")]["value"].iloc[0])
    leg = toy._edge_id(acc, "Concentrate", exporter, "Smelting", importer, year,
                       toy.GROSS)

    customs = toy.make_feeds(truth, toy.SEED)["customs"]
    row = pd.DataFrame([{"Year": year, "Process": "261790",
                         "Region": toy._CUSTOMS_CODE[exporter],
                         "Partner": toy._CUSTOMS_CODE[importer],
                         "Unit": "kt", "Value": factor * output}])
    customs.raw = pd.concat([customs.raw, row], ignore_index=True)

    feeds = _feed_run_ids(run_root, base["imp"])
    feeds["customs"] = _register_feed(run_root, customs, f"{label}_customs")
    with _variant(label):
        runs = _run_stages(run_root, toy.run_stage2(run_root, feeds))
    prior, a_star, sigma = _edge_values(run_root, runs["opt"], leg)
    table = pd.DataFrame([{
        "leg": f"Concentrate[{exporter}] → Smelting[{importer}] {year} {toy.GROSS}",
        "output": output, "truth": float(toy.truth_vector(truth, acc)[leg]),
        "prior": prior, "a_star": a_star, "posterior_over_prior": a_star / prior,
        "z": (a_star - prior) / sigma}])
    return {"table": table, "runs": {"feed": feeds, **runs}}


# =============================================================================
# Break it 3: a loose solver tolerance
# =============================================================================

def _solve_year(run_root, const_run_id: str, year: str, gap_rel: float):
    """One year of ``const_run_id`` solved at relative gap ``gap_rel``, the other
    tolerances those of :func:`toy.run_stage4`; nothing is persisted."""
    tree = toy.run_tree(run_root)
    backend = ClarabelBackend(
        tol_feas            = toy.TOL_FEAS,
        tol_gap_rel         = gap_rel,
        tol_gap_abs         = toy.TOL_GAP_ABS,
        reduced_tol_feas    = toy.REDUCED_TOL_FEAS,
        reduced_tol_gap_rel = gap_rel,
        reduced_tol_gap_abs = toy.REDUCED_TOL_GAP_ABS,
        max_iter            = toy.MAX_ITER)
    solver = Solver(backend, const_dir=str(tree["const"]), opt_dir=str(tree["opt"]),
                    acc_dir=str(tree["acc"]), verbose_first_solve=False,
                    prior_mode="cold", qp_variables=toy.QP_VARIABLES)
    return solver.solve_all(const_run_id, timesteps=[year],
                            feasibility_check="none")[year]


def break_tolerance(run_root, year: str = "2003", gap_rel: float = 1e-3,
                    tight_gap_rel: float = 1e-6, bands: bool = True, top: int = 6,
                    base: Optional[dict] = None, label: str = "toy_tolerance") -> dict:
    """Solve ``year`` at relative gap ``gap_rel`` and at ``tight_gap_rel`` (ADR 0039) and
    compare the two period slices edge by edge, against each edge's prior σ.

    Without ``bands`` the toy QP has no inequality cone: Clarabel solves its KKT system
    directly (0 iterations) and both gaps return the same solution to the last digit.
    With ``bands`` Stage 3 gains copper's non-negativity rows (``FlowBoundsBuilder``,
    ADR 0039, a const run under ``label``), the interior-point iteration runs and the
    gap decides where it stops.

    Returns
    -------
    dict
        ``{"table", "summary", "runs"}``. ``table`` holds the ``top`` edges by
        ``|Δ|/σ``: ``edge``, the two ``a`` values, ``diff``, ``sigma_prior`` and
        ``diff_over_sigma``; ``summary`` the iteration counts and objectives of both
        solves, ``max_abs_diff``, ``max_diff_over_sigma`` and ``n_over_sigma`` (edges
        differing by more than their σ); ``runs`` the const run solved.
    """
    base = base or base_ids(run_root)
    tree = toy.run_tree(run_root)
    const_id = base["const"]
    if bands:
        def builders(ctx) -> list:
            return toy.builders(ctx) + [FlowBoundsBuilder()]
        with _variant(label, _with_builders(builders)):
            const_id = toy.run_stage3(run_root, base["imp"])
    loose, tight = (_solve_year(run_root, const_id, year, g)
                    for g in (gap_rel, tight_gap_rel))
    const = ConstraintRunRegistry(str(tree["const"])).load(const_id)
    _, sigma = condition_prior(const.a_prior_global, const.sigma_a)
    cols  = np.asarray(loose.cols_t)
    diff  = np.abs(loose.F_opt - tight.F_opt)
    ratio = diff / sigma[cols]
    order = np.argsort(-ratio, kind="stable")[:top]
    table = pd.DataFrame({
        "edge":            toy.load_acc(tree).edge_meta.loc[cols[order]].apply(
            lambda r: f"{r.from_sector}[{r.from_region}] → {r.to_sector}[{r.to_region}]"
                      f" {r.valuation}", axis=1).to_numpy(),
        "a_loose":         loose.F_opt[order],
        "a_tight":         tight.F_opt[order],
        "diff":            diff[order],
        "sigma_prior":     sigma[cols][order],
        "diff_over_sigma": ratio[order]})
    summary = {"year": year, "gap_rel": gap_rel, "tight_gap_rel": tight_gap_rel,
               "iterations": (loose.iterations, tight.iterations),
               "objective": (loose.objective, tight.objective),
               "max_abs_diff": float(diff.max()),
               "max_diff_over_sigma": float(ratio.max()),
               "n_over_sigma": int((diff > sigma[cols]).sum())}
    return {"table": table, "summary": summary, "runs": {"const": const_id}}


# =============================================================================
# The bridge: three seams, each a few lines (#61, notebook 09)
# =============================================================================

class ScrapShareSeed(BaseImputer):
    """Seam 1, a ``BaseImputer``: seed an unwritten ``Scrap → Smelting`` leg (content) at
    ``share`` of the concentrate input its smelter already holds, σ = ``cv`` · seed. The
    ledger enforces no-overwrite and the scope declared here; nothing else is written."""

    def __init__(self, share: float = 0.10, cv: float = 0.5, id: str = "scrap_seed"):
        self.share, self.cv, self.id = share, cv, id

    def prepare(self, acc) -> None:
        em, content = acc.edge_meta, acc.valuation_roles.content
        self._scope = ((em["from_sector"] == "Scrap") & (em["to_sector"] == "Smelting")
                       & (em["valuation"] == content)).to_numpy()
        feed = em[(em["from_sector"] == "Concentrate") & (em["to_sector"] == "Smelting")
                  & (em["valuation"] == content)]
        self._inputs = {k: g["edge_id"].to_numpy()
                        for k, g in feed.groupby(["to_region", "time"])}

    def writer_scope(self, acc) -> np.ndarray:
        return self._scope

    def impute(self, ledger, step: int, acc) -> None:
        em = acc.edge_meta
        for e in np.flatnonzero(self._scope):
            src = self._inputs[(em["to_region"].iat[e], em["time"].iat[e])]
            src = src[np.isfinite(ledger.f[src])]
            if src.size and np.isnan(ledger.f[e]):
                v = self.share * float(ledger.f[src].sum())
                ledger.write(e, v, self.cv * v, step, "scrap_share_seed",
                             list(src), [self.share] * len(src))


class NetStockRow(BaseConstraintBuilder):
    """Seam 2, a ``BaseConstraintBuilder``: one soft row per region and year, the net
    stock change ``Metal → Stock`` (content) = 0 at ``sigma`` kt Content. The row's
    ``row_meta`` carries its identity; nothing parses its id."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def build_static(self, acc) -> ConstraintBlock:
        em, content = acc.edge_meta, acc.valuation_roles.content
        hit = em[(em["to_sector"] == "Stock") & (em["valuation"] == content)]
        m, meta = len(hit), pd.DataFrame({c: [None] * len(hit) for c in ROW_META_COLS})
        meta["constraint_id"] = [build_constraint_id("sum", "netstock", r, year=t)
                                 for r, t in zip(hit["to_region"], hit["time"])]
        meta["type"], meta["variant"] = "sum", "netstock"
        meta["is_hard"], meta["sense"] = False, "eq"
        meta["time"], meta["valuation"] = hit["time"].to_numpy(), content
        meta["sector"], meta["region"] = "Stock", hit["to_region"].to_numpy()
        G = sp.csr_matrix((np.ones(m), (np.arange(m), hit["edge_id"].to_numpy())),
                          shape=(m, acc.n_edges))
        return ConstraintBlock(G, np.zeros(m), np.full(m, self.sigma), meta)


def fourth_feed(run_root, label: str = "toy_ministry", seed: int = toy.SEED) -> str:
    """Seam 3: a synthetic fourth feed through the Stage 1 API. The *Ministry* reports
    the smelters' refined output (content) in exactly the two region-years the Bureau did
    not publish, noisy at the Bureau's pedigree. Returns its feed run id."""
    truth = toy.truth(seed).set_index(toy._EDGE_KEY)["value"]
    rows = [(t, r, "smelter", "refined",
             float(truth.loc[(t, toy.CONTENT, "Smelting", r, "Metal", r)]))
            for r, t in toy._BUREAU_GAPS]
    raw = pd.DataFrame(rows, columns=["Year", "Region", "Process", "Item", "_truth"]
                       ).assign(Unit=toy._BUREAU_UNIT)
    meta = dataclasses.replace(toy._metadata("bureau"), feed_name="ministry")
    eps = np.random.default_rng([seed, 4]).standard_normal(len(raw))
    cv  = meta.to_pedigree_scores().total_cv()
    raw = raw.assign(Value=raw.pop("_truth") * (1 + cv * eps))
    feed = toy.SyntheticFeed("ministry", raw, toy._maps(
        {"smelter": "Smelting", "refined": "Metal"}, {r: r for r in toy.REGIONS},
        {toy._BUREAU_UNIT: toy.CONTENT},
        [("Year", "src_time"), ("Unit", "src_valuation"), ("Region", "src_region_from"),
         ("Region", "src_region_to"), ("Process", "src_sector_from"),
         ("Item", "src_sector_to")]), meta)
    return _register_feed(run_root, feed, label, seed)


def run_seams(run_root, base: Optional[dict] = None, label: str = "toy_seams") -> dict:
    """Stages 2–4 of the toy with the three seams in: the Ministry joins the feeds, the
    ``ScrapShareSeed`` joins the terminal imputers (before the ``TimeImputer``) and the
    ``NetStockRow`` joins the builders. Returns ``{"runs": {...}, "n_seeded": int}``,
    ``n_seeded`` the edges the seed wrote."""
    base  = base or base_ids(run_root)
    feeds = _feed_run_ids(run_root, base["imp"])
    feeds["ministry"] = fourth_feed(run_root, f"{label}_ministry")
    spec = dataclasses.replace(
        toy.CASE.imp, terminal=lambda ctx: [ScrapShareSeed()] + toy.terminal(ctx))
    case = dataclasses.replace(
        toy.CASE, imp=spec, const=dataclasses.replace(
            toy.CASE.const, builders=lambda ctx: toy.builders(ctx) + [NetStockRow()]))
    with _variant(label, case):
        runs = _run_stages(run_root, toy.run_stage2(run_root, feeds))
    tree = toy.run_tree(run_root)
    log  = ImputationRunRegistry(str(tree["imp"])).load(
        runs["imp"]).imputation_log.as_dataframe()
    return {"runs": {"feed": feeds, **runs},
            "n_seeded": int((log["imputer_id"] == "scrap_seed").sum())}


# =============================================================================
# Your own case (#197): the helpers of chapter 10
# =============================================================================
# A case outside the package needs three things its author does not want to type out:
# a params family in the template's schema, a feed with its concordance maps, and a
# Stage 1 that builds the acc from the registered spec and registers the feeds.

def params_from_template(template, rows: "list[dict]", dest) -> Path:
    """Write ``rows`` as a params family under the header of ``template``.

    ``template`` is the shipped ``params_template.csv`` (``;``-separated; its ``#`` lines
    are the column reference and two worked examples). The copy keeps the template's
    columns and delimiter, and none of its examples: a row names columns of the template,
    every other column is left blank.

    Raises
    ------
    KeyError
        A row names a column the template does not have; nothing is written.
    """
    lines = [ln for ln in Path(template).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    columns = lines[0].split(";")
    for row in rows:
        unknown = sorted(set(row) - set(columns))
        if unknown:
            raise KeyError(f"{unknown} are not columns of {template}; its columns are "
                           f"{';'.join(columns)}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows({c: row.get(c, "") for c in columns} for row in rows)
    return dest


def register_feeds(data_dir, acc_id: str, specs: "list[FeedSpec]") -> "dict[str, str]":
    """Stage 1 of a case of your own, through the public API: build the accounting
    structure from the acc spec registered under ``<data_dir>/input/acc/<acc_id>`` (the
    structure and its build record go to ``<data_dir>/proc/acc``), then run each
    :class:`~pysuto.feedpipe.FeedSpec` through :func:`~pysuto.feedpipe.run_feed`, which
    reads the feed's registered ``values.csv``, ``conc.xlsx`` and ``meta.xlsx``.
    Returns ``{feed_id: feed_run_id}``."""
    data_dir  = Path(data_dir)
    inputs    = InputRegistries.initialize(root=str(data_dir / "input"))
    acc_dir   = data_dir / "proc" / "acc"
    spec      = read_acc_spec_dir(inputs.acc.resolve(acc_id))
    acc_store.save(AccountingStructure.from_spec(spec, acc_id, storage_dir=str(acc_dir)))
    write_build_record(str(acc_dir), inputs.acc.entry(acc_id))
    return {s.feed_id: run_feed(s, acc_id=acc_id, data_dir=str(data_dir),
                                diag=False).run_id for s in specs}


def run_case(case: str, data_dir, *args: str) -> "dict[str, str]":
    """Run ``python -m pysuto run --case <case> --data-dir <data_dir> <args>`` with this
    interpreter, from the current working directory, and return ``{stage: run id}`` from
    its stdout. The log lines it also prints are skipped; ``feed`` holds the feed run ids
    as the command prints them, and a stage the command did not run maps to ``""``.
    ``case`` is anything ``--case`` takes: an alias, a module or a path such as
    ``./my_case.py``.

    Raises
    ------
    RuntimeError
        The command exited non-zero; the message carries the end of its stderr.
    """
    cmd = [sys.executable, "-m", "pysuto", "run", "--case", str(case),
           "--data-dir", str(data_dir), *args]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        tail = "\n".join(done.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"`python -m pysuto run --case {case}` exited "
                           f"{done.returncode}:\n{tail}")
    lines = [ln.split(None, 1) + [""] for ln in done.stdout.splitlines() if ln.strip()]
    return {ln[0]: ln[1].strip() for ln in lines if ln[0] in ALL_STEPS}


def check_10_1(table: pd.DataFrame, tailings: float, year: str = "2002") -> str:
    """Exercise 10.1: the ``Content`` the Mill books into ``Tailings`` in ``year``, read
    from a PSUT table, to 1 %."""
    rows = table[(table["period"] == year) & (table["valuation"] == toy.CONTENT)
                 & (table["to_sector"] == "Tailings")]
    if len(rows) != 1:
        raise ValueError(f"expected one Mill → Tailings row in {year}, found {len(rows)}")
    expected = float(rows["val"].iloc[0])
    toy._close(tailings, expected, f"Mill → Tailings {year}", rtol=1e-2)
    return f"✓ 10.1  Mill → Tailings {year}: {expected:.3f} {toy.UNITS[toy.CONTENT]}"
