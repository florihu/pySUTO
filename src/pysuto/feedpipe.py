"""
feedpipe.py
===========
Canonical four-step DataFeed lifecycle:

    A  Read metadata (Excel) → DataFeedMetadata
    B  Build ConcordanceTable
    C  Build WeightMatrix
    D  Apply + propagate uncertainty  →  PropagationResult
       Register + persist everything  →  DataFeedRegistry

All matrix operations use scipy.sparse (CSR format). The driver of a feed build is
:func:`run_feed` over a :class:`FeedSpec`; the ``step_*`` functions are its pieces.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from pysuto.acc import AccountingStructure
from pysuto.input_registry import InputRegistries
from pysuto.meta import DataFeedMetadata
from pysuto.conc import ConcordanceTable, WeightMatrix, node_ids
from pysuto.unc import PropagationResult
from pysuto.registry import DataFeedRegistry, stage_timer
from pysuto.logging import add_run_handler, get_logger
from pysuto.lp import P_DEFAULT

# The Lp exponent a feed's pedigree is propagated with (recorded in the feed sidecar).
FEED_P = P_DEFAULT

_log = get_logger("pysuto.feedpipe")


# ===========================================================================
# STEP A — Read metadata
# ===========================================================================

def step_A_read_metadata(excel_path: str) -> DataFeedMetadata:
    """
    Parse the Excel sidecar that describes this dataset.

    Returns a DataFeedMetadata with all identification, pedigree,
    and provenance fields populated.
    """
    _log.info("A  reading metadata: %s", excel_path)
    meta = DataFeedMetadata.from_excel(excel_path)
    _log.info("   feed_name=%s  iedc_type=%s  version=%s",
              meta.feed_name, meta.iedc_type, meta.version)
    return meta


# ===========================================================================
# STEP B — Build ConcordanceTable
# ===========================================================================

def step_B_build_concordance(
    raw_df:           pd.DataFrame,
    acc_struct,
    sector_map_df:    pd.DataFrame,
    region_map_df:    pd.DataFrame,
    time_map_df:      pd.DataFrame,
    valuation_map_df: pd.DataFrame,
    var_map_df:       pd.DataFrame,
    feed_id:          str,
    aux_col:          Optional[str] = None,
    aux_sides:        Optional[dict] = None,
    allow_unmatched_edges: bool = True,
    user_weights_df:  Optional[pd.DataFrame] = None,
    leg_filter:       Optional[dict] = None,
    drop_unweighted_fans: bool = False,
) -> ConcordanceTable:
    """
    Build the src→root edge mapping (ConcordanceTable) from raw data.

    Parameters
    ----------
    aux_col / aux_sides : the raw column that tells supply rows from use rows, and its
        values mapped to the endpoint they classify (``"from"`` / ``"to"``); the
        vocabulary is the case's (#119), passed through to ConcordanceTable.from_raw.
    allow_unmatched_edges : passed through to ConcordanceTable.from_raw so
        that rows that cannot be matched to a root edge are collected in
        ct.unmatched_raw_rows rather than raising an exception.
    user_weights_df : optional DataFrame with columns [root_sector_from,
        root_sector_to, root_region_from, root_region_to, root_time, weight].
        Rows matching that full bilateral-route + year key override the default
        equal-split weight; other fan legs keep the uniform default (weight=1,
        normalised per src_edge group). Used for BACI-trade-share weighting of
        the ICSG producer fan-out (ADR 0002 option D).
    leg_filter / drop_unweighted_fans : fan policy (ADR 0008), passed through
        to ConcordanceTable.from_raw — restrict qualifier fans to their input
        sector's legs, and do not distribute fans that have no weights.

    Returns a validated ConcordanceTable ready for weight-matrix assembly.
    """
    _log.info("B  building concordance  feed='%s'  raw_rows=%d", feed_id, len(raw_df))
    if user_weights_df is not None and not user_weights_df.empty:
        _log.info("   user_weights: %d entries", len(user_weights_df))
    ct = ConcordanceTable.from_raw(
        raw_df                = raw_df,
        acc_struct            = acc_struct,
        sector_map_df         = sector_map_df,
        region_map_df         = region_map_df,
        time_map_df           = time_map_df,
        valuation_map_df      = valuation_map_df,
        var_map_df            = var_map_df,
        aux_col               = aux_col,
        aux_sides             = aux_sides,
        allow_unmatched_edges = allow_unmatched_edges,
        id                    = feed_id,
        user_weights_df       = user_weights_df,
        leg_filter            = leg_filter,
        drop_unweighted_fans  = drop_unweighted_fans,
    )
    unmatched = getattr(ct, "unmatched_raw_rows", [])
    undistributed = getattr(ct, "undistributed_raw_rows", [])
    _log.info("   src_edges=%d  root_edges=%d  conc_rows=%d  unmatched=%d  undistributed=%d",
              ct.n_src_edges, ct.n_root_edges, len(ct.tab), len(unmatched), len(undistributed))
    if unmatched:
        _log.warning("   %d raw rows could not be matched to a root edge", len(unmatched))
    if undistributed:
        _log.warning("   %d raw rows matched but were not distributed "
                     "(unweighted fans, ADR 0008)", len(undistributed))
    return ct


# ===========================================================================
# STEP C — Build WeightMatrix
# ===========================================================================

def step_C_build_weight_matrix(ct: ConcordanceTable) -> WeightMatrix:
    """
    Assemble the sparse W[src_edge, root_edge] matrix from the concordance.

    The weight matrix is stored internally as a scipy.sparse.csr_matrix.
    Returns a WeightMatrix whose .apply_with_uncertainty() method is ready.
    """
    _log.info("C  building weight matrix")
    wm = WeightMatrix(concordance=ct, id=ct.id)
    _log.info("   shape=%s  nnz=%d", wm.weight_matrix.shape, wm.weight_matrix.nnz)
    return wm


# ===========================================================================
# STEP D — Apply + propagate
# ===========================================================================

def step_D_apply_and_propagate(
    wm:       WeightMatrix,
    F_src:    np.ndarray,
    metadata: DataFeedMetadata,
    p:        float = P_DEFAULT,
    cv_root_scale: Optional[np.ndarray] = None,
    unit:     str = "kt",
) -> PropagationResult:
    """
    1.  Derive PedigreeScores from metadata and attach to the weight matrix.
    2.  Store F_src on the weight matrix.
    3.  Run the combined apply + Lp uncertainty propagation (scipy.sparse).

    Propagation formula (power-mean / Lp norm):
        F_root      = W.T @ F_src
        σ_root[j]   = CV · ( Σ_i (|W_ij| · |f_i|)^p )^(1/p)
        cv_root     = σ_root / |F_root|

    ``unit`` names the feed's valuation in the log (``ConcordanceTable.unit``).

    Returns
    -------
    PropagationResult with F_root, sigma_root, cv_root, cv_src.
    """
    _log.info("D  propagating uncertainty  F_src_nnz=%d  F_src_sum=%.1f %s",
              int(np.count_nonzero(F_src)), float(F_src.sum()), unit)
    pg = metadata.to_pedigree_scores()
    wm.set_pedigree(pg)
    wm.set_F_src(F_src)
    result = wm.apply_with_uncertainty(p=p, cv_root_scale=cv_root_scale)
    _log.info("   cv_src=%.2f%%  F_root_sum=%.1f %s  mean_cv_root=%.2f%%",
              result.cv_src * 100,
              float(result.F_root.sum()), unit,
              float(np.nanmean(result.cv_root)) * 100)
    return result


# ===========================================================================
# MASS-CONSERVATION GUARD — feed read-in robustness
# ===========================================================================

def matched_raw_total(raw_df: pd.DataFrame, ct: ConcordanceTable,
                      value_col: str = "Value") -> float:
    """Total raw ``value_col`` over the raw rows that matched ≥1 src edge.

    This is the mass the concordance is *responsible for projecting*. Raw rows
    with no src edge are excluded — their drop is coverage loss (governed by
    ``allow_unmatched_edges``), not a projection error, so they must not count
    against the conservation check.
    """
    matched = raw_df.index.isin(set(ct.tab["raw_id"].dropna()))
    return float(raw_df.loc[matched, value_col].sum())


def assert_mass_conserved(projected_total: float, raw_total: float, *,
                          label: str = "feed", tol: float = 0.01) -> None:
    """Raise ``ValueError`` if projected feed mass deviates from the matched raw
    input mass by more than ``tol`` (relative).

    Catches single-endpoint fan-out **replication** (one raw row duplicated
    across many src edges → N× inflation) and any non-mass-preserving weight
    matrix. A legitimate fan-in (several raw rows summing into one src edge)
    still conserves the total, so it passes. See
    ``docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md``.
    """
    if raw_total == 0:
        return
    rel = abs(projected_total - raw_total) / abs(raw_total)
    if rel > tol:
        factor = projected_total / raw_total if raw_total else float("inf")
        raise ValueError(
            f"{label}: feed mass not conserved — projected {projected_total:,.1f} "
            f"vs matched raw {raw_total:,.1f} ({factor:.2f}× / {rel:.1%} deviation "
            f"> tol {tol:.1%}). Likely single-endpoint fan-out replication "
            f"(ADR 0002) or a non-normalised weight matrix."
        )


def build_F_src(raw_df: pd.DataFrame, ct: ConcordanceTable, *,
                label: str = "feed", value_col: str = "Value") -> np.ndarray:
    """Aggregate raw values to the src-edge level aligned to ``ct.src_edges``.

    Dedup on ``(raw_id, src edge)`` so a single raw row fanning across N root
    edges (Mapping 2 fan-out — ``ct.tab`` holds one row per src×root pair) is
    counted **once**, while distinct raw rows that resolve to the *same* src
    edge (Mapping 1 raw-row aggregation) are **summed**. Then assert mass
    conservation. See ADR 0002 and GLOSSARY.md "Two mapping levels".
    """
    src_raw = ct.tab[["raw_id"] + list(ConcordanceTable.SRC_COLS)].drop_duplicates()
    merged = src_raw.merge(
        raw_df.reset_index().rename(columns={"index": "_raw_idx"}),
        left_on="raw_id", right_on="_raw_idx", how="left",
    )
    src_cols = list(ConcordanceTable.SRC_COLS)
    agg = merged.groupby(src_cols)[value_col].sum().reset_index()

    F_src = np.zeros(ct.n_src_edges)
    if len(agg):
        pos = pd.MultiIndex.from_frame(ct.src_edges[src_cols]).get_indexer(
            pd.MultiIndex.from_frame(agg[src_cols]))
        known = pos >= 0
        F_src[pos[known]] = agg[value_col].to_numpy()[known]

    assert_mass_conserved(float(F_src.sum()),
                          matched_raw_total(raw_df, ct, value_col),
                          label=label)
    return F_src


# ===========================================================================
# REGISTRATION — store everything
# ===========================================================================

def register_feed(
    wm:        WeightMatrix,
    metadata:  DataFeedMetadata,
    result:    PropagationResult,
    registry:  DataFeedRegistry,
    *,
    parents:   dict,
    F_src:     Optional[np.ndarray] = None,
    label:     Optional[str] = None,
    stage_config:  Optional[dict] = None,
    stage_summary: Optional[dict] = None,
):
    """
    Persist the weight matrix, metadata sidecar, result arrays and the
    source flow vector under a freshly minted ``run_id``, then flush
    the registry index to disk.

    ``parents`` follows the harmonized sidecar schema:
    ``{"inputs": {"acc": {"id", "version"},
                  "feed": {"id", "version"}}, "upstream": {}}``.

    Returns the registry entry (with ``.run_dir`` and ``.run_id``).
    """
    _log.info("E  registering feed '%s'", metadata.feed_name)
    entry = registry.register(
        wm, metadata,
        parents       = parents,
        result        = result,
        F_src         = F_src,
        label         = label,
        stage_config  = stage_config,
        stage_summary = stage_summary,
    )
    registry.save()
    _log.info("   run_id        → %s", entry.run_id)
    _log.info("   run dir       → %s", entry.run_dir)
    return entry


# ===========================================================================
# EDGE SCOPE AND ROW PENALTY
# ===========================================================================

def _apply_edge_scope_mask(ct, acc_struct, mask_name: str) -> None:
    """Filter ``ct.tab`` in-place to drop rows whose resolved root edge is not
    in ``acc_struct.masks[mask_name]``.

    Raises
    ------
    ValueError
        If ``mask_name`` is not in ``acc_struct.masks`` or a concordance row names no
        root edge: an unscoped feed would carry what the scope exists to exclude
        (ADR 0014).
    """
    if not mask_name:
        return
    masks = getattr(acc_struct, "masks", None) or {}
    if mask_name not in masks:
        raise ValueError(
            f"edge_scope_mask={mask_name!r} is not present in acc_struct.masks "
            f"(available: {sorted(masks)})"
        )

    scope = masks[mask_name]
    n_before = len(ct.tab)

    # Resolve each tab row to its edge_id via (from_id, to_id, time, valuation).
    tab = ct.tab
    from_id = node_ids(acc_struct.node_to_id, tab["root_sector_from"],
                       tab["root_entity_from"], tab["root_region_from"])
    to_id   = node_ids(acc_struct.node_to_id, tab["root_sector_to"],
                       tab["root_entity_to"], tab["root_region_to"])
    keys = pd.DataFrame({
        "from_id":   from_id,
        "to_id":     to_id,
        "time":      tab["root_time"].astype(str).values,
        "valuation": tab["root_valuation"].astype(str).values,
    })

    ei = acc_struct.edge_index[["from_id", "to_id", "time", "valuation", "edge_id"]].copy()
    ei["time"]      = ei["time"].astype(str)
    ei["valuation"] = ei["valuation"].astype(str)
    joined = keys.merge(ei, on=["from_id", "to_id", "time", "valuation"], how="left")
    edge_ids = joined["edge_id"].to_numpy()

    unresolved = pd.isna(edge_ids)
    if unresolved.any():
        raise ValueError(
            f"edge_scope_mask={mask_name!r}: {int(unresolved.sum())} concordance rows "
            f"name no root edge of the accounting structure"
        )
    in_scope = np.asarray(scope)[edge_ids.astype(int)].astype(bool)

    ct.tab = ct.tab.loc[in_scope].reset_index(drop=True)
    n_after = len(ct.tab)
    if n_after != n_before:
        _log.info(
            "edge_scope_mask=%r: kept %d / %d concordance rows (%d dropped)",
            mask_name, n_after, n_before, n_before - n_after,
        )


def _build_row_penalty_scale(acc_struct, metadata, row_region=None):
    """Build a per-root-edge CV multiplier that pushes ROW-touching edges to the
    worst geographical pedigree score (``geographical_correlation = 4``).

    A root edge is ROW-touching iff its from- or to-node region is ``row_region``,
    the case's label for its ROW region (copper: ``ROW``; #119). It must
    be one of the structure's regions when the feed opts into the penalty.
    The region is decoded inline from ``acc_struct`` (a downward dependency) —
    the same node-index join used in ``acc.py`` / ``masks.py`` — so feedpipe
    stays within the one-way import direction (no ``param_relation`` import).

    Scaling ROW edges' CV by ``ratio = cv(geo=4)/cv(geo=baseline)`` is exactly
    equivalent to re-propagating those rows with ``geographical_correlation=4``.

    Returns ``(cv_root_scale, summary, row_mask)`` or ``(None, None, None)``
    when the feed does not opt into the penalty.
    """
    if not getattr(metadata, "row_penalty", False):
        return None, None, None
    regions = set(acc_struct.node_index["region"].astype(str))
    if row_region not in regions:
        raise ValueError(f"the feed opts into the ROW penalty, so row_region must be "
                         f"one of the structure's regions; got {row_region!r}")

    pg_norm = metadata.to_pedigree_scores()
    pg_pen  = dataclasses.replace(pg_norm, geographical_correlation=4)
    cv_norm = pg_norm.total_cv()
    cv_pen  = pg_pen.total_cv()
    ratio   = cv_pen / cv_norm if cv_norm > 0 else 1.0

    ei = acc_struct.edge_index
    rf = acc_struct.node_index.loc[ei["from_id"].values, "region"].to_numpy().astype(str)
    rt = acc_struct.node_index.loc[ei["to_id"].values,   "region"].to_numpy().astype(str)
    row_mask = (rf == row_region) | (rt == row_region)

    cv_root_scale = np.where(row_mask, ratio, 1.0)
    summary = {"row_penalty": {
        "dim":         "geographical_correlation",
        "n_row_edges": int(row_mask.sum()),   # ROW-touching edges in the acc
        "n_edges":     int(row_mask.size),
        "cv_norm":     float(cv_norm),
        "cv_pen":      float(cv_pen),
        "ratio":       float(ratio),
    }}
    return cv_root_scale, summary, row_mask


def propagate_with_row_penalty(wm, F_src, metadata, acc_struct,
                               p: float = FEED_P,
                               cv_root_scale_extra=None, row_region=None,
                               unit: str = "kt"):
    """Run step-D propagation and apply the optional ROW pedigree penalty.

    Wraps :func:`step_D_apply_and_propagate` so the penalty is wired identically
    in every :func:`run_feed` build and the per-feed open-coded scripts. When the
    feed opts into ``row_penalty``, ROW-touching edges are propagated with a
    geo->4 cv multiplier, the result is stamped with a ``row_penalized``
    provenance mask (only edges this feed actually carries), and an INFO line is
    logged.

    ``cv_root_scale_extra`` is a second per-root-edge CV multiplier the caller
    may supply, combined multiplicatively with the ROW ratio. It exists for
    feeds whose rows are part observation and part inference: ADR 0027's
    re-export re-attribution prices the inferred component through it. ``None``
    (the default) leaves every caller's behaviour bit-identical.

    ``row_region`` is the case's ROW region label, required when the feed
    opts into the penalty (see :func:`_build_row_penalty_scale`). ``unit`` is the
    feed's valuation label for the log (``ConcordanceTable.unit``).

    Returns ``(result, stage_summary)`` where ``stage_summary`` is the
    row_penalty summary dict (or ``None`` when the penalty is off).
    """
    cv_root_scale, row_summary, row_mask = _build_row_penalty_scale(acc_struct, metadata,
                                                                     row_region)
    if cv_root_scale_extra is not None:
        extra = np.asarray(cv_root_scale_extra, dtype=float)
        cv_root_scale = extra if cv_root_scale is None else cv_root_scale * extra
    result = step_D_apply_and_propagate(wm, F_src, metadata, p=p,
                                        cv_root_scale=cv_root_scale, unit=unit)
    if row_mask is not None:
        result.row_penalized = row_mask & (result.F_root != 0)
        denom = float(np.nansum(np.abs(result.F_root)))
        share = (float(np.nansum(np.abs(result.F_root[row_mask]))) / denom
                 if denom > 0 else 0.0)
        rp = row_summary["row_penalty"]
        # edges this feed actually carries flow on (matches result.row_penalized)
        rp["n_penalized"] = int(result.row_penalized.sum())
        rp["value_share"] = share
        _log.info("D  row_penalty: %d/%d fed ROW edges (value_share=%.1f%%) "
                  "geo->4 cv %.4f->%.4f (ratio %.3f)",
                  rp["n_penalized"], rp["n_row_edges"], share * 100,
                  rp["cv_norm"], rp["cv_pen"], rp["ratio"])
    return result, row_summary


# ===========================================================================
# ONE FEED BUILD — FeedSpec + run_feed (#143)
# ===========================================================================

@dataclass
class FeedBuild:
    """What a :class:`FeedSpec` hook sees of one feed build.

    :func:`run_feed` fills the fields in step order, so a hook finds every field its
    step precedes: ``user_weights`` sees up to ``meta``, ``post_propagate`` all of them.
    ``input_ref`` is the parent block's ``inputs`` dict; a hook may add entries to it.
    ``registry_dir`` is the feed-run registry (``<data_dir>/proc/feed/registry``), where a
    hook finds the runs of the feeds built before it.
    """
    spec:       "FeedSpec"
    acc:        object
    inputs:     InputRegistries
    input_root: str
    feed_dir:   str
    input_ref:  dict
    registry_dir: str = ""
    raw_df:     Optional[pd.DataFrame] = None
    tables:     Optional[dict] = None
    meta:       Optional[DataFeedMetadata] = None
    ct:         Optional[ConcordanceTable] = None
    wm:         Optional[WeightMatrix] = None
    F_src:      Optional[np.ndarray] = None
    result:     Optional[PropagationResult] = None


@dataclass(frozen=True)
class FeedSpec:
    """One feed's policy for :func:`run_feed` (GLOSSARY.md "Case").

    Attributes
    ----------
    feed_id : str
        The input registry's feed id, the label of the run and of its diagnostics.
    load_raw : callable
        ``load_raw(values_csv_path) -> DataFrame``: the raw rows, with ``Year`` a string.
    load_tables : callable
        ``load_tables(conc_xlsx_path) -> dict`` with ``sector_map``, ``region_map``,
        ``time_map``, ``valuation_map`` and ``var_map``.
    row_region : str or None
        The case's ROW region, for the ROW pedigree penalty (#119).
    aux_col, aux_sides :
        The raw column that tells supply rows from use rows and its vocabulary.
    allow_unmatched_edges, drop_unweighted_fans :
        Passed to :func:`step_B_build_concordance`.
    edge_scope : bool
        Apply the metadata sidecar's ``Edge_scope_mask`` to the concordance (ADR 0014).
    leg_filter : callable or None
        ``leg_filter(tables) -> dict``: the fan policy (ADR 0008).
    user_weights : callable or None
        ``user_weights(build) -> (DataFrame or None, {input_ref key: registry entry})``:
        producer weights, and the inputs they were built from.
    cv_scale : callable or None
        ``cv_scale(acc, feed_dir) -> ndarray or None``: a per-root-edge CV multiplier.
    post_propagate : callable or None
        ``post_propagate(build)``: edits ``build.result`` after propagation (σ hooks such
        as ICSG's observed zeros) and may add entries to ``build.input_ref``.
    knobs : dict or None
        The module constants the hooks read, ``{name: value}`` (JSON-serialisable): the
        feed run's sidecar records them in ``stage_config["knobs"]`` (#210).
    """
    feed_id:               str
    load_raw:              Callable
    load_tables:           Callable
    row_region:            Optional[str] = None
    aux_col:               Optional[str] = None
    aux_sides:             Optional[dict] = None
    allow_unmatched_edges: bool = True
    drop_unweighted_fans:  bool = False
    edge_scope:            bool = False
    leg_filter:            Optional[Callable] = None
    user_weights:          Optional[Callable] = None
    cv_scale:              Optional[Callable] = None
    post_propagate:        Optional[Callable] = None
    knobs:                 Optional[dict] = None


def check_file(path: str, label: str) -> None:
    """Raise ``FileNotFoundError`` naming the missing input file ``path``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nMissing {label}:\n  {path}\n"
            f"Place the file there or update the input registry's current/ directory."
        )


def load_acc_structure(acc_id: str, acc_dir: str) -> AccountingStructure:
    """Load the registered accounting structure ``acc_id`` from ``acc_dir``."""
    _log.info("=" * 60)
    _log.info("A  Loading AccountingStructure")
    _log.info("   id  : %s", acc_id)
    _log.info("   dir : %s", acc_dir)

    acc_path = os.path.join(acc_dir, acc_id)
    if not os.path.isdir(acc_path):
        available = (
            "\n".join(
                f"  • {d}" for d in sorted(os.listdir(acc_dir))
                if os.path.isdir(os.path.join(acc_dir, d))
            )
            if os.path.isdir(acc_dir) else
            f"  ('{acc_dir}' does not exist)"
        )
        raise FileNotFoundError(
            f"\nAccounting structure '{acc_id}' not found at '{acc_path}'.\n"
            f"Register it first (scripts/acc.py).\n"
            f"Available structures:\n{available}"
        )

    acc = AccountingStructure.load(id=acc_id, storage_dir=acc_dir)
    _log.info("   nodes           : %d", acc.n_nodes)
    _log.info("   admissible edges: %d", len(acc.edge_index))
    _log.info("   time range      : %s – %s", acc.time_range[0], acc.time_range[-1])
    _log.info("   valuations      : %s", acc.valuation)
    return acc


def build_feed_concordance(spec: FeedSpec, raw_df: pd.DataFrame, tables: dict, acc,
                           user_weights_df: Optional[pd.DataFrame] = None
                           ) -> ConcordanceTable:
    """Step B for ``spec``: its concordance tables and fan policy over ``raw_df``."""
    _log.info("=" * 60)
    _log.info("C  Building ConcordanceTable")

    leg_filter = spec.leg_filter(tables) if spec.leg_filter is not None else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ct = step_B_build_concordance(
            raw_df                = raw_df,
            acc_struct            = acc,
            sector_map_df         = tables["sector_map"],
            region_map_df         = tables["region_map"],
            time_map_df           = tables["time_map"],
            valuation_map_df      = tables["valuation_map"],
            var_map_df            = tables["var_map"],
            feed_id               = spec.feed_id,
            aux_col               = spec.aux_col,
            aux_sides             = spec.aux_sides,
            allow_unmatched_edges = spec.allow_unmatched_edges,
            user_weights_df       = user_weights_df,
            leg_filter            = leg_filter,
            drop_unweighted_fans  = spec.drop_unweighted_fans,
        )

    for w in caught:
        _log.warning("   ⚠  %s", w.message)
    return ct


@stage_timer()
def run_feed(spec: FeedSpec, *, acc_id: str, data_dir: str = "data",
             run_log: Optional[str] = None, diag: bool = True):
    """Build, propagate and register one feed run (Stage 1) from ``spec``.

    Steps A → F of the three copper drivers, once: the registered accounting structure,
    the feed's raw rows, concordance workbook and metadata sidecar, the concordance and
    weight matrix, the source vector and its propagated root values, the registered
    run, and (``diag``) its diagnostics.

    Parameters
    ----------
    spec : FeedSpec
    acc_id : str
        The accounting structure, registered under ``<data_dir>/proc/acc``. Required: the
        caller names it; no config file is read.
    data_dir : str
        Holds ``input/`` (the input registries) and ``proc/`` (``proc/acc``, and the
        feed registry at ``proc/feed/registry``, where the run is written).
    run_log : str, optional
        A file that collects the process's log records for this call; the handler is
        removed again on return.
    diag : bool
        Render the feed diagnostics into the run's ``diag/`` directory.

    Returns
    -------
    entry
        The feed run's registry entry (``.run_id``, ``.run_dir``).
    """
    root = logging.getLogger()
    handler, root_level = None, root.level
    if run_log is not None:
        handler = add_run_handler(run_log)
    try:
        return _run_feed(spec, acc_id, data_dir, run_log, diag)
    finally:
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
            root.setLevel(root_level)


def _feed_stage_config(spec: FeedSpec) -> dict:
    """What the feed run's sidecar records of its configuration (#210): the Lp exponent
    ``p`` it propagated with, the spec's policy fields, each hook's qualified name and
    the spec's ``knobs``."""
    hooks = {f.name: f"{getattr(spec, f.name).__module__}."
                     f"{getattr(spec, f.name).__qualname__}"
             for f in dataclasses.fields(spec) if callable(getattr(spec, f.name))}
    return {"p":                     FEED_P,
            "row_region":            spec.row_region,
            "aux_col":               spec.aux_col,
            "aux_sides":             spec.aux_sides,
            "allow_unmatched_edges": spec.allow_unmatched_edges,
            "drop_unweighted_fans":  spec.drop_unweighted_fans,
            "edge_scope":            spec.edge_scope,
            "hooks":                 hooks,
            "knobs":                 spec.knobs or {}}


def _run_feed(spec, acc_id, data_dir, run_log, diag):
    input_root   = os.path.join(data_dir, "input")
    acc_dir      = os.path.join(data_dir, "proc", "acc")
    registry_dir = os.path.join(data_dir, "proc", "feed", "registry")

    _log.info("=" * 60)
    _log.info("feed=%s  acc=%s", spec.feed_id, acc_id)
    if run_log is not None:
        _log.info("run log → %s", run_log)

    # 0 — input registries (scan-and-bump every kind)
    inputs    = InputRegistries.initialize(root=input_root)
    feed_dir  = inputs.feed.resolve(spec.feed_id)
    raw_file  = str(feed_dir / "values.csv")
    conc_file = str(feed_dir / "conc.xlsx")
    meta_file = str(feed_dir / "meta.xlsx")
    input_ref = {
        "acc":  inputs.acc.entry(acc_id),
        "feed": inputs.feed.entry(spec.feed_id),
    }
    _log.info("   input feed : %s v%d (code=%s)",
              input_ref["feed"]["id"], input_ref["feed"]["version"],
              input_ref["feed"]["code"])
    _log.info("   input acc  : %s v%d", input_ref["acc"]["id"],
              input_ref["acc"]["version"])

    # A — accounting structure
    acc = load_acc_structure(acc_id, acc_dir)
    b = FeedBuild(spec=spec, acc=acc, inputs=inputs, input_root=input_root,
                  feed_dir=str(feed_dir), input_ref=input_ref, registry_dir=registry_dir)

    # B — raw data + concordance tables + metadata sidecar
    _log.info("=" * 60)
    _log.info("B  Loading inputs")
    _log.info("   raw data     : %s", raw_file)
    _log.info("   concordance  : %s", conc_file)
    _log.info("   meta sidecar : %s", meta_file)
    b.raw_df = spec.load_raw(raw_file)
    b.tables = spec.load_tables(conc_file)
    if os.path.isfile(meta_file):
        b.meta = step_A_read_metadata(meta_file)
    else:
        # default pedigree scores and no Edge_scope_mask: a scoped feed then raises
        # below, since it would build unscoped (ADR 0014)
        _log.warning("   meta sidecar not found — using default pedigree scores")
        b.meta = DataFeedMetadata(feed_name=spec.feed_id)
    user_weights_df = None
    if spec.user_weights is not None:
        user_weights_df, extra_inputs = spec.user_weights(b)
        input_ref.update(extra_inputs)
    _log.info("   rows loaded  : %d", len(b.raw_df))
    _log.info("   sector map   : %d entries", len(b.tables["sector_map"]))
    _log.info("   region map   : %d entries", len(b.tables["region_map"]))
    _log.info("   time map     : %d entries", len(b.tables["time_map"]))

    # C — concordance + weight matrix
    b.ct = build_feed_concordance(spec, b.raw_df, b.tables, acc, user_weights_df)
    if spec.edge_scope:
        # Edge scope (meta.xlsx Edge_scope_mask), e.g. BACI's cross_region (ADR 0014):
        # this driver builds the ct directly, so it applies the scope itself.
        if not b.meta.edge_scope_mask:
            raise ValueError(f"feed {spec.feed_id!r} is scoped (edge_scope) but its "
                             f"meta sidecar names no Edge_scope_mask (ADR 0014)")
        _apply_edge_scope_mask(b.ct, acc, b.meta.edge_scope_mask)
    _log.info("=" * 60)
    _log.info("C  Building WeightMatrix")
    b.wm = step_C_build_weight_matrix(b.ct)

    # D — source flow vector + uncertainty propagation
    b.F_src = build_F_src(b.raw_df, b.ct, label=spec.feed_id)
    _log.info("=" * 60)
    _log.info("D  F_src vector")
    _log.info("   length    : %d", len(b.F_src))
    _log.info("   non-zero  : %d", int(np.count_nonzero(b.F_src)))
    _log.info("   total %s  : %.1f", b.ct.unit, float(b.F_src.sum()))
    _log.info("=" * 60)
    _log.info("D  Propagating uncertainty")
    cv_extra = spec.cv_scale(acc, str(feed_dir)) if spec.cv_scale is not None else None
    b.result, row_summary = propagate_with_row_penalty(
        b.wm, b.F_src, b.meta, acc, p=FEED_P, cv_root_scale_extra=cv_extra,
        row_region=spec.row_region, unit=b.ct.unit)
    if spec.post_propagate is not None:
        spec.post_propagate(b)

    # E — register + persist
    _log.info("=" * 60)
    _log.info("E  Registering feed")
    registry = DataFeedRegistry(odir=registry_dir)
    registry.load_index()
    b.wm.odir = registry_dir
    entry = register_feed(
        b.wm, b.meta, b.result, registry,
        parents       = {"inputs": input_ref, "upstream": {}},
        F_src         = b.F_src,
        stage_config  = _feed_stage_config(spec),
        stage_summary = row_summary,
    )
    _log.info("   run_id           %s", entry.run_id)
    _log.info("   run dir          %s", entry.run_dir)

    # F — diagnostics
    if diag:
        from pysuto.diag import FeedDiagnostics   # diag/ is a leaf: imported on use
        _log.info("=" * 60)
        _log.info("F  Generating diagnostics")
        d = FeedDiagnostics(
            feed_run_id = entry.run_id,
            feed_id     = spec.feed_id,
            acc_id      = acc_id,
            raw_df      = b.raw_df,
            ct          = b.ct,
            F_src       = b.F_src,
            result      = b.result,
            conc_file   = conc_file,
        )
        d.plot(odir=os.path.join(entry.run_dir, "diag"))
        d.summary()
        _log.info("   written to: %s/", entry.run_dir)
    _log.info("=" * 60)
    _log.info("Done.")
    return entry
