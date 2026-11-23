"""
anchors.py
==========
The Stage 3 anchor families: soft-sum rows that pin a group of edges to the model's own
Stage 2 prior (#145, #26 §3). A case passes its sector labels, valuation and thresholds;
nothing here names a sector of any application. Every family is built through
:meth:`SumConstraintBuilder.from_mask_anchor`.

* :func:`scale_total_cv`, :func:`scale_total_resolver` — the cv of one scale total
  (ADR 0016 provenance classification, ADR 0019 contradicted nodes, ADR 0025 floor).
* :func:`scale_total_anchors` — the supply and use total-output scale totals (ADR 0006).
* :func:`scrap_level_anchors` — the level of an SB feed into a consuming process (ADR 0031).
* :func:`nodata_anchor_exempt`, :func:`nodata_exempt_mask` — the per-edge exemption of
  the no-data σ anchor (ADR 0013 amendment 3).

A scale total is a soft row, not a control total (ADR 0006); the σ floor of the resolver
applies to feed-backed totals only, never globally (ADR 0025).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from pysuto.flow_node_balance import contradicted_flow_nodes
from pysuto.logging import get_logger
from pysuto.relation.param_relation import CONC_ACCEPTS, expand_edge_index
from pysuto.relation.params import read_params

from .sum import SumConstraintBuilder

_log = get_logger(__name__)

# Scale totals are node-scoped, year-locked and unit-coherent.
SCALE_GROUP_KEYS = ("sector_from", "region_from", "time", "valuation")
# A level anchor is grouped on its consuming node, so row_meta.sector is that node.
LEVEL_GROUP_KEYS = ("sector_to", "region_to", "time", "valuation")


def scale_total_cv(feed_share: float, c_abs: float, *, anchor_cv: float,
                   fabricated_cv: float, feed_share_observed: float,
                   abs_floor: float) -> Optional[float]:
    """cv for one scale-total group (ADR 0016 classification, ADR 0025 floor).

    Fabricated groups (feed share below the threshold) keep ``fabricated_cv``.
    Feed-backed groups get the cv that makes ``cv·|c|`` equal the feeds'
    reporting-resolution floor on small totals, ``max(anchor_cv, floor/|c|)``,
    so σ_c = max(anchor_cv·|c|, floor). ``None`` means "use the builder's
    default cv" (a feed-backed group with |c| = 0 cannot be floored via cv).
    """
    if feed_share < feed_share_observed:
        return fabricated_cv
    if c_abs <= 0.0:
        return None
    return max(anchor_cv, abs_floor / c_abs)


def scale_total_resolver(acc, agg_result, *, anchor_cv: float, fabricated_cv: float,
                         feed_share_observed: float, abs_floor: float,
                         unclosed_sectors: tuple = (),
                         unclosed_anchor_cv: Optional[float] = None, logger=None):
    """The scale totals' σ resolver (ADR 0016, 0019, 0025): ``(key_map, edge_ids) ->
    cv`` for ``SumConstraintBuilder.from_mask_anchor``, over ``agg_result``.

    Parameters
    ----------
    acc, agg_result : the accounting structure and the imputed aggregation; the feed
        share of a group reads ``agg_result.coverage`` (GLOSSARY.md "Coverage").
    anchor_cv, fabricated_cv, feed_share_observed, abs_floor : the four knobs of
        :func:`scale_total_cv`.
    unclosed_sectors : flow sectors judged by ``contradicted_flow_nodes`` (ADR 0019).
    unclosed_anchor_cv : the cv of a contradicted node's anchor; ``None`` runs the
        detector only.
    logger : logger of the detector's message (default: this module's).
    """
    logger = logger or _log
    # Provenance-aware σ_c: loosen scale totals whose prior group-sum is
    # fabricated (feed-backed share of Σ|f_agg| < feed_share_observed). Uses
    # agg_result.coverage — the pre-imputation "did any feed observe this
    # edge?" flag (GLOSSARY.md §Coverage), so imputations never count.
    # NaN reads as 0 here (and in from_mask_anchor's nansum); the engine rejects a NaN
    # f_agg, but only after the builders exist.
    f_agg = agg_result.f_agg
    cov   = np.asarray(agg_result.coverage, dtype=bool)
    f_agg_signed = np.nan_to_num(np.asarray(f_agg, dtype=float))
    absf = np.abs(f_agg_signed)

    # ADR 0019: the same detection Stage 2 used to σ-inflate the edges.
    unclosed, _ = contradicted_flow_nodes(acc, agg_result, sectors=unclosed_sectors)
    logger.info("ADR 0019: %d contradicted %s node-year(s) — anchor action: %s",
                len(unclosed), "/".join(unclosed_sectors),
                f"usetotout cv={unclosed_anchor_cv}"
                if unclosed_anchor_cv is not None else "none (detector only)")

    def resolve(key_map, edge_ids):
        # Contradicted node: the outflow anchor loses authority so the hard
        # balance can settle at the reliable inflow prior (ADR 0019).
        if (unclosed_anchor_cv is not None
                and (key_map.get("sector_from"), key_map.get("region_from"),
                     key_map.get("time")) in unclosed):
            return unclosed_anchor_cv
        ids = np.asarray(edge_ids, dtype=np.int64)
        tot = absf[ids].sum()
        if tot <= 0.0:
            return fabricated_cv
        feed_share = absf[ids[cov[ids]]].sum() / tot
        return scale_total_cv(feed_share, abs(float(f_agg_signed[ids].sum())),
                              anchor_cv=anchor_cv, fabricated_cv=fabricated_cv,
                              feed_share_observed=feed_share_observed,
                              abs_floor=abs_floor)
    return resolve


def scale_total_anchors(acc, f_agg, resolver, *, anchor_cv: float, p: float,
                        supply_prefix: str = "supplytotout",
                        use_prefix: str = "usetotout", logger=None) -> list:
    """The supply and the use total-output scale totals, in that order (ADR 0006).

    Every process's and every flow's total-output scale is pinned to the model's own
    Stage 2 prior, ``Σ (group edges) = Σ f_agg``, grouped by
    ``(sector_from, region_from, time, valuation)``:

    * supply total output (Process → Flow useful output): one row per process;
    * use total output (Flow → anything: use and final demand): one row per flow,
      summed over all destinations.

    ``resolver`` is :func:`scale_total_resolver`'s result; ``anchor_cv`` the fallback cv;
    ``p`` the builders' Lp exponent. |c| ≈ 0 groups are dropped by ``from_mask_anchor``.
    """
    (logger or _log).info("Registering SumConstraintBuilder — supply/use total-output "
                          "scale totals")
    ei_exp      = expand_edge_index(acc.edge_index, acc.node_index)
    supply_mask = (ei_exp["entity_from"].eq("Process")
                   & ei_exp["entity_to"].eq("Flow")).to_numpy()
    use_mask    = ei_exp["entity_from"].eq("Flow").to_numpy()
    return [SumConstraintBuilder.from_mask_anchor(
                acc,
                mask          = mask,
                anchor_values = f_agg,
                group_keys    = SCALE_GROUP_KEYS,
                sigma_cv      = anchor_cv,
                id_prefix     = prefix,
                sigma_cv_resolver = resolver,
                p             = p,
            ) for mask, prefix in ((supply_mask, supply_prefix), (use_mask, use_prefix))]


def scrap_level_anchors(acc, f_agg, coverage, resolver, *, source_sector: str,
                        consumers: Mapping[str, str], anchor_cv: float, p: float,
                        logger=None) -> list:
    """Level anchors of a boundary sector's feed into consuming processes (ADR 0031).

    One family per ``consumers`` entry ``{id_prefix: consuming sector}``, in its order.
    A family's edges run ``source_sector → consumer`` in the content valuation
    (``acc.valuation_roles.content``) and are COVERED by a feed (``coverage``), grouped
    by ``(sector_to, region_to, time, valuation)``. The families are registered
    separately so each route is anchored at its own level, not only their sum.

    Content only: the gross twin is derived and carries no information of its own.
    Covered only: a row asserts a level a feed REPORTED, which makes every group
    feed-backed, so the ADR 0025 floor of ``resolver`` applies unconditionally.
    """
    ei_exp  = expand_edge_index(acc.edge_index, acc.node_index)
    content = acc.valuation_roles.content
    cov     = np.asarray(coverage, dtype=bool)
    scope   = ((ei_exp["sector_from"].eq(source_sector)
                & ei_exp["valuation"].eq(content)).to_numpy() & cov)
    masks   = {prefix: scope & ei_exp["sector_to"].eq(sector).to_numpy()
               for prefix, sector in consumers.items()}
    (logger or _log).info("Registering SumConstraintBuilder — scrap-level anchors (%s)",
                          ", ".join(f"{k} {int(m.sum())} edges" for k, m in masks.items()))
    return [SumConstraintBuilder.from_mask_anchor(
                acc,
                mask          = mask,
                anchor_values = f_agg,
                group_keys    = LEVEL_GROUP_KEYS,
                sigma_cv      = anchor_cv,
                id_prefix     = prefix,
                sigma_cv_resolver = resolver,
                p             = p,
            ) for prefix, mask in masks.items()]


def nodata_exempt_mask(acc, exclude_sectors, data_aware_sectors, region_years) -> np.ndarray:
    """Per-edge exemption from the no-data σ anchor (ADR 0013 amendment 3).

    An edge touching ``exclude_sectors`` (either endpoint) is spared, except that
    one touching ``data_aware_sectors`` is spared only where its
    ``(from_region, time)`` is in ``region_years`` — the region-years in which a
    relation actually sets its magnitude. Elsewhere a zero prior means "no flow",
    and the anchor applies. ``region_years`` holds ``(region, time)`` string pairs.
    """
    em = acc.edge_meta
    fs, ts = em["from_sector"].to_numpy(), em["to_sector"].to_numpy()
    spared = np.isin(fs, list(exclude_sectors)) | np.isin(ts, list(exclude_sectors))
    aware = np.isin(fs, list(data_aware_sectors)) | np.isin(ts, list(data_aware_sectors))
    keys = zip(em["from_region"].astype(str), em["time"].astype(str))
    listed = np.fromiter((k in region_years for k in keys), dtype=bool, count=len(em))
    return spared & (~aware | listed)


def nodata_anchor_exempt(acc, params_csv: str, *, param_prefix: str, exclude: tuple,
                         data_aware: tuple, region_key: str = "a_region_from",
                         time_key: str = "a_time", accepts=CONC_ACCEPTS,
                         logger=None) -> tuple:
    """``(mask, n)``: the no-data anchor's per-edge exemption and the number of
    region-years it spares (``(None, 0)`` when ``data_aware`` is empty).

    ADR 0013 amendment 3: edges touching ``exclude`` sectors are spared from the
    no-data σ anchor, but those touching ``data_aware`` sectors only in region-years in
    which a relation sets their magnitude — the region-years of the rows of
    ``params_csv`` whose param id starts with ``param_prefix`` (selector columns
    ``region_key`` and ``time_key``).
    """
    if not data_aware:
        return None, 0
    rows = read_params(params_csv, accepts)
    region_years = {(r.selectors[region_key], r.selectors[time_key])
                    for r in rows if r.param_id.startswith(param_prefix)}
    mask = nodata_exempt_mask(acc, exclude, data_aware, region_years)
    (logger or _log).info("No-data anchor: data-aware sectors spared in %d region-years "
                          "with a relation-set magnitude; %d edges spared in total",
                          len(region_years), int(mask.sum()))
    return mask, len(region_years)
