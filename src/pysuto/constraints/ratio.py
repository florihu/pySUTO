"""
ratio.py
========
Two ratio-style builders that share enough implementation surface to live
together:

* :class:`RatioConstraintBuilder` — Type-3 soft equality: r·a_i − a_j = 0.
* :class:`RatioBoundBuilder`      — Type-4 ratio-bounds: l ≤ a_i / a_j ≤ u.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.logging import get_logger
from pysuto.relation.param_relation import (
    ParamRelationResolver,
    expand_edge_index,
)
from pysuto.relation.params import read_params
from pysuto.relation.ratio_relation import agg_buckets, ratio_relation
from pysuto.relation.relation import Relation

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block
from .emit import emit, ratio_scale_index, ratio_sigma_c
from .naming import TYPE_RATIO, _safe, build_constraint_id, empty_row_meta_dict, render_edge
from .sum  import BOUND_C_INACTIVE

_log = get_logger(__name__)


# The param_types each builder reads. The stripping ratio gets no band: its equality row
# WR = k·Ore already ties WR to Ore (ADR 0038), and [value_min, value_max] is the
# uncertainty band of k, reported beside the flow, not a bound — so RatioBoundBuilder
# does not accept it.
ACCEPTS_EQ = frozenset({"concentration", "efficiency", "stripping_ratio", "stock_bound",
                        "scrap_share"})
ACCEPTS_BOUND = ACCEPTS_EQ - {"stripping_ratio"}

# Net-stock bands are centred at 0 (value_nominal=0) with a symmetric ±p95 half-band
# (value_max). Read that half-band as the ratio-space σ_r via p95 ≈ Z95·σ, treating
# the net-change/throughput ratio as ~N(0, σ). See plan A2 / docs/rep.
Z95_STOCK = 1.645


def _decode_edge(ei_exp_idx, eid: int) -> dict:
    """Decoded (sector/region from→to + valuation + time) for one edge_id.

    ``ei_exp_idx`` is :func:`expand_edge_index` output indexed by ``edge_id``.
    """
    r = ei_exp_idx.loc[eid]
    return {
        "sector_from": str(r["sector_from"]),
        "region_from": str(r["region_from"]),
        "sector_to":   str(r["sector_to"]),
        "region_to":   str(r["region_to"]),
        "valuation":   str(r["valuation"]),
        "time":        str(r["time"]),
    }


def _ratio_struct_cols(num: dict, denom: dict) -> dict:
    """row_meta num_*/denom_* columns from two decoded edges."""
    return {
        "num_sector_from":   num["sector_from"], "num_region_from": num["region_from"],
        "num_sector_to":     num["sector_to"],   "num_region_to":   num["region_to"],
        "num_valuation":     num["valuation"],
        "denom_sector_from": denom["sector_from"], "denom_region_from": denom["region_from"],
        "denom_sector_to":   denom["sector_to"],   "denom_region_to":   denom["region_to"],
        "denom_valuation":   denom["valuation"],
    }


def _relative_sigma(sigma: float, r: float, param_id: str | None) -> float:
    """σ in ratio units (``ParamRow.sigma``) → the relative σ_r = σ/|r| (#29).

    At a prior that sits on the ratio the symmetric scale
    ``max(Σ|a₀,num|, |r|·Σ|a₀,denom|)`` is |r|·Σ|a₀,denom|, so only σ/|r| gives the
    linearised σ_c = σ·Σ|a₀,denom|. A stated-exact ratio (σ = 0) is relative 0 at
    any r, so a zero ratio stated as exact (the ``concwruse`` shape) keeps the
    floor; a zero ratio with σ > 0 has no relative σ and raises.
    """
    if sigma == 0.0:
        return 0.0
    if r == 0.0:
        raise ValueError(
            f"RatioConstraintBuilder: param {param_id!r} has value_nominal = 0 with "
            f"σ = {sigma} — a zero ratio has no relative σ."
        )
    return sigma / abs(r)


class RatioConstraintBuilder(BaseConstraintBuilder):
    """
    Soft equality ratio constraint: a_i − r · a_j = ε_k

    Loaded from the same params CSV schema as ParamSource.
    The CSV uncertainty modes (sd / min_max / pedigree) state σ in *ratio units*;
    :meth:`from_csv` divides it by |r| into the *relative* σ_r each pair carries
    (#29), and :meth:`build_dynamic` rescales that to flow space symmetrically
    (Gauss–Newton linearization of the log-ratio error model at the prior, REP
    analysis_opt_20260611_075036 §8.2, where σ_log = σ/r):

        σ_c = max(σ_r, sigma_rel_floor) · max(Σ|a₀,num|, |r|·Σ|a₀,denom|, scale_abs_floor)

    Net-stock rows (``scale_mode="denominator"``, r = 0) keep σ_r in ratio units:
    their scale is the denominator alone, which the ratio is already relative to.
    """

    def __init__(self, pairs: list[Relation],
                 sigma_rel_floor: float,
                 scale_abs_floor: float,
                 hard: bool = False):
        """
        pairs : list of ratio :class:`~pysuto.relation.relation.Relation` objects
            (numerator edges first, ``n_num`` of them; denominator coefficients −r;
            ``sigma`` = relative σ_r = σ/|r|, in ratio units on scale_mode="denominator"
            rows)
        sigma_rel_floor : lower bound on σ_r (caps |z| at 1/floor on a 100 %
            ratio violation).
        scale_abs_floor : lower bound on the flow-space scale (tonnage), so
            both-zero-prior pairs keep a finite σ_c.
        hard : emit hard equalities (``σ_c = 0``) instead of soft rows — for a
            relation that defines an edge nothing else observes (the stripping
            ratio of the waste-rock satellite, ADR 0038).
        """
        self._pairs = pairs
        self._hard = bool(hard)
        self._sigma_rel_floor = float(sigma_rel_floor)
        self._scale_abs_floor = float(scale_abs_floor)
        self._cached_static: ConstraintBlock | None = None
        self._scale_idx = None   # ratio_scale_index(pairs), built on first build_dynamic

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: str | None = None,
        *,
        sigma_rel_floor: float,
        scale_abs_floor: float,
        hard: bool = False,
        covered: np.ndarray | None = None,
    ) -> "RatioConstraintBuilder":
        """
        Load from params CSV (same schema as ParamSource.from_params).
        Resolves A→B edge pairs and computes the relative σ_r = σ/|r| from unc_type
        (σ in ratio units, as ``ParamRow.sigma`` returns it). Raises ``ValueError``
        on a zero ratio with σ > 0, and on a soft row with no σ rule (``unc_type`` empty).

        ``mask`` (optional): name of a mask in ``acc_struct.masks``. Pairs
        where either edge falls outside the mask are dropped.

        ``covered`` (optional): per-edge feed coverage (``agg_result.coverage``). A
        net-stock row (``param_type == "stock_bound"``) is centred at zero, so where
        a feed reports its stock leg (the numerator, b side) it would pull that report
        toward zero: no row is emitted there (#98, ADR 0040 am.).
        """
        resolver = ParamRelationResolver(acc_struct)
        ei_idx = expand_edge_index(acc_struct.edge_index, acc_struct.node_index).set_index("edge_id")

        if mask is not None:
            if mask not in acc_struct.masks:
                raise ValueError(
                    f"RatioConstraintBuilder(mask={mask!r}): mask not in "
                    f"acc_struct.masks. Available: {sorted(acc_struct.masks)}"
                )
            scope = acc_struct.masks[mask]
        else:
            scope = None

        pairs = []
        n_no_nominal = 0
        for row in read_params(path, ACCEPTS_EQ):
            nominal = row.value_nominal
            if np.isnan(nominal):
                n_no_nominal += 1
                continue
            param_id = row.param_id or None
            # Net-stock bands (param_type=="stock_bound", nominal r=0, centred at 0):
            # read the symmetric ±p95 half-band (value_max) as the ratio-space σ_r and
            # tag scale_mode="denominator" so build_dynamic scales σ_c off the THROUGHPUT
            # denominator (|r|·den = 0 here would otherwise collapse it to the abs floor
            # → quasi-hard). All other ratios keep the CSV unc modes + symmetric scale.
            is_stock = row.param_type == "stock_bound"
            if is_stock:
                vmax = row.value_max
                if np.isnan(vmax) and not hard:
                    raise ValueError(
                        f"RatioConstraintBuilder: stock_bound param {param_id!r} has no "
                        f"value_max — its σ_r would be 0 (a stated-exact net stock), "
                        f"only the sigma_rel_floor.")
                sigma = abs(vmax) / Z95_STOCK if not np.isnan(vmax) else 0.0
                scale_mode = "denominator"
            else:
                # a hard row carries σ_c = 0 whatever σ is; a soft row needs a σ rule
                sigma = 0.0 if hard else _relative_sigma(row.sigma(), nominal, param_id)
                scale_mode = "symmetric"
            # a zero-centred net-stock row has no place on a reported stock leg (#98)
            skip_reported = is_stock and covered is not None

            # A user param_id, if present, becomes the grammar `variant` slot
            # (underscore-free); otherwise the ratio variant is "ratioeq".
            variant  = _safe(param_id, field="param_id") if param_id else "ratioeq"

            # agg rows pool every matched a-side edge against every matched
            # b-side edge per (region_to, time, valuation) bucket → one soft row
            #   Σ b − r·Σ a = ε   (pooled denominator; set-scoped).
            if row.agg:
                buckets, asz, bsz = agg_buckets(row, acc_struct, scope)
                if not buckets:
                    _log.warning(
                        "RatioConstraintBuilder %s: agg row had no overlapping "
                        "(region, time, valuation) buckets (a=%d b=%d) — skipped",
                        variant, asz, bsz,
                    )
                    continue
                for bk in buckets:
                    if skip_reported and covered[bk["b_edges"]].any():
                        continue
                    # Σ b (coeff +1) − r·Σ a (coeff −r) into one row; set-scoped, so
                    # num_*/denom_* stay None (no single numerator/denominator edge).
                    # The bucket's region is the row's group key and is carried as a
                    # column, so the cross-year identity never needs the id.
                    pairs.append(ratio_relation(
                        bk["b_edges"], bk["a_edges"], nominal,
                        variant      = variant,
                        specs        = (_safe(bk["region"], field="region"), bk["valuation"]),
                        time         = bk["time"],
                        valuation    = bk["valuation"],
                        hard         = hard,
                        sigma        = sigma,
                        scale_mode   = scale_mode,
                        meta         = {"region": bk["region"]},
                        agg          = True,
                        source       = param_id or "",
                    ))
                continue

            for r in resolver.resolve_row(row):
                if scope is not None and not (scope[r.a_edge_id] and scope[r.b_edge_id]):
                    continue
                if skip_reported and covered[r.b_edge_id]:
                    continue
                # edge b (coeff +1) is the numerator; edge a (coeff −r) the denom.
                num   = _decode_edge(ei_idx, r.b_edge_id)
                denom = _decode_edge(ei_idx, r.a_edge_id)
                if is_stock and num["valuation"] != denom["valuation"]:
                    raise ValueError(
                        f"RatioConstraintBuilder: stock_bound param {param_id!r} pairs a "
                        f"{num['valuation']} numerator with a {denom['valuation']} "
                        f"denominator — its σ_c (ratio σ × Σ|a₀,denom|) would be in the "
                        f"denominator's unit while the row is in the numerator's.")
                pairs.append(ratio_relation(
                    [r.b_edge_id], [r.a_edge_id], nominal,
                    variant      = variant,
                    specs        = (
                        render_edge(num["sector_from"], num["region_from"],
                                    num["sector_to"], num["region_to"]),
                        render_edge(denom["sector_from"], denom["region_from"],
                                    denom["sector_to"], denom["region_to"]),
                        num["valuation"], denom["valuation"],
                    ),
                    time         = num["time"],
                    valuation    = num["valuation"],
                    hard         = hard,
                    sigma        = sigma,
                    scale_mode   = scale_mode,
                    meta         = _ratio_struct_cols(num, denom),
                    source       = param_id or "",
                ))

        if n_no_nominal:
            _log.info("RatioConstraintBuilder: %d param row(s) without value_nominal "
                      "skipped", n_no_nominal)
        _log.info("RatioConstraintBuilder: %d pairs from %s", len(pairs), os.path.basename(path))
        return cls(pairs, sigma_rel_floor=sigma_rel_floor, scale_abs_floor=scale_abs_floor,
                   hard=hard)

    def build_static(self, acc_struct) -> ConstraintBlock:
        block = emit(self._pairs, acc_struct.n_edges)
        if self._pairs:
            _log.info("RatioConstraintBuilder.build_static: %d rows", len(self._pairs))
        self._cached_static = block
        return block

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Rescale the relative σ_r to flow space against the prior ``f_agg``
        (symmetric: max of numerator-side and r·denominator-side prior mass,
        floored), then deactivate rows whose ``time`` does not match ``t`` by
        setting ``sigma_c = SIGMA_C_INACTIVE`` (:func:`pysuto.constraints.emit.ratio_sigma_c`).

        Mirrors :meth:`SumConstraintBuilder.build_dynamic`. ``c`` is left
        unchanged (always 0 for ratio_eq).
        """
        static = self._cached_static
        if static is None:
            static = self.build_static(acc_struct)
        if static.G_rows.shape[0] == 0:
            return _empty_block(acc_struct.n_edges)
        if self._hard:
            # Equalities hold in every year's solve they appear in (their edges are
            # year-local), like the mass balances: no σ scaling, no deactivation.
            return static

        if self._scale_idx is None:
            self._scale_idx = ratio_scale_index(self._pairs)
        sigma_c = ratio_sigma_c(self._scale_idx, static, f_agg, t,
                                self._sigma_rel_floor, self._scale_abs_floor)
        return ConstraintBlock(
            G_rows   = static.G_rows,
            c        = static.c,
            sigma_c  = sigma_c,
            row_meta = static.row_meta,
        )


# =============================================================================
# RatioBoundBuilder
# =============================================================================

class RatioBoundBuilder(BaseConstraintBuilder):
    """
    Hard ratio bound (le) constraints from the same params CSV.

    For each row with a finite value_min / value_max (the ratio band; bound_lower and
    bound_upper are read by nobody):
      Upper: a_i − r_u · a_j ≤ 0  →  G row: +1 at edge_i, −r_u at edge_j
      Lower: −a_i + r_l · a_j ≤ 0 →  G row: −1 at edge_i, +r_l at edge_j
    """

    def __init__(self, bound_rows: list[dict], kind: str = "unknown"):
        self._bound_rows = bound_rows
        self._kind = kind
        self._cached_static: ConstraintBlock | None = None

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: str | None = None,
        kind: str = "unknown",
    ) -> "RatioBoundBuilder":
        resolver = ParamRelationResolver(acc_struct)

        if mask is not None:
            if mask not in acc_struct.masks:
                raise ValueError(
                    f"RatioBoundBuilder(mask={mask!r}): mask not in "
                    f"acc_struct.masks. Available: {sorted(acc_struct.masks)}"
                )
            scope = acc_struct.masks[mask]
        else:
            scope = None

        ei_idx = expand_edge_index(acc_struct.edge_index, acc_struct.node_index).set_index("edge_id")

        bound_rows = []
        # structural lower bound < 0 ⇔ a signed edge (acc bounds; ADR 0039)
        signed           = np.asarray(acc_struct.bounds.l, dtype=float) < 0.0
        n_signed_skipped = 0
        for row in read_params(path, ACCEPTS_BOUND):
            nominal = row.value_nominal
            bl      = row.value_min
            bu      = row.value_max
            if np.isnan(bl) and np.isnan(bu):
                continue
            if not np.isnan(bl) and not np.isnan(bu) and bl > bu:
                raise ValueError(
                    f"RatioBoundBuilder: param {row.param_id!r} has "
                    f"value_min={bl} > value_max={bu} — empty ratio band."
                )

            # A user param_id, if present, becomes the grammar `variant` slot
            # (underscore-free); otherwise the bound variant is "bound".
            param_id = row.param_id or None
            variant  = _safe(param_id, field="param_id") if param_id else "bound"

            if row.agg:
                buckets, asz, bsz = agg_buckets(row, acc_struct, scope)
                if asz == 0 or bsz == 0:
                    _log.warning(
                        "RatioBoundBuilder %s: agg row matched no edges (a=%d b=%d) — skipped",
                        param_id, asz, bsz,
                    )
                    continue
                if not buckets:
                    _log.warning(
                        "RatioBoundBuilder %s: agg row had no overlapping (region, time, "
                        "valuation) buckets between a (%d edges) and b (%d edges) — skipped",
                        param_id, asz, bsz,
                    )
                    continue
                for bk in buckets:
                    bound_rows.append({
                        "variant":     variant,
                        "agg":         True,
                        "edge_i_list": bk["b_edges"],
                        "edge_j_list": bk["a_edges"],
                        "region":      bk["region"],
                        "time":        bk["time"],
                        "valuation":   bk["valuation"],
                        "bl":          bl,
                        "bu":          bu,
                    })
                continue

            # A one-signed band (bl ≥ 0, or an upper bound alone) is satisfiable
            # for a negative pair only at zero: on a signed edge (structural
            # l < 0 — the stock-change legs) it would forbid the leg's negative
            # half, so no hard band is emitted there; the soft ratio equality
            # carries the ratio (ADR 0041). A band that spans zero (the stock
            # bands, bl < 0) is sign-safe and stays hard.
            one_signed = np.isnan(bl) or bl >= 0.0
            for r in resolver.resolve_row(row):
                if scope is not None and not (scope[r.a_edge_id] and scope[r.b_edge_id]):
                    continue
                if one_signed and (signed[r.a_edge_id] or signed[r.b_edge_id]):
                    n_signed_skipped += 1
                    continue
                bound_rows.append({
                    "variant":  variant,
                    "agg":      False,
                    "edge_i":   r.b_edge_id,
                    "edge_j":   r.a_edge_id,
                    "num":      _decode_edge(ei_idx, r.b_edge_id),
                    "denom":    _decode_edge(ei_idx, r.a_edge_id),
                    "bl":       bl,
                    "bu":       bu,
                })

        n_agg = sum(1 for br in bound_rows if br.get("agg"))
        _log.info("RatioBoundBuilder: %d bound rows from %s (%d agg, %d per-pair; "
                  "%d one-signed pairs skipped on signed edges)",
                  len(bound_rows), os.path.basename(path),
                  n_agg, len(bound_rows) - n_agg, n_signed_skipped)
        return cls(bound_rows, kind=kind)

    def build_static(self, acc_struct) -> ConstraintBlock:
        n_edges = acc_struct.n_edges
        if not self._bound_rows:
            return _empty_block(n_edges)

        rows_list, cols_list, vals_list = [], [], []
        meta_list = []
        row_id = 0

        for br in self._bound_rows:
            variant = br["variant"]
            if br.get("agg"):
                ei_list = br["edge_i_list"]   # denominator edges (b)
                ej_list = br["edge_j_list"]   # numerator edges (a)
                t_row   = br["time"]
                valuation   = br["valuation"]
                # set-scoped: no single edge. Region is in the id (so a shared
                # variant, e.g. one eff param across regions, stays unique per
                # (region, valuation, year)) AND in the `region` column, which
                # is what identifies the row across years; num_*/denom_* stay None.
                spec_tokens = (_safe(br["region"], field="region"), valuation)
                struct_cols = {"region": br["region"]}
            else:
                ei_list = [br["edge_i"]]
                ej_list = [br["edge_j"]]
                num, denom = br["num"], br["denom"]
                t_row     = num["time"]
                valuation = num["valuation"]
                spec_tokens = (
                    render_edge(num["sector_from"], num["region_from"],
                                num["sector_to"], num["region_to"]),
                    render_edge(denom["sector_from"], denom["region_from"],
                                denom["sector_to"], denom["region_to"]),
                    num["valuation"], denom["valuation"],
                )
                struct_cols = _ratio_struct_cols(num, denom)
            bu, bl = br["bu"], br["bl"]

            def _emit(direction: str, ei_sign: float, ej_sign: float, r_coeff: float):
                nonlocal row_id
                for ej in ej_list:
                    rows_list.append(row_id); cols_list.append(ej); vals_list.append(ej_sign * r_coeff)
                for ei in ei_list:
                    rows_list.append(row_id); cols_list.append(ei); vals_list.append(ei_sign)
                meta = empty_row_meta_dict()
                meta.update({
                    "constraint_id": build_constraint_id(
                        TYPE_RATIO, variant, *spec_tokens, direction, year=t_row,
                    ),
                    "type":          TYPE_RATIO,
                    "variant":       variant,
                    "is_hard":       True,
                    "sense":         "le",
                    "time":          t_row,
                    "valuation":     valuation,
                    "ratio_kind":    self._kind,
                    "bound_dir":     direction,
                    **struct_cols,
                })
                meta_list.append(meta)
                row_id += 1

            if not np.isnan(bu):
                _emit("upper", ei_sign=1.0, ej_sign=-1.0, r_coeff=bu)
            if not np.isnan(bl):
                _emit("lower", ei_sign=-1.0, ej_sign=1.0, r_coeff=bl)

        if row_id == 0:
            return _empty_block(n_edges)

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(row_id, n_edges))
        _log.info("RatioBoundBuilder.build_static: %d rows", row_id)
        block = ConstraintBlock(
            G_rows   = G,
            c        = np.zeros(row_id),
            sigma_c  = np.zeros(row_id),
            row_meta = pd.DataFrame(meta_list),
        )
        self._cached_static = block
        return block

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Deactivate ratio_bound rows whose ``time`` does not match ``t`` by
        setting ``c = BOUND_C_INACTIVE``. The hard inequality
        ``A x ≤ BOUND_C_INACTIVE`` is then trivially satisfied (worst-case
        |A x| ≲ 4e12 ≪ 1e15 given u_max = 1e9 and typical bound ratios).
        σ_c stays 0 for hard rows.
        """
        static = self._cached_static
        if static is None:
            static = self.build_static(acc_struct)
        if static.G_rows.shape[0] == 0:
            return _empty_block(acc_struct.n_edges)

        c        = static.c.copy()
        t_str    = str(t)
        times    = static.row_meta["time"].astype(str).to_numpy()
        inactive = times != t_str
        c[inactive] = BOUND_C_INACTIVE

        return ConstraintBlock(
            G_rows   = static.G_rows,
            c        = c,
            sigma_c  = static.sigma_c,
            row_meta = static.row_meta,
        )


# =============================================================================
# FlowBoundsBuilder
# =============================================================================

