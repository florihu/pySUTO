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
from pysuto.param_relation import (
    A_FIELDS,
    B_FIELDS,
    WILDCARD,
    ParamRelationResolver,
    expand_edge_index,
    float_or,
    get_edge_cache,
    sigma_from_row,
)

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block
from .naming import TYPE_RATIO, _safe, build_constraint_id, empty_row_meta_dict, render_edge
from .sum  import SIGMA_C_INACTIVE, BOUND_C_INACTIVE

_log = get_logger(__name__)


_AGG_TRUE = {"true", "1", "yes", "y", "t"}

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


def _is_agg(row) -> bool:
    """True if the row's `agg` cell evaluates to a true-ish value."""
    return str(row.get("agg", "")).strip().lower() in _AGG_TRUE


def _match_side_edges(row, prefix: str, acc_struct) -> np.ndarray:
    """Return edge_ids matching the ``{a_|b_}``-prefixed selectors on `row`.

    Wildcards / "same" are treated as 'match any value' — `agg` rows do not
    pair sides per-edge, so the cross-side "same" constraint has no meaning
    here. Time and valuation are returned alongside the edge ids by callers
    via ``acc_struct.edge_index`` for bucketing.
    """
    ei_exp, _el, _itt, _fv, _mc, edge_ids_arr = get_edge_cache(acc_struct)
    fields = A_FIELDS if prefix == "a" else B_FIELDS
    mask = np.ones(len(ei_exp), dtype=bool)
    for col_key, ei_col, _ in fields:
        val = str(row.get(col_key, "*")).strip()
        if val in WILDCARD or val == "same":
            continue
        mask &= (ei_exp[ei_col].astype(str).to_numpy() == val)
    return edge_ids_arr[mask]


def _agg_buckets(row, acc_struct, scope):
    """Match a/b selectors and bucket both sides by ``(region_to, time, valuation)``.

    Returns ``(buckets, a_size, b_size)`` where each bucket is
    ``{"region", "time", "valuation", "a_edges": [...], "b_edges": [...]}`` for
    every key present on *both* sides. Bucketing on ``region_to`` scopes the
    pool per process region: input use-edges carry ``region_to = process region``
    and the product supply-edge carries ``region_to = product region`` (the
    same-region recipe makes the two equal), so cross-region pooling cannot leak.
    For stock-bounds rows ``region_to`` is pinned by the selector, so adding it to
    the key leaves their bucketing unchanged.
    """
    ei_exp = get_edge_cache(acc_struct)[0]
    r_all = ei_exp["region_to"].astype(str).to_numpy()
    t_all = ei_exp["time"].astype(str).to_numpy()
    v_all = ei_exp["valuation"].astype(str).to_numpy()
    a_edges = _match_side_edges(row, "a", acc_struct)
    b_edges = _match_side_edges(row, "b", acc_struct)
    if scope is not None:
        a_edges = a_edges[scope[a_edges]]
        b_edges = b_edges[scope[b_edges]]
    asz, bsz = int(a_edges.size), int(b_edges.size)
    if asz == 0 or bsz == 0:
        return [], asz, bsz
    ra, ta, va = r_all[a_edges], t_all[a_edges], v_all[a_edges]
    rb, tb, vb = r_all[b_edges], t_all[b_edges], v_all[b_edges]
    keys = sorted(set(zip(ra, ta, va)) & set(zip(rb, tb, vb)))
    buckets = []
    for r, t, v in keys:
        a_sub = a_edges[(ra == r) & (ta == t) & (va == v)]
        b_sub = b_edges[(rb == r) & (tb == t) & (vb == v)]
        buckets.append({"region": r, "time": t, "valuation": v,
                        "a_edges": a_sub.tolist(), "b_edges": b_sub.tolist()})
    return buckets, asz, bsz


class RatioConstraintBuilder(BaseConstraintBuilder):
    """
    Soft free-cone ratio constraint: a_i − r · a_j = ε_k

    Loaded from the same params CSV schema as ParametricImputer.
    The CSV uncertainty modes (sd / min_max / pedigree) yield the *relative*
    ratio-space σ_r per pair; :meth:`build_dynamic` rescales it to flow space
    symmetrically (Gauss–Newton linearization of the log-ratio error model at
    the prior, REP analysis_opt_20260611_075036 §8.2):

        σ_c = max(σ_r, sigma_rel_floor) · max(Σ|a₀,num|, |r|·Σ|a₀,denom|, scale_abs_floor)
    """

    def __init__(self, pairs: list[dict],
                 sigma_rel_floor: float = 0.02,
                 scale_abs_floor: float = 1.0):
        """
        pairs : list of dicts with keys:
            constraint_id, edge_i, edge_j, r, sigma_c   (sigma_c = relative σ_r)
        sigma_rel_floor : lower bound on σ_r (caps |z| at 1/floor on a 100 %
            ratio violation).
        scale_abs_floor : lower bound on the flow-space scale (tonnage), so
            both-zero-prior pairs keep a finite σ_c.
        """
        self._pairs = pairs
        self._sigma_rel_floor = float(sigma_rel_floor)
        self._scale_abs_floor = float(scale_abs_floor)

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: str | None = None,
        sigma_rel_floor: float = 0.02,
        scale_abs_floor: float = 1.0,
    ) -> "RatioConstraintBuilder":
        """
        Load from params CSV (same schema as ParametricImputer.from_csv).
        Resolves A→B edge pairs and computes the relative σ_r from unc_type.

        ``mask`` (optional): name of a mask in ``acc_struct.masks``. Pairs
        where either edge falls outside the mask are dropped.
        """
        df = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("*")
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
        for _, row in df.iterrows():
            nominal = float_or(row.get("value_nominal", "nan"), np.nan)
            if np.isnan(nominal):
                continue
            # Net-stock bands (param_type=="stock_bound", nominal r=0, centred at 0):
            # read the symmetric ±p95 half-band (value_max) as the ratio-space σ_r and
            # tag scale_mode="denominator" so build_dynamic scales σ_c off the THROUGHPUT
            # denominator (|r|·den = 0 here would otherwise collapse it to the abs floor
            # → quasi-hard). All other ratios keep the CSV unc modes + symmetric scale.
            if str(row.get("param_type", "")).strip() == "stock_bound":
                vmax = float_or(row.get("value_max", "nan"), np.nan)
                sigma = abs(vmax) / Z95_STOCK if not np.isnan(vmax) else 0.0
                scale_mode = "denominator"
            else:
                try:
                    sigma = sigma_from_row(row, nominal)
                except Exception:
                    sigma = 0.0
                scale_mode = "symmetric"

            # A user param_id, if present, becomes the grammar `variant` slot
            # (underscore-free); otherwise the ratio variant is "ratioeq".
            param_id = str(row.get("param_id", "")).strip() or None
            variant  = _safe(param_id, field="param_id") if param_id else "ratioeq"

            # agg rows pool every matched a-side edge against every matched
            # b-side edge per (region_to, time, valuation) bucket → one soft row
            #   Σ b − r·Σ a = ε   (pooled denominator; set-scoped).
            if _is_agg(row):
                buckets, asz, bsz = _agg_buckets(row, acc_struct, scope)
                if not buckets:
                    _log.warning(
                        "RatioConstraintBuilder %s: agg row had no overlapping "
                        "(region, time, valuation) buckets (a=%d b=%d) — skipped",
                        variant, asz, bsz,
                    )
                    continue
                for bk in buckets:
                    cid = build_constraint_id(
                        TYPE_RATIO, variant,
                        _safe(bk["region"], field="region"), bk["valuation"],
                        year=bk["time"],
                    )
                    pairs.append({
                        "constraint_id": cid,
                        "variant":     variant,
                        "agg":         True,
                        "edge_i_list": bk["b_edges"],
                        "edge_j_list": bk["a_edges"],
                        "r":           nominal,
                        "sigma_c":     sigma,
                        "scale_mode":  scale_mode,
                        "time":        bk["time"],
                        "valuation":   bk["valuation"],
                        "region":      bk["region"],
                    })
                continue

            for r in resolver.resolve_row(row):
                if scope is not None and not (scope[r.a_edge_id] and scope[r.b_edge_id]):
                    continue
                # edge_i (coeff +1) is the numerator; edge_j (coeff −r) the denom.
                num   = _decode_edge(ei_idx, r.b_edge_id)
                denom = _decode_edge(ei_idx, r.a_edge_id)
                cid = build_constraint_id(
                    TYPE_RATIO, variant,
                    render_edge(num["sector_from"], num["region_from"],
                                num["sector_to"], num["region_to"]),
                    render_edge(denom["sector_from"], denom["region_from"],
                                denom["sector_to"], denom["region_to"]),
                    num["valuation"], denom["valuation"],
                    year=num["time"],
                )
                pairs.append({
                    "constraint_id": cid,
                    "variant":  variant,
                    "edge_i":   r.b_edge_id,
                    "edge_j":   r.a_edge_id,
                    "r":        nominal,
                    "sigma_c":  sigma,
                    "scale_mode": scale_mode,
                    "time":     num["time"],
                    "num":      num,
                    "denom":    denom,
                })

        _log.info("RatioConstraintBuilder: %d pairs from %s", len(pairs), os.path.basename(path))
        return cls(pairs, sigma_rel_floor=sigma_rel_floor, scale_abs_floor=scale_abs_floor)

    @property
    def source_paths(self) -> list[str]:
        return []

    def build_static(self, acc_struct) -> ConstraintBlock:
        n_edges = acc_struct.n_edges
        if not self._pairs:
            return _empty_block(n_edges)

        M = len(self._pairs)
        rows_list, cols_list, vals_list = [], [], []
        meta_list = []

        for row_id, p in enumerate(self._pairs):
            if p.get("agg"):
                # Σ b (coeff +1) − r·Σ a (coeff −r) into one row; set-scoped, so
                # num_*/denom_* stay None (no single numerator/denominator edge).
                for ei in p["edge_i_list"]:
                    rows_list.append(row_id); cols_list.append(ei); vals_list.append(1.0)
                for ej in p["edge_j_list"]:
                    rows_list.append(row_id); cols_list.append(ej); vals_list.append(-p["r"])
                valuation = p["valuation"]
                struct = {}
            else:
                rows_list += [row_id, row_id]
                cols_list += [p["edge_i"], p["edge_j"]]
                vals_list += [1.0, -p["r"]]
                num, denom = p["num"], p["denom"]
                valuation = num["valuation"]
                struct = _ratio_struct_cols(num, denom)
            meta = empty_row_meta_dict()
            meta.update({
                "constraint_id": p["constraint_id"],
                "type":          TYPE_RATIO,
                "variant":       p["variant"],
                "is_hard":       False,
                "cone_type":     "free",
                "time":          p["time"],
                "valuation":     valuation,
                **struct,
            })
            meta_list.append(meta)

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(M, n_edges))
        _log.info("RatioConstraintBuilder.build_static: %d rows", M)
        block = ConstraintBlock(
            G_rows   = G,
            c        = np.zeros(M),
            sigma_c  = np.array([p["sigma_c"] for p in self._pairs]),
            row_meta = pd.DataFrame(meta_list),
        )
        self._cached_static = block
        return block

    def _scale_index(self):
        """Cached flattened (row, col) index arrays for the numerator and
        denominator sides of every pair, plus |r| per row — used to evaluate
        the symmetric flow-space scale in one vectorized pass per timestep."""
        cached = getattr(self, "_scale_idx", None)
        if cached is None:
            rn, cn, rd, cd, rabs, den_mode = [], [], [], [], [], []
            for row_id, p in enumerate(self._pairs):
                num_cols = p["edge_i_list"] if p.get("agg") else [p["edge_i"]]
                den_cols = p["edge_j_list"] if p.get("agg") else [p["edge_j"]]
                rn += [row_id] * len(num_cols); cn += list(num_cols)
                rd += [row_id] * len(den_cols); cd += list(den_cols)
                rabs.append(abs(p["r"]))
                den_mode.append(p.get("scale_mode") == "denominator")
            cached = (np.asarray(rn, dtype=int), np.asarray(cn, dtype=int),
                      np.asarray(rd, dtype=int), np.asarray(cd, dtype=int),
                      np.asarray(rabs), np.asarray(den_mode, dtype=bool))
            self._scale_idx = cached
        return cached

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Rescale the relative σ_r to flow space against the prior ``f_agg``
        (symmetric: max of numerator-side and r·denominator-side prior mass,
        floored), then deactivate rows whose ``time`` does not match ``t`` by
        setting ``sigma_c = SIGMA_C_INACTIVE``.

        Mirrors :meth:`SumConstraintBuilder.build_dynamic`. ``c`` is left
        unchanged (always 0 for ratio_eq).
        """
        static = getattr(self, "_cached_static", None) or self.build_static(acc_struct)
        if static.G_rows.shape[0] == 0:
            return _empty_block(acc_struct.n_edges)

        rn, cn, rd, cd, rabs, den_mode = self._scale_index()
        M = static.G_rows.shape[0]
        num_scale = np.zeros(M)
        den_scale = np.zeros(M)
        np.add.at(num_scale, rn, np.abs(f_agg[cn]))
        np.add.at(den_scale, rd, np.abs(f_agg[cd]))
        # Symmetric ratios (r≠0): scale by max(|num|, |r|·|den|). Net-stock bands
        # (scale_mode="denominator", r=0): scale by the throughput |denominator|
        # alone — else |r|·den=0 collapses σ_c to the abs floor (quasi-hard). The
        # abs floor still keeps σ_c finite & soft when the throughput collapses
        # (flat-prior-zero pin), avoiding re-infeasibility. See plan A2.
        scale   = np.where(den_mode, den_scale, np.maximum(num_scale, rabs * den_scale))
        scale   = np.maximum(scale, self._scale_abs_floor)
        sigma_r = np.maximum(static.sigma_c, self._sigma_rel_floor)
        sigma_c = sigma_r * scale

        t_str    = str(t)
        times    = static.row_meta["time"].astype(str).to_numpy()
        inactive = times != t_str
        sigma_c[inactive] = SIGMA_C_INACTIVE

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
    Hard non-negative-cone ratio bound constraints from the same params CSV.

    For each row with finite bound_lower / bound_upper:
      Upper: a_i − r_u · a_j ≤ 0  →  G row: +1 at edge_i, −r_u at edge_j
      Lower: −a_i + r_l · a_j ≤ 0 →  G row: −1 at edge_i, +r_l at edge_j
    """

    def __init__(self, bound_rows: list[dict], kind: str = "unknown"):
        self._bound_rows = bound_rows
        self._kind = kind

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: str | None = None,
        kind: str = "unknown",
    ) -> "RatioBoundBuilder":
        df = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("*")
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
        for _, row in df.iterrows():
            # Stripping ratio is soft-only (ADR 0005): its [value_min, value_max] is a
            # plausible range, not an empirical hard WR/Ore bound. The imputer fills
            # WR = max(0, CO − Ore) (ASSUMPTIONS.md), which a hard WR ≥ value_min·Ore bound
            # would make infeasible for crude_ore < grade-implied-ore regions. Skip it.
            if str(row.get("param_type", "")).strip() == "stripping_ratio":
                continue
            nominal = float_or(row.get("value_nominal", "nan"), np.nan)
            bl      = float_or(row.get("value_min", "nan"), np.nan)
            bu      = float_or(row.get("value_max", "nan"), np.nan)
            if np.isnan(bl) and np.isnan(bu):
                continue
            if not np.isnan(bl) and not np.isnan(bu) and bl > bu:
                raise ValueError(
                    f"RatioBoundBuilder: param {row.get('param_id', '?')!r} has "
                    f"value_min={bl} > value_max={bu} — empty ratio band."
                )

            # A user param_id, if present, becomes the grammar `variant` slot
            # (underscore-free); otherwise the bound variant is "bound".
            param_id = str(row.get("param_id", "")).strip() or None
            variant  = _safe(param_id, field="param_id") if param_id else "bound"

            if _is_agg(row):
                buckets, asz, bsz = _agg_buckets(row, acc_struct, scope)
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
                        "nominal":     nominal if not np.isnan(nominal) else 1.0,
                        "bl":          bl,
                        "bu":          bu,
                    })
                continue

            for r in resolver.resolve_row(row):
                if scope is not None and not (scope[r.a_edge_id] and scope[r.b_edge_id]):
                    continue
                bound_rows.append({
                    "variant":  variant,
                    "agg":      False,
                    "edge_i":   r.b_edge_id,
                    "edge_j":   r.a_edge_id,
                    "num":      _decode_edge(ei_idx, r.b_edge_id),
                    "denom":    _decode_edge(ei_idx, r.a_edge_id),
                    "nominal":  nominal if not np.isnan(nominal) else 1.0,
                    "bl":       bl,
                    "bu":       bu,
                })

        n_agg = sum(1 for br in bound_rows if br.get("agg"))
        _log.info("RatioBoundBuilder: %d bound rows from %s (%d agg, %d per-pair)",
                  len(bound_rows), os.path.basename(path),
                  n_agg, len(bound_rows) - n_agg)
        return cls(bound_rows, kind=kind)

    @property
    def source_paths(self) -> list[str]:
        return []

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
                # set-scoped: no single edge. Region is in the id so a shared
                # variant (e.g. one eff param across regions) stays unique per
                # (region, valuation, year).
                spec_tokens = (_safe(br["region"], field="region"), valuation)
                struct_cols = {}                      # num_*/denom_* stay None
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
                    "cone_type":     "nonneg",
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
        static = getattr(self, "_cached_static", None) or self.build_static(acc_struct)
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

