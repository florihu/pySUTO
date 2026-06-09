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


class RatioConstraintBuilder(BaseConstraintBuilder):
    """
    Soft free-cone ratio constraint: a_i − r · a_j = ε_k

    Loaded from the same params CSV schema as ParametricImputer.
    sigma_c comes from the three standard uncertainty modes (sd / min_max / pedigree).
    """

    def __init__(self, pairs: list[dict]):
        """
        pairs : list of dicts with keys:
            constraint_id, edge_i, edge_j, r, sigma_c
        """
        self._pairs = pairs

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: str | None = None,
    ) -> "RatioConstraintBuilder":
        """
        Load from params CSV (same schema as ParametricImputer.from_csv).
        Resolves A→B edge pairs and computes sigma_c from unc_type column.

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
            try:
                sigma = sigma_from_row(row, nominal)
            except Exception:
                sigma = 0.0

            # A user param_id, if present, becomes the grammar `variant` slot
            # (underscore-free); otherwise the ratio variant is "ratioeq".
            param_id = str(row.get("param_id", "")).strip() or None
            variant  = _safe(param_id, field="param_id") if param_id else "ratioeq"

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
                    "time":     num["time"],
                    "num":      num,
                    "denom":    denom,
                })

        _log.info("RatioConstraintBuilder: %d pairs from %s", len(pairs), os.path.basename(path))
        return cls(pairs)

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
            rows_list += [row_id, row_id]
            cols_list += [p["edge_i"], p["edge_j"]]
            vals_list += [1.0, -p["r"]]
            num, denom = p["num"], p["denom"]
            meta = empty_row_meta_dict()
            meta.update({
                "constraint_id": p["constraint_id"],
                "type":          TYPE_RATIO,
                "variant":       p["variant"],
                "is_hard":       False,
                "cone_type":     "free",
                "time":          p["time"],
                "valuation":     num["valuation"],
                **_ratio_struct_cols(num, denom),
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

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Deactivate ratio_eq rows whose ``time`` does not match ``t`` by
        setting ``sigma_c = SIGMA_C_INACTIVE``. In-year rows keep the
        per-pair σ_c assembled in :meth:`build_static`.

        Mirrors :meth:`SumConstraintBuilder.build_dynamic`. ``c`` is left
        unchanged (always 0 for ratio_eq).
        """
        static = getattr(self, "_cached_static", None) or self.build_static(acc_struct)
        if static.G_rows.shape[0] == 0:
            return _empty_block(acc_struct.n_edges)

        sigma_c  = static.sigma_c.copy()
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

        ei_exp, _el, _itt, _fv, _mc, _eid = get_edge_cache(acc_struct)
        t_all = ei_exp["time"].astype(str).to_numpy()
        v_all = ei_exp["valuation"].astype(str).to_numpy()
        ei_idx = expand_edge_index(acc_struct.edge_index, acc_struct.node_index).set_index("edge_id")

        bound_rows = []
        for _, row in df.iterrows():
            nominal = float_or(row.get("value_nominal", "nan"), np.nan)
            bl      = float_or(row.get("bound_lower", "nan"), np.nan)
            bu      = float_or(row.get("bound_upper", "nan"), np.nan)
            if np.isnan(bl) and np.isnan(bu):
                continue

            # A user param_id, if present, becomes the grammar `variant` slot
            # (underscore-free); otherwise the bound variant is "bound".
            param_id = str(row.get("param_id", "")).strip() or None
            variant  = _safe(param_id, field="param_id") if param_id else "bound"

            if _is_agg(row):
                a_edges = _match_side_edges(row, "a", acc_struct)
                b_edges = _match_side_edges(row, "b", acc_struct)
                if scope is not None:
                    a_edges = a_edges[scope[a_edges]]
                    b_edges = b_edges[scope[b_edges]]
                if a_edges.size == 0 or b_edges.size == 0:
                    _log.warning(
                        "RatioBoundBuilder %s: agg row matched no edges (a=%d b=%d) — skipped",
                        param_id, a_edges.size, b_edges.size,
                    )
                    continue
                t_a, v_a = t_all[a_edges], v_all[a_edges]
                t_b, v_b = t_all[b_edges], v_all[b_edges]
                keys = sorted(set(zip(t_a, v_a)) & set(zip(t_b, v_b)))
                if not keys:
                    _log.warning(
                        "RatioBoundBuilder %s: agg row had no overlapping (time, valuation) "
                        "buckets between a (%d edges) and b (%d edges) — skipped",
                        param_id, a_edges.size, b_edges.size,
                    )
                    continue
                for t, v in keys:
                    a_sub = a_edges[(t_a == t) & (v_a == v)]
                    b_sub = b_edges[(t_b == t) & (v_b == v)]
                    bound_rows.append({
                        "variant":     variant,
                        "agg":         True,
                        "edge_i_list": b_sub.tolist(),
                        "edge_j_list": a_sub.tolist(),
                        "time":        t,
                        "valuation":   v,
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
                spec_tokens = (valuation,)            # set-scoped: no single edge
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
                    rows_list.append(row_id); cols_list.append(ej); vals_list.append(ej_sign)
                for ei in ei_list:
                    rows_list.append(row_id); cols_list.append(ei); vals_list.append(ei_sign * r_coeff)
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
                _emit("upper", ei_sign=-1.0, ej_sign=1.0, r_coeff=bu)
            if not np.isnan(bl):
                _emit("lower", ei_sign=1.0, ej_sign=-1.0, r_coeff=bl)

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

