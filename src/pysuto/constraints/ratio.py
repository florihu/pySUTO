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
    float_or,
    get_edge_cache,
    resolve_period_selector,
    sigma_from_row,
)

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block

_log = get_logger(__name__)


_AGG_TRUE = {"true", "1", "yes", "y", "t"}


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
    Soft free-cone ratio constraint: a_i − r(t) · a_j = ε_k

    One structural row per resolved ``(edge_i, edge_j)`` pair. The
    coefficient ``r(t)`` is genuinely time-varying — ore grade declines
    over time, so the column carrying ``−r(t)`` is written per period by
    :meth:`build_dynamic`. Rows are active only in periods where the CSV
    actually supplies a value for that pair.
    """

    def __init__(self, pairs: list[dict]):
        """
        pairs : list of dicts with keys:
            constraint_id, edge_i, edge_j, r_by_t (dict t_idx → float),
            sigma_by_t (dict t_idx → float)
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

        Rows resolving to the same structural ``(edge_i, edge_j)`` pair
        are merged into one constraint row; each contributing CSV row
        injects its ``(r, σ)`` at the period selected by ``a_time`` /
        ``b_time`` (wildcard → every period). Pairs whose either edge
        falls outside ``mask`` are dropped.
        """
        df = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("*")
        resolver = ParamRelationResolver(acc_struct)
        time_range = list(getattr(acc_struct, "time_range", []))

        if mask is not None:
            if mask not in acc_struct.masks:
                raise ValueError(
                    f"RatioConstraintBuilder(mask={mask!r}): mask not in "
                    f"acc_struct.masks. Available: {sorted(acc_struct.masks)}"
                )
            scope = acc_struct.masks[mask]
        else:
            scope = None

        pairs_by_key: dict[tuple, dict] = {}

        for _, row in df.iterrows():
            nominal = float_or(row.get("value_nominal", "nan"), np.nan)
            if np.isnan(nominal):
                continue
            try:
                sigma = sigma_from_row(row, nominal)
            except Exception:
                sigma = 0.0

            param_id = str(row.get("param_id", "")).strip() or None
            t_idxs   = resolve_period_selector(row.get("a_time", "*"), time_range)
            if not t_idxs:
                continue

            for rr in resolver.resolve_row(row):
                if scope is not None and not (scope[rr.a_edge_id] and scope[rr.b_edge_id]):
                    continue
                key = (rr.b_edge_id, rr.a_edge_id)
                entry = pairs_by_key.get(key)
                if entry is None:
                    entry = {
                        "constraint_id": param_id or f"ratio_{rr.a_edge_id}_{rr.b_edge_id}",
                        "edge_i":   rr.b_edge_id,
                        "edge_j":   rr.a_edge_id,
                        "r_by_t":     {},
                        "sigma_by_t": {},
                    }
                    pairs_by_key[key] = entry
                for ti in t_idxs:
                    entry["r_by_t"][ti]     = nominal
                    entry["sigma_by_t"][ti] = sigma

        pairs = list(pairs_by_key.values())
        _log.info(
            "RatioConstraintBuilder: %d structural pairs from %s",
            len(pairs), os.path.basename(path),
        )
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
            # Default static coefficient is the first period's r value
            # (any period that has a value works for the sparsity pattern;
            # build_dynamic overwrites per-t).
            ts_with_r = sorted(p["r_by_t"].keys())
            r_default = p["r_by_t"][ts_with_r[0]] if ts_with_r else 0.0
            rows_list += [row_id, row_id]
            cols_list += [p["edge_i"], p["edge_j"]]
            vals_list += [1.0, -r_default]
            meta_list.append({
                "constraint_id": p["constraint_id"],
                "type":          "ratio_eq",
                "is_hard":       False,
                "cone_type":     "free",
            })

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(M, n_edges))
        _log.info("RatioConstraintBuilder.build_static: %d rows", M)
        # Default static σ_c is the first period's σ (placeholder).
        sigma_static = np.array([
            p["sigma_by_t"][sorted(p["sigma_by_t"])[0]] if p["sigma_by_t"] else 0.0
            for p in self._pairs
        ])
        block = ConstraintBlock(
            G_rows   = G,
            c        = np.zeros(M),
            sigma_c  = sigma_static,
            row_meta = pd.DataFrame(meta_list),
        )
        self._cached_static = block
        return block

    def build_dynamic(self, acc_struct, a_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Per-period overlay. Re-emits G_rows with ``−r(t)`` for every pair
        that has a parameter at this period; pairs without a parameter
        at ``t`` are marked ``active = False`` (G_rows still carries the
        placeholder coefficient — the engine drops the row from the
        objective via the activation mask).
        """
        static = getattr(self, "_cached_static", None) or self.build_static(acc_struct)
        if static.G_rows.shape[0] == 0:
            return _empty_block(acc_struct.n_edges)

        time_range = list(getattr(acc_struct, "time_range", []))
        t_str = str(t)
        try:
            ti = [str(x) for x in time_range].index(t_str)
        except ValueError:
            ti = None

        M = static.G_rows.shape[0]
        sigma_c = static.sigma_c.copy()
        active  = np.zeros(M, dtype=bool)

        # Rewrite the −r entry per row using the period coefficient.
        G = static.G_rows.tocsr(copy=True)
        for row_id, p in enumerate(self._pairs):
            r_t = p["r_by_t"].get(ti)
            if r_t is None:
                continue
            active[row_id] = True
            sigma_c[row_id] = p["sigma_by_t"].get(ti, sigma_c[row_id])

            # The pair has two entries in G_rows: +1 at edge_i, −r at edge_j.
            # Find the slot whose column == edge_j and overwrite with −r_t.
            cols_row = G.indices[G.indptr[row_id]: G.indptr[row_id + 1]]
            data_row = G.data[G.indptr[row_id]: G.indptr[row_id + 1]]
            for k, col in enumerate(cols_row):
                if col == p["edge_j"]:
                    data_row[k] = -r_t

        return ConstraintBlock(
            G_rows   = G,
            c        = static.c,
            sigma_c  = sigma_c,
            row_meta = static.row_meta,
            active   = active,
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
        time_range = list(getattr(acc_struct, "time_range", []))

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
        v_all = ei_exp["valuation"].astype(str).to_numpy()

        # Deduplicate per structural (edge_i, edge_j); merge per-period
        # bound values from each CSV row.
        agg_by_key: dict[tuple, dict]   = {}
        pair_by_key: dict[tuple, dict]  = {}

        for _, row in df.iterrows():
            nominal = float_or(row.get("value_nominal", "nan"), np.nan)
            bl      = float_or(row.get("bound_lower", "nan"), np.nan)
            bu      = float_or(row.get("bound_upper", "nan"), np.nan)
            if np.isnan(bl) and np.isnan(bu):
                continue

            param_id = str(row.get("param_id", "")).strip() or None
            t_idxs   = resolve_period_selector(row.get("a_time", "*"), time_range)
            if not t_idxs:
                continue

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
                v_a = v_all[a_edges]
                v_b = v_all[b_edges]
                vals = sorted(set(v_a.tolist()) & set(v_b.tolist()))
                if not vals:
                    _log.warning(
                        "RatioBoundBuilder %s: agg row had no overlapping valuation "
                        "buckets between a (%d edges) and b (%d edges) — skipped",
                        param_id, a_edges.size, b_edges.size,
                    )
                    continue
                for v in vals:
                    a_sub = a_edges[v_a == v]
                    b_sub = b_edges[v_b == v]
                    key = ("agg", tuple(sorted(b_sub.tolist())),
                           tuple(sorted(a_sub.tolist())), v)
                    entry = agg_by_key.get(key)
                    if entry is None:
                        entry = {
                            "param_id":    param_id or f"bound_agg_{v}",
                            "agg":         True,
                            "edge_i_list": b_sub.tolist(),
                            "edge_j_list": a_sub.tolist(),
                            "nominal":     nominal if not np.isnan(nominal) else 1.0,
                            "bl_by_t":     {},
                            "bu_by_t":     {},
                        }
                        agg_by_key[key] = entry
                    for ti in t_idxs:
                        if not np.isnan(bl):
                            entry["bl_by_t"][ti] = bl
                        if not np.isnan(bu):
                            entry["bu_by_t"][ti] = bu
                continue

            for rr in resolver.resolve_row(row):
                if scope is not None and not (scope[rr.a_edge_id] and scope[rr.b_edge_id]):
                    continue
                key = ("pair", rr.b_edge_id, rr.a_edge_id)
                entry = pair_by_key.get(key)
                if entry is None:
                    entry = {
                        "param_id": param_id or f"bound_{rr.a_edge_id}_{rr.b_edge_id}",
                        "agg":      False,
                        "edge_i":   rr.b_edge_id,
                        "edge_j":   rr.a_edge_id,
                        "nominal":  nominal if not np.isnan(nominal) else 1.0,
                        "bl_by_t":  {},
                        "bu_by_t":  {},
                    }
                    pair_by_key[key] = entry
                for ti in t_idxs:
                    if not np.isnan(bl):
                        entry["bl_by_t"][ti] = bl
                    if not np.isnan(bu):
                        entry["bu_by_t"][ti] = bu

        bound_rows = list(agg_by_key.values()) + list(pair_by_key.values())
        n_agg = sum(1 for br in bound_rows if br.get("agg"))
        _log.info("RatioBoundBuilder: %d bound rows from %s (%d agg, %d per-pair)",
                  len(bound_rows), os.path.basename(path),
                  n_agg, len(bound_rows) - n_agg)
        return cls(bound_rows, kind=kind)

    @property
    def source_paths(self) -> list[str]:
        return []

    def build_static(self, acc_struct) -> ConstraintBlock:
        """
        One structural row per (bound entry, direction) — ``upper`` and
        ``lower`` get their own rows. Each row's ``ratio_slot`` column on
        ``row_meta`` carries the spec index so ``build_dynamic`` can
        write the period coefficient.
        """
        n_edges = acc_struct.n_edges
        if not self._bound_rows:
            return _empty_block(n_edges)

        rows_list, cols_list, vals_list = [], [], []
        meta_list = []
        row_id = 0
        # Per-row metadata: spec index (back-ref) + direction (upper/lower).
        self._row_specs: list[tuple[int, str]] = []

        for spec_idx, br in enumerate(self._bound_rows):
            if br.get("agg"):
                ei_list = br["edge_i_list"]
                ej_list = br["edge_j_list"]
            else:
                ei_list = [br["edge_i"]]
                ej_list = [br["edge_j"]]

            bu_default = (
                br["bu_by_t"][sorted(br["bu_by_t"])[0]] if br["bu_by_t"] else None
            )
            bl_default = (
                br["bl_by_t"][sorted(br["bl_by_t"])[0]] if br["bl_by_t"] else None
            )

            if bu_default is not None:
                for ej in ej_list:
                    rows_list.append(row_id); cols_list.append(ej); vals_list.append(1.0)
                for ei in ei_list:
                    rows_list.append(row_id); cols_list.append(ei); vals_list.append(-bu_default)
                meta_list.append({
                    "constraint_id": f"{br['param_id']}_upper",
                    "type":          "ratio_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
                    "ratio_kind":    self._kind,
                    "bound_dir":     "upper",
                })
                self._row_specs.append((spec_idx, "upper"))
                row_id += 1

            if bl_default is not None:
                for ej in ej_list:
                    rows_list.append(row_id); cols_list.append(ej); vals_list.append(-1.0)
                for ei in ei_list:
                    rows_list.append(row_id); cols_list.append(ei); vals_list.append(bl_default)
                meta_list.append({
                    "constraint_id": f"{br['param_id']}_lower",
                    "type":          "ratio_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
                    "ratio_kind":    self._kind,
                    "bound_dir":     "lower",
                })
                self._row_specs.append((spec_idx, "lower"))
                row_id += 1

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

    def build_dynamic(self, acc_struct, a_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Per-period overlay. For each row, rewrite the bound coefficient
        (``−bu(t)`` on upper rows, ``+bl(t)`` on lower rows) using the
        spec's stored per-period value. Mark inactive when no bound is
        defined at this period.
        """
        static = getattr(self, "_cached_static", None) or self.build_static(acc_struct)
        if static.G_rows.shape[0] == 0:
            return _empty_block(acc_struct.n_edges)

        time_range = list(getattr(acc_struct, "time_range", []))
        t_str = str(t)
        try:
            ti = [str(x) for x in time_range].index(t_str)
        except ValueError:
            ti = None

        M = static.G_rows.shape[0]
        active = np.zeros(M, dtype=bool)
        G = static.G_rows.tocsr(copy=True)

        for row_id, (spec_idx, direction) in enumerate(self._row_specs):
            spec = self._bound_rows[spec_idx]
            ei_list = spec["edge_i_list"] if spec.get("agg") else [spec["edge_i"]]

            if direction == "upper":
                value = spec["bu_by_t"].get(ti)
                if value is None:
                    continue
                coeff = -value
            else:
                value = spec["bl_by_t"].get(ti)
                if value is None:
                    continue
                coeff = value

            active[row_id] = True
            cols_row = G.indices[G.indptr[row_id]: G.indptr[row_id + 1]]
            data_row = G.data[G.indptr[row_id]: G.indptr[row_id + 1]]
            ei_set = set(int(e) for e in ei_list)
            for k, col in enumerate(cols_row):
                if int(col) in ei_set:
                    data_row[k] = coeff

        return ConstraintBlock(
            G_rows   = G,
            c        = static.c,
            sigma_c  = static.sigma_c,
            row_meta = static.row_meta,
            active   = active,
        )


# =============================================================================
# FlowBoundsBuilder
# =============================================================================

