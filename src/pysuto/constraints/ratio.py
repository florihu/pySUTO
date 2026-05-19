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
from pysuto.param_relation import ParamRelationResolver, float_or, sigma_from_row

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block

_log = get_logger(__name__)


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

            param_id = str(row.get("param_id", "")).strip() or None

            for r in resolver.resolve_row(row):
                if scope is not None and not (scope[r.a_edge_id] and scope[r.b_edge_id]):
                    continue
                pairs.append({
                    "constraint_id": param_id or f"ratio_{r.a_edge_id}_{r.b_edge_id}",
                    "edge_i":  r.b_edge_id,
                    "edge_j":  r.a_edge_id,
                    "r":       nominal,
                    "sigma_c": sigma,
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
        # Time per row is taken from edge_i (edge_i and edge_j share the
        # same time by construction in the param_relation B-resolver).
        edge_time = acc_struct.edge_index["time"].astype(str).to_numpy()

        for row_id, p in enumerate(self._pairs):
            rows_list += [row_id, row_id]
            cols_list += [p["edge_i"], p["edge_j"]]
            vals_list += [1.0, -p["r"]]
            meta_list.append({
                "constraint_id": p["constraint_id"],
                "type":          "ratio_eq",
                "is_hard":       False,
                "cone_type":     "free",
                "time":          edge_time[p["edge_i"]],
            })

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(M, n_edges))
        _log.info("RatioConstraintBuilder.build_static: %d rows", M)
        return ConstraintBlock(
            G_rows   = G,
            c        = np.zeros(M),
            sigma_c  = np.array([p["sigma_c"] for p in self._pairs]),
            row_meta = pd.DataFrame(meta_list),
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

    def __init__(self, bound_rows: list[dict]):
        self._bound_rows = bound_rows

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: str | None = None,
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

        bound_rows = []
        for _, row in df.iterrows():
            nominal = float_or(row.get("value_nominal", "nan"), np.nan)
            bl      = float_or(row.get("bound_lower", "nan"), np.nan)
            bu      = float_or(row.get("bound_upper", "nan"), np.nan)
            if np.isnan(bl) and np.isnan(bu):
                continue

            param_id = str(row.get("param_id", "")).strip() or None

            for r in resolver.resolve_row(row):
                if scope is not None and not (scope[r.a_edge_id] and scope[r.b_edge_id]):
                    continue
                bound_rows.append({
                    "param_id": param_id or f"bound_{r.a_edge_id}_{r.b_edge_id}",
                    "edge_i":   r.b_edge_id,
                    "edge_j":   r.a_edge_id,
                    "nominal":  nominal if not np.isnan(nominal) else 1.0,
                    "bl":       bl,
                    "bu":       bu,
                })

        _log.info("RatioBoundBuilder: %d bound rows from %s", len(bound_rows), os.path.basename(path))
        return cls(bound_rows)

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
        edge_time = acc_struct.edge_index["time"].astype(str).to_numpy()

        for br in self._bound_rows:
            ei, ej = br["edge_i"], br["edge_j"]
            bu, bl = br["bu"], br["bl"]
            t_row = edge_time[ei]

            if not np.isnan(bu):
                rows_list += [row_id, row_id]
                cols_list += [ei, ej]
                vals_list += [1.0, -bu]
                meta_list.append({
                    "constraint_id": f"{br['param_id']}_upper",
                    "type":          "ratio_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
                    "time":          t_row,
                })
                row_id += 1

            if not np.isnan(bl):
                rows_list += [row_id, row_id]
                cols_list += [ei, ej]
                vals_list += [-1.0, bl]
                meta_list.append({
                    "constraint_id": f"{br['param_id']}_lower",
                    "type":          "ratio_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
                    "time":          t_row,
                })
                row_id += 1

        if row_id == 0:
            return _empty_block(n_edges)

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(row_id, n_edges))
        _log.info("RatioBoundBuilder.build_static: %d rows", row_id)
        return ConstraintBlock(
            G_rows   = G,
            c        = np.zeros(row_id),
            sigma_c  = np.zeros(row_id),
            row_meta = pd.DataFrame(meta_list),
        )


# =============================================================================
# FlowBoundsBuilder
# =============================================================================

