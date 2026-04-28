"""
constraints.py
==============
Plugin-based constraint engine for pySUTO.

Architecture mirrors ImputationEngine: register BaseConstraintBuilder subclasses,
call run() to produce a ConstraintRun with G (time-invariant), c(t), sigma_c(t),
row_meta, u, l, a0, sigma_a.

Five built-in builders
----------------------
MassBalanceBuilder    Hard zero-cone: Σ supply_i − Σ use_i = 0 per (Flow|Process) node×time
SumConstraintBuilder  Soft free:      Σ coeff_i · a_i = c_k + ε  (from acc_struct or CSV)
RatioConstraintBuilder Soft free:     a_i − r · a_j = ε           (from params CSV)
RatioBoundBuilder     Hard non-neg:   a_i − r_u·a_j ≤ 0           (from same params CSV)
FlowBoundsBuilder     Hard non-neg:   l ≤ a ≤ u                    (from acc_struct.bounds)

ConstraintBlock output is solver-neutral.  A thin transform script maps cone_type
values ("zero", "nonneg", "free") to Clarabel's cone objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .impute import (
    _build_edge_lookup,
    _expand_edge_index,
    _enumerate_a_edges,
    _resolve_b_ids,
    _sigma_from_row,
    _float_or,
    _WILDCARD,
    _A_FIELDS,
    _B_FIELDS,
)

try:
    from .logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

_log = get_logger(__name__)


# =============================================================================
# ConstraintBlock  — output of a single builder for one phase (static/dynamic)
# =============================================================================

@dataclass
class ConstraintBlock:
    """
    Sparse block of constraint rows from one builder.

    G_rows    : (M_k × E) sparse CSR matrix
    c         : (M_k,) RHS values
    sigma_c   : (M_k,) per-row uncertainty (0 for hard constraints)
    row_meta  : DataFrame with one row per G row; columns:
                  constraint_id (str), type (str), is_hard (bool), cone_type (str)
    """
    G_rows:   sp.csr_matrix
    c:        np.ndarray
    sigma_c:  np.ndarray
    row_meta: pd.DataFrame


def _empty_block(n_edges: int) -> ConstraintBlock:
    return ConstraintBlock(
        G_rows   = sp.csr_matrix((0, n_edges)),
        c        = np.empty(0),
        sigma_c  = np.empty(0),
        row_meta = pd.DataFrame(columns=["constraint_id", "type", "is_hard", "cone_type"]),
    )


def _concat_blocks(blocks: list[ConstraintBlock], n_edges: int) -> ConstraintBlock:
    non_empty = [b for b in blocks if b.G_rows.shape[0] > 0]
    if not non_empty:
        return _empty_block(n_edges)
    return ConstraintBlock(
        G_rows   = sp.vstack([b.G_rows for b in non_empty], format="csr"),
        c        = np.concatenate([b.c for b in non_empty]),
        sigma_c  = np.concatenate([b.sigma_c for b in non_empty]),
        row_meta = pd.concat([b.row_meta for b in non_empty], ignore_index=True),
    )


# =============================================================================
# BaseConstraintBuilder
# =============================================================================

class BaseConstraintBuilder(ABC):
    """
    Abstract base for all constraint builders.

    build_static()  — rows whose G coefficients never change across timesteps.
                      Called once per run.
    build_dynamic() — rows whose c or sigma_c depend on f_agg / sigma_agg at time t.
                      Default: returns empty block (override only when needed).
    """

    @abstractmethod
    def build_static(self, acc_struct) -> ConstraintBlock:
        ...

    def build_dynamic(
        self,
        acc_struct,
        f_agg:     np.ndarray,
        sigma_agg: np.ndarray,
        t:         str,
    ) -> ConstraintBlock:
        return _empty_block(acc_struct.n_edges)

    @property
    def source_paths(self) -> list[str]:
        return []


# =============================================================================
# MassBalanceBuilder
# =============================================================================

class MassBalanceBuilder(BaseConstraintBuilder):
    """
    Hard zero-cone constraint: Σ supply_edges − Σ use_edges = 0
    for every (Flow | Process) node that has ≥1 supply edge AND ≥1 use edge,
    per timestep.

    SB nodes are open-boundary endpoints and are excluded.
    G is time-invariant: one row per (balanced_node, time) pair.
    """

    _BALANCE_ENTITIES = {"Flow", "Process"}

    def build_static(self, acc_struct) -> ConstraintBlock:
        ei        = acc_struct.edge_index
        node_idx  = acc_struct.node_index
        n_edges   = acc_struct.n_edges
        times     = acc_struct.time_range

        # Expand edge_index with decoded node attributes
        ni = node_idx.loc[ei["from_id"].values]
        nj = node_idx.loc[ei["to_id"].values]
        ef_arr = ni["entity"].values
        et_arr = nj["entity"].values
        sf_arr = acc_struct.supply_edge_mask
        uf_arr = acc_struct.use_edge_mask
        t_arr  = ei["time"].values

        rows_list, cols_list, vals_list = [], [], []
        c_list, sigma_list, meta_list   = [], [], []
        row_id = 0

        for time in times:
            t_mask = (t_arr == str(time))
            # Any node that has at least one incoming AND one outgoing edge is balanced.
            # Incoming: edges where to_id == n  (supply edges for Flow nodes,
            #                                    use edges for Process nodes)
            # Outgoing: edges where from_id == n (supply edges for Process nodes,
            #                                     use edges for Flow nodes)
            nodes_with_incoming = set(ei.loc[t_mask, "to_id"].values)
            nodes_with_outgoing = set(ei.loc[t_mask, "from_id"].values)
            balanced = nodes_with_incoming & nodes_with_outgoing

            for nid in sorted(balanced):
                entity = str(node_idx.loc[nid, "entity"])
                if entity not in self._BALANCE_ENTITIES:
                    continue

                # incoming edges: to_id == nid  → coefficient +1
                inc_mask = t_mask & (ei["to_id"].values == nid)
                # outgoing edges: from_id == nid → coefficient −1
                out_mask = t_mask & (ei["from_id"].values == nid)

                inc_ids = ei.loc[inc_mask, "edge_id"].values
                out_ids = ei.loc[out_mask, "edge_id"].values

                if len(inc_ids) == 0 or len(out_ids) == 0:
                    continue

                for eid in inc_ids:
                    rows_list.append(row_id); cols_list.append(eid); vals_list.append(1.0)
                for eid in out_ids:
                    rows_list.append(row_id); cols_list.append(eid); vals_list.append(-1.0)

                c_list.append(0.0)
                sigma_list.append(0.0)
                meta_list.append({
                    "constraint_id": f"mass_balance_node{nid}_t{time}",
                    "type": "mass_balance",
                    "is_hard": True,
                    "cone_type": "zero",
                })
                row_id += 1

        if row_id == 0:
            return _empty_block(n_edges)

        G = sp.csr_matrix(
            (vals_list, (rows_list, cols_list)),
            shape=(row_id, n_edges),
        )
        return ConstraintBlock(
            G_rows   = G,
            c        = np.array(c_list),
            sigma_c  = np.array(sigma_list),
            row_meta = pd.DataFrame(meta_list),
        )


# =============================================================================
# SumConstraintBuilder
# =============================================================================

class SumConstraintBuilder(BaseConstraintBuilder):
    """
    Soft free-cone sum constraint: Σ coeff_i · a_i = c_k + ε_k

    Two construction modes:
      from_acc_struct() — auto-derives "useful supply" balance from acc_struct topology
      from_csv()        — loads user-defined sum constraints from a CSV file

    G rows are static (built once).  c and sigma_c are dynamic when
    sigma_ck_mode == "fraction_of_flow" (depends on f_agg per timestep).
    For modes "sd" and "pedigree" the sigma_c is static too.
    """

    def __init__(self, specs: list[dict], sigma_fraction: float = 0.0):
        """
        specs : list of dicts, each with keys:
            constraint_id, description,
            edge_ids: list[int], coefficients: list[float],
            c_k: float, sigma_ck_mode: str,
            sigma_ck_value: float, sigma_ck_fraction: float
        """
        self._specs = specs
        self._sigma_fraction = sigma_fraction  # fallback

    # ------------------------------------------------------------------
    # Factory: auto-derive from acc_struct topology
    # ------------------------------------------------------------------

    @classmethod
    def from_acc_struct(
        cls,
        acc_struct,
        sigma_fraction: float = 0.05,
        entity_filter: Optional[set] = None,
    ) -> "SumConstraintBuilder":
        """
        Auto-generate one soft sum constraint per (Flow|Process) balance node
        per timestep:  Σ useful_supply_i = Σ use_inputs_i  (excluding SB edges).

        entity_filter : set of entity strings to include (default: {Flow, Process})
        sigma_fraction: σ_c = sigma_fraction × |sum of supply flow magnitudes|
        """
        if entity_filter is None:
            entity_filter = {"Flow", "Process"}

        ei       = acc_struct.edge_index
        ni       = acc_struct.node_index
        sf_mask  = acc_struct.supply_edge_mask
        uf_mask  = acc_struct.use_edge_mask
        sb_mask  = acc_struct.sb_edge_mask
        t_arr    = ei["time"].values
        times    = acc_struct.time_range

        specs = []
        for time in times:
            t_mask = (t_arr == str(time))

            supply_to_nodes = set(ei.loc[t_mask & sf_mask & ~sb_mask, "to_id"].values)
            use_from_nodes  = set(ei.loc[t_mask & uf_mask, "from_id"].values)
            balanced = supply_to_nodes & use_from_nodes

            for nid in sorted(balanced):
                entity = str(ni.loc[nid, "entity"])
                if entity not in entity_filter:
                    continue

                sup_ids = ei.loc[
                    t_mask & sf_mask & ~sb_mask & (ei["to_id"].values == nid),
                    "edge_id",
                ].values.tolist()
                use_ids = ei.loc[
                    t_mask & uf_mask & (ei["from_id"].values == nid),
                    "edge_id",
                ].values.tolist()

                if not sup_ids or not use_ids:
                    continue

                all_ids   = sup_ids + use_ids
                all_coeff = [1.0] * len(sup_ids) + [-1.0] * len(use_ids)

                specs.append({
                    "constraint_id":   f"useful_supply_node{nid}_t{time}",
                    "description":     f"Useful supply balance node {nid} t={time}",
                    "edge_ids":        all_ids,
                    "coefficients":    all_coeff,
                    "c_k":             0.0,
                    "sigma_ck_mode":   "fraction_of_flow",
                    "sigma_ck_value":  0.0,
                    "sigma_ck_fraction": sigma_fraction,
                    "_supply_ids":     sup_ids,  # stored for sigma calc
                })

        return cls(specs, sigma_fraction=sigma_fraction)

    # ------------------------------------------------------------------
    # Factory: load from CSV
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str, acc_struct) -> "SumConstraintBuilder":
        """
        Load user-defined sum constraints from a long-format CSV.

        Required columns:
            constraint_id, coefficient, c_k, sigma_ck_mode
        Optional:
            description, sigma_ck_value, sigma_ck_fraction
        Edge selector columns (all support * wildcard):
            sector_from, entity_from, region_from,
            sector_to,   entity_to,   region_to,
            time, valuation
        """
        _NUMERIC_COLS = {"c_k", "coefficient", "sigma_ck_value", "sigma_ck_fraction"}

        def _read_cell(val, default):
            s = str(val).strip()
            return default if s in ("", "nan", "None", "*") else float(s)

        df = pd.read_csv(path, dtype=str).fillna("*")
        ei_exp      = _expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        edge_lookup = _build_edge_lookup(ei_exp)
        field_vals  = {
            col: sorted(str(v) for v in ei_exp[col].unique())
            for col in [
                "sector_from", "entity_from", "region_from",
                "sector_to",   "entity_to",   "region_to",
                "time",        "valuation",
            ]
        }

        # Map CSV selector columns to the 8 A-field keys expected by _enumerate_a_edges
        _col_map = {
            "sector_from": "a_sector_from", "entity_from": "a_entity_from",
            "region_from": "a_region_from", "sector_to":   "a_sector_to",
            "entity_to":   "a_entity_to",   "region_to":   "a_region_to",
            "time":        "a_time",         "valuation":   "a_valuation",
        }

        grouped = df.groupby("constraint_id", sort=False)
        specs   = []

        for cid, grp in grouped:
            all_ids, all_coeff = [], []
            desc = str(grp["description"].iloc[0]) if "description" in grp.columns else str(cid)
            c_k  = _read_cell(grp["c_k"].iloc[0], 0.0)
            mode = str(grp.get("sigma_ck_mode", pd.Series(["sd"])).iloc[0]).strip()
            sv   = _read_cell(grp.get("sigma_ck_value",    pd.Series(["0"])).iloc[0], 0.0)
            sf   = _read_cell(grp.get("sigma_ck_fraction", pd.Series(["0"])).iloc[0], 0.0)

            for _, row in grp.iterrows():
                # Build a pseudo-row with "a_*" keys for _enumerate_a_edges
                a_row = {}
                for src_col, dst_col in _col_map.items():
                    a_row[dst_col] = str(row.get(src_col, "*")).strip()
                a_row = pd.Series(a_row)
                coeff = float(row["coefficient"])

                for a_tuple in _enumerate_a_edges(a_row, field_vals):
                    eid = edge_lookup.get(a_tuple)
                    if eid is not None:
                        all_ids.append(eid)
                        all_coeff.append(coeff)

            if not all_ids:
                _log.warning("SumConstraintBuilder: constraint '%s' matched no edges", cid)
                continue

            specs.append({
                "constraint_id":     str(cid),
                "description":       desc,
                "edge_ids":          all_ids,
                "coefficients":      all_coeff,
                "c_k":               c_k,
                "sigma_ck_mode":     mode,
                "sigma_ck_value":    sv,
                "sigma_ck_fraction": sf,
                "_supply_ids":       [e for e, c in zip(all_ids, all_coeff) if c > 0],
            })

        return cls(specs)

    # ------------------------------------------------------------------

    @property
    def source_paths(self) -> list[str]:
        return []

    def build_static(self, acc_struct) -> ConstraintBlock:
        n_edges = acc_struct.n_edges
        if not self._specs:
            return _empty_block(n_edges)

        rows_list, cols_list, vals_list = [], [], []
        meta_list = []

        for row_id, spec in enumerate(self._specs):
            for eid, coeff in zip(spec["edge_ids"], spec["coefficients"]):
                rows_list.append(row_id); cols_list.append(eid); vals_list.append(coeff)
            meta_list.append({
                "constraint_id": spec["constraint_id"],
                "type":          "sum_eq",
                "is_hard":       False,
                "cone_type":     "free",
            })

        M = len(self._specs)
        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(M, n_edges))

        # For non-dynamic sigma modes, fill sigma_c here
        sigma_c = np.zeros(M)
        c       = np.array([s["c_k"] for s in self._specs])

        for i, spec in enumerate(self._specs):
            mode = spec["sigma_ck_mode"]
            if mode == "sd":
                sigma_c[i] = spec["sigma_ck_value"]
            elif mode == "pedigree":
                sigma_c[i] = spec["sigma_ck_value"]  # pre-computed at load time

        return ConstraintBlock(
            G_rows   = G,
            c        = c,
            sigma_c  = sigma_c,
            row_meta = pd.DataFrame(meta_list),
        )

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock:
        # Only needed for fraction_of_flow mode — update sigma_c using current f_agg
        has_fraction = any(s["sigma_ck_mode"] == "fraction_of_flow" for s in self._specs)
        if not has_fraction:
            return _empty_block(acc_struct.n_edges)

        static = self.build_static(acc_struct)
        sigma_c = static.sigma_c.copy()

        for i, spec in enumerate(self._specs):
            if spec["sigma_ck_mode"] == "fraction_of_flow":
                sup_ids = spec.get("_supply_ids", spec["edge_ids"])
                total   = float(np.nansum(np.abs(f_agg[sup_ids]))) if len(sup_ids) else 0.0
                sigma_c[i] = spec["sigma_ck_fraction"] * total

        return ConstraintBlock(
            G_rows   = static.G_rows,
            c        = static.c,
            sigma_c  = sigma_c,
            row_meta = static.row_meta,
        )


# =============================================================================
# RatioConstraintBuilder
# =============================================================================

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
    def from_csv(cls, path: str, acc_struct) -> "RatioConstraintBuilder":
        """
        Load from params CSV (same schema as ParametricImputer.from_csv).
        Resolves A→B edge pairs and computes sigma_c from unc_type column.
        """
        df = pd.read_csv(path, dtype=str).fillna("*")
        ei_exp      = _expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        edge_lookup = _build_edge_lookup(ei_exp)
        field_vals  = {
            col: sorted(str(v) for v in ei_exp[col].unique())
            for col in [c for _, c, _ in _A_FIELDS]
        }

        pairs = []
        for _, row in df.iterrows():
            nominal = _float_or(row.get("value_nominal", "nan"), np.nan)
            if np.isnan(nominal):
                continue
            try:
                sigma = _sigma_from_row(row, nominal)
            except Exception:
                sigma = 0.0

            param_id = str(row.get("param_id", "")).strip() or None

            for a_tuple in _enumerate_a_edges(row, field_vals):
                b_ids = _resolve_b_ids(row, a_tuple, edge_lookup, field_vals)
                a_eid = edge_lookup.get(a_tuple)
                if a_eid is None:
                    continue
                for b_eid in b_ids:
                    pairs.append({
                        "constraint_id": param_id or f"ratio_{a_eid}_{b_eid}",
                        "edge_i":  b_eid,
                        "edge_j":  a_eid,
                        "r":       nominal,
                        "sigma_c": sigma,
                    })

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
            meta_list.append({
                "constraint_id": p["constraint_id"],
                "type":          "ratio_eq",
                "is_hard":       False,
                "cone_type":     "free",
            })

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(M, n_edges))
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
    def from_csv(cls, path: str, acc_struct) -> "RatioBoundBuilder":
        df = pd.read_csv(path, dtype=str).fillna("*")
        ei_exp      = _expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        edge_lookup = _build_edge_lookup(ei_exp)
        field_vals  = {
            col: sorted(str(v) for v in ei_exp[col].unique())
            for col in [c for _, c, _ in _A_FIELDS]
        }

        bound_rows = []
        for _, row in df.iterrows():
            nominal = _float_or(row.get("value_nominal", "nan"), np.nan)
            bl      = _float_or(row.get("bound_lower", "nan"), np.nan)
            bu      = _float_or(row.get("bound_upper", "nan"), np.nan)
            if np.isnan(bl) and np.isnan(bu):
                continue

            param_id = str(row.get("param_id", "")).strip() or None

            for a_tuple in _enumerate_a_edges(row, field_vals):
                b_ids = _resolve_b_ids(row, a_tuple, edge_lookup, field_vals)
                a_eid = edge_lookup.get(a_tuple)
                if a_eid is None:
                    continue
                for b_eid in b_ids:
                    bound_rows.append({
                        "param_id": param_id or f"bound_{a_eid}_{b_eid}",
                        "edge_i":   b_eid,
                        "edge_j":   a_eid,
                        "nominal":  nominal if not np.isnan(nominal) else 1.0,
                        "bl":       bl,
                        "bu":       bu,
                    })

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

        for br in self._bound_rows:
            ei, ej = br["edge_i"], br["edge_j"]
            bu, bl = br["bu"], br["bl"]

            if not np.isnan(bu):
                rows_list += [row_id, row_id]
                cols_list += [ei, ej]
                vals_list += [1.0, -bu]
                meta_list.append({
                    "constraint_id": f"{br['param_id']}_upper",
                    "type":          "ratio_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
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
                })
                row_id += 1

        if row_id == 0:
            return _empty_block(n_edges)

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(row_id, n_edges))
        return ConstraintBlock(
            G_rows   = G,
            c        = np.zeros(row_id),
            sigma_c  = np.zeros(row_id),
            row_meta = pd.DataFrame(meta_list),
        )


# =============================================================================
# FlowBoundsBuilder
# =============================================================================

class FlowBoundsBuilder(BaseConstraintBuilder):
    """
    Hard non-neg-cone flow bound constraints from acc_struct.bounds.

    For each edge with non-default bounds:
      Upper: a_i ≤ u_i  →  G row: +1 at edge_i,  c = u_i
      Lower: a_i ≥ l_i  →  G row: −1 at edge_i,  c = −l_i

    Produces u and l arrays (passed through to ConstraintRun).
    """

    def build_static(self, acc_struct) -> ConstraintBlock:
        bounds  = acc_struct.bounds
        l, u    = bounds.l, bounds.u
        n_edges = acc_struct.n_edges

        default_l = bounds.DEFAULT_LOWER
        default_u = bounds.DEFAULT_UPPER

        non_default = (l != default_l) | (u != default_u)
        edge_ids    = np.where(non_default)[0]

        if len(edge_ids) == 0:
            return _empty_block(n_edges)

        rows_list, cols_list, vals_list = [], [], []
        c_list, meta_list = [], []
        row_id = 0

        for eid in edge_ids:
            if u[eid] != default_u:
                rows_list.append(row_id); cols_list.append(eid); vals_list.append(1.0)
                c_list.append(u[eid])
                meta_list.append({
                    "constraint_id": f"flow_bound_{eid}_upper",
                    "type":          "flow_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
                })
                row_id += 1

            if l[eid] != default_l:
                rows_list.append(row_id); cols_list.append(eid); vals_list.append(-1.0)
                c_list.append(-l[eid])
                meta_list.append({
                    "constraint_id": f"flow_bound_{eid}_lower",
                    "type":          "flow_bound",
                    "is_hard":       True,
                    "cone_type":     "nonneg",
                })
                row_id += 1

        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(row_id, n_edges))
        return ConstraintBlock(
            G_rows   = G,
            c        = np.array(c_list),
            sigma_c  = np.zeros(row_id),
            row_meta = pd.DataFrame(meta_list),
        )


# =============================================================================
# ConstraintRun  — output of ConstraintEngine.run()
# =============================================================================

@dataclass
class ConstraintRun:
    """
    Full output of one ConstraintEngine.run() call.

    G          : sparse (M × E), time-invariant
    row_meta   : DataFrame — one row per G row (constraint_id, type, is_hard, cone_type)
    a0         : (E,) — f_agg at the first timestep (t=0 prior for optimiser)
    sigma_a    : (E,) — sigma_agg
    u          : (E,) — flow upper bounds
    l          : (E,) — flow lower bounds
    timestep_data : dict mapping timestep label → {"c": ..., "sigma_c": ...}
    run_id     : unique identifier
    acc_id     : AccountingStructure id
    agg_id     : AggregationResult id (if available)
    created_at : ISO timestamp
    """
    G:              sp.csr_matrix
    row_meta:       pd.DataFrame
    a0:             np.ndarray
    sigma_a:        np.ndarray
    u:              np.ndarray
    l:              np.ndarray
    timestep_data:  dict
    run_id:         str
    acc_id:         str
    agg_id:         str
    created_at:     str


# =============================================================================
# ConstraintRunRegistry
# =============================================================================

class ConstraintRunRegistry:
    """
    Lightweight JSON registry for ConstraintRun objects.

    Mirrors DataFeedRegistry: save() writes files, load() reloads, summary_df() lists runs.

    Directory layout:
        {storage_dir}/
        ├── registry.json
        └── {run_id}/
            ├── G.npz
            ├── row_meta.parquet
            ├── a0.npy
            ├── sigma_a.npy
            ├── u.npy
            ├── l.npy
            └── timesteps/
                └── t_{label}.npz
    """

    def __init__(self, storage_dir: str = "data/proc/const"):
        self.storage_dir = storage_dir
        self._index: dict[str, dict] = {}
        self._index_path = os.path.join(storage_dir, "registry.json")
        os.makedirs(storage_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        if os.path.exists(self._index_path):
            with open(self._index_path) as f:
                self._index = json.load(f)

    def _save_index(self):
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def register(self, run: ConstraintRun) -> None:
        run_dir = os.path.join(self.storage_dir, run.run_id)
        os.makedirs(run_dir, exist_ok=True)
        ts_dir  = os.path.join(run_dir, "timesteps")
        os.makedirs(ts_dir, exist_ok=True)

        sp.save_npz(os.path.join(run_dir, "G.npz"), run.G)
        run.row_meta.to_parquet(os.path.join(run_dir, "row_meta.parquet"), index=False)
        np.save(os.path.join(run_dir, "a0.npy"),      run.a0)
        np.save(os.path.join(run_dir, "sigma_a.npy"), run.sigma_a)
        np.save(os.path.join(run_dir, "u.npy"),       run.u)
        np.save(os.path.join(run_dir, "l.npy"),       run.l)

        for label, td in run.timestep_data.items():
            np.savez(
                os.path.join(ts_dir, f"t_{label}.npz"),
                c=td["c"], sigma_c=td["sigma_c"],
            )

        self._index[run.run_id] = {
            "acc_id":     run.acc_id,
            "agg_id":     run.agg_id,
            "created_at": run.created_at,
            "n_rows":     run.G.shape[0],
            "n_edges":    run.G.shape[1],
            "timesteps":  list(run.timestep_data.keys()),
        }
        self._save_index()
        _log.info("ConstraintRun '%s' registered (%d G rows)", run.run_id, run.G.shape[0])

    def load(self, run_id: str) -> ConstraintRun:
        if run_id not in self._index:
            raise KeyError(f"Run '{run_id}' not in registry")
        run_dir = os.path.join(self.storage_dir, run_id)
        ts_dir  = os.path.join(run_dir, "timesteps")

        G        = sp.load_npz(os.path.join(run_dir, "G.npz"))
        row_meta = pd.read_parquet(os.path.join(run_dir, "row_meta.parquet"))
        a0       = np.load(os.path.join(run_dir, "a0.npy"))
        sigma_a  = np.load(os.path.join(run_dir, "sigma_a.npy"))
        u        = np.load(os.path.join(run_dir, "u.npy"))
        l        = np.load(os.path.join(run_dir, "l.npy"))

        meta    = self._index[run_id]
        ts_data = {}
        for label in meta["timesteps"]:
            npz = np.load(os.path.join(ts_dir, f"t_{label}.npz"))
            ts_data[label] = {"c": npz["c"], "sigma_c": npz["sigma_c"]}

        return ConstraintRun(
            G             = G,
            row_meta      = row_meta,
            a0            = a0,
            sigma_a       = sigma_a,
            u             = u,
            l             = l,
            timestep_data = ts_data,
            run_id        = run_id,
            acc_id        = meta["acc_id"],
            agg_id        = meta["agg_id"],
            created_at    = meta["created_at"],
        )

    def summary_df(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self._index, orient="index").reset_index(
            names="run_id"
        )


# =============================================================================
# ConstraintEngine
# =============================================================================

class ConstraintEngine:
    """
    Orchestrates an ordered list of BaseConstraintBuilder instances.

    Usage::

        engine = ConstraintEngine(acc_struct)
        engine.register(MassBalanceBuilder())
        engine.register(SumConstraintBuilder.from_acc_struct(acc_struct))
        engine.register(RatioConstraintBuilder.from_csv("params.csv", acc_struct))
        engine.register(RatioBoundBuilder.from_csv("params.csv", acc_struct))
        engine.register(FlowBoundsBuilder())

        run = engine.run(agg_result, timesteps=["2010", "2011", ...])

    run.G is built once from all static blocks.
    run.timestep_data[t] = {"c": ..., "sigma_c": ...} is assembled per timestep
    by overlaying dynamic blocks onto the static c / sigma_c.
    """

    def __init__(self, acc_struct, storage_dir: str = "data/proc/const"):
        self.acc_struct  = acc_struct
        self._builders:  list[BaseConstraintBuilder] = []
        self._registry   = ConstraintRunRegistry(storage_dir)

    def register(self, builder: BaseConstraintBuilder) -> None:
        self._builders.append(builder)

    def run(
        self,
        agg_result,
        timesteps: list,
        agg_id:    str = "",
        run_id:    Optional[str] = None,
        persist:   bool = True,
    ) -> ConstraintRun:
        """
        Build G once, then assemble c(t) / sigma_c(t) for each timestep.

        Parameters
        ----------
        agg_result  : AggregationResult with f_agg and sigma_agg
        timesteps   : ordered list of time labels (strings or ints)
        agg_id      : provenance identifier for the agg_result
        run_id      : explicit run ID; auto-generated if None
        persist     : if True, save to registry
        """
        acc        = self.acc_struct
        n_edges    = acc.n_edges
        f_agg      = agg_result.f_agg
        sigma_agg  = agg_result.sigma_agg

        # ── 1. Build static G (time-invariant) ────────────────────────
        static_blocks = [b.build_static(acc) for b in self._builders]
        static_full   = _concat_blocks(static_blocks, n_edges)
        G_static      = static_full.G_rows
        c_static      = static_full.c.copy()
        sc_static     = static_full.sigma_c.copy()
        row_meta      = static_full.row_meta.copy()
        M             = G_static.shape[0]

        _log.info("ConstraintEngine: G shape (%d, %d)", M, n_edges)

        # ── 2. Per-timestep c and sigma_c ─────────────────────────────
        timestep_data: dict[str, dict] = {}
        for t in timesteps:
            t_label = str(t)
            # Identify rows belonging to this timestep from dynamic blocks
            c_t      = c_static.copy()
            sigma_c_t = sc_static.copy()

            # Track which static rows each dynamic block covers
            row_offset = 0
            for b, sb in zip(self._builders, static_blocks):
                dyn = b.build_dynamic(acc, f_agg, sigma_agg, t_label)
                bM  = sb.G_rows.shape[0]
                if dyn.G_rows.shape[0] == bM and bM > 0:
                    c_t[row_offset: row_offset + bM]       = dyn.c
                    sigma_c_t[row_offset: row_offset + bM] = dyn.sigma_c
                row_offset += bM

            timestep_data[t_label] = {"c": c_t, "sigma_c": sigma_c_t}

        # ── 3. Assemble ConstraintRun ──────────────────────────────────
        if run_id is None:
            run_id = (
                f"const_{acc.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                f"_{uuid.uuid4().hex[:8]}"
            )

        cr = ConstraintRun(
            G             = G_static,
            row_meta      = row_meta,
            a0            = f_agg.copy(),
            sigma_a       = sigma_agg.copy(),
            u             = acc.bounds.u.copy(),
            l             = acc.bounds.l.copy(),
            timestep_data = timestep_data,
            run_id        = run_id,
            acc_id        = acc.id,
            agg_id        = agg_id,
            created_at    = datetime.utcnow().isoformat(),
        )

        if persist:
            self._registry.register(cr)

        return cr
