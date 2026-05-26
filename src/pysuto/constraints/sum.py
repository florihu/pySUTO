"""
sum.py
======
``SumConstraintBuilder`` — Type-2 soft-sum rows: Σ values across a CSV-defined
edge group equals a per-timestep target.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.logging import get_logger
from pysuto.param_relation import WILDCARD, ParamRelationResolver

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block

_log = get_logger(__name__)


# σ_c sentinel used to deactivate a soft sum-constraint row in timesteps
# whose year does not match the row's spec time. Effective when squared in
# the objective: (1/σ_c)² = 1e-20, four orders below any realistic floor,
# so the row's slack ε is essentially unpenalised → constraint inactive.
SIGMA_C_INACTIVE: float = 1e10


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

    def __init__(self, specs: list[dict], sigma_fraction: float = 0.0, p: float = 1.3):
        """
        specs : list of dicts, each with keys:
            constraint_id, description,
            edge_ids: list[int], coefficients: list[float],
            c_k: float, sigma_ck_mode: str,
            sigma_ck_value: float, sigma_ck_fraction: float
        """
        self._specs = specs
        self._sigma_fraction = sigma_fraction  # fallback
        self._p = p

    @staticmethod
    def _resolve_scope_mask(acc_struct, mask_name: Optional[str]) -> np.ndarray:
        if mask_name is None:
            return np.ones(acc_struct.n_edges, dtype=bool)
        if mask_name not in acc_struct.masks:
            raise ValueError(
                f"SumConstraintBuilder(mask={mask_name!r}): mask not in "
                f"acc_struct.masks. Available: {sorted(acc_struct.masks)}"
            )
        return acc_struct.masks[mask_name]

    # ------------------------------------------------------------------
    # Factory: auto-derive from acc_struct topology
    # ------------------------------------------------------------------

    @classmethod
    def from_acc_struct(
        cls,
        acc_struct,
        sigma_fraction: float = 0.05,
        entity_filter: Optional[set] = None,
        mask: Optional[str] = None,
    ) -> "SumConstraintBuilder":
        """
        Auto-generate one soft sum constraint per (Flow|Process) balance node
        per timestep:  Σ useful_supply_i = Σ use_inputs_i  (excluding SB edges).

        entity_filter : set of entity strings to include (default: {Flow, Process})
        sigma_fraction: σ_c = sigma_fraction × |sum of supply flow magnitudes|
        mask          : if set, only edges in ``acc_struct.masks[mask]`` are
                        considered. Nodes with no remaining supply or no
                        remaining use after filtering are skipped.
        """
        if entity_filter is None:
            entity_filter = {"Flow", "Process"}

        ei       = acc_struct.edge_index
        ni       = acc_struct.node_index
        sf_mask  = acc_struct.masks["supply"]
        uf_mask  = acc_struct.masks["use"]
        sb_mask  = acc_struct.masks["sb"]
        t_arr    = ei["time"].values
        v_arr    = ei["valuation"].values
        times    = acc_struct.time_range
        valuations = list(acc_struct.valuation)
        scope    = cls._resolve_scope_mask(acc_struct, mask)

        specs = []
        for time in times:
            t_mask_base = (t_arr == str(time)) & scope
            for valuation in valuations:
                t_mask = t_mask_base & (v_arr == valuation)

                # Anchor each row on supply edges of this valuation. Splitting
                # by valuation keeps every emitted row unit-coherent.
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
                    n_row = ni.loc[nid]

                    specs.append({
                        "constraint_id":   f"useful_supply_node{nid}_t{time}_{valuation}",
                        "description":     f"Useful supply balance node {nid} t={time} val={valuation}",
                        "edge_ids":        all_ids,
                        "coefficients":    all_coeff,
                        "c_k":             0.0,
                        "sigma_ck_mode":   "fraction_of_flow",
                        "sigma_ck_value":  0.0,
                        "sigma_ck_fraction": sigma_fraction,
                        "_supply_ids":     sup_ids,
                        "time":            str(time),
                        "valuation":       valuation,
                        "sector":          str(n_row["sector"]),
                        "region":          str(n_row["region"]),
                        "entity":          str(n_row["entity"]),
                        "entity_sub":      str(n_row.get("entity_sub", "")) if "entity_sub" in ni.columns else None,
                    })

        return cls(specs, sigma_fraction=sigma_fraction)

    # ------------------------------------------------------------------
    # Factory: entity/entity_sub-routed sum constraints
    # ------------------------------------------------------------------

    @classmethod
    def from_acc_struct_subentity(
        cls,
        acc_struct,
        entity_filter: set,
        *,
        rhs_in_entity: set = frozenset(),
        rhs_out_entity: set = frozenset(),
        rhs_out_entity_sub: set = frozenset(),
        lhs_neg_out_entity_sub: set = frozenset(),
        sigma_mode: str = "p_mean",
        sigma_fraction: float = 0.05,
        p: float = 1.3,
        mask: Optional[str] = None,
    ) -> "SumConstraintBuilder":
        """
        Generate one soft sum constraint per (node, time) pair for nodes whose
        entity is in ``entity_filter``.

        Each incident edge is assigned a role based on the entity / entity_sub of
        the *other* node.  Unmatched edges default to LHS +1.

        Parameters
        ----------
        rhs_in_entity        : incoming edge other-node entity  → RHS (_rhs_ids)
        rhs_out_entity       : outgoing edge other-node entity  → RHS (_rhs_ids)
        rhs_out_entity_sub   : outgoing edge other-node entity_sub → RHS (_rhs_ids)
        lhs_neg_out_entity_sub : outgoing edge other-node entity_sub → LHS −1
        sigma_mode           : "p_mean" (Lp / power-mean propagation over
                               _rhs_ids sigma_agg with exponent p)
                               or "fraction_of_flow"
        sigma_fraction       : used only when sigma_mode="fraction_of_flow"
        """
        ei    = acc_struct.edge_index
        ni    = acc_struct.node_index
        t_arr = ei["time"].values
        v_arr = ei["valuation"].values
        times = acc_struct.time_range
        valuations = list(acc_struct.valuation)
        scope = cls._resolve_scope_mask(acc_struct, mask)

        # Pre-build lookup arrays for the other-node's entity and entity_sub
        from_entity     = ni.loc[ei["from_id"].values, "entity"].values
        from_entity_sub = ni.loc[ei["from_id"].values, "entity_sub"].values
        to_entity       = ni.loc[ei["to_id"].values,   "entity"].values
        to_entity_sub   = ni.loc[ei["to_id"].values,   "entity_sub"].values
        eids            = ei["edge_id"].values
        from_ids        = ei["from_id"].values
        to_ids          = ei["to_id"].values

        specs = []
        for time in times:
            t_mask_base = (t_arr == str(time)) & scope
            for valuation in valuations:
                t_mask = t_mask_base & (v_arr == valuation)

                # Nodes of the right entity type that appear at (time, valuation)
                node_ids = sorted(set(
                    ni.index[ni["entity"].isin(entity_filter)].tolist()
                ) & (
                    set(from_ids[t_mask]) | set(to_ids[t_mask])
                ))

                for nid in node_ids:
                    lhs_pos, lhs_neg, rhs_ids = [], [], []

                    in_mask = t_mask & (to_ids == nid)
                    for eid, other_entity, other_sub in zip(
                        eids[in_mask], from_entity[in_mask], from_entity_sub[in_mask]
                    ):
                        if other_entity in rhs_in_entity:
                            rhs_ids.append(eid)
                        else:
                            lhs_pos.append(eid)

                    out_mask = t_mask & (from_ids == nid)
                    for eid, other_entity, other_sub in zip(
                        eids[out_mask], to_entity[out_mask], to_entity_sub[out_mask]
                    ):
                        other_sub_str = str(other_sub) if other_sub is not None and str(other_sub) != "nan" else None
                        if other_entity in rhs_out_entity or other_sub_str in rhs_out_entity_sub:
                            rhs_ids.append(eid)
                        elif other_sub_str in lhs_neg_out_entity_sub:
                            lhs_neg.append(eid)
                        else:
                            lhs_pos.append(eid)

                    lhs_edges = lhs_pos + lhs_neg
                    if not lhs_edges or not rhs_ids:
                        continue

                    all_ids   = lhs_pos + lhs_neg
                    all_coeff = [1.0] * len(lhs_pos) + [-1.0] * len(lhs_neg)
                    n_row = ni.loc[nid]

                    spec: dict = {
                        "constraint_id":   f"sub_entity_node{nid}_t{time}_{valuation}",
                        "description":     f"Subentity sum node {nid} t={time} val={valuation}",
                        "edge_ids":        all_ids,
                        "coefficients":    all_coeff,
                        "c_k":             0.0,
                        "sigma_ck_mode":   sigma_mode,
                        "sigma_ck_value":  0.0,
                        "sigma_ck_fraction": sigma_fraction,
                        "_rhs_ids":        rhs_ids,
                        "time":            str(time),
                        "valuation":       valuation,
                        "sector":          str(n_row["sector"]),
                        "region":          str(n_row["region"]),
                        "entity":          str(n_row["entity"]),
                        "entity_sub":      str(n_row.get("entity_sub", "")) if "entity_sub" in ni.columns else None,
                    }
                    if sigma_mode == "fraction_of_flow":
                        spec["_supply_ids"] = rhs_ids
                    specs.append(spec)

        _log.info(
            "SumConstraintBuilder.from_acc_struct_subentity: %d specs (entity_filter=%s)",
            len(specs), sorted(entity_filter),
        )
        return cls(specs, sigma_fraction=sigma_fraction, p=p)

    # ------------------------------------------------------------------
    # Factory: load from CSV
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        mask: Optional[str] = None,
    ) -> "SumConstraintBuilder":
        """
        Load user-defined sum constraints from a long-format CSV.

        ``mask`` (optional): name of a mask in ``acc_struct.masks``. Resolved
        edges that fall outside the mask are dropped from each spec. A
        constraint whose remaining edges are empty is skipped with a warning.

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
        resolver = ParamRelationResolver(acc_struct)
        scope = cls._resolve_scope_mask(acc_struct, mask)

        # Map CSV selector columns to the 8 A-field keys the resolver expects.
        _col_map = {
            "sector_from": "a_sector_from", "entity_from": "a_entity_from",
            "region_from": "a_region_from", "sector_to":   "a_sector_to",
            "entity_to":   "a_entity_to",   "region_to":   "a_region_to",
            "time":        "a_time",         "valuation":   "a_valuation",
        }

        edge_val = acc_struct.edge_index["valuation"].values

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
                # Build a pseudo-row with "a_*" keys for the resolver.
                a_row = pd.Series({
                    dst: str(row.get(src, "*")).strip()
                    for src, dst in _col_map.items()
                })
                coeff = float(row["coefficient"])

                for eid in resolver.match_a_edges(a_row):
                    if not scope[eid]:
                        continue
                    all_ids.append(eid)
                    all_coeff.append(coeff)

            if not all_ids:
                _log.warning("SumConstraintBuilder: constraint '%s' matched no edges", cid)
                continue

            # Resolve spec time: if every CSV row in the group names the same
            # single year (no wildcards), the constraint is year-locked to
            # that year. Otherwise (mixed years, or any wildcard) it's
            # treated as time-agnostic and stays active across all timesteps.
            time_vals = (
                grp["time"].astype(str).str.strip().unique().tolist()
                if "time" in grp.columns else []
            )
            if len(time_vals) == 1 and time_vals[0] not in WILDCARD:
                spec_time = time_vals[0]
            else:
                spec_time = None

            # Per-valuation split: if a CSV constraint resolves to edges
            # spanning multiple valuations, emit one sub-spec per valuation.
            # Keeps every row in G unit-coherent (hard rule: no Mass+CC mixing).
            ids_arr   = np.asarray(all_ids, dtype=int)
            coeff_arr = np.asarray(all_coeff, dtype=float)
            row_vals  = edge_val[ids_arr]
            unique_vals = sorted(set(row_vals.tolist()))

            if len(unique_vals) > 1:
                _log.warning(
                    "SumConstraintBuilder: constraint '%s' resolves to %d valuations %s — "
                    "auto-splitting into one row per valuation",
                    cid, len(unique_vals), unique_vals,
                )

            for val in unique_vals:
                mask_v = (row_vals == val)
                sub_ids   = ids_arr[mask_v].tolist()
                sub_coeff = coeff_arr[mask_v].tolist()
                sub_cid   = f"{cid}_{val}" if len(unique_vals) > 1 else str(cid)
                specs.append({
                    "constraint_id":     sub_cid,
                    "description":       desc,
                    "edge_ids":          sub_ids,
                    "coefficients":      sub_coeff,
                    "c_k":               c_k,
                    "sigma_ck_mode":     mode,
                    "sigma_ck_value":    sv,
                    "sigma_ck_fraction": sf,
                    "_supply_ids":       [e for e, c in zip(sub_ids, sub_coeff) if c > 0],
                    "time":              spec_time,
                    "valuation":         val,
                    "sector":            None,
                    "region":            None,
                    "entity":            None,
                    "entity_sub":        None,
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
                "time":          spec.get("time"),
                "valuation":     spec.get("valuation"),
                "sector":        spec.get("sector"),
                "region":        spec.get("region"),
                "entity":        spec.get("entity"),
                "entity_sub":    spec.get("entity_sub"),
            })

        M = len(self._specs)
        G = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(M, n_edges))
        _log.info("SumConstraintBuilder.build_static: %d rows", M)

        # For non-dynamic sigma modes, fill sigma_c here
        sigma_c = np.zeros(M)
        c       = np.array([s["c_k"] for s in self._specs])

        for i, spec in enumerate(self._specs):
            mode = spec["sigma_ck_mode"]
            if mode == "sd":
                sigma_c[i] = spec["sigma_ck_value"]
            elif mode == "pedigree":
                sigma_c[i] = spec["sigma_ck_value"]  # pre-computed at load time
            # "fraction_of_flow" and "p_mean" are both dynamic — leave as 0.0 placeholder

        block = ConstraintBlock(
            G_rows   = G,
            c        = c,
            sigma_c  = sigma_c,
            row_meta = pd.DataFrame(meta_list),
        )
        self._cached_static = block
        return block

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock:
        """
        Assemble (c, σ_c) for this builder at timestep ``t``.

        Rows whose spec has ``time != t`` are *deactivated* for this
        timestep: ``c = 0`` and ``σ_c = SIGMA_C_INACTIVE`` so the row's
        slack ε is essentially unpenalised in the objective. Rows with
        ``spec["time"] is None`` (time-agnostic, e.g. hand-written CSV
        constraints with wildcard time) stay active for every timestep.

        This is what turns the per-timestep loop in
        ``ConstraintEngine.run`` into real per-year solves — without it,
        ``c(t)`` and ``σ_c(t)`` would be identical for every t.
        """
        has_fraction = any(s["sigma_ck_mode"] == "fraction_of_flow" for s in self._specs)
        has_p_mean   = any(s["sigma_ck_mode"] == "p_mean" for s in self._specs)
        any_per_year = any(s.get("time") is not None for s in self._specs)
        if not (has_fraction or has_p_mean or any_per_year):
            return _empty_block(acc_struct.n_edges)

        static  = getattr(self, "_cached_static", None) or self.build_static(acc_struct)
        c_out   = static.c.copy()
        sigma_c = static.sigma_c.copy()
        t_str   = str(t)

        for i, spec in enumerate(self._specs):
            spec_time = spec.get("time")
            active = (spec_time is None) or (str(spec_time) == t_str)

            if not active:
                c_out[i]   = 0.0
                sigma_c[i] = SIGMA_C_INACTIVE
                continue

            mode = spec["sigma_ck_mode"]
            if mode == "fraction_of_flow":
                sup_ids = spec.get("_supply_ids", spec["edge_ids"])
                total   = float(np.nansum(np.abs(f_agg[sup_ids]))) if len(sup_ids) else 0.0
                sigma_c[i] = spec["sigma_ck_fraction"] * total
            elif mode == "p_mean":
                rhs = spec.get("_rhs_ids", [])
                if rhs:
                    c_out[i]   = float(np.nansum(f_agg[rhs]))
                    sigma_c[i] = float(np.nansum(sigma_agg[rhs] ** self._p) ** (1.0 / self._p))

        return ConstraintBlock(
            G_rows   = static.G_rows,
            c        = c_out,
            sigma_c  = sigma_c,
            row_meta = static.row_meta,
        )


# =============================================================================
# RatioConstraintBuilder
# =============================================================================

