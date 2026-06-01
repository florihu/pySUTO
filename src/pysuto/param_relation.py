"""
param_relation.py
=================
CSV-driven A→B parametric edge-relation resolver.

Owns the params-template CSV column schema (A/B selector fields,
wildcards, "same" directive), the edge-index expansion + cache, the
A/B enumeration, and the σ-from-row recipe.

Consumed as a shared seam by Stage 2 (`ParametricImputer.from_csv`)
and Stage 3 (`RatioConstraintBuilder.from_csv`,
`RatioBoundBuilder.from_csv`). Each adapter iterates a CSV and calls
`ParamRelationResolver.resolve_row(row)`; the resolver yields a
`ResolvedRelation` per matched `(a_edge_id, b_edge_id)` pair without
deriving σ, deduping, or applying any skip policy. Adapters apply
their own row-projection and policy.

Public API
----------
WILDCARD, A_FIELDS, B_FIELDS, A_PRODUCT_THRESHOLD     Schema constants
expand_edge_index(edge_index, node_index)             Decode sector/entity/region
get_edge_cache(acc_struct)                            Per-acc_struct edge cache
sigma_from_row(row, nominal)                          σ via unc_type ∈ {sd, min_max, pedigree}
float_or(val, default)                                Lenient float parse
specificity(row)                                      Count non-wildcard A-fields
ResolvedRelation                                      Resolver output dataclass
ParamRelationResolver(acc_struct)                     Resolver entry point
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .unc import PedigreeScores


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

WILDCARD = {"*", "nan", "None", ""}

# A-selector field → (ei_exp column, counterpart column for "same" constraint)
#
# Under the structural model, edge matching ignores time (admissibility is
# time-independent and edge_index has no `time` column). The `a_time`/`b_time`
# CSV columns survive on the resolved row and are read by consumers as a
# *period selector* (which timestep the value applies to: wildcard → broadcast
# to all timesteps, specific year → that period only).
A_FIELDS = [
    ("a_sector_from", "sector_from", "sector_to"),
    ("a_entity_from", "entity_from", "entity_to"),
    ("a_region_from", "region_from", "region_to"),
    ("a_sector_to",   "sector_to",   "sector_from"),
    ("a_entity_to",   "entity_to",   "entity_from"),
    ("a_region_to",   "region_to",   "region_from"),
    ("a_valuation",   "valuation",   None),
]

# B-selector field → (ei_exp column, corresponding A-edge column for "same")
B_FIELDS = [
    ("b_sector_from", "sector_from", "sector_from"),
    ("b_entity_from", "entity_from", "entity_from"),
    ("b_region_from", "region_from", "region_from"),
    ("b_sector_to",   "sector_to",   "sector_to"),
    ("b_entity_to",   "entity_to",   "entity_to"),
    ("b_region_to",   "region_to",   "region_to"),
    ("b_valuation",   "valuation",   "valuation"),
]

# Above which estimated A-product size to switch from enumerate-with-dict-lookup
# to a numpy bitwise-mask strategy (faster for very wildcard-heavy selectors).
A_PRODUCT_THRESHOLD = 100_000

_A_SPEC_KEYS = [k for k, _, _ in A_FIELDS]
_A_EI_POS    = {ec: i for i, (_, ec, _) in enumerate(A_FIELDS)}

_MASK_COLS = [ec for _, ec, _ in A_FIELDS]  # ei_exp columns used by A-selector matching


# ---------------------------------------------------------------------------
# Schema-level helpers (public)
# ---------------------------------------------------------------------------

def float_or(val, default: float) -> float:
    """Lenient float parse: returns `default` for blank / 'nan' / 'None' / non-numeric."""
    s = str(val).strip()
    return float(s) if s and s not in ("nan", "None", "") else default


def sigma_from_row(row: pd.Series, nominal: float) -> float:
    """
    Derive σ from a params-CSV row according to its `unc_type` column.

    Modes:
      "sd"       — σ = value_sd
      "min_max"  — σ = (value_max − value_min) / 4
      "pedigree" — σ = nominal · PedigreeScores(...).total_cv()  (Laner et al. 2016)
    """
    unc = str(row.get("unc_type", "")).strip()
    if unc == "sd":
        return float(row["value_sd"])
    if unc == "min_max":
        return (float(row["value_max"]) - float(row["value_min"])) / 4.0
    if unc == "pedigree":
        ps = PedigreeScores(
            reliability              = int(row["pedigree_reliability"]),
            completeness             = int(row["pedigree_completeness"]),
            temporal_correlation     = int(row["pedigree_temporal_correlation"]),
            geographical_correlation = int(row["pedigree_geographical_correlation"]),
            other_correlation        = int(row["pedigree_other_correlation"]),
        )
        return ps.total_cv() * nominal
    raise ValueError(f"Unknown unc_type: {unc!r}")


def specificity(row: pd.Series) -> int:
    """Count A-selector fields that are not wildcards (higher = more specific)."""
    return sum(
        str(row.get(k, "*")).strip() not in WILDCARD
        for k in _A_SPEC_KEYS
    )


def resolve_period_selector(
    val: str,
    time_range: list,
    counterpart_idx: Optional[list[int]] = None,
) -> list[int]:
    """
    Interpret a CSV ``a_time`` / ``b_time`` cell as a *period selector* and
    return the list of ``t_idx`` (positions into ``time_range``) the row
    applies to.

    Rules
    -----
    * Wildcard (``*``, ``nan``, ``None``, ``""``) → every period.
    * Literal year (e.g. ``"2020"``) → that period if present in
      ``time_range``; empty list otherwise.
    * ``"same"`` (B-side only) → mirror ``counterpart_idx`` exactly. Pass
      the A-side resolved indices in via ``counterpart_idx``; raises if
      the directive is used without a counterpart.
    """
    s = str(val).strip()
    if s in WILDCARD:
        return list(range(len(time_range)))
    if s == "same":
        if counterpart_idx is None:
            raise ValueError("'same' period selector requires counterpart_idx")
        return list(counterpart_idx)
    tr_str = [str(t) for t in time_range]
    try:
        return [tr_str.index(s)]
    except ValueError:
        return []


# ---------------------------------------------------------------------------
# Edge-index expansion + per-acc_struct cache
# ---------------------------------------------------------------------------

def expand_edge_index(edge_index: pd.DataFrame, node_index: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of edge_index with decoded sector/entity/region columns added."""
    ei = edge_index.copy().reset_index(drop=True)
    from_vals = node_index.loc[ei["from_id"].values]
    to_vals   = node_index.loc[ei["to_id"].values]
    ei["sector_from"] = from_vals["sector"].values
    ei["entity_from"] = from_vals["entity"].values
    ei["region_from"] = from_vals["region"].values
    ei["sector_to"]   = to_vals["sector"].values
    ei["entity_to"]   = to_vals["entity"].values
    ei["region_to"]   = to_vals["region"].values
    return ei


def _build_edge_lookup(ei_exp: pd.DataFrame) -> dict:
    """O(1) lookup: (sfrom,efrom,rfrom,sto,eto,rto,val) → edge_id (structural)."""
    cols = ["sector_from", "entity_from", "region_from",
            "sector_to",   "entity_to",   "region_to", "valuation"]
    arr  = ei_exp[cols].to_numpy(dtype=str)
    eids = ei_exp["edge_id"].to_numpy(dtype=int)
    return {tuple(arr[i]): int(eids[i]) for i in range(len(arr))}


def _build_mask_cache(ei_exp: pd.DataFrame) -> dict:
    """
    Pre-compute one bool ndarray per (column, value) pair and per
    "same-field" constraint, so mask application is O(n/64) bitwise AND
    instead of O(n) string comparison per call.
    """
    cache: dict = {}
    arrays: dict = {}
    for col in _MASK_COLS:
        arr = ei_exp[col].to_numpy(dtype=str)
        arrays[col] = arr
        for val in np.unique(arr):
            cache[(col, val)] = arr == val
    # "same" masks: ei_col == counterpart (used by A-selector "same" keyword)
    for _, ei_col, counterpart in A_FIELDS:
        if counterpart is not None and ei_col in arrays and counterpart in arrays:
            cache[("__same__", ei_col, counterpart)] = (
                arrays[ei_col] == arrays[counterpart]
            )
    return cache


# Module-level cache keyed by id(acc_struct); valid for the lifetime of the
# acc_struct object. The resolver also caches per instance so a single
# resolver doesn't re-dispatch through this dict on every resolve_row call.
_ec_cache: dict = {}


def get_edge_cache(acc_struct) -> tuple:
    """
    Build once per acc_struct:
      (ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr)
    Subsequent calls with the same acc_struct return the cached tuple.
    """
    key = id(acc_struct)
    if key not in _ec_cache:
        ei_exp       = expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        edge_lookup  = _build_edge_lookup(ei_exp)
        id_to_tuple  = {eid: tup for tup, eid in edge_lookup.items()}
        field_vals   = {col: sorted(str(v) for v in ei_exp[col].unique()) for col in _MASK_COLS}
        mask_cache   = _build_mask_cache(ei_exp)
        edge_ids_arr = ei_exp["edge_id"].to_numpy(dtype=int)
        _ec_cache[key] = (ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr)
    return _ec_cache[key]


# ---------------------------------------------------------------------------
# A/B enumeration
# ---------------------------------------------------------------------------

def _enumerate_a_edges(row: pd.Series, field_vals: dict):
    """
    Yield all N-tuples that match the A-selector of a CSV row, where N = len(A_FIELDS).
    Fixed fields → single value; wildcard → all unique values; same → copy counterpart.
    itertools.product runs only over non-same dimensions.
    """
    n = len(A_FIELDS)
    dims: list = []
    same_to: dict = {}
    for i, (col_key, ei_col, counterpart) in enumerate(A_FIELDS):
        v = str(row.get(col_key, "*")).strip()
        if v in WILDCARD:
            dims.append(field_vals[ei_col])
        elif v == "same" and counterpart is not None:
            cpart_pos = next(j for j, (_, ec, _) in enumerate(A_FIELDS) if ec == counterpart)
            same_to[i] = cpart_pos
            dims.append(None)
        else:
            dims.append((v,))
    free_pos  = [i for i in range(n) if dims[i] is not None]
    free_dims = [dims[i] for i in free_pos]
    for combo in itertools.product(*free_dims):
        t: list = [None] * n
        for idx, pos in enumerate(free_pos):
            t[pos] = combo[idx]
        for pos, cp in same_to.items():
            t[pos] = t[cp]
        if None not in t:
            yield tuple(t)


def _resolve_b_ids(row: pd.Series, a_tuple: tuple, edge_lookup: dict, field_vals: dict) -> list:
    """
    Enumerate all B-tuples matching the B-selector given a resolved A-edge tuple,
    and return the corresponding edge_ids via O(1) dict lookup.
    """
    dims: list = []
    for col_key, ei_col, a_col in B_FIELDS:
        v = str(row.get(col_key, "*")).strip()
        if v in WILDCARD:
            dims.append(field_vals[ei_col])
        elif v == "same":
            dims.append((a_tuple[_A_EI_POS[a_col]],))
        else:
            dims.append((v,))
    result = []
    for combo in itertools.product(*dims):
        eid = edge_lookup.get(combo)
        if eid is not None:
            result.append(eid)
    return result


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedRelation:
    """
    One (a_edge, b_edge) match produced by ParamRelationResolver.

    a_edge_id    Edge id matched by the row's A-selector.
    b_edge_id    Edge id matched by the row's B-selector under that A.
    specificity  Count of non-wildcard A-fields on the originating row.
                 Constant across all yields from a single row.
    row          The originating pd.Series. Adapters project to their own
                 record format from here.
    """
    a_edge_id:   int
    b_edge_id:   int
    specificity: int
    row:         pd.Series


class ParamRelationResolver:
    """
    Resolves a params-CSV row against an accounting structure's edge index.

    For each input row, yields one ResolvedRelation per (a_edge, b_edge)
    match. Does not derive σ. Does not dedup by specificity. Does not
    apply any skip-on-error policy. Adapters apply their own
    row-projection and policy.

    For consumers that only need the A-side (e.g. SumConstraintBuilder,
    which sums coefficients over an A-matched edge set without any
    paired B edge), call `match_a_edges(row)` directly.
    """

    def __init__(self, acc_struct) -> None:
        self._cache = get_edge_cache(acc_struct)

    def _match_a(self, row: pd.Series) -> list:
        """Return [(a_edge_id, a_tuple), …] for the row's A-selector."""
        ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr = self._cache
        n = len(ei_exp)

        # Estimate product size to choose enumeration strategy.
        prod_size = 1
        for col_key, ei_col, _ in A_FIELDS:
            v = str(row.get(col_key, "*")).strip()
            if v in WILDCARD or v == "same":
                prod_size *= len(field_vals[ei_col])
                if prod_size > A_PRODUCT_THRESHOLD:
                    break

        if prod_size <= A_PRODUCT_THRESHOLD:
            return [
                (a_id, a_tuple)
                for a_tuple in _enumerate_a_edges(row, field_vals)
                if (a_id := edge_lookup.get(a_tuple)) is not None
            ]

        a_mask_np = np.ones(n, dtype=bool)
        for col_key, ei_col, counterpart in A_FIELDS:
            val = str(row.get(col_key, "*")).strip()
            if val in WILDCARD:
                continue
            if val == "same" and counterpart is not None:
                m = mask_cache.get(("__same__", ei_col, counterpart))
            else:
                m = mask_cache.get((ei_col, val))
            if m is None:
                a_mask_np = np.zeros(n, dtype=bool)
                break
            a_mask_np &= m
        return [
            (int(aid), id_to_tuple[int(aid)])
            for aid in edge_ids_arr[a_mask_np]
        ]

    def match_a_edges(self, row: pd.Series) -> list[int]:
        """
        Return the list of A edge_ids matching this row's A-selector.

        Used by adapters that only need A-side matching (no paired B edge),
        e.g. SumConstraintBuilder summing coefficients over a matched set.
        """
        return [aid for aid, _ in self._match_a(row)]

    def resolve_row(self, row: pd.Series) -> Iterator[ResolvedRelation]:
        edge_lookup = self._cache[1]
        field_vals  = self._cache[3]
        spec = specificity(row)
        for a_id, a_tuple in self._match_a(row):
            for b_id in _resolve_b_ids(row, a_tuple, edge_lookup, field_vals):
                yield ResolvedRelation(
                    a_edge_id   = int(a_id),
                    b_edge_id   = int(b_id),
                    specificity = spec,
                    row         = row,
                )
