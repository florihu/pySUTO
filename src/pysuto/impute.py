"""
impute.py
=========
Stage 2c — Imputation pipeline.

Classes
-------
ImputationLog           Per-edge fill provenance table.
BaseImputer             Abstract base — implement impute().
ImputationEngine        Ordered pipeline; updates AggregationResult in-place.

ParametricRelation      Lambda-based directed fill (target ← sources).
ParametricImputer       Bidirectional CSV pairs + lambda directed relations.

ParamRef                Reference to a params-CSV value with optional inverse.
SourceTerm              One additive term in a FormulaRelation.
FormulaRelation         Directed multi-source balance equation.
FormulaImputer          Fills missing edges via FormulaRelation balance equations.

TimeImputer             Temporal interpolation / extrapolation per edge series.

FlatPriorImputer        Final-fallback: flat prior (f=0, large σ) for residual NaN.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Callable, Literal, Optional, Union

import itertools

import numpy as np
import pandas as pd

from .agg import AggregationResult
from .logging import get_logger
from .unc import PedigreeScores

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_edge_index(edge_index: pd.DataFrame, node_index: pd.DataFrame) -> pd.DataFrame:
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


def _build_group_ids(keys: list[tuple]) -> tuple[np.ndarray, int]:
    """
    Hash a list of per-edge tuple keys to dense integer group ids.

    Returns ``(group_id, n_groups)`` where ``group_id[i]`` is the
    zero-based id of edge i's group. Order of first occurrence determines
    id assignment, which keeps results deterministic.
    """
    group_id = np.empty(len(keys), dtype=np.int64)
    seen: dict = {}
    for i, k in enumerate(keys):
        g = seen.get(k)
        if g is None:
            g = len(seen)
            seen[k] = g
        group_id[i] = g
    return group_id, len(seen)


def _sigma_from_row(row: pd.Series, nominal: float) -> float:
    """Derive σ from a params-CSV row according to unc_type."""
    unc = str(row.get("unc_type", "")).strip()
    if unc == "sd":
        return float(row["value_sd"])
    if unc == "min_max":
        return (float(row["value_max"]) - float(row["value_min"])) / 4.0
    if unc == "pedigree":
        ps = PedigreeScores(
            reliability=int(row["pedigree_reliability"]),
            completeness=int(row["pedigree_completeness"]),
            temporal_correlation=int(row["pedigree_temporal_correlation"]),
            geographical_correlation=int(row["pedigree_geographical_correlation"]),
            other_correlation=int(row["pedigree_other_correlation"]),
        )
        return ps.total_cv() * nominal
    raise ValueError(f"Unknown unc_type: {unc!r}")


def _float_or(val, default: float) -> float:
    s = str(val).strip()
    return float(s) if s and s not in ("nan", "None", "") else default


def _apply_cv_cap(sigma_val: float, f_val: float, cv_cap: float) -> tuple[float, bool]:
    """
    Cap σ at cv_cap × |f|. Returns (sigma_capped, was_capped).

    cv_cap <= 0 or cv_cap == inf disables the cap. f == 0 disables the cap.
    """
    if (cv_cap <= 0.0 or not np.isfinite(cv_cap)
            or f_val == 0.0 or not np.isfinite(sigma_val) or not np.isfinite(f_val)):
        return sigma_val, False
    cap = cv_cap * abs(f_val)
    if sigma_val > cap:
        return cap, True
    return sigma_val, False


def _lp_combine(values, p: float) -> float:
    """
    Power-mean (Lp) combiner: ( Σ |x_i|^p )^(1/p).

    With p=2 this matches Gaussian quadrature. With p<2 large terms are
    weighted relatively less, suppressing σ inflation when source σ values
    have correlated structure. The pySUTO range is p ∈ [1.1, 1.5];
    default p = 1.3.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    s = float(np.sum(np.abs(arr) ** p))
    if s <= 0.0:
        return 0.0
    return s ** (1.0 / p)


def _reg_sensitivity(coeff: float, q: float, alpha_min: float) -> float:
    """
    Bounded effective sensitivity used in ParametricImputer σ propagation.

        s(α) = |α| / (|α|^q + α_min^q)^(1/q)

    Asymptotes to 1 for |α| >> α_min (caps coefficient amplification of σ),
    and to |α|/α_min for |α| << α_min (linearly damps very small coefficients).
    q controls the knee sharpness (q=3 is a moderate default).
    """
    a = abs(float(coeff))
    am = abs(float(alpha_min))
    if a == 0.0 and am == 0.0:
        return 0.0
    return a / (a ** q + am ** q) ** (1.0 / q)


def _time_sort_key(t):
    try:
        return (0, int(t))
    except (ValueError, TypeError):
        return (1, str(t))


_WILDCARD = {"*", "nan", "None", ""}

# Keep legacy alias so _apply_a_mask / _apply_b_mask still work
_A_WILDCARD = _WILDCARD

_A_SPEC_KEYS = [
    "a_sector_from", "a_entity_from", "a_region_from",
    "a_sector_to",   "a_entity_to",   "a_region_to",
    "a_time",        "a_valuation",
]


def _specificity(row: pd.Series) -> int:
    """Count A-selector fields that are not wildcards (higher = more specific)."""
    return sum(
        str(row.get(k, "*")).strip() not in _WILDCARD
        for k in _A_SPEC_KEYS
    )


# ---------------------------------------------------------------------------
# Mask cache — pre-built boolean arrays for fast selector matching
# ---------------------------------------------------------------------------

_MASK_COLS = [
    "sector_from", "entity_from", "region_from",
    "sector_to",   "entity_to",   "region_to",
    "time",        "valuation",
]


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
    for _, ei_col, counterpart in _A_FIELDS:
        if counterpart is not None and ei_col in arrays and counterpart in arrays:
            cache[("__same__", ei_col, counterpart)] = (
                arrays[ei_col] == arrays[counterpart]
            )
    return cache


# Module-level cache for the expanded edge index and derived lookup structures.
# Keyed by id(acc_struct); valid for the lifetime of the acc_struct object.
_ec_cache: dict = {}


def _get_edge_cache(acc_struct) -> tuple:
    """
    Build once per acc_struct:
      (ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr)
    Subsequent calls with the same acc_struct return the cached result.
    """
    key = id(acc_struct)
    if key not in _ec_cache:
        ei_exp       = _expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        edge_lookup  = _build_edge_lookup(ei_exp)
        id_to_tuple  = {eid: tup for tup, eid in edge_lookup.items()}
        field_vals   = {col: sorted(str(v) for v in ei_exp[col].unique()) for col in _MASK_COLS}
        mask_cache   = _build_mask_cache(ei_exp)
        edge_ids_arr = ei_exp["edge_id"].to_numpy(dtype=int)
        _ec_cache[key] = (ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr)
    return _ec_cache[key]


_A_PRODUCT_THRESHOLD = 100_000



# ---------------------------------------------------------------------------
# ImputationLog
# ---------------------------------------------------------------------------

class ImputationLog:
    """
    Provenance table for every imputed edge.

    Each entry is a dict with keys:
        edge_id, step, method, source_edges, coefficients,
        f_before, f_after, sigma_before, sigma_after,
        description, param_id
    """

    _COLS = [
        "edge_id", "step", "method", "source_edges", "coefficients",
        "f_before", "f_after", "sigma_before", "sigma_after",
        "description", "param_id",
    ]

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def append(self, entry: dict) -> None:
        self._entries.append(entry)

    def extend(self, entries: list[dict]) -> None:
        self._entries.extend(entries)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._entries, columns=self._COLS)

    def for_edge(self, edge_id: int) -> pd.DataFrame:
        return self.as_dataframe().query("edge_id == @edge_id").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ImputationLog({len(self._entries)} entries)"


# ---------------------------------------------------------------------------
# ImputationMetadata
# ---------------------------------------------------------------------------

class ImputationMetadata:
    """Run-level summary: one record per imputer step."""

    _COLS = [
        "run_timestamp", "step", "imputer_type", "input_paths",
        "elapsed_seconds", "edges_filled",
        "edges_missing_after", "pct_additionally_filled",
    ]

    def __init__(self) -> None:
        self._records: list[dict] = []

    def append(self, record: dict) -> None:
        self._records.append(record)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=self._COLS)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"ImputationMetadata({len(self._records)} steps)"


# ---------------------------------------------------------------------------
# BaseImputer
# ---------------------------------------------------------------------------

class BaseImputer(ABC):
    """
    Abstract base for all imputers.

    impute() receives the current f and sigma arrays (modified in-place),
    the missing mask at the START of this step (bool, True where NaN),
    the pipeline step index for logging, and the accounting structure.
    Returns a list of log-entry dicts (one per filled edge).
    """

    @abstractmethod
    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        ...

    @property
    def source_paths(self) -> list[str]:
        return []

    def config(self) -> dict:
        """
        Return a JSON-serialisable snapshot of this imputer's configuration.

        Default impl: every public attribute (no leading underscore) whose
        value is JSON-able. Subclasses can override for richer detail.
        """
        import json as _json
        out: dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                _json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)
        return out


# ---------------------------------------------------------------------------
# ImputationEngine
# ---------------------------------------------------------------------------

class ImputationEngine:
    """
    Runs registered imputers in registration order.

    Each step receives the state (f, σ) left by the previous step.
    Stops early if no missing edges remain.
    Attaches the completed ImputationLog to agg_result.imputation_log.
    """

    def __init__(self, acc_struct) -> None:
        self._acc_struct = acc_struct
        self._imputers: list[BaseImputer] = []

    @property
    def imputers(self) -> list[BaseImputer]:
        """Read-only view of registered imputers (in registration order)."""
        return list(self._imputers)

    def register(self, imputer: BaseImputer) -> None:
        if hasattr(imputer, "prepare"):
            imputer.prepare(self._acc_struct)
        self._imputers.append(imputer)

    def run(self, agg_result: AggregationResult) -> AggregationResult:
        f     = agg_result.f_agg.copy()
        sigma = agg_result.sigma_agg.copy()
        log   = ImputationLog()
        meta  = ImputationMetadata()
        run_ts = datetime.now().isoformat(timespec="seconds")

        n_total = len(f)
        for step, imputer in enumerate(self._imputers):
            missing = np.isnan(f)
            n_missing_before = int(missing.sum())
            if n_missing_before == 0:
                logger.info("step %d: no missing edges — stopping early", step)
                break
            cov_before = (n_total - n_missing_before) / n_total

            # Hand the (read-only) aggregation context to any imputer that
            # opts in via attach_agg_result(); FlatPriorImputer uses this
            # to read per-edge coverage for density-aware CV.
            if hasattr(imputer, "attach_agg_result"):
                imputer.attach_agg_result(agg_result)

            t0 = time.perf_counter()
            entries = imputer.impute(f, sigma, missing, step, self._acc_struct)
            elapsed = time.perf_counter() - t0
            log.extend(entries)

            # Post-step bounds enforcement: warn and clip any newly-filled out-of-bounds values
            newly_filled = np.where(missing & ~np.isnan(f))[0]
            if newly_filled.size > 0:
                bl = self._acc_struct.bounds.l[newly_filled]
                bu = self._acc_struct.bounds.u[newly_filled]
                fv = f[newly_filled]
                clip_needed = (fv < bl) | (fv > bu)
                if clip_needed.any():
                    _, _, id_to_tuple, _, _, _ = _get_edge_cache(self._acc_struct)
                    for e_id in newly_filled[clip_needed]:
                        lo  = float(self._acc_struct.bounds.l[e_id])
                        hi  = float(self._acc_struct.bounds.u[e_id])
                        val = float(f[e_id])
                        clipped_val = float(np.clip(val, lo, hi))
                        tup = id_to_tuple[int(e_id)]
                        warnings.warn(
                            f"step {step} ({type(imputer).__name__}): "
                            f"edge {e_id} ({tup[0]}[{tup[2]}]→{tup[3]}[{tup[5]}], "
                            f"t={tup[6]}, {tup[7]}): "
                            f"imputed {val:.4g} clipped to {clipped_val:.4g} "
                            f"(bounds [{lo:.4g}, {hi:.4g}])",
                            UserWarning,
                            stacklevel=2,
                        )
                        f[e_id] = clipped_val

            n_missing_after = int(np.isnan(f).sum())
            n_filled = n_missing_before - n_missing_after
            cov_after = (n_total - n_missing_after) / n_total
            logger.info(
                "step %d (%s): filled %d edges  →  coverage %.1f%% → %.1f%%  (△+%.1f%%)",
                step, type(imputer).__name__, n_filled,
                100 * cov_before, 100 * cov_after, 100 * (cov_after - cov_before),
            )
            meta.append({
                "run_timestamp": run_ts,
                "step": step,
                "imputer_type": type(imputer).__name__,
                "input_paths": "; ".join(imputer.source_paths),
                "elapsed_seconds": round(elapsed, 4),
                "edges_filled": n_filled,
                "edges_missing_after": n_missing_after,
                "pct_additionally_filled": round(100 * n_filled / n_total, 2),
            })

        agg_result.f_agg               = f
        agg_result.sigma_agg           = sigma
        agg_result.imputation_log      = log
        agg_result.imputation_metadata = meta
        return agg_result


# ---------------------------------------------------------------------------
# ParametricRelation  (lambda-based, directed)
# ---------------------------------------------------------------------------

@dataclass
class ParametricRelation:
    """
    Directed fill: f_target = Σ coeff_i × f_source_i

    target      Predicate over expanded edge_index (ei_exp) → bool mask.
                Identifies which edges to fill.
    sources     Predicate over expanded ei_exp → bool mask.
                Identifies candidate source edges.
                Within each target-edge context, sources are filtered to
                edges with the same region_from and time as the target.
    coefficients list[float] — one coefficient per matched source edge.
                If len == 1, the single coefficient is applied to the sum
                of all matched sources.
    epsilon     Relative model uncertainty added in quadrature.
    param_id    Optional string (logged as param_id in ImputationLog).
    """

    description: str
    target: Callable
    sources: Callable
    coefficients: list[float]
    epsilon: float = 0.0
    param_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ParametricImputer
# ---------------------------------------------------------------------------

# A-selector field → (ei_exp column, counterpart column for "same" constraint)
_A_FIELDS = [
    ("a_sector_from", "sector_from", "sector_to"),
    ("a_entity_from", "entity_from", "entity_to"),
    ("a_region_from", "region_from", "region_to"),
    ("a_sector_to",   "sector_to",   "sector_from"),
    ("a_entity_to",   "entity_to",   "entity_from"),
    ("a_region_to",   "region_to",   "region_from"),
    ("a_time",        "time",        None),
    ("a_valuation",   "valuation",   None),
]

# B-selector field → (ei_exp column, corresponding A-edge column)
_B_FIELDS = [
    ("b_sector_from", "sector_from", "sector_from"),
    ("b_entity_from", "entity_from", "entity_from"),
    ("b_region_from", "region_from", "region_from"),
    ("b_sector_to",   "sector_to",   "sector_to"),
    ("b_entity_to",   "entity_to",   "entity_to"),
    ("b_region_to",   "region_to",   "region_to"),
    ("b_time",        "time",        "time"),
    ("b_valuation",   "valuation",   "valuation"),
]


# Position of each ei_exp column in the 8-tuple (sector_from=0 … valuation=7)
_A_EI_POS = {ec: i for i, (_, ec, _) in enumerate(_A_FIELDS)}


def _build_edge_lookup(ei_exp: pd.DataFrame) -> dict:
    """O(1) lookup: (sfrom,efrom,rfrom,sto,eto,rto,time,val) → edge_id."""
    cols = ["sector_from", "entity_from", "region_from",
            "sector_to",   "entity_to",   "region_to", "time", "valuation"]
    arr  = ei_exp[cols].to_numpy(dtype=str)
    eids = ei_exp["edge_id"].to_numpy(dtype=int)
    return {tuple(arr[i]): int(eids[i]) for i in range(len(arr))}


def _enumerate_a_edges(row: pd.Series, field_vals: dict):
    """
    Yield all 8-tuples that match the A-selector of a CSV row.
    Fixed fields → single value; wildcard → all unique values; same → copy counterpart.
    itertools.product runs only over non-same dimensions.
    """
    dims: list = []
    same_to: dict = {}
    for i, (col_key, ei_col, counterpart) in enumerate(_A_FIELDS):
        v = str(row.get(col_key, "*")).strip()
        if v in _WILDCARD:
            dims.append(field_vals[ei_col])
        elif v == "same" and counterpart is not None:
            cpart_pos = next(j for j, (_, ec, _) in enumerate(_A_FIELDS) if ec == counterpart)
            same_to[i] = cpart_pos
            dims.append(None)
        else:
            dims.append((v,))
    free_pos  = [i for i in range(8) if dims[i] is not None]
    free_dims = [dims[i] for i in free_pos]
    for combo in itertools.product(*free_dims):
        t: list = [None] * 8
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
    for col_key, ei_col, a_col in _B_FIELDS:
        v = str(row.get(col_key, "*")).strip()
        if v in _WILDCARD:
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


class ParametricImputer(BaseImputer):
    """
    Two instantiation modes:

    1. from_csv(path, acc_struct)  — precomputes bidirectional (a_id, b_id, coeff, σ)
       pairs from a params-template CSV.  Each pair is filled in whichever
       direction the missing mask requires: f_b = coeff·f_a or f_a = f_b/coeff.

    2. ParametricImputer(relations=[...])  — directed fills from ParametricRelation
       lambda objects.  Sources are matched per target by region_from and time.
    """

    def __init__(
        self,
        relations: Optional[list[ParametricRelation]] = None,
        _csv_pairs: Optional[list] = None,
        _source_path: Optional[str] = None,
        cv_cap: float = float("inf"),
        p: float = 1.3,
        q: float = 3.0,
        alpha_min: float = 0.05,
    ) -> None:
        self._relations  = relations or []
        # Each tuple: (a_id, b_id, coeff, sigma_param, epsilon, param_id, desc)
        # Note: per-row `bound_lower / bound_upper` from the CSV are *ratio bounds*
        # (consumed by RatioBoundBuilder in constraints.py); they are NOT absolute
        # caps on the filled target flow and are deliberately not read here.
        self._csv_pairs  = _csv_pairs or []
        self._source_path: Optional[str] = _source_path
        self._ei_exp:    Optional[pd.DataFrame] = None  # lazy cache for lambda relations
        self.cv_cap     = float(cv_cap)
        self.p          = float(p)
        self.q          = float(q)
        self.alpha_min  = float(alpha_min)

    @property
    def source_paths(self) -> list[str]:
        return [self._source_path] if self._source_path else []

    def config(self) -> dict:
        return {
            "cv_cap":         self.cv_cap,
            "p":              self.p,
            "q":              self.q,
            "alpha_min":      self.alpha_min,
            "source_path":    self._source_path,
            "n_csv_pairs":    len(self._csv_pairs),
            "n_relations":    len(self._relations),
            "relations_desc": [r.description for r in self._relations],
        }

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        cv_cap: float = float("inf"),
        p: float = 1.3,
        q: float = 3.0,
        alpha_min: float = 0.05,
    ) -> "ParametricImputer":
        """
        Load a params-template CSV (semicolon-delimited) and precompute
        (edge_a_id, edge_b_id) pairs for bidirectional imputation.

        Comment lines (starting with #) and rows with blank param_id are skipped.
        """
        df = pd.read_csv(path, sep=";", comment="#", dtype=str)
        df = df.dropna(subset=["param_id"])
        df = df[~df["param_id"].str.startswith("#")]

        ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr = (
            _get_edge_cache(acc_struct)
        )
        n = len(ei_exp)
        # dict[(a_id, b_id)] → (pair_tuple, specificity)
        best: dict[tuple, tuple] = {}

        for _, row in df.iterrows():
            nominal = float(row["value_nominal"])
            try:
                sigma_p = _sigma_from_row(row, nominal)
            except Exception as exc:
                logger.warning("param_id=%s: sigma error (%s) — skipped", row["param_id"], exc)
                continue
            epsilon  = _float_or(row.get("epsilon"), 0.0)
            # bound_lower / bound_upper are *ratio* bounds — consumed by
            # RatioBoundBuilder for constraint building, not by this imputer.
            param_id = str(row["param_id"])
            desc     = str(row.get("description", ""))
            spec     = _specificity(row)
            n_a      = 0

            # Estimate product size to decide enumeration strategy
            prod_size = 1
            for col_key, ei_col, _ in _A_FIELDS:
                v = str(row.get(col_key, "*")).strip()
                if v in _WILDCARD or v == "same":
                    prod_size *= len(field_vals[ei_col])
                    if prod_size > _A_PRODUCT_THRESHOLD:
                        break

            if prod_size <= _A_PRODUCT_THRESHOLD:
                # Small product: iterate combos via dict lookup
                a_iter: list = [
                    (a_id, a_tuple)
                    for a_tuple in _enumerate_a_edges(row, field_vals)
                    if (a_id := edge_lookup.get(a_tuple)) is not None
                ]
            else:
                # Large product: numpy bitwise mask — O(n/64) per field
                a_mask_np = np.ones(n, dtype=bool)
                for col_key, ei_col, counterpart in _A_FIELDS:
                    val = str(row.get(col_key, "*")).strip()
                    if val in _WILDCARD:
                        continue
                    if val == "same" and counterpart is not None:
                        m = mask_cache.get(("__same__", ei_col, counterpart))
                    else:
                        m = mask_cache.get((ei_col, val))
                    if m is None:
                        a_mask_np = np.zeros(n, dtype=bool)
                        break
                    a_mask_np &= m
                a_iter = [
                    (int(aid), id_to_tuple[int(aid)])
                    for aid in edge_ids_arr[a_mask_np]
                ]

            for a_id, a_tuple in a_iter:
                n_a += 1
                for b_id in _resolve_b_ids(row, a_tuple, edge_lookup, field_vals):
                    key  = (a_id, b_id)
                    pair = (a_id, b_id, nominal, sigma_p, epsilon, param_id, desc)
                    if key not in best or spec > best[key][1]:
                        if key in best:
                            logger.debug(
                                "param_id=%s: replacing lower-specificity pair %s (spec %d→%d)",
                                param_id, key, best[key][1], spec,
                            )
                        best[key] = (pair, spec)

            if n_a == 0:
                logger.warning("param_id=%s: no A-edges matched", param_id)

        pairs = [v[0] for v in best.values()]
        logger.info("from_csv: loaded %d bidirectional pairs from %s", len(pairs), path)
        return cls(
            _csv_pairs=pairs,
            _source_path=str(path),
            cv_cap=cv_cap,
            p=p,
            q=q,
            alpha_min=alpha_min,
        )

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        entries: list[dict] = []
        cv_cap  = self.cv_cap
        p       = self.p
        q       = self.q
        am      = self.alpha_min

        # ── CSV pairs (bidirectional) ──────────────────────────────────
        for a_id, b_id, coeff, sig_p, epsilon, param_id, desc in self._csv_pairs:
            a_nan = np.isnan(f[a_id])
            b_nan = np.isnan(f[b_id])

            if not a_nan and b_nan:
                f_b = float(coeff * f[a_id])
                # Forward (A→B): bare coefficient. σ_b = Lp(|coeff|·σ_a, σ_param, ε·f_b)
                s_b   = _lp_combine(
                    [coeff * sigma[a_id], sig_p, epsilon * f_b], p,
                )
                s_b, capped = _apply_cv_cap(s_b, f_b, cv_cap)
                f[b_id]     = f_b
                sigma[b_id] = s_b
                entries.append(_log(
                    b_id, step,
                    "parametric_capped" if capped else "parametric",
                    [a_id], [coeff], f_b, s_b, desc, param_id,
                ))

            elif a_nan and not b_nan:
                inv = 1.0 / coeff
                f_a = float(inv * f[b_id])
                # Backward (B→A): regularized sensitivity caps σ amplification
                # when |inv| ≫ 1 (e.g. 1/ore_grade ≈ 100s).
                s_inv = _reg_sensitivity(inv, q, am)
                s_a   = _lp_combine(
                    [s_inv * sigma[b_id], sig_p, epsilon * f_a], p,
                )
                s_a, capped = _apply_cv_cap(s_a, f_a, cv_cap)
                f[a_id]     = f_a
                sigma[a_id] = s_a
                entries.append(_log(
                    a_id, step,
                    "parametric_capped" if capped else "parametric",
                    [b_id], [inv], f_a, s_a, desc, param_id,
                ))

        # ── Lambda relations (directed: target ← sources) ─────────────
        if self._relations:
            if self._ei_exp is None:
                self._ei_exp = _expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
            ei_exp = self._ei_exp
            for rel in self._relations:
                t_mask = rel.target(ei_exp)
                t_ids  = ei_exp[t_mask]["edge_id"].values

                for t_id in t_ids:
                    if not np.isnan(f[t_id]):
                        continue

                    t_row = ei_exp[ei_exp["edge_id"] == t_id].iloc[0]
                    t_rfrom = t_row["region_from"]
                    t_time  = t_row["time"]

                    # Sources filtered to same region_from and same time
                    s_mask = (
                        rel.sources(ei_exp)
                        & (ei_exp["region_from"].astype(str) == str(t_rfrom))
                        & (ei_exp["time"].astype(str) == str(t_time))
                    )
                    s_ids = ei_exp[s_mask]["edge_id"].values

                    if len(s_ids) == 0:
                        continue

                    # Resolve coefficients
                    if len(rel.coefficients) == 1:
                        coeffs = [rel.coefficients[0]] * len(s_ids)
                    elif len(rel.coefficients) == len(s_ids):
                        coeffs = list(rel.coefficients)
                    else:
                        logger.warning(
                            "ParametricRelation '%s': %d coefficients but %d sources "
                            "for target edge %d — skipped",
                            rel.description, len(rel.coefficients), len(s_ids), t_id,
                        )
                        continue

                    if any(np.isnan(f[s]) for s in s_ids):
                        continue

                    f_t = float(sum(c * f[s] for c, s in zip(coeffs, s_ids)))
                    # Lambda (target ← sources): bare coefficients (forward by definition)
                    # σ_t = ( Σ_l |α_l·σ_l|^p + |ε·f_t|^p )^(1/p)
                    terms = [c * sigma[s] for c, s in zip(coeffs, s_ids)]
                    terms.append(rel.epsilon * f_t)
                    s_t = _lp_combine(terms, p)
                    s_t, capped = _apply_cv_cap(s_t, f_t, cv_cap)

                    f[t_id]     = f_t
                    sigma[t_id] = s_t
                    entries.append(_log(
                        t_id, step,
                        "parametric_capped" if capped else "parametric",
                        list(s_ids), coeffs, f_t, s_t,
                        rel.description, rel.param_id,
                    ))

        return entries


# ---------------------------------------------------------------------------
# FormulaImputer
# ---------------------------------------------------------------------------

@dataclass
class ParamRef:
    """
    Reference to a value in a loaded params CSV.

    param_id    Key matching the param_id column of the CSV.
    transform   "identity" (default) uses value_nominal directly;
                "inverse" uses 1 / value_nominal.
    """
    param_id: str
    transform: Literal["identity", "inverse"] = "identity"


@dataclass
class SourceTerm:
    """
    One additive term in a FormulaRelation.

    selector        Predicate over expanded edge_index (ei_exp) → bool mask.
    coefficient     float or ParamRef; negative = subtraction.
    region_binding  Constrains source region relative to the target edge:
                      "none"                  — no region constraint
                      "from_matches_target_from" — source.region_from == target.region_from
                      "from_matches_target_to"   — source.region_from == target.region_to
                      "to_matches_target_from"   — source.region_to   == target.region_from
                      "to_matches_target_to"     — source.region_to   == target.region_to
    time_binding    If True, source.time must equal target.time.
    aggregate       "single" (expect exactly one matching edge) or
                    "sum"    (sum over all matching edges).
    """
    selector:       Callable
    coefficient:    Union[float, "ParamRef"] = 1.0
    region_binding: Literal[
        "none",
        "from_matches_target_from", "from_matches_target_to",
        "to_matches_target_from",   "to_matches_target_to",
    ] = "none"
    time_binding:   bool                           = True
    aggregate:      Literal["single", "sum"]       = "single"


@dataclass
class FormulaRelation:
    """
    Directed multi-source balance equation:
        f_target = clip(Σ_i coeff_i × f_source_i, clip_lower, clip_upper)

    description   Human-readable string (logged).
    target        Predicate over ei_exp → bool mask; identifies target edges.
    terms         list[SourceTerm]; sources with region/time binding.
    clip_lower    Physical lower bound; None = no lower clip.
    clip_upper    Physical upper bound; None = no upper clip.
    epsilon       Relative model uncertainty added in quadrature.
    """
    description:  str
    target:       Callable
    terms:        list[SourceTerm]
    clip_lower:   Optional[float] = 0.0
    clip_upper:   Optional[float] = None
    epsilon:      float           = 0.0


class FormulaImputer(BaseImputer):
    """
    Fills missing edges using FormulaRelation balance equations.

    params_csv  Optional path to a params-template CSV; if provided, its
                value_nominal column is loaded so ParamRef coefficients can
                be resolved.  Pass a dict str→float for programmatic supply.
    """

    def __init__(
        self,
        relations: list[FormulaRelation],
        params_csv: Optional[str] = None,
        params: Optional[dict[str, float]] = None,
        cv_cap: float = float("inf"),
        p: float = 1.3,
    ) -> None:
        self._relations = relations
        self._params: dict[str, float] = dict(params or {})
        if params_csv is not None:
            self._load_params_csv(params_csv)
        # Lazy cache — populated on first impute() call or by prepare()
        self._ei_exp:     Optional[pd.DataFrame] = None
        self._mask_cache: Optional[dict]         = None
        self._eid_to_pos: Optional[dict]         = None
        self._ei_rf:      Optional[np.ndarray]   = None
        self._ei_rt:      Optional[np.ndarray]   = None
        self._ei_time:    Optional[np.ndarray]   = None
        self._precomp:    Optional[list]         = None
        self.cv_cap        = float(cv_cap)
        self.p             = float(p)
        self._params_csv_path: Optional[str] = str(params_csv) if params_csv else None

    @property
    def source_paths(self) -> list[str]:
        return [self._params_csv_path] if self._params_csv_path else []

    def config(self) -> dict:
        return {
            "cv_cap":          self.cv_cap,
            "p":               self.p,
            "params_csv":      self._params_csv_path,
            "n_relations":     len(self._relations),
            "relations_desc":  [r.description for r in self._relations],
        }

    def prepare(self, acc_struct) -> None:
        ei_exp, _, _, _, mask_cache, _ = _get_edge_cache(acc_struct)
        eids_np = ei_exp["edge_id"].to_numpy(dtype=int)
        self._ei_exp     = ei_exp
        self._mask_cache = mask_cache
        self._eid_to_pos = dict(zip(eids_np.tolist(), range(len(eids_np))))
        self._ei_rf      = ei_exp["region_from"].to_numpy(dtype=str)
        self._ei_rt      = ei_exp["region_to"].to_numpy(dtype=str)
        self._ei_time    = ei_exp["time"].to_numpy(dtype=str)
        self._precomp    = [
            (rel.target(ei_exp), [term.selector(ei_exp) for term in rel.terms])
            for rel in self._relations
        ]

    def _load_params_csv(self, path: str) -> None:
        df = pd.read_csv(path, sep=";", comment="#", dtype=str)
        df = df.dropna(subset=["param_id"])
        df = df[~df["param_id"].str.startswith("#")]
        for _, row in df.iterrows():
            try:
                self._params[str(row["param_id"])] = float(row["value_nominal"])
            except (ValueError, KeyError):
                pass

    def _resolve_coeff(self, coeff: Union[float, ParamRef]) -> Optional[float]:
        if isinstance(coeff, ParamRef):
            val = self._params.get(coeff.param_id)
            if val is None:
                logger.warning("ParamRef '%s' not found in loaded params", coeff.param_id)
                return None
            return (1.0 / val) if coeff.transform == "inverse" else val
        return float(coeff)

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        # ── Lazy one-time setup ──────────────────────────────────────────
        if self._ei_exp is None:
            self.prepare(acc_struct)

        ei_exp     = self._ei_exp
        cache      = self._mask_cache
        eid_to_pos = self._eid_to_pos
        ei_rf      = self._ei_rf
        ei_rt      = self._ei_rt
        ei_time_np = self._ei_time
        edge_ids   = ei_exp["edge_id"].values
        entries: list[dict] = []

        _rb_col = {
            "from_matches_target_from": "region_from",
            "from_matches_target_to":   "region_from",
            "to_matches_target_from":   "region_to",
            "to_matches_target_to":     "region_to",
        }
        _rb_side = {
            "from_matches_target_from": "from",
            "from_matches_target_to":   "to",
            "to_matches_target_from":   "from",
            "to_matches_target_to":     "to",
        }

        for rel_idx, rel in enumerate(self._relations):
            t_mask_rel, term_base_masks = self._precomp[rel_idx]
            missing_now = np.isnan(f)
            t_ids = edge_ids[t_mask_rel & missing_now[edge_ids]]

            for t_id in t_ids:
                _pos   = eid_to_pos[int(t_id)]
                t_rf   = str(ei_rf[_pos])
                t_rt   = str(ei_rt[_pos])
                t_time = str(ei_time_np[_pos])

                f_val       = 0.0
                lp_terms:   list[float] = []
                src_ids:    list[int]   = []
                src_coeffs: list[float] = []
                ok = True

                for i, term in enumerate(rel.terms):
                    # Start from pre-computed base selector mask
                    s_mask = term_base_masks[i]

                    # Region binding via cache lookup
                    rb = term.region_binding
                    if rb != "none" and rb in _rb_col:
                        col = _rb_col[rb]
                        val = t_rf if _rb_side[rb] == "from" else t_rt
                        rm  = cache.get((col, val))
                        s_mask = s_mask & rm if rm is not None else np.zeros(len(edge_ids), dtype=bool)

                    # Time binding via cache lookup
                    if term.time_binding:
                        tm = cache.get(("time", t_time))
                        s_mask = s_mask & tm if tm is not None else np.zeros(len(edge_ids), dtype=bool)

                    s_ids = edge_ids[s_mask]

                    if term.aggregate == "single" and len(s_ids) != 1:
                        ok = False
                        break
                    if any(np.isnan(f[s]) for s in s_ids):
                        ok = False
                        break

                    coeff = self._resolve_coeff(term.coefficient)
                    if coeff is None:
                        ok = False
                        break

                    for s in s_ids:
                        f_val   += coeff * f[s]
                        lp_terms.append(coeff * sigma[s])
                        src_ids.append(int(s))
                        src_coeffs.append(coeff)

                if not ok:
                    continue

                lp_terms.append(rel.epsilon * f_val)

                # Clip to acc.bounds (authoritative; FormulaRelation.clip_lower/clip_upper ignored)
                clipped = False
                lo = float(acc_struct.bounds.l[int(t_id)])
                hi = float(acc_struct.bounds.u[int(t_id)])
                if f_val < lo:
                    f_val   = lo
                    clipped = True
                if f_val > hi:
                    f_val   = hi
                    clipped = True

                s_val = _lp_combine(lp_terms, self.p)
                s_val, capped = _apply_cv_cap(s_val, f_val, self.cv_cap)
                f[t_id]     = f_val
                sigma[t_id] = s_val
                if clipped and capped:
                    method = "formula_clipped_capped"
                elif clipped:
                    method = "formula_clipped"
                elif capped:
                    method = "formula_capped"
                else:
                    method = "formula"
                entries.append(_log(
                    t_id, step, method,
                    src_ids, src_coeffs, f_val, s_val,
                    rel.description, None,
                ))

        return entries


# ---------------------------------------------------------------------------
# TimeImputer
# ---------------------------------------------------------------------------

class TimeImputer(BaseImputer):
    """
    Temporal interpolation / extrapolation for missing edges.

    Edges are grouped by (from_id, to_id, valuation).  Within each group,
    the sorted time series is scanned for gaps and filled according to method.

    Parameters
    ----------
    window          Number of adjacent time steps to search (both directions
                    for "linear" and "mean"; only backward for "locf";
                    only forward for "nocb").  Override per-direction with
                    window_before / window_after.
    method          "linear"  — two-point linear interpolation.
                    "locf"    — last observation carried forward.
                    "nocb"    — next observation carried backward.
                    "mean"    — unweighted mean within window.
    min_obs         Minimum non-NaN observations required within window.
    gap_inflation   σ multiplier per time-step gap (default 1.0 = none).
    window_before   Override backward window (default = window).
    window_after    Override forward window (default = window).
    """

    def __init__(
        self,
        window: int = 5,
        method: Literal["linear", "locf", "nocb", "mean"] = "linear",
        min_obs: int = 1,
        gap_inflation: float = 1.0,
        window_before: Optional[int] = None,
        window_after:  Optional[int] = None,
        interpolate_only: bool = False,
        p: float = 1.3,
    ) -> None:
        self.window           = window
        self.method           = method
        self.min_obs          = min_obs
        self.gap_inflation    = gap_inflation
        self.wb               = window_before if window_before is not None else window
        self.wa               = window_after  if window_after  is not None else window
        self.interpolate_only = interpolate_only
        self.p                = float(p)

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        ei = acc_struct.edge_index

        # Build sorted time ordering
        all_times = sorted(set(ei["time"].tolist()), key=_time_sort_key)
        t_to_idx  = {t: i for i, t in enumerate(all_times)}

        # Group edges by series key → {time_idx: edge_id}
        fid_arr = ei["from_id"].to_numpy(dtype=int)
        tid_arr = ei["to_id"].to_numpy(dtype=int)
        val_arr = ei["valuation"].to_numpy(dtype=str)
        tim_arr = ei["time"].values
        eid_arr = ei["edge_id"].to_numpy(dtype=int)
        series: dict[tuple, dict[int, int]] = {}
        for i in range(len(fid_arr)):
            key = (fid_arr[i], tid_arr[i], val_arr[i])
            series.setdefault(key, {})[t_to_idx[tim_arr[i]]] = eid_arr[i]

        entries: list[dict] = []

        for key, t_map in series.items():
            sorted_tidxs = sorted(t_map.keys())

            for t_target in sorted_tidxs:
                e_id = t_map[t_target]
                if not np.isnan(f[e_id]):
                    continue

                entry = self._fill_one(
                    f, sigma, t_map, t_target, e_id, step,
                )
                if entry is not None:
                    entries.append(entry)

        return entries

    def _fill_one(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        t_map: dict[int, int],
        t_target: int,
        e_id: int,
        step: int,
    ) -> Optional[dict]:
        method = self.method

        # Collect known observations before and after within window
        before = [
            (t, t_map[t]) for t in sorted(t_map)
            if t < t_target and (t_target - t) <= self.wb and not np.isnan(f[t_map[t]])
        ]
        after = [
            (t, t_map[t]) for t in sorted(t_map)
            if t > t_target and (t - t_target) <= self.wa and not np.isnan(f[t_map[t]])
        ]

        f_val: Optional[float] = None
        s_val: Optional[float] = None
        src:   list[int]       = []
        wts:   list[float]     = []
        meth:  str             = method

        if method == "locf":
            if not before:
                return None
            t_obs, s_id = before[-1]
            gap   = t_target - t_obs
            f_val = float(f[s_id])
            s_val = float(sigma[s_id]) * self.gap_inflation ** gap
            src, wts, meth = [s_id], [1.0], "locf"

        elif method == "nocb":
            if not after:
                return None
            t_obs, s_id = after[0]
            gap   = t_obs - t_target
            f_val = float(f[s_id])
            s_val = float(sigma[s_id]) * self.gap_inflation ** gap
            src, wts, meth = [s_id], [1.0], "nocb"

        elif method == "mean":
            candidates = before + after
            if len(candidates) < self.min_obs:
                return None
            n     = len(candidates)
            w     = 1.0 / n
            f_val = float(sum(w * f[eid] for _, eid in candidates))
            # σ_mean = ( Σ_i |w·σ_i|^p )^(1/p)
            s_val = _lp_combine([w * sigma[eid] for _, eid in candidates], self.p)
            src   = [eid for _, eid in candidates]
            wts   = [w] * n
            meth  = "time_mean"

        else:  # linear
            if before and after:
                t_a, a_id = before[-1]
                t_b, b_id = after[0]
                gap  = t_b - t_a
                lam  = (t_target - t_a) / gap
                f_val = float((1 - lam) * f[a_id] + lam * f[b_id])
                # σ_linear = ( |(1-λ)·σ_a|^p + |λ·σ_b|^p )^(1/p) · γ_0^gap
                s_val = _lp_combine(
                    [(1 - lam) * sigma[a_id], lam * sigma[b_id]], self.p,
                ) * (self.gap_inflation ** gap)
                src, wts, meth = [a_id, b_id], [1 - lam, lam], "time_linear"
            elif self.interpolate_only:
                return None
            elif before:
                t_obs, s_id = before[-1]
                gap   = t_target - t_obs
                f_val = float(f[s_id])
                s_val = float(sigma[s_id]) * self.gap_inflation ** gap
                src, wts, meth = [s_id], [1.0], "time_linear"
            elif after:
                t_obs, s_id = after[0]
                gap   = t_obs - t_target
                f_val = float(f[s_id])
                s_val = float(sigma[s_id]) * self.gap_inflation ** gap
                src, wts, meth = [s_id], [1.0], "time_linear"
            else:
                return None

        if f_val is None:
            return None

        f[e_id]     = f_val
        sigma[e_id] = s_val
        return self._make_entry(e_id, step, meth, src, wts, f_val, s_val)

    @staticmethod
    def _make_entry(
        e_id: int, step: int, method: str,
        src: list[int], coeffs: list[float],
        f_val: float, s_val: float,
    ) -> dict:
        return _log(e_id, step, method, src, coeffs, f_val, s_val, "", None)


# ---------------------------------------------------------------------------
# FlatPriorImputer
# ---------------------------------------------------------------------------

class FlatPriorImputer(BaseImputer):
    """
    Final-fallback imputer: fills any residual-NaN edges with a flat prior
    sized by an accounting-mask CV × group scale, with two refinements
    introduced after `REP/analysis_optim_const_copper_…_20260513_112419.md`.

        f_j     = 0
        σ_j     = CV(j) · scale(j)

    **CV(j) — coverage-aware, per-mask interpolation.**
    Each mask (e.g. `supply`, `use`) carries a pair ``(cv_lo, cv_hi)``. For a
    given edge j the CV is interpolated by the coverage density of its
    ``(sector_from, region_from)`` neighbourhood::

        ρ(j)   = fraction of edges in (sector_from, region_from) with
                 ``agg_result.coverage == True``
        CV(j)  = cv_lo[mask(j)] + (cv_hi[mask(j)] - cv_lo[mask(j)]) · (1 - ρ(j))

    Data-rich neighbourhoods (ρ ≈ 1) get the tight CV; data-poor
    neighbourhoods (ρ ≈ 0) get the loose CV. Masks are resolved in
    insertion order, first-match-wins; unmatched edges receive
    ``default_cv`` with no density modulation. If no coverage array has
    been attached at impute time (``attach_agg_result`` was not called),
    every edge is treated as fully covered (ρ = 1, CV = cv_lo) and a
    warning is logged.

    **scale(j) — 3-level cascade.**
    Cascading fallback so an edge whose narrow group has no observed flow
    still receives a meaningful scale (the prior single-level rule
    collapsed σ to 0 on ~40 k copper edges, which then pinned them at
    σ_a = 1e-6 via the constraints-side floor)::

        level 1: mean(|f_agg|) over (sector_from, entity_from,
                                    region_from, valuation)
        level 2: mean(|f_agg|) over (sector_from, valuation)
                 — used when level 1 is empty
        level 3: median(|f_agg|) over finite, non-zero entries globally
                 — used when both narrower groups are empty;
                   falls back to 1.0 only if the whole array is empty.

    Only finite, non-zero |f_agg| values are counted at every level (so
    upstream-imputed zeros don't pull the scale down to zero).

    Register LAST in the pipeline; ``ConstraintEngine.run()`` will refuse a
    NaN-bearing agg_result without it.
    """

    _SUPPORTED_MASKS = ("supply", "use", "tradable", "sb")

    def __init__(
        self,
        cv_by_mask: dict[str, tuple[float, float]],
        default_cv: float,
        acc_struct,
        p: float = 1.3,
    ) -> None:
        """
        Parameters
        ----------
        cv_by_mask : dict[str, tuple[float, float]]
            Mapping ``mask_name → (cv_lo, cv_hi)``. Insertion order encodes
            priority (first-matching mask wins). For backward compatibility
            a scalar ``cv`` is accepted and treated as ``(cv, cv)`` (no
            density modulation).
        default_cv : float
            CV applied to edges that match no listed mask. Used as a single
            value, no density modulation.
        acc_struct : AccountingStructure
            Provides edge masks (``{name}_edge_mask``) and the edge / node
            index DataFrames for group-key construction.
        p : float
            Lp exponent for downstream uncertainty propagation (recorded
            in ``config()``; not used inside this imputer directly).
        """
        # Normalise scalar CV → (lo, hi) so back-compat callers still work.
        self.cv_by_mask: dict[str, tuple[float, float]] = {}
        for name, val in cv_by_mask.items():
            if isinstance(val, (tuple, list)):
                lo, hi = float(val[0]), float(val[1])
            else:
                lo = hi = float(val)
            self.cv_by_mask[name] = (lo, hi)

        self.default_cv = float(default_cv)
        self.p = float(p)
        self._source_path: Optional[str] = None
        # agg_result.coverage attached by ImputationEngine.run via
        # attach_agg_result() before impute() is called.
        self._coverage: Optional[np.ndarray] = None

        for name in self.cv_by_mask:
            if name not in self._SUPPORTED_MASKS:
                raise ValueError(
                    f"Unknown mask name {name!r}; supported: {self._SUPPORTED_MASKS}"
                )

        # Pre-resolve mask name → bool ndarray (length n_edges) once.
        self._masks: list[tuple[str, np.ndarray]] = []
        for name in self.cv_by_mask:  # insertion order
            attr = f"{name}_edge_mask"
            mask = getattr(acc_struct, attr, None)
            if mask is None:
                raise AttributeError(
                    f"acc_struct has no attribute '{attr}' for mask '{name}'"
                )
            self._masks.append((name, np.asarray(mask, dtype=bool)))

        self._n_edges = int(acc_struct.n_edges)

        # Pre-compute three integer group-id arrays per edge:
        #   level 1: (sector_from, entity_from, region_from, valuation) — narrow scale group
        #   level 2: (sector_from, valuation)                           — broader scale fallback
        #   density: (sector_from, region_from)                         — coverage-density neighbourhood
        ei_exp = _expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        if len(ei_exp) != self._n_edges:
            raise ValueError(
                f"acc_struct.n_edges={self._n_edges} mismatches edge_index length={len(ei_exp)}"
            )

        sec_from = ei_exp["sector_from"].astype(str).to_numpy()
        ent_from = ei_exp["entity_from"].astype(str).to_numpy()
        reg_from = ei_exp["region_from"].astype(str).to_numpy()
        val_arr  = ei_exp["valuation"].astype(str).to_numpy()

        self._group_id,         self._n_groups        = _build_group_ids(
            list(zip(sec_from, ent_from, reg_from, val_arr))
        )
        self._group_id_level2,  self._n_groups_level2 = _build_group_ids(
            list(zip(sec_from, val_arr))
        )
        self._density_group_id, self._n_density_groups = _build_group_ids(
            list(zip(sec_from, reg_from))
        )

    def attach_agg_result(self, agg_result) -> None:
        """
        Receive a reference to the current `AggregationResult` so the
        imputer can read per-edge ``coverage`` for density-aware CV.

        Called by ``ImputationEngine.run`` before each step. Other imputers
        without this method are simply skipped by the engine.
        """
        cov = getattr(agg_result, "coverage", None)
        if cov is None:
            self._coverage = None
            return
        cov = np.asarray(cov, dtype=bool)
        if cov.shape != (self._n_edges,):
            raise ValueError(
                f"agg_result.coverage shape {cov.shape} != (n_edges={self._n_edges},)"
            )
        self._coverage = cov

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        default_cv: float,
        p: float = 1.3,
    ) -> "FlatPriorImputer":
        """
        Load a CV-by-mask CSV (comma-delimited).

        New schema (preferred)::

            mask_name, cv_lo, cv_hi, description

        Legacy schema (still accepted; treated as cv_lo == cv_hi)::

            mask_name, cv, description

        Row order encodes priority (first-matching mask wins).
        """
        df = pd.read_csv(path, comment="#")
        has_lo_hi = {"cv_lo", "cv_hi"}.issubset(df.columns)
        has_cv    = "cv" in df.columns

        if has_lo_hi:
            df = df.dropna(subset=["mask_name", "cv_lo", "cv_hi"])
            mode = "cv_lo/cv_hi"
        elif has_cv:
            df = df.dropna(subset=["mask_name", "cv"])
            mode = "legacy single-cv"
        else:
            raise ValueError(
                f"flat_prior_cv CSV at {path!r} must contain either "
                f"(cv_lo, cv_hi) or legacy `cv` columns."
            )

        cv_by_mask: dict[str, tuple[float, float]] = {}
        for _, row in df.iterrows():
            name = str(row["mask_name"]).strip()
            if not name:
                continue
            if has_lo_hi:
                cv_by_mask[name] = (float(row["cv_lo"]), float(row["cv_hi"]))
            else:
                cv = float(row["cv"])
                cv_by_mask[name] = (cv, cv)

        logger.info("FlatPriorImputer.from_csv(%s): schema=%s, %d mask rows",
                    path, mode, len(cv_by_mask))

        inst = cls(cv_by_mask=cv_by_mask, default_cv=default_cv, acc_struct=acc_struct, p=p)
        inst._source_path = str(path)
        return inst

    @property
    def source_paths(self) -> list[str]:
        return [self._source_path] if self._source_path else []

    def config(self) -> dict:
        return {
            "cv_by_mask":  {k: list(v) for k, v in self.cv_by_mask.items()},
            "default_cv":  self.default_cv,
            "p":           self.p,
            "source_path": self._source_path,
        }

    def _coverage_density(self) -> np.ndarray:
        """
        Per-edge fraction of `coverage == True` over the edge's
        (sector_from, region_from) neighbourhood.

        Returns an all-ones array (and logs a warning) when no coverage
        has been attached, so downstream code degrades gracefully to the
        tight `cv_lo` end of the range.
        """
        if self._coverage is None:
            logger.warning(
                "FlatPriorImputer: no coverage attached; "
                "treating every edge as fully covered (ρ=1, CV=cv_lo). "
                "ImputationEngine should call attach_agg_result() before impute()."
            )
            return np.ones(self._n_edges, dtype=float)

        cov = self._coverage.astype(float)
        sums   = np.bincount(self._density_group_id, weights=cov,
                             minlength=self._n_density_groups)
        counts = np.bincount(self._density_group_id,
                             minlength=self._n_density_groups).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            density_per_group = np.where(counts > 0, sums / counts, 1.0)
        return density_per_group[self._density_group_id]

    def _resolve_cv(self) -> np.ndarray:
        """
        Per-edge CV vector with two layers:

        1.  First-match-wins assignment of the edge's mask, using
            ``(cv_lo, cv_hi)`` for that mask.
        2.  Density-aware interpolation within the masked range::

                CV(j) = cv_lo + (cv_hi − cv_lo) · (1 − ρ(j))

            where ρ(j) is computed by ``_coverage_density``.

        Edges that match no listed mask receive ``self.default_cv``
        unchanged (single value, no density modulation).
        """
        cv       = np.full(self._n_edges, self.default_cv, dtype=float)
        assigned = np.zeros(self._n_edges, dtype=bool)
        density  = self._coverage_density()  # ρ(j) in [0, 1]

        for name, mask in self._masks:
            cv_lo, cv_hi = self.cv_by_mask[name]
            new = mask & ~assigned
            if not new.any():
                continue
            cv[new] = cv_lo + (cv_hi - cv_lo) * (1.0 - density[new])
            assigned |= new
        return cv

    def _resolve_scale(self, f: np.ndarray) -> np.ndarray:
        """
        Per-edge scale via 3-level cascade — see class docstring for the
        full rule. Only finite, non-zero ``|f|`` values contribute to any
        level; upstream-imputed zeros never pull the scale down to zero.
        """
        finite_nonzero = np.isfinite(f) & (f != 0.0)
        abs_f          = np.where(finite_nonzero, np.abs(f), 0.0)
        weight         = finite_nonzero.astype(float)

        # Level 1 — narrow group
        sums1   = np.bincount(self._group_id, weights=abs_f, minlength=self._n_groups)
        counts1 = np.bincount(self._group_id, weights=weight, minlength=self._n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean1 = np.where(counts1 > 0, sums1 / counts1, 0.0)

        # Level 2 — broader (sector_from, valuation) group
        sums2   = np.bincount(self._group_id_level2, weights=abs_f, minlength=self._n_groups_level2)
        counts2 = np.bincount(self._group_id_level2, weights=weight, minlength=self._n_groups_level2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean2 = np.where(counts2 > 0, sums2 / counts2, 0.0)

        # Level 3 — global median of finite, non-zero |f|
        if finite_nonzero.any():
            global_scale = float(np.median(abs_f[finite_nonzero]))
        else:
            global_scale = 1.0  # whole array empty → trivial unit scale

        scale_l1 = mean1[self._group_id]
        scale_l2 = mean2[self._group_id_level2]

        scale = np.where(scale_l1 > 0, scale_l1,
                         np.where(scale_l2 > 0, scale_l2, global_scale))
        return scale

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        idx = np.where(missing)[0]
        if idx.size == 0:
            return []

        cv_per_edge    = self._resolve_cv()
        scale_per_edge = self._resolve_scale(f)
        sigma_target   = cv_per_edge * scale_per_edge

        f[idx]     = 0.0
        sigma[idx] = sigma_target[idx]

        desc = "no data, no imputer matched: flat prior (cv·scale)"
        return [
            _log(int(j), step, "flat_prior", [], [],
                 0.0, float(sigma_target[j]), desc, None)
            for j in idx
        ]


# ---------------------------------------------------------------------------
# Shared log-entry constructor
# ---------------------------------------------------------------------------

def _log(
    edge_id: int, step: int, method: str,
    source_edges: list[int], coefficients: list[float],
    f_after: float, sigma_after: float,
    description: str, param_id: Optional[str],
) -> dict:
    return {
        "edge_id":      edge_id,
        "step":         step,
        "method":       method,
        "source_edges": list(source_edges),
        "coefficients": list(coefficients),
        "f_before":     np.nan,
        "f_after":      float(f_after),
        "sigma_before": np.nan,
        "sigma_after":  float(sigma_after),
        "description":  description,
        "param_id":     param_id,
    }
