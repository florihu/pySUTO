"""
formula.py
==========
Formula imputer — directed multi-source balance equations.

* :class:`ParamRef`        — reference to a params-CSV value (optional inverse).
* :class:`SourceTerm`      — one additive term in a :class:`FormulaRelation`.
* :class:`FormulaRelation` — directed multi-source balance equation.
* :class:`FormulaImputer`  — fills missing edges via :class:`FormulaRelation`
                              balance equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from pysuto.logging import get_logger
from pysuto.lp import lp_combine
from pysuto.param_relation import get_edge_cache

from .base import BaseImputer, _apply_cv_cap, _log

logger = get_logger(__name__)


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
    missing         NaN-source policy: "block" (default) aborts the fill when
                    any matched source value is missing; "zero" treats missing
                    sources as 0 (GLOSSARY.md "missing counts as 0" — for
                    subtract/export terms whose absence means "no such flow
                    observed", not "unknown"). Structural failures (aggregate
                    "single" matching ≠ 1 edge) always block.
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
    missing:        Literal["block", "zero"]       = "block"


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
        id: Optional[str] = None,
    ) -> None:
        self.id = id or type(self).__name__
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
        ei_exp, _, _, _, mask_cache, _ = get_edge_cache(acc_struct)
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
        df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype=str)
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
                    if term.missing == "zero":
                        # NaN sources count as 0 (contribute nothing) — the
                        # structural single-edge check above still applies.
                        s_ids = [s for s in s_ids if not np.isnan(f[s])]
                    elif any(np.isnan(f[s]) for s in s_ids):
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

                s_val = lp_combine(lp_terms, self.p)
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

