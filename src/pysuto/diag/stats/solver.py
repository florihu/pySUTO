"""
solver.py
=========
``SolverStatsDiagnostic`` — post-solve row-level CSV tables per ``year``.

Two deliverables per selected year:

**Edge table** (``edges_<year>.csv``) — one row per edge active in ``year``, with
full endpoint metadata plus ``a_prior`` (= ``a0``), ``a_post`` (= ``F_opt``),
``sigma_a``, the per-edge **defect** ``= a_post − a_prior`` (the prior departure
``a*−a₀`` of SD-T-09), ``defect_abs`` and ``z = defect/σ_a`` (SD-T-03; NaN where
``σ_a ≤ 0``).

**Constraint table** (``constraints_<year>.csv``) — one row per constraint row
(hard + soft) active in ``year``, built by the stage-harmonised
:func:`pysuto.diag.stats._core.constraint_table`: ``lhs_prior = G·a0``,
``lhs_post = c_star`` (= ``G·a*``), ``residual = lhs_post − c`` (= the slack the
solver calls ``eps`` on a soft row; see README §"Cone order"), ``z = residual/σ_c``
(SD-T-02). The Stage-3 twin :class:`ConstStatsDiagnostic` emits the same schema
minus ``lhs_post`` (``constraints_prior_<year>.csv``).

Terminology: **defect** = per-edge ``a*−a₀``; **residual** = per-constraint
``G·a − c`` (reserved — see ``OptimResult.residuals``). They are distinct.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._core import constraint_table

_EDGE_META_COLS = ["edge_id", "from_sector", "from_region", "from_entity",
                   "to_sector", "to_region", "to_entity", "valuation"]


def resolve_years(years, available) -> list[str]:
    """Coerce a ``years`` spec to a validated list of year strings.

    ``years`` is a sequence of year-likes (coerced via ``str()``) or the literal
    ``"all"`` → every entry of ``available`` (the run's timestep keys). An
    unknown year raises ``KeyError`` listing what is available.
    """
    avail = [str(k) for k in available]
    if isinstance(years, str):
        if years != "all":
            raise ValueError(f"years must be a sequence or 'all', got {years!r}")
        return avail
    out = [str(y) for y in years]
    unknown = [y for y in out if y not in avail]
    if unknown:
        raise KeyError(f"no timestep for year(s) {unknown} (have {avail})")
    return out


class SolverStatsDiagnostic:
    """Post-solve row-level edge + constraint CSVs per ``year``.

    Parameters
    ----------
    results : dict[timestep -> OptimResult]
        The (reassembled) per-timestep solver results, as exposed by
        ``SolverDiag.results``. Each ``OptimResult`` supplies global-length
        ``F_opt`` and ``c_star``.
    constraint_run : ConstraintRun
        Supplies ``G``, ``a0``, ``sigma_a``, ``row_meta`` and ``timestep_data``.
    acc_struct :
        ``AccountingStructure`` — supplies ``edge_meta``.
    years : sequence of str | "all"
        The years both deliverables are emitted for (default ``("1995",)``).
    """

    def __init__(self, results, constraint_run, acc_struct, years=("1995",)) -> None:
        self._res = results
        self._cr = constraint_run
        self._acc = acc_struct
        self.years = resolve_years(years, results)
        self._lhs_prior = None  # cached G·a0 (year-independent)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result_for_year(self, year: str):
        """The ``OptimResult`` whose timestep matches ``year`` (key may be int/str)."""
        for k in self._res:
            if str(k) == year:
                return self._res[k]
        raise KeyError(f"no solver result for year {year!r} "
                       f"(have {[str(k) for k in self._res]})")

    def _timestep_data_for_year(self, year: str) -> dict:
        for k in self._cr.timestep_data:
            if str(k) == year:
                return self._cr.timestep_data[k]
        raise KeyError(f"no constraint timestep_data for year {year!r}")

    def _lhs_prior_full(self) -> np.ndarray:
        """``G·a0`` over all rows — computed once, cached."""
        if self._lhs_prior is None:
            a0 = np.asarray(self._cr.a0, dtype=float)
            self._lhs_prior = np.asarray(self._cr.G.dot(a0)).ravel()
        return self._lhs_prior

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def edge_table(self, year: str) -> pd.DataFrame:
        """One row per edge with ``edge_meta.time == year``.

        ``F_opt``/``a0``/``σ_a`` are global ``(E,)`` vectors aligned with the
        positional order of ``edge_meta``.
        """
        em = self._acc.edge_meta.reset_index(drop=True)
        res = self._result_for_year(year)

        a_post  = np.asarray(res.F_opt, dtype=float)
        a_prior = np.asarray(self._cr.a0, dtype=float)
        sigma_a = np.asarray(self._cr.sigma_a, dtype=float)

        defect = a_post - a_prior
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigma_a > 0, defect / sigma_a, np.nan)

        cols = [c for c in _EDGE_META_COLS if c in em.columns]
        out = em[cols].copy()
        out["a_prior"]    = a_prior
        out["a_post"]     = a_post
        out["sigma_a"]    = sigma_a
        out["defect"]     = defect
        out["defect_abs"] = np.abs(defect)
        out["z"]          = z
        mask = em["time"].astype(str) == year
        return out[mask.to_numpy()].reset_index(drop=True)

    def constraint_table(self, year: str) -> pd.DataFrame:
        """One row per constraint row (hard + soft) active in ``year``."""
        td = self._timestep_data_for_year(year)
        res = self._result_for_year(year)
        return constraint_table(
            self._cr.row_meta, td["c"], td["sigma_c"],
            self._lhs_prior_full(), year, lhs_post=res.c_star,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, odir: str) -> list[str]:
        """Write ``edges_<year>.csv`` + ``constraints_<year>.csv`` per year."""
        os.makedirs(odir, exist_ok=True)
        paths: list[str] = []
        for year in self.years:
            p_edges = os.path.join(odir, f"edges_{year}.csv")
            p_const = os.path.join(odir, f"constraints_{year}.csv")
            self.edge_table(year).to_csv(p_edges, index=False)
            self.constraint_table(year).to_csv(p_const, index=False)
            print(f"   Edges       ({year}) → {p_edges}  ✓")
            print(f"   Constraints ({year}) → {p_const}  ✓")
            paths += [p_edges, p_const]
        return paths
