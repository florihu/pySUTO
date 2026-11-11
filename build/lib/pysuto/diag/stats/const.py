"""
const.py
========
``ConstStatsDiagnostic`` — Stage-3 (constraint-build) twin of
:class:`SolverStatsDiagnostic`, emitting the **prior** row-level constraint CSV
per ``year``.

At the constraint stage there is no solution yet (``a* = a0``), so the per-edge
**defect** ``a*−a0`` is identically zero and the edge table is solver-only. What
*is* meaningful pre-solve is each constraint's prior misfit: ``residual =
lhs_prior − c`` with ``lhs_prior = G·a0`` (the same quantity
:class:`ConstDiagnostic` plots as ``eps0``; note ConstDiagnostic uses the opposite
sign ``c − G·a0`` — here we follow the codebase-reserved residual convention
``G·a − c``).

The table is built by the shared :func:`pysuto.diag.stats._core.constraint_table`
— the exact builder the solver stage adds ``lhs_post = c_star`` to — so both
stages emit one harmonised row-level schema over **all** constraint rows
(``constraints_prior_<year>.csv`` here vs ``constraints_<year>.csv`` post-solve).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._core import constraint_table
from .solver import resolve_years


class ConstStatsDiagnostic:
    """Prior row-level constraint table per ``year``.

    Parameters
    ----------
    constraint_run : ConstraintRun
        Supplies ``G``, ``a0``, ``row_meta`` and ``timestep_data``.
    years : sequence of str | "all"
        The years the table is emitted for (default ``("1995",)``).
    """

    def __init__(self, constraint_run, years=("1995",)) -> None:
        self._cr = constraint_run
        self.years = resolve_years(years, constraint_run.timestep_data)
        self._lhs_prior = None  # cached G·a0 (year-independent)

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

    def constraint_table(self, year: str) -> pd.DataFrame:
        """One row per constraint row (hard + soft) active in ``year``."""
        td = self._timestep_data_for_year(year)
        return constraint_table(
            self._cr.row_meta, td["c"], td["sigma_c"],
            self._lhs_prior_full(), year,
        )

    def write(self, odir: str) -> list[str]:
        """Write ``constraints_prior_<year>.csv`` per year."""
        os.makedirs(odir, exist_ok=True)
        paths: list[str] = []
        for year in self.years:
            p = os.path.join(odir, f"constraints_prior_{year}.csv")
            self.constraint_table(year).to_csv(p, index=False)
            print(f"   Constraints (prior, {year}) → {p}  ✓")
            paths.append(p)
        return paths
