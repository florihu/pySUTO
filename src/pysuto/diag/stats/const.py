"""
const.py
========
``ConstStatsDiagnostic`` — Stage-3 (constraint-build) companion to
:class:`SolverStatsDiagnostic`, emitting the **prior** soft-constraint
residual/violation CSV for one ``year``.

At the constraint stage there is no solution yet (``a* = a0``), so the per-edge
**defect** ``a*−a0`` is identically zero and is *not* emitted here — that table is
solver-only. What *is* meaningful pre-solve is the **prior residual** of each soft
constraint: ``residual = G·a0 − c`` (the misfit of the warm-start prior ``a0``
against the constraint target). This is the same quantity :class:`ConstDiagnostic`
plots as ``eps0`` (note: ConstDiagnostic uses the opposite sign ``c − G·a0``; here
we follow the codebase-reserved residual convention ``G·a − c``).

The table is built by the shared :func:`pysuto.diag.stats._core.soft_constraint_table`
with the achieved LHS set to ``G·a0`` — the exact same builder the solver stage feeds
``c_star`` into, so both stages emit one harmonised schema
``[type, valuation, c, lhs, residual, sigma_c, z]``.
"""

from __future__ import annotations

import os

import numpy as np

from ._core import soft_constraint_table


class ConstStatsDiagnostic:
    """Prior soft-constraint residual table for a single ``year``.

    Parameters
    ----------
    constraint_run : ConstraintRun
        Supplies ``G``, ``a0``, ``row_meta`` and ``timestep_data``.
    year : str
        The single year the table is restricted to (default ``"1995"``).
    """

    def __init__(self, constraint_run, year: str = "1995") -> None:
        self._cr = constraint_run
        self.year = str(year)

    def _timestep_data_for_year(self) -> dict:
        for k in self._cr.timestep_data:
            if str(k) == self.year:
                return self._cr.timestep_data[k]
        raise KeyError(f"no constraint timestep_data for year {self.year!r}")

    def violations_table(self):
        """One row per soft constraint active in ``year`` (indexed by constraint_id).

        Constraint stage: the achieved LHS is the prior ``G·a0``.
        """
        cr = self._cr
        td = self._timestep_data_for_year()
        lhs = np.asarray(cr.G.dot(np.asarray(cr.a0, dtype=float))).ravel()
        return soft_constraint_table(cr.row_meta, td["c"], td["sigma_c"], lhs, self.year)

    def write(self, odir: str) -> list[str]:
        """Write the prior-violation CSV into ``odir`` and return its path."""
        os.makedirs(odir, exist_ok=True)
        p = os.path.join(odir, f"constraint_violations_prior_{self.year}.csv")
        self.violations_table().to_csv(p)
        print(f"   Constraint violations (prior) → {p}  ✓")
        return [p]
