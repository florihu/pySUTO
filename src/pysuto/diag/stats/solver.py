"""
solver.py
=========
``SolverStatsDiagnostic`` — post-solve per-bucket CSV tables for one ``year``.

Where :class:`StatsDiagnostic` (Stage 2) tabulates the imputed-val distribution,
this Stage-4 diagnostic tabulates two solver outputs:

**Defect tables** (``defect_from_<year>.csv`` / ``defect_to_<year>.csv``) — the
per-edge *defect* ``= F_opt − a0`` (the prior departure ``a*−a₀`` of SD-T-09),
summarised per ``(sector, region, valuation)`` by the from/to endpoint. Columns
match :class:`StatsDiagnostic` plus ``median_abs_z`` where ``z = (F_opt − a0)/σ_a``
(SD-T-03). Every solved edge contributes (no masking): ``n_nan = 0``,
``val_pct = 100`` — the count columns are kept only for schema parity with the
Stage-2 tables.

**Constraint-violation table** (``constraint_violations_<year>.csv``) — one row per
**soft** constraint (free cone — the old ``sum_eq`` ∪ ``ratio_eq``) active in ``year``,
indexed by ``constraint_id``, with the stage-neutral schema ``[type, variant, valuation,
c, lhs, residual, sigma_c, z]`` (shared with :class:`ConstStatsDiagnostic`). Here ``lhs = c_star`` (=
``G·a*`` on a soft row), ``residual = lhs − c`` is the row's residual ``G·a − c`` (= the
slack the solver calls ``eps``; see README §"Cone order"), and ``z = residual/σ_c`` (SD-T-02).

Terminology: **defect** = per-edge ``a*−a₀``; **residual** = per-constraint
``G·a − c`` (reserved — see ``OptimResult.residuals``). They are distinct.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._core import grouped_stats, soft_constraint_table


class SolverStatsDiagnostic:
    """Post-solve defect + constraint-violation CSVs for a single ``year``.

    Parameters
    ----------
    results : dict[timestep -> OptimResult]
        The (reassembled) per-timestep solver results, as exposed by
        ``SolverDiag.results``. Each ``OptimResult`` supplies global-length
        ``F_opt`` and ``c_star``.
    constraint_run : ConstraintRun
        Supplies ``a0``, ``sigma_a``, ``row_meta`` and ``timestep_data``.
    acc_struct :
        ``AccountingStructure`` — supplies ``edge_meta``.
    year : str
        The single year both deliverables are restricted to (default ``"1995"``).
    """

    def __init__(self, results, constraint_run, acc_struct, year: str = "1995") -> None:
        self._res = results
        self._cr = constraint_run
        self._acc = acc_struct
        self.year = str(year)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result_for_year(self):
        """The ``OptimResult`` whose timestep matches ``self.year`` (key may be int/str)."""
        for k in self._res:
            if str(k) == self.year:
                return self._res[k]
        raise KeyError(f"no solver result for year {self.year!r} "
                       f"(have {[str(k) for k in self._res]})")

    def _timestep_data_for_year(self) -> dict:
        for k in self._cr.timestep_data:
            if str(k) == self.year:
                return self._cr.timestep_data[k]
        raise KeyError(f"no constraint timestep_data for year {self.year!r}")

    # ------------------------------------------------------------------
    # Defect tables
    # ------------------------------------------------------------------

    def _edge_frame(self) -> pd.DataFrame:
        """Per-edge frame (year-filtered) carrying ``defect`` and ``z`` columns.

        ``F_opt``/``a0``/``σ_a`` are global ``(E,)`` vectors aligned with the
        positional order of ``edge_meta`` (mirrors SD-T-09).
        """
        em = self._acc.edge_meta.reset_index(drop=True)
        res = self._result_for_year()

        f_opt = np.asarray(res.F_opt, dtype=float)
        a0 = np.asarray(self._cr.a0, dtype=float)
        sa = np.asarray(self._cr.sigma_a, dtype=float)

        defect = f_opt - a0
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sa > 0, defect / sa, np.nan)

        em = em.copy()
        em["defect"] = defect
        em["z"] = z
        return em[em["time"].astype(str) == self.year]

    def defect_from_table(self) -> pd.DataFrame:
        """Defect distribution grouped by the ``from`` (sector, region, valuation)."""
        return grouped_stats(self._edge_frame(), "from", "defect", z_col="z")

    def defect_to_table(self) -> pd.DataFrame:
        """Defect distribution grouped by the ``to`` (sector, region, valuation)."""
        return grouped_stats(self._edge_frame(), "to", "defect", z_col="z")

    # ------------------------------------------------------------------
    # Constraint-violation table
    # ------------------------------------------------------------------

    def violations_table(self) -> pd.DataFrame:
        """One row per soft constraint active in ``year`` (indexed by constraint_id).

        Solver stage: the achieved LHS is ``c_star`` (= ``G·a*`` on a soft row).
        """
        td = self._timestep_data_for_year()
        res = self._result_for_year()
        return soft_constraint_table(
            self._cr.row_meta, td["c"], td["sigma_c"], res.c_star, self.year,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, odir: str) -> list[str]:
        """Write the three CSVs into ``odir`` and return their paths."""
        os.makedirs(odir, exist_ok=True)
        p_from = os.path.join(odir, f"defect_from_{self.year}.csv")
        p_to   = os.path.join(odir, f"defect_to_{self.year}.csv")
        p_viol = os.path.join(odir, f"constraint_violations_{self.year}.csv")
        self.defect_from_table().to_csv(p_from)
        self.defect_to_table().to_csv(p_to)
        self.violations_table().to_csv(p_viol)
        print(f"   Defect from (sector,region) → {p_from}  ✓")
        print(f"   Defect to   (sector,region) → {p_to}  ✓")
        print(f"   Constraint violations       → {p_viol}  ✓")
        return [p_from, p_to, p_viol]
