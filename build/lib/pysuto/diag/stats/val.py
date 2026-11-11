"""
val.py
======
``StatsDiagnostic`` — post-imputation per-bucket val *distribution* CSVs.

Where :class:`CoverageDiagnostic` counts how many real vals exist per bucket, this
diagnostic summarises *what those vals look like*: their distribution per
``(sector, region, valuation)`` group, for a single ``year``.

A "val" keeps the branch meaning (see :class:`CoverageDiagnostic`): a post-imputation
``f_agg`` that is **not** a flat-prior default fill and is not NaN — i.e. ``has_val``.
For the distribution every non-val edge is treated as **missing** (``val_value = NaN``):
non-vals are excluded from the percentile/mean stats but **counted**, so ``val_pct``
reports val-coverage.

``val_pct`` is **not** the glossary ``Coverage`` (= observed by a feed); it is the looser
val-fraction that also credits fan/time/parametric/formula imputations — only flat-prior
fills and NaN are excluded.

Two CSVs, differing only in which endpoint defines the group:

File A — ``stats_from_<year>.csv``
    Grouped by the **from** ``(sector, region)`` + ``valuation``.
File B — ``stats_to_<year>.csv``
    Grouped by the **to** ``(sector, region)`` + ``valuation``.

Scope: **all** ``year`` edges (no supply/use or SB filtering); each edge contributes its
``f_agg`` to whichever from/to endpoint bucket it lands in.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._core import grouped_stats
from .coverage import CoverageDiagnostic


class StatsDiagnostic:
    """Per-bucket val-distribution CSVs. See module docstring for the data model.

    Parameters
    ----------
    agg_result : AggregationResult
        Result after ``ImputationEngine.run()`` (``f_agg`` + ``imputation_log``).
    acc_struct :
        ``AccountingStructure`` — supplies ``edge_meta``.
    year : str
        The single year both tables are restricted to (default ``"1995"``).
    """

    def __init__(self, agg_result, acc_struct, year: str = "1995") -> None:
        self._cov = CoverageDiagnostic(agg_result, acc_struct, year=year)
        self.year = str(year)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _stats_table(self, side: str) -> pd.DataFrame:
        """Distribution stats grouped by the ``side`` (``"from"``/``"to"``) endpoint."""
        e = self._cov.edges
        e = e[e["time"].astype(str) == self.year].copy()
        if e.empty:
            return pd.DataFrame()

        e["val_value"] = np.where(e["has_val"].to_numpy(), e["f_agg"].to_numpy(), np.nan)
        return grouped_stats(e, side, "val_value")

    def stats_from_table(self) -> pd.DataFrame:
        """File A — stats grouped by the ``from`` (sector, region)."""
        return self._stats_table("from")

    def stats_to_table(self) -> pd.DataFrame:
        """File B — stats grouped by the ``to`` (sector, region)."""
        return self._stats_table("to")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, odir: str) -> list[str]:
        """Write both CSVs into ``odir`` and return their paths."""
        os.makedirs(odir, exist_ok=True)
        p1 = os.path.join(odir, f"stats_from_{self.year}.csv")
        p2 = os.path.join(odir, f"stats_to_{self.year}.csv")
        self.stats_from_table().to_csv(p1)
        self.stats_to_table().to_csv(p2)
        print(f"   Stats from (sector,region) → {p1}  ✓")
        print(f"   Stats to   (sector,region) → {p2}  ✓")
        return [p1, p2]
