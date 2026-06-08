"""
coverage.py
===========
``CoverageDiagnostic`` — post-imputation system-coverage CSVs.

Emits two CSV summaries that count, per bucket, how many edges carry a *real*
value after imputation — where "real" deliberately **excludes** the flat-prior
fallback fill (``f=0``). This exposes the genuine data coverage that the filled
``f_agg`` would otherwise hide.

A "val" (``has_val``):
    An edge has a val unless its **final** imputation-log entry used a flat-prior
    fallback method (``flat_prior`` / ``flat_prior_noisy_zero``) or its ``f_agg``
    is NaN. Observed edges (never in the log) and fan/time/parametric fills all
    count as vals. "Final" = the log row with the **maximum** ``step`` for that
    edge_id (``step`` is the global, monotonic pipeline-step counter; terminal
    imputers carry ``pass=None``, so ``step`` alone is the correct order key).

File 1 — ``coverage_supply_use_<year>.csv``
    Commodity-centric supply/use coverage. Scope = edges with a ``Flow``
    endpoint. ``supply`` = the ``Flow`` is the destination (``Process→Flow``);
    ``use`` = the ``Flow`` is the origin (``Flow→Process`` *and* the final-demand
    ``Flow→FD`` edges). Columns = the ``Flow`` (commodity) sector. Rows =
    ``(counterparty-region, valuation, direction)``.

File 2 — ``coverage_system_boundary_<year>.csv``
    Raw/secondary boundary-crossing coverage. Scope = edges with an ``SB``
    endpoint whose ``entity_sub != "FD"`` (final demand lives in File 1).
    ``input`` = ``from_entity=="SB"``; ``output`` = ``to_entity=="SB"``.
    Columns = the SB ``entity_sub`` category (``Nature`` / ``SEM``). Rows =
    ``(non-SB region, valuation, direction)``.

Both files: cell = count of ``has_val`` edges, empty buckets = ``0``; restricted
to a single ``year`` with ``valuation`` carried as a row-index level.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


class CoverageDiagnostic:
    """Post-imputation coverage CSVs. See module docstring for the data model.

    Parameters
    ----------
    agg_result : AggregationResult
        Result after ``ImputationEngine.run()``; must expose ``f_agg`` and
        ``imputation_log``.
    acc_struct :
        ``AccountingStructure`` — supplies ``edge_meta`` (edge↔node join with
        ``from_*`` / ``to_*`` attributes incl. ``entity_sub``).
    year : str
        The single year both tables are restricted to (default ``"1995"``).
    """

    _NO_VAL_METHODS = {"flat_prior", "flat_prior_noisy_zero"}

    def __init__(self, agg_result, acc_struct, year: str = "1995") -> None:
        self.agg_result = agg_result
        self.acc        = acc_struct
        self.year       = str(year)
        self._edges     = self._build_edge_frame()

    # ------------------------------------------------------------------
    # Per-edge frame + has_val
    # ------------------------------------------------------------------

    def _build_edge_frame(self) -> pd.DataFrame:
        em = self.acc.edge_meta.reset_index(drop=True).copy()
        # entity_sub may be absent on older accs — guarantee both endpoints have it.
        for col in ("from_entity_sub", "to_entity_sub"):
            if col not in em.columns:
                em[col] = ""
        f = np.asarray(self.agg_result.f_agg, dtype=float)
        em["f_agg"]   = f[em["edge_id"].to_numpy()]
        em["has_val"] = self._compute_has_val(em)
        return em

    @property
    def edges(self) -> pd.DataFrame:
        """Per-edge frame with ``f_agg`` + ``has_val`` (one row per edge)."""
        return self._edges

    def _compute_has_val(self, em: pd.DataFrame) -> np.ndarray:
        finite = np.isfinite(em["f_agg"].to_numpy())
        log = self.agg_result.imputation_log.as_dataframe()
        if log.empty:
            return finite
        log = log.copy()
        log["step"] = pd.to_numeric(log["step"], errors="coerce")
        # Final fill per edge = the highest global pipeline step.
        final    = log.loc[log.groupby("edge_id")["step"].idxmax()]
        flat_ids = set(
            final.loc[final["method"].isin(self._NO_VAL_METHODS), "edge_id"]
            .astype(int)
            .tolist()
        )
        no_val_method = em["edge_id"].astype(int).isin(flat_ids).to_numpy()
        return finite & ~no_val_method

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def supply_use_table(self) -> pd.DataFrame:
        """File 1 — supply/use coverage keyed by commodity (Flow) sector."""
        e = self._edges[self._edges["time"].astype(str) == self.year].copy()
        from_flow = e["from_entity"].to_numpy() == "Flow"
        to_flow   = e["to_entity"].to_numpy() == "Flow"
        e = e[from_flow ^ to_flow].copy()            # exactly one Flow endpoint
        to_flow = e["to_entity"].to_numpy() == "Flow"
        e["direction"] = np.where(to_flow, "supply", "use")
        e["commodity"] = np.where(to_flow, e["to_sector"],  e["from_sector"])
        e["region"]    = np.where(to_flow, e["from_region"], e["to_region"])
        return self._pivot(e, "commodity")

    def system_boundary_table(self) -> pd.DataFrame:
        """File 2 — boundary coverage keyed by SB entity_sub (Nature/SEM)."""
        e = self._edges[self._edges["time"].astype(str) == self.year].copy()
        from_sb = e["from_entity"].to_numpy() == "SB"
        to_sb   = e["to_entity"].to_numpy() == "SB"
        e = e[from_sb ^ to_sb].copy()                # exactly one SB endpoint
        from_sb = e["from_entity"].to_numpy() == "SB"
        e["category"] = np.where(from_sb, e["from_entity_sub"], e["to_entity_sub"])
        e = e[e["category"] != "FD"].copy()          # FD belongs to File 1
        from_sb = e["from_entity"].to_numpy() == "SB"
        e["direction"] = np.where(from_sb, "input", "output")
        e["region"]    = np.where(from_sb, e["to_region"], e["from_region"])
        return self._pivot(e, "category")

    @staticmethod
    def _pivot(e: pd.DataFrame, col: str) -> pd.DataFrame:
        if e.empty:
            return pd.DataFrame()
        table = e.pivot_table(
            values="has_val",
            index=["region", "valuation", "direction"],
            columns=col,
            aggfunc="sum",
            fill_value=0,
        )
        return table.astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, odir: str) -> list[str]:
        """Write both CSVs into ``odir`` and return their paths."""
        os.makedirs(odir, exist_ok=True)
        p1 = os.path.join(odir, f"coverage_supply_use_{self.year}.csv")
        p2 = os.path.join(odir, f"coverage_system_boundary_{self.year}.csv")
        self.supply_use_table().to_csv(p1)
        self.system_boundary_table().to_csv(p2)
        print(f"   Coverage supply/use        → {p1}  ✓")
        print(f"   Coverage system-boundary   → {p2}  ✓")
        return [p1, p2]
