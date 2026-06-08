"""
test_coverage_diag.py
=====================
Tests for :class:`pysuto.diag.CoverageDiagnostic` — the post-imputation
system-coverage CSVs.

Fixture topology (``CovAcc``) exercises every edge shape that the File 1 /
File 2 partition must separate, plus a genuinely-observed ``0.0`` flow to prove
the "no val" rule is driven by the imputation-log *method*, not by ``f == 0``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.agg import AggregationResult
from pysuto.diag import CoverageDiagnostic
from pysuto.impute import ImputationLog


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class CovAcc:
    """Minimal acc exposing ``edge_meta`` like the real AccountingStructure.

    Nodes:
        0 Mining /Process/A          4 Scrap  /SB/A (entity_sub SEM)
        1 Cathode/Flow   /A          5 Slag   /SB/A (entity_sub SEM)
        2 FU     /SB     /A (FD)     6 Anode  /Flow /A
        3 CO     /SB     /A (Nature)

    Edges (edge_id order) — all 1995/Mass unless noted:
        0 0→1            supply, commodity Cathode      f=10  observed   → val
        1 1→0            use,    commodity Cathode      f=0   flat_prior → no val
        2 1→2 (Flow→FD)  use,    commodity Cathode      f=5   observed   → val  (File 1 only)
        3 3→0 (Nature→P) input,  category  Nature       f=3   parametric → val
        4 4→0 (Scrap→P)  input,  category  SEM          f=7   observed   → val
        5 0→5 (P→Slag)   output, category  SEM          f=0   flat_prior → no val
        6 0→1  CC        supply, commodity Cathode      f=8   observed   → val
        7 0→1  Mass 1996 (other year, filtered out)     f=9   observed
        8 0→6            supply, commodity Anode         f=0   observed   → val (observed zero)
    """

    def __init__(self) -> None:
        self.node_index = pd.DataFrame(
            {
                "sector":     ["Mining", "Cathode", "FU", "CO", "Scrap", "Slag", "Anode"],
                "entity":     ["Process", "Flow", "SB", "SB", "SB", "SB", "Flow"],
                "region":     ["A", "A", "A", "A", "A", "A", "A"],
                "entity_sub": ["", "", "FD", "Nature", "SEM", "SEM", ""],
            },
            index=[0, 1, 2, 3, 4, 5, 6],
        )
        self.edge_index = pd.DataFrame(
            {
                "edge_id":   [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "from_id":   [0, 1, 1, 3, 4, 0, 0, 0, 0],
                "to_id":     [1, 0, 2, 0, 0, 5, 1, 1, 6],
                "time":      ["1995", "1995", "1995", "1995", "1995",
                              "1995", "1995", "1996", "1995"],
                "valuation": ["Mass", "Mass", "Mass", "Mass", "Mass",
                              "Mass", "CC", "Mass", "Mass"],
            }
        )

    @property
    def edge_meta(self) -> pd.DataFrame:
        ei = self.edge_index.reset_index(drop=True)
        nf = self.node_index.loc[ei["from_id"].values].reset_index(drop=True).add_prefix("from_")
        nt = self.node_index.loc[ei["to_id"].values].reset_index(drop=True).add_prefix("to_")
        return pd.concat([ei, nf, nt], axis=1)


def _agg(f_values):
    f = np.array(f_values, dtype=float)
    F = f[np.newaxis, :]
    return AggregationResult(
        f_agg=f.copy(), sigma_agg=np.ones_like(f),
        Omega=np.where(np.isfinite(F), 1.0, 0.0), F=F, Sigma=np.ones_like(F),
        feed_ids=["test"], coverage=np.isfinite(f),
    )


def _log(entries):
    log = ImputationLog()
    for e in entries:
        log.append({
            "edge_id": e["edge_id"], "step": e["step"],
            "imputer_id": e.get("imputer_id", float("nan")),
            "pass": e.get("pass", float("nan")), "method": e["method"],
            "source_edges": [], "coefficients": [],
            "f_before": np.nan, "f_after": e["f_after"],
            "sigma_before": np.nan, "sigma_after": 1.0,
            "description": "", "param_id": None,
        })
    return log


@pytest.fixture
def diag():
    acc = CovAcc()
    ar  = _agg([10.0, 0.0, 5.0, 3.0, 7.0, 0.0, 8.0, 9.0, 0.0])
    ar.imputation_log = _log([
        {"edge_id": 1, "step": 5, "method": "flat_prior", "f_after": 0.0},
        {"edge_id": 3, "step": 2, "method": "parametric", "f_after": 3.0},
        {"edge_id": 5, "step": 5, "method": "flat_prior", "f_after": 0.0},
    ])
    return CoverageDiagnostic(ar, acc_struct=acc, year="1995")


# ---------------------------------------------------------------------------
# File 1 — supply/use
# ---------------------------------------------------------------------------

def test_file1_index_levels_and_valuation(diag):
    t = diag.supply_use_table()
    assert list(t.index.names) == ["region", "valuation", "direction"]
    valuations = t.index.get_level_values("valuation").unique().tolist()
    assert "Mass" in valuations and "CC" in valuations   # valuation is indexed


def test_file1_supply_use_counts(diag):
    t = diag.supply_use_table()
    # supply Cathode Mass: e0 observed → 1
    assert t.loc[("A", "Mass", "supply"), "Cathode"] == 1
    # use Cathode Mass: e1 flat_prior (no val), e2 observed (val) → 1
    assert t.loc[("A", "Mass", "use"), "Cathode"] == 1
    # supply Cathode CC: e6 observed → 1
    assert t.loc[("A", "CC", "supply"), "Cathode"] == 1


def test_file1_includes_final_demand_use(diag):
    """Flow→FD edge (e2) shows in File 1 as use coverage of its commodity."""
    t = diag.supply_use_table()
    # The only val-bearing use/Cathode edge is the final-demand edge e2.
    assert t.loc[("A", "Mass", "use"), "Cathode"] == 1


def test_file1_observed_zero_counts_as_val(diag):
    """e8 is an observed 0.0 supply edge — counts as a val (method rule, not f==0)."""
    t = diag.supply_use_table()
    assert t.loc[("A", "Mass", "supply"), "Anode"] == 1


def test_file1_excludes_other_years(diag):
    """e7 (1996) must not inflate any 1995 bucket; only its year is dropped."""
    t = diag.supply_use_table()
    # supply Cathode Mass 1995 stays at 1 (e0), the 1996 e7 is filtered out.
    assert t.loc[("A", "Mass", "supply"), "Cathode"] == 1


def test_file1_no_sb_only_edges(diag):
    """No boundary categories (Nature/SEM/FU) leak into File 1 columns."""
    t = diag.supply_use_table()
    assert set(t.columns) <= {"Cathode", "Anode"}


# ---------------------------------------------------------------------------
# File 2 — system boundary
# ---------------------------------------------------------------------------

def test_file2_columns_are_nature_sem_only(diag):
    t = diag.system_boundary_table()
    assert set(t.columns) == {"Nature", "SEM"}
    assert "FD" not in t.columns


def test_file2_input_output_counts(diag):
    t = diag.system_boundary_table()
    # input Nature: e3 parametric → val
    assert t.loc[("A", "Mass", "input"), "Nature"] == 1
    # input SEM: e4 observed Scrap→Process → val
    assert t.loc[("A", "Mass", "input"), "SEM"] == 1
    # output SEM: e5 flat_prior → no val → empty bucket renders as 0
    assert t.loc[("A", "Mass", "output"), "SEM"] == 0


def test_file2_excludes_final_demand(diag):
    """The Flow→FD edge (e2) is absent from File 2 (it lives in File 1)."""
    t = diag.system_boundary_table()
    assert "FD" not in t.columns
    # FD edges would have direction 'output'; only SEM output exists here.
    outputs = t.xs("output", level="direction").columns.tolist()
    assert "FD" not in outputs


def test_file2_empty_buckets_are_zero_int(diag):
    t = diag.system_boundary_table()
    assert t.values.dtype.kind == "i"          # integer counts, not float/NaN
    assert (t.values >= 0).all()


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------

def test_write_emits_both_csvs(diag, tmp_path):
    paths = diag.write(str(tmp_path))
    assert len(paths) == 2
    names = {p.split("/")[-1] for p in paths}
    assert names == {
        "coverage_supply_use_1995.csv",
        "coverage_system_boundary_1995.csv",
    }
    for p in paths:
        assert pd.read_csv(p).shape[0] > 0
