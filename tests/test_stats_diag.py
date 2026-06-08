"""
test_stats_diag.py
==================
Tests for :class:`pysuto.diag.StatsDiagnostic` — the post-imputation val
*distribution* CSVs (per ``(sector, region, valuation)`` by from/to endpoint).

Reuses the ``CovAcc`` topology of ``tests/test_coverage_diag.py`` so the val rule
(method-driven ``has_val``, observed ``0.0`` is a val) is exercised identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.agg import AggregationResult
from pysuto.diag import StatsDiagnostic
from pysuto.impute import ImputationLog


# ---------------------------------------------------------------------------
# Fixture (mirrors tests/test_coverage_diag.py::CovAcc)
# ---------------------------------------------------------------------------

class CovAcc:
    """Minimal acc exposing ``edge_meta`` like the real AccountingStructure.

    Edges (edge_id order) — all 1995/Mass unless noted:
        0 Mining→Cathode   supply  f=10  observed   → val
        1 Cathode→Mining   use     f=0   flat_prior → no val
        2 Cathode→FU       use FD  f=5   observed   → val
        3 CO→Mining        input   f=3   parametric → val
        4 Scrap→Mining     input   f=7   observed   → val
        5 Mining→Slag      output  f=0   flat_prior → no val
        6 Mining→Cathode   supply CC  f=8   observed → val
        7 Mining→Cathode   1996       f=9   observed → filtered out (year)
        8 Mining→Anode     supply  f=0   observed   → val (observed zero)
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
    return StatsDiagnostic(ar, acc_struct=acc, year="1995")


_COLS = ["min", "p25", "p50", "p75", "p95", "mean", "max", "sum",
         "n_val", "n_nan", "val_pct"]


# ---------------------------------------------------------------------------
# Index / columns
# ---------------------------------------------------------------------------

def test_index_levels_and_columns(diag):
    t = diag.stats_from_table()
    assert list(t.index.names) == ["sector", "region", "valuation"]
    assert list(t.columns) == _COLS
    valuations = t.index.get_level_values("valuation").unique().tolist()
    assert "Mass" in valuations and "CC" in valuations


# ---------------------------------------------------------------------------
# Dist excludes non-vals, counts them; observed-zero is a val
# ---------------------------------------------------------------------------

def test_from_mining_mass_counts_and_stats(diag):
    """(Mining,A,Mass): e0=10(val), e5=NaN(flat_prior), e8=0(observed zero, val)."""
    t = diag.stats_from_table()
    row = t.loc[("Mining", "A", "Mass")]
    assert row["n_val"] == 2          # e0, e8 — observed zero counts
    assert row["n_nan"] == 1          # e5 flat_prior excluded from dist
    assert row["val_pct"] == pytest.approx(100.0 * 2 / 3)
    assert row["min"] == 0.0          # observed zero contributes 0.0
    assert row["max"] == 10.0
    assert row["mean"] == pytest.approx(5.0)
    assert row["sum"] == pytest.approx(10.0)   # 10 + 0 over the two vals


def test_flat_prior_only_group_kept_zero_coverage(diag):
    """to-side (Slag,A,Mass): only e5 (flat_prior) → n_val 0, NaN stats, val_pct 0."""
    t = diag.stats_to_table()
    row = t.loc[("Slag", "A", "Mass")]
    assert row["n_val"] == 0
    assert row["n_nan"] == 1
    assert row["val_pct"] == 0.0
    assert np.isnan(row["min"]) and np.isnan(row["mean"])


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

def test_percentiles_match_numpy(diag):
    t = diag.stats_from_table()
    row = t.loc[("Mining", "A", "Mass")]
    vals = np.array([10.0, 0.0])      # the two non-NaN val_values
    assert row["p25"] == pytest.approx(np.quantile(vals, 0.25))
    assert row["p50"] == pytest.approx(np.quantile(vals, 0.50))
    assert row["p75"] == pytest.approx(np.quantile(vals, 0.75))
    assert row["p95"] == pytest.approx(np.quantile(vals, 0.95))


# ---------------------------------------------------------------------------
# from vs to differ
# ---------------------------------------------------------------------------

def test_from_vs_to_differ(diag):
    """Supply edge e0 Mining→Cathode: from=(Mining,A), to=(Cathode,A)."""
    tf = diag.stats_from_table()
    tt = diag.stats_to_table()
    # The value 10 lands in the from-Mining bucket and the to-Cathode bucket.
    assert tf.loc[("Mining", "A", "Mass"), "max"] == 10.0
    assert tt.loc[("Cathode", "A", "Mass"), "max"] == 10.0
    # ...and Cathode is not a from-Mass key carrying that 10.
    assert tf.loc[("Cathode", "A", "Mass"), "max"] == 5.0   # only e2 there


# ---------------------------------------------------------------------------
# Year filter
# ---------------------------------------------------------------------------

def test_year_filter_drops_1996(diag):
    """e7 (1996, Mining→Cathode, f=9) must not appear in any 1995 bucket."""
    tt = diag.stats_to_table()
    # to-Cathode Mass would be 10 (e0) only; if e7 leaked, max would be ≥9 in CC? No —
    # e7 is Mass, to Cathode; it would push the Mass bucket's count/max. Assert it didn't.
    assert tt.loc[("Cathode", "A", "Mass"), "n_val"] == 1
    assert tt.loc[("Cathode", "A", "Mass"), "max"] == 10.0


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------

def test_write_emits_both_csvs(diag, tmp_path):
    paths = diag.write(str(tmp_path))
    assert len(paths) == 2
    names = {p.split("/")[-1] for p in paths}
    assert names == {"stats_from_1995.csv", "stats_to_1995.csv"}
    for p in paths:
        assert pd.read_csv(p).shape[0] > 0
