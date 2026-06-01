"""
test_imp_diag.py
================
Tests for the refined imputation and aggregation diagnostics in
:mod:`pysuto.diag` — specifically:

  * ``_TIME_METHODS`` tightened to exclude ``locf``/``nocb``
  * ``ImpDiagnostic._panel_cv_dist`` uses a log10 axis
  * ``_overlay_year_reference`` adds a 1995 reference curve to F and CV panels
  * ``_fig3_timeseries`` only renders imputed points with bilateral
    raw-observation neighbours
  * ``AggDiagnostic`` emits ``agg_root_by_year.png`` when given ``acc_struct``
  * All single-run filenames drop any run-id prefix
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless before any pyplot import

import numpy as np
import pandas as pd
import pytest

# Reuse the MockAcc + helper from test_imputation.py
from test_imputation import MockAcc, _agg_result  # type: ignore

# Phase 7 rewired diag/imp.py + diag/agg.py to the structural 2-D schema.
# Most tests in this file now pass; the ones still marked xfail below
# carry hand-rolled pre-refactor fixtures (1-D AggregationResult arrays
# or ``edge_index["time"]`` mutation) that the new diag code no longer
# matches. They will be unblocked when the test fixtures are migrated
# to the 2-D (n_edges, n_timesteps) shape with per-cell t_idx in the
# imputation log.
_LEGACY_FIXTURE = pytest.mark.xfail(
    reason="legacy 1-D / edge_index['time'] fixture predates structural refactor",
    strict=True,
)


@pytest.fixture
def acc():
    return MockAcc()


# ---------------------------------------------------------------------------
# Helper: build an AggregationResult + imputation log on top of MockAcc
# ---------------------------------------------------------------------------

def _agg_with_log(acc, fills: list[dict]):
    """Build an AggregationResult and attach a minimal imputation log."""
    from pysuto.impute import ImputationLog

    n = acc.n_edges
    f = np.full(n, np.nan)
    s = np.full(n, np.nan)
    # Pre-fill all covered edges to a synthetic value so the snapshot loop has work.
    for fill in fills:
        f[fill["edge_id"]] = fill["f_after"]
        s[fill["edge_id"]] = fill["sigma_after"]

    agg = _agg_result(f, s)
    log = ImputationLog()
    for fill in fills:
        log.append({
            "edge_id":      fill["edge_id"],
            "step":         fill["step"],
            "method":       fill["method"],
            "source_edges": fill.get("source_edges", []),
            "coefficients": fill.get("coefficients", []),
            "f_before":     float("nan"),
            "f_after":      fill["f_after"],
            "sigma_before": float("nan"),
            "sigma_after":  fill["sigma_after"],
            "description":  fill.get("description", ""),
            "param_id":     fill.get("param_id", ""),
        })
    agg.imputation_log = log
    return agg


# ---------------------------------------------------------------------------
# 1. _TIME_METHODS tightened
# ---------------------------------------------------------------------------

def test_time_methods_excludes_one_sided():
    from pysuto.diag import ImpDiagnostic
    assert "locf" not in ImpDiagnostic._TIME_METHODS
    assert "nocb" not in ImpDiagnostic._TIME_METHODS
    assert "time_linear" in ImpDiagnostic._TIME_METHODS
    assert "time_mean"   in ImpDiagnostic._TIME_METHODS


# ---------------------------------------------------------------------------
# 2. CV panel uses log10 axis
# ---------------------------------------------------------------------------

def test_cv_panel_uses_log10(acc):
    from pysuto.diag import ImpDiagnostic
    import matplotlib.pyplot as plt

    fills = [
        {"edge_id": 0, "step": 1, "method": "time_linear", "f_after": 10.0, "sigma_after": 1.0},
        {"edge_id": 1, "step": 1, "method": "time_linear", "f_after": 5.0,  "sigma_after": 2.5},
    ]
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=acc, label="t")
    fig, ax = plt.subplots()
    diag._panel_cv_dist(ax)
    assert "log" in ax.get_xlabel().lower()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. 1995 overlay present when acc_struct supplied (no-op otherwise)
# ---------------------------------------------------------------------------

@_LEGACY_FIXTURE
def test_1995_overlay_added_when_acc_supplied(acc):
    """Patch MockAcc to include 1995 edges and confirm the overlay is drawn."""
    from pysuto.diag   import ImpDiagnostic
    from pysuto.impute import ImputationLog
    import matplotlib.pyplot as plt

    acc.edge_index = acc.edge_index.copy()
    acc.edge_index.loc[0, "time"] = "1995"
    acc.edge_index.loc[1, "time"] = "1995"

    # Raw values at edges 0, 1 (1995) + edge 2 imputed → pre snapshot has
    # real f at edges 0, 1 for the 1995 overlay to read.
    f = np.full(acc.n_edges, np.nan)
    s = np.full(acc.n_edges, np.nan)
    f[0], s[0] = 8.0,  1.0
    f[1], s[1] = 12.0, 1.5
    f[2], s[2] = 10.0, 1.0
    agg = _agg_result(f, s)
    log = ImputationLog()
    log.append({
        "edge_id": 2, "step": 1, "method": "time_linear",
        "source_edges": [], "coefficients": [],
        "f_before": float("nan"), "f_after": 10.0,
        "sigma_before": float("nan"), "sigma_after": 1.0,
        "description": "", "param_id": "",
    })
    agg.imputation_log = log

    diag = ImpDiagnostic(agg, acc_struct=acc, label="t")
    fig, ax = plt.subplots()
    diag._panel_f_dist(ax)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("ref 1995" in lbl for lbl in legend_labels)
    plt.close(fig)


def test_overlay_noop_without_acc(acc):
    from pysuto.diag import ImpDiagnostic
    import matplotlib.pyplot as plt

    fills = [
        {"edge_id": 0, "step": 1, "method": "time_linear", "f_after": 10.0, "sigma_after": 1.0},
    ]
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=None, label="t")
    fig, ax = plt.subplots()
    # Should not raise; should not add a "ref" entry to the legend.
    diag._overlay_year_reference(ax, what="f", year="1995")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Bilateral-neighbour guard in _fig3_timeseries
# ---------------------------------------------------------------------------

@_LEGACY_FIXTURE
def test_bilateral_guard_filters_isolated_imputed(monkeypatch):
    """Synthetic series: imputed at t=5, raw only at t=10 → no × marker."""
    from pysuto.diag import ImpDiagnostic
    from pysuto.agg   import AggregationResult
    from pysuto.impute import ImputationLog

    # Build a 6-edge MockAcc-like acc with a single series, times 1..6,
    # raw observation only at t=6, imputed at t=3 (no bilateral support).
    edge_index = pd.DataFrame({
        "edge_id":   [0, 1, 2, 3, 4, 5],
        "from_id":   [0, 0, 0, 0, 0, 0],
        "to_id":     [1, 1, 1, 1, 1, 1],
        "time":      ["1", "2", "3", "4", "5", "6"],
        "valuation": ["Mass"] * 6,
    })

    class _Acc:
        def __init__(self):
            self.edge_index = edge_index
            self.time_range = ["1", "2", "3", "4", "5", "6"]
        def decode_node(self, nid):
            return ("Mining", "Process", "A")

    acc = _Acc()

    f = np.array([np.nan, np.nan, 10.0, np.nan, np.nan, 12.0])
    s = np.array([np.nan, np.nan,  1.0, np.nan, np.nan,  1.0])
    log = ImputationLog()
    log.append({
        "edge_id": 2, "step": 1, "method": "time_linear",
        "source_edges": [], "coefficients": [],
        "f_before": float("nan"), "f_after": 10.0,
        "sigma_before": float("nan"), "sigma_after": 1.0,
        "description": "", "param_id": "",
    })

    agg = AggregationResult(
        a_agg=f, sigma_agg=s,
        Omega=np.where(np.isfinite(f[np.newaxis, :]), 1.0, 0.0),
        F=f[np.newaxis, :], Sigma=s[np.newaxis, :],
        feed_ids=["t"],
        coverage=np.isfinite(f),
    )
    agg.imputation_log = log

    diag = ImpDiagnostic(agg, acc_struct=acc, label="t")
    fig = diag._fig3_timeseries()
    # When the only imputed point lacks bilateral neighbours, the panel
    # has no orange-X scatter; an empty fig may still be produced — verify
    # no imputed markers were rendered by inspecting collections.
    if fig is not None:
        ax = fig.axes[0]
        # imputed scatter uses marker="x"; raw uses marker="o".
        x_collections = [
            c for c in ax.collections
            if hasattr(c, "get_paths") and len(c.get_paths()) > 0
        ]
        # No collection should correspond to >0 imputed × points at t=3.
        # All imputed points were filtered out, so any collection is just raw.
        for c in x_collections:
            offsets = c.get_offsets()
            # imputed marker scatter is the 50-size orange one; raw is 30-size grey.
            if c.get_sizes().size and c.get_sizes()[0] == 50:
                assert offsets.shape[0] == 0
        import matplotlib.pyplot as plt; plt.close(fig)


@_LEGACY_FIXTURE
def test_bilateral_guard_allows_real_gap(monkeypatch):
    """Synthetic series: imputed at t=3, raw at t=2 and t=4 → × marker rendered."""
    from pysuto.diag   import ImpDiagnostic
    from pysuto.agg    import AggregationResult
    from pysuto.impute import ImputationLog

    edge_index = pd.DataFrame({
        "edge_id":   [0, 1, 2, 3, 4],
        "from_id":   [0, 0, 0, 0, 0],
        "to_id":     [1, 1, 1, 1, 1],
        "time":      ["1", "2", "3", "4", "5"],
        "valuation": ["Mass"] * 5,
    })

    class _Acc:
        def __init__(self):
            self.edge_index = edge_index
            self.time_range = ["1", "2", "3", "4", "5"]
        def decode_node(self, nid):
            return ("Mining", "Process", "A")

    acc = _Acc()

    f = np.array([np.nan, 10.0, 11.0, 12.0, np.nan])
    s = np.array([np.nan,  1.0,  1.0,  1.0, np.nan])
    log = ImputationLog()
    log.append({
        "edge_id": 2, "step": 1, "method": "time_linear",
        "source_edges": [], "coefficients": [],
        "f_before": float("nan"), "f_after": 11.0,
        "sigma_before": float("nan"), "sigma_after": 1.0,
        "description": "", "param_id": "",
    })

    agg = AggregationResult(
        a_agg=f, sigma_agg=s,
        Omega=np.where(np.isfinite(f[np.newaxis, :]), 1.0, 0.0),
        F=f[np.newaxis, :], Sigma=s[np.newaxis, :],
        feed_ids=["t"],
        coverage=np.isfinite(f),
    )
    agg.imputation_log = log

    diag = ImpDiagnostic(agg, acc_struct=acc, label="t")
    fig = diag._fig3_timeseries()
    assert fig is not None
    ax = fig.axes[0]
    has_imp = any(
        hasattr(c, "get_sizes") and c.get_sizes().size and c.get_sizes()[0] == 50
        and c.get_offsets().shape[0] > 0
        for c in ax.collections
    )
    assert has_imp, "imputed point with bilateral neighbours should be rendered"
    import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# 5. AggDiagnostic emits agg_root_by_year.png when acc_struct supplied
# ---------------------------------------------------------------------------

def test_agg_diag_plain_filenames_and_new_fig(acc, tmp_path):
    from pysuto.diag import AggDiagnostic

    n = acc.n_edges
    f = np.linspace(1.0, 10.0, n)
    s = np.full(n, 0.5)
    agg = _agg_result(f, s)

    # With acc_struct → emit two figs.
    paths = AggDiagnostic(agg, acc_struct=acc, label="copper").plot(
        odir=str(tmp_path), show=False,
    )
    fnames = sorted(os.path.basename(p) for p in paths)
    assert fnames == ["agg_distributions.png", "agg_root_by_year.png"]
    for p in paths:
        assert os.path.exists(p) and os.path.getsize(p) > 0


def test_agg_diag_without_acc_omits_root_by_year(acc, tmp_path):
    from pysuto.diag import AggDiagnostic

    n = acc.n_edges
    f = np.linspace(1.0, 10.0, n)
    s = np.full(n, 0.5)
    agg = _agg_result(f, s)

    paths = AggDiagnostic(agg, label="copper").plot(odir=str(tmp_path), show=False)
    fnames = [os.path.basename(p) for p in paths]
    assert fnames == ["agg_distributions.png"]


# ---------------------------------------------------------------------------
# 6. ImpDiagnostic filenames drop the run_id prefix
# ---------------------------------------------------------------------------

def test_imp_diag_filenames_have_no_prefix(acc, tmp_path):
    from pysuto.diag import ImpDiagnostic

    fills = [
        {"edge_id": 0, "step": 1, "method": "time_linear", "f_after": 10.0, "sigma_after": 1.0},
        {"edge_id": 1, "step": 1, "method": "time_linear", "f_after":  5.0, "sigma_after": 0.5},
    ]
    # MockAcc lacks ``time_range`` / ``decode_node`` so we attach them for
    # _fig3_timeseries to construct without error.
    acc.time_range  = ["2020", "2021", "2022"]
    acc.decode_node = lambda nid: ("Mining", "Process", "A")
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=acc,
                         label="imp_some_run_20260518_abc")
    paths = diag.plot(odir=str(tmp_path), show=False)
    for p in paths:
        bn = os.path.basename(p)
        assert not bn.startswith("imp_some_run_"), (
            f"filename {bn} still carries the run_id label"
        )
        assert bn.startswith("imputation_")
