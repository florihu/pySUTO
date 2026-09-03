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
            "imputer_id":   fill.get("imputer_id", float("nan")),
            "pass":         fill.get("pass", float("nan")),
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

    # year=None → all-year view, where the dashed 1995 reference overlay applies
    # (under the default year="1995" focus the overlay is redundant and skipped).
    diag = ImpDiagnostic(agg, acc_struct=acc, label="t", year=None)
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
        f_agg=f, sigma_agg=s,
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
        f_agg=f, sigma_agg=s,
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


# ---------------------------------------------------------------------------
# 7. Year focus + sector/region |f| boxplots
# ---------------------------------------------------------------------------

class _MetaAcc:
    """Accounting-structure stub exposing ``edge_meta`` (sector/region) plus a
    1996 edge so the 1995 focus has something to exclude."""

    def __init__(self):
        self.edge_index = pd.DataFrame({
            "edge_id":   [0, 1, 2, 3, 4, 5],
            "from_id":   [0, 0, 1, 1, 0, 0],
            "to_id":     [1, 1, 0, 0, 1, 1],
            "time":      ["1995", "1995", "1995", "1995", "1996", "1996"],
            "valuation": ["Mass", "CC", "Mass", "CC", "Mass", "Mass"],
        })
        self.n_edges = 6
        self.time_range = ["1995", "1996"]

    def decode_node(self, nid):
        return ("Mining", "Process", "A")

    @property
    def edge_meta(self):
        return self.edge_index.assign(
            from_sector=["Mining", "Mining", "Ore", "Ore", "Mining", "Mining"],
            to_sector=["Ore", "Ore", "Mining", "Mining", "Ore", "Ore"],
            from_region=["A", "A", "B", "B", "A", "A"],
            to_region=["B", "B", "A", "A", "B", "B"],
        )


def _meta_agg_with_empty_log():
    from pysuto.impute import ImputationLog
    f = np.arange(1.0, 7.0)          # 6 finite flows
    s = np.full(6, 0.5)
    agg = _agg_result(f, s)
    agg.imputation_log = ImputationLog()
    return agg


def test_year_filter_restricts_to_year():
    from pysuto.diag import ImpDiagnostic

    acc = _MetaAcc()
    agg = _meta_agg_with_empty_log()

    d95 = ImpDiagnostic(agg, acc_struct=acc, year="1995")
    assert d95._year_ids == {0, 1, 2, 3}
    assert int(d95._year_mask.sum()) == 4

    # year=None → no filter at all.
    d_all = ImpDiagnostic(agg, acc_struct=acc, year=None)
    assert d_all._year_mask is None and d_all._year_ids is None

    # A year absent from the data falls back to all-year (no empty panels).
    d_missing = ImpDiagnostic(agg, acc_struct=acc, year="2050")
    assert d_missing._year_mask is None and d_missing._year_ids is None


def test_sector_region_boxplots_produced(tmp_path):
    from pysuto.diag import ImpDiagnostic

    acc = _MetaAcc()
    agg = _meta_agg_with_empty_log()
    paths = ImpDiagnostic(agg, acc_struct=acc, label="t", year="1995").plot(
        odir=str(tmp_path), show=False,
    )
    names = {os.path.basename(p) for p in paths}
    for fn in (
        "imputation_box_from_sector.png", "imputation_box_to_sector.png",
        "imputation_box_from_region.png", "imputation_box_to_region.png",
    ):
        assert fn in names, f"missing {fn}"
        assert os.path.getsize(os.path.join(str(tmp_path), fn)) > 0


def test_sector_region_boxplots_skipped_without_edge_meta(acc, tmp_path):
    """MockAcc has no edge_meta → the sector/region boxplots are not emitted."""
    from pysuto.diag import ImpDiagnostic

    fills = [
        {"edge_id": 0, "step": 1, "method": "time_linear", "f_after": 10.0, "sigma_after": 1.0},
    ]
    acc.time_range  = ["2020", "2021", "2022"]
    acc.decode_node = lambda nid: ("Mining", "Process", "A")
    paths = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=acc).plot(
        odir=str(tmp_path), show=False,
    )
    names = {os.path.basename(p) for p in paths}
    assert not any(n.startswith("imputation_box_") for n in names)


# ---------------------------------------------------------------------------
# 8. Canonical short step ids
# ---------------------------------------------------------------------------

def test_step_ids_use_imputer_and_pass(acc):
    """With imputer_id + pass on the log, step ids read ``sN <short>·pP``."""
    from pysuto.diag import ImpDiagnostic

    fills = [
        {"edge_id": 0, "step": 0, "imputer_id": "FormulaImputer", "pass": 1,
         "method": "formula", "f_after": 10.0, "sigma_after": 1.0},
        {"edge_id": 1, "step": 1, "imputer_id": "TimeImputer",
         "method": "time_linear", "f_after": 5.0, "sigma_after": 0.5},
    ]
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=None, label="t")
    assert diag._step_ids[0] == "s0 formula·p1"   # cascade step shows pass
    assert diag._step_ids[1] == "s1 time"         # terminal step has no pass


def test_step_ids_fall_back_to_method(acc):
    """Without imputer_id the step id falls back to ``sN <dominant-method>``."""
    from pysuto.diag import ImpDiagnostic

    fills = [
        {"edge_id": 0, "step": 1, "method": "time_linear", "f_after": 10.0, "sigma_after": 1.0},
    ]
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=None, label="t")
    assert diag._step_ids[1] == "s1 time_linear"


def test_sigma_cv_boxplots_keyed_by_step(acc):
    """σ/CV boxplots show 'covered' + one box per step (not per method)."""
    from pysuto.diag import ImpDiagnostic

    fills = [
        {"edge_id": 0, "step": 0, "imputer_id": "FormulaImputer", "pass": 1,
         "method": "formula", "f_after": 10.0, "sigma_after": 1.0},
        {"edge_id": 1, "step": 1, "imputer_id": "TimeImputer",
         "method": "time_linear", "f_after": 5.0, "sigma_after": 0.5},
    ]
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=None, label="t")

    for fig_fn in (diag._fig2_sigma_box, diag._fig_cv_box):
        fig = fig_fn()
        assert fig is not None
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        # Boxes are keyed by step id, not raw method name. (No 'covered'
        # baseline here: every edge in the fixture is imputed, so the
        # pre-snapshot carries no observed σ.)
        assert "s0 formula·p1" in labels
        assert "s1 time" in labels
        assert "time_linear" not in labels
        assert "formula" not in labels


def test_imputed_count_panel_uses_step_ids(acc):
    from pysuto.diag import ImpDiagnostic
    import matplotlib.pyplot as plt

    fills = [
        {"edge_id": 0, "step": 0, "imputer_id": "FormulaImputer", "pass": 1,
         "method": "formula", "f_after": 10.0, "sigma_after": 1.0},
    ]
    diag = ImpDiagnostic(_agg_with_log(acc, fills), acc_struct=None, label="t")
    fig, ax = plt.subplots()
    diag._panel_imputed_count(ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "s0 formula·p1" in labels
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Region boxplots stack valuations as rows
# ---------------------------------------------------------------------------

def test_region_boxplots_stack_as_rows_sector_stays_cols():
    """Region |f| boxplots are 2 rows × 1 col; sector plots stay 1 row × 2 cols."""
    from pysuto.diag import ImpDiagnostic

    acc = _MetaAcc()
    agg = _meta_agg_with_empty_log()
    diag = ImpDiagnostic(agg, acc_struct=acc, label="t", year="1995")

    fig_region = diag._fig_box_from_region()
    rows, cols = fig_region.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (2, 1)

    fig_sector = diag._fig_box_from_sector()
    rows, cols = fig_sector.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (1, 2)


def test_agg_root_by_year_ignores_year_filter(tmp_path):
    """The 1995 focus restricts agg_distributions but agg_root_by_year stays
    all-year and is still emitted."""
    from pysuto.diag import AggDiagnostic

    acc = _MetaAcc()
    f = np.arange(1.0, 7.0)
    s = np.full(6, 0.5)
    agg = _agg_result(f, s)
    paths = AggDiagnostic(agg, acc_struct=acc, label="copper", year="1995").plot(
        odir=str(tmp_path), show=False,
    )
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["agg_distributions.png", "agg_root_by_year.png"]


# ---------------------------------------------------------------------------
# Cross-valuation consistency gate (report-only)
# ---------------------------------------------------------------------------

def test_cross_valuation_check_flags_impossible_pair(acc, tmp_path):
    """A feed-covered CC/Mass twin with CC > Mass (the mapped-self-trade
    poisoning signature) is reported; healthy pairs are not."""
    from pysuto.diag.imp import ImpDiagnostic

    agg = _agg_with_log(acc, [
        {"edge_id": 4, "step": 0, "method": "flat_prior",
         "f_after": 0.0, "sigma_after": 1e6},
    ])
    # edges 0/2 = Mining_A→Ore_A Mass/CC (2020); 1/3 = the B pair
    agg.f_agg[[0, 2, 1, 3]] = [0.03, 5678.0, 200.0, 100.0]
    agg.sigma_agg[[0, 2, 1, 3]] = [0.005, 389.0, 5.0, 5.0]
    agg.coverage[[0, 1, 2, 3]] = True

    out = ImpDiagnostic(agg, acc_struct=acc, label="t").cross_valuation_check(
        odir=str(tmp_path))
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["from_sector"], row["from_region"]) == ("Mining", "A")
    assert row["implied_conc"] > 1.0
    assert os.path.exists(os.path.join(str(tmp_path), "cross_valuation_check.csv"))


def test_cross_valuation_check_clean_and_uncovered_ignored(acc):
    """Consistent pairs pass; a CC>Mass pair is ignored unless BOTH sides are
    feed-covered (imputed values are not observations)."""
    from pysuto.diag.imp import ImpDiagnostic

    agg = _agg_with_log(acc, [
        {"edge_id": 4, "step": 0, "method": "flat_prior",
         "f_after": 0.0, "sigma_after": 1e6},
    ])
    agg.f_agg[[0, 2]] = [200.0, 100.0]     # healthy covered pair
    agg.sigma_agg[[0, 2]] = [5.0, 5.0]
    agg.coverage[[0, 2]] = True
    agg.f_agg[[1, 3]] = [0.03, 5678.0]     # impossible, but Mass side uncovered
    agg.sigma_agg[[1, 3]] = [0.005, 389.0]
    agg.coverage[3] = True

    out = ImpDiagnostic(agg, acc_struct=acc, label="t").cross_valuation_check()
    assert out.empty
