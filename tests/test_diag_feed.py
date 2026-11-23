"""
test_diag_feed.py
=================
``FeedDiagnostics`` (Stage 1): ``summary()`` and the figure's panels on a five-row raw
frame, the unmatched-reason breakdown with and without the concordance workbook, and
the valuation unit in the labels (#280, U1a-G6-7, U1a-S5-3).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # headless before any pyplot import

import numpy as np
import pandas as pd

from pysuto.diag.feed import FeedDiagnostics


def _raw():
    return pd.DataFrame({
        "Year":   ["2018", "2018", "2019", "2019", "2019"],
        "Region": ["Chile", "Peru", "Chile", "Atlantis", "Narnia"],
        "Value":  [10.0, 20.0, 30.0, 40.0, 50.0],
    })


def _diag(tmp_path=None, conc_file=None, unit=None):
    raw = _raw()
    ct = SimpleNamespace(
        # rows 3 and 4 matched no edge; the others fan onto two root edges each
        tab=pd.DataFrame({"raw_id": [0, 1, 2], "src_sector_from": ["Mining"] * 3}),
        unmatched_raw_rows=[3, 4], n_src_edges=3, n_root_edges=6)
    if unit is not None:
        ct.unit = unit
    result = SimpleNamespace(cv_src=0.05, cv_root=np.array([0.05, 0.04, np.nan, 0.0]),
                             F_root=np.array([1.0, 2.0, 0.0, 3.0]))
    return FeedDiagnostics(feed_run_id="feed_x_1", feed_id="x", acc_id="acc",
                           raw_df=raw, ct=ct, F_src=np.array([10.0, 20.0, 30.0]),
                           result=result, conc_file=conc_file)


def test_summary_counts_rows_and_value_and_names_the_top_unmatched_regions(capsys):
    text = _diag().summary()
    assert capsys.readouterr().out.strip() == text.strip()
    assert "Raw rows total  :        5" in text
    assert "Rows matched    :        3  (60.0 %)" in text
    assert "Rows unmatched  :        2  (40.0 %)" in text
    assert "Value matched   :       60.0 kt" in text
    assert "Value coverage  :   40.0 %" in text
    assert "CV_src          :   5.00 %" in text
    assert "Atlantis" in text and "Narnia" in text


def test_summary_and_labels_carry_the_feeds_valuation_unit():
    diag = _diag(unit="kt CC")
    assert "Value total     :      150.0 kt CC" in diag.summary()
    assert diag.unit == "kt CC"
    assert _diag().unit == "kt"            # a concordance without a unit reads as gross


def test_the_figure_is_written_and_logs_instead_of_printing(tmp_path, capsys, caplog):
    with caplog.at_level(logging.INFO, logger="pysuto.diag.feed"):
        path = _diag(unit="kt CC").plot(odir=str(tmp_path))
    assert path == str(tmp_path / "feed.png") and (tmp_path / "feed.png").stat().st_size > 0
    assert capsys.readouterr().out == ""
    assert any(str(tmp_path) in r.getMessage() for r in caplog.records)


def test_the_unmatched_reason_names_regions_the_workbook_does_not_map(tmp_path):
    import matplotlib.pyplot as plt
    conc = tmp_path / "conc.xlsx"
    pd.DataFrame({"src_region": ["Chile", "Peru", "Atlantis"]}).to_excel(
        conc, sheet_name="region_map", index=False)
    fig, ax = plt.subplots()
    _diag(conc_file=str(conc))._panel_unmatched_reason(ax)
    reasons = {t.get_text() for t in ax.get_yticklabels()}
    plt.close(fig)
    assert reasons == {"reason unknown", "unmapped region"}      # Atlantis mapped, Narnia not


def test_without_a_workbook_every_unmatched_row_is_reason_unknown():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    _diag()._panel_unmatched_reason(ax)
    reasons = {t.get_text() for t in ax.get_yticklabels()}
    plt.close(fig)
    assert reasons == {"reason unknown"}


def test_a_fully_matched_feed_says_so():
    import matplotlib.pyplot as plt
    diag = _diag()
    diag._unmatched_idx = set()
    fig, ax = plt.subplots()
    diag._panel_unmatched_reason(ax)
    assert any("All rows matched" in t.get_text() for t in ax.texts)
    plt.close(fig)
