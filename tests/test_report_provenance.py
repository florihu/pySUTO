"""The report scripts read their runs from sidecars, not from typed constants (#221).

``table3_intervals.csv`` names its opt and uncprop runs beside it and ``manuscript_numbers.py``
refuses a stale one; its base runs default to ``numbers.json``'s ``tolerance_pair``; the
Fig. 5d legend takes its tolerances from the pair's sidecars. Toy inputs in ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "report"))
import manuscript_numbers as M  # noqa: E402
import results_figs as RF  # noqa: E402
import table3_intervals as T  # noqa: E402


def _intervals_csv(tmp_path, opt, uncprop):
    csv = tmp_path / "table3_intervals.csv"
    csv.write_text("row\n")
    T.write_run_record(csv, opt, uncprop)
    return csv


def test_an_intervals_csv_of_the_same_opt_run_and_one_of_its_uncprop_runs_passes(tmp_path):
    csv = _intervals_csv(tmp_path, "opt_a", "uncprop_1")
    M.check_intervals_run(csv, "opt_a", ["uncprop_0", "uncprop_1"])


def test_an_intervals_csv_that_names_another_opt_run_is_refused(tmp_path):
    csv = _intervals_csv(tmp_path, "opt_old", "uncprop_1")
    with pytest.raises(ValueError, match="opt_old"):
        M.check_intervals_run(csv, "opt_new", ["uncprop_1"])


def test_an_intervals_csv_of_an_uncprop_run_not_built_from_the_opt_run_is_refused(tmp_path):
    csv = _intervals_csv(tmp_path, "opt_a", "uncprop_other")
    with pytest.raises(ValueError, match="uncprop_other"):
        M.check_intervals_run(csv, "opt_a", ["uncprop_1"])


def test_an_intervals_csv_that_records_no_run_is_refused(tmp_path):
    csv = tmp_path / "table3_intervals.csv"
    csv.write_text("row\n")
    with pytest.raises(ValueError, match="re-run it"):
        M.check_intervals_run(csv, "opt_a", ["uncprop_1"])


def test_the_two_scripts_agree_on_the_record_beside_the_csv():
    assert T.RUN_SUFFIX == M.INTERVALS_RUN_SUFFIX


def test_the_intervals_csv_resolves_from_the_repository_not_the_working_directory(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (M.REPO / M.INTERVALS_CSV).parent.is_dir()


def test_the_base_runs_default_to_the_tolerance_pair_of_numbers_json(tmp_path):
    nj = tmp_path / "numbers.json"
    nj.write_text(json.dumps({"tolerance_pair": ["opt_old", "opt_new"]}))
    assert M.resolve_base_runs(None, None, nj) == ("opt_old", "opt_new")
    assert M.resolve_base_runs("opt_x", None, nj) == ("opt_x", "opt_new")
    assert M.resolve_base_runs("opt_x", "opt_y", tmp_path / "absent.json") == ("opt_x", "opt_y")


def test_no_base_runs_without_a_tolerance_pair_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tolerance_pair"):
        M.resolve_base_runs(None, None, tmp_path / "absent.json")


def _opt_sidecar(proc, run, cfg):
    (proc / run).mkdir(parents=True)
    (proc / run / "metadata.json").write_text(json.dumps({"stage_config": cfg}))


def test_the_tolerance_pair_label_reads_each_runs_recorded_gap(tmp_path):
    _opt_sidecar(tmp_path, "opt_old", {})                  # before ADR 0039: no record
    _opt_sidecar(tmp_path, "opt_new", {"tol_gap_rel": 1e-6})
    assert RF.tolerance_pair_label(["opt_old", "opt_new"], str(tmp_path)) == "gap 1e-3 → 1e-6"
    _opt_sidecar(tmp_path, "opt_mid", {"tol_gap_rel": 1e-8})
    assert RF.tolerance_pair_label(["opt_new", "opt_mid"], str(tmp_path)) == "gap 1e-6 → 1e-8"


def test_a_figure_is_saved_as_png_and_as_a_pdf_with_embedded_truetype_fonts(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("toy")
    png = tmp_path / "fig_toy.png"
    RF._save(fig, png)
    plt.close(fig)
    pdf = tmp_path / "fig_toy.pdf"
    assert png.stat().st_size > 0
    raw = pdf.read_bytes()
    assert raw.startswith(b"%PDF")
    assert b"/FontFile2" in raw          # Type 42 (TrueType) is embedded, not Type 3
