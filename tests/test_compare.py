"""
test_compare.py
===============
``python -m pysuto compare`` (#157) on the toy case: a chain compared with itself, and
with a second chain of the same inputs, is identical; one perturbed array is named
with its stage, file, key and first index; a stage only one chain has is skipped; a leaf
with two children needs both leaves; the command writes nothing under the data directory.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from pysuto.__main__ import main
from pysuto.compare import CompareError, first_difference, lineage
from pysuto.tutorial import toy


def _toy_chain(root) -> dict:
    """The six toy stages into ``root``; no PNG (Kaleido needs Chrome)."""
    ids = {"feed": toy.run_stage1(root)}
    ids["imp"]     = toy.run_stage2(root, ids["feed"])
    ids["const"]   = toy.run_stage3(root, ids["imp"])
    ids["opt"]     = toy.run_stage4(root, ids["const"])
    ids["uncprop"] = toy.run_stage5(root, ids["opt"])
    ids["psut"]    = toy.run_stage6(root, ids["opt"], png=False)
    return ids


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    """Two toy chains of the same inputs in one data directory."""
    root = tmp_path_factory.mktemp("twin")
    return {"root": root, "a": _toy_chain(root), "b": _toy_chain(root)}


def _run(root, base, new, capsys):
    code = main(["compare", "--data-dir", str(root), "--base", *base, "--new", *new])
    return code, capsys.readouterr()


def _tree(root) -> dict:
    return {os.path.join(top, f): os.path.getmtime(os.path.join(top, f))
            for top, _, files in os.walk(root) for f in files}


def test_a_chain_compared_with_itself_is_identical_and_the_command_writes_nothing(
        twin, capsys):
    a, before = twin["a"], _tree(twin["root"])
    code, out = _run(twin["root"], [a["psut"], a["uncprop"]],
                     [a["psut"], a["uncprop"]], capsys)
    assert code == 0
    assert out.out.rstrip().endswith("chains identical")
    for stage in ("feed", "imp", "const", "opt", "uncprop", "psut"):
        assert f"{stage:8s}identical" in out.out
    assert _tree(twin["root"]) == before


def test_a_second_chain_of_the_same_inputs_is_bit_identical(twin, capsys):
    a, b = twin["a"], twin["b"]
    assert a["opt"] != b["opt"]
    code, out = _run(twin["root"], [a["psut"], a["uncprop"]],
                     [b["psut"], b["uncprop"]], capsys)
    assert code == 0, out.out
    assert "DIFFERENT" not in out.out


def test_one_perturbed_array_is_named_with_its_stage_file_key_and_first_index(
        twin, tmp_path, capsys):
    root = tmp_path / "copy"
    shutil.copytree(twin["root"], root)
    a, b = twin["a"], twin["b"]
    path = root / "proc" / "opt" / b["opt"] / "timesteps" / "t_2003.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["F_opt"][4] += 1e-9
    np.savez(path, **arrays)
    code, out = _run(root, [a["psut"]], [b["psut"]], capsys)
    assert code == 1
    assert "opt     DIFFERENT" in out.out
    assert "timesteps/t_2003.npz::F_opt: first at index (4,)" in out.out
    assert "(1 of 84 differ)" in out.out
    assert "chains DIFFER" in out.out
    assert "const   identical" in out.out         # the stages upstream stay identical


def test_a_dtype_a_shape_and_a_nan_position_are_each_a_difference():
    a = np.array([1.0, np.nan, 3.0])
    assert first_difference(a, a.copy()) is None
    assert "dtype" in first_difference(a, a.astype(np.float32))
    assert "shape" in first_difference(a, a[:2])
    assert "index (1,)" in first_difference(a, np.array([1.0, 2.0, 3.0]))
    assert "index (0, 1)" in first_difference(np.zeros((2, 2)),
                                              np.array([[0, 1.], [0, 0]]))
    nat = np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")
    assert first_difference(nat, nat.copy()) is None
    obj = np.array(["x", None, "z"], dtype=object)
    assert first_difference(obj, obj.copy()) is None
    assert "index (2,)" in first_difference(obj, np.array(["x", None, "y"], dtype=object))


def test_a_perturbed_parquet_column_and_a_changed_input_version_are_named(
        twin, tmp_path, capsys):
    root = tmp_path / "copy"
    shutil.copytree(twin["root"], root)
    a, b = twin["a"], twin["b"]
    log = root / "proc" / "imp" / b["imp"] / "imputation_log.parquet"
    frame = pd.read_parquet(log)
    col = next(c for c in frame.columns if frame[c].dtype.kind == "f")
    frame.loc[frame.index[1], col] = frame[col].iloc[1] + 1.0
    frame.to_parquet(log)
    meta = root / "proc" / "const" / b["const"] / "metadata.json"
    meta.write_text(meta.read_text().replace('"version": 1', '"version": 2', 1))
    code, out = _run(root, [a["psut"]], [b["psut"]], capsys)
    assert code == 1
    assert f"imputation_log.parquet::[{col}]: first at index (1,)" in out.out
    assert "const   DIFFERENT" in out.out and "parents.inputs" in out.out


def _reverse_feed_rows(path) -> list:
    """Reverse the feed order of a stored imp's per-feed rows, consistently (#211)."""
    with np.load(path, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    for key in ("F", "Sigma", "Omega", "feed_ids"):
        arrays[key] = arrays[key][::-1].copy()
    np.savez(path, **arrays)
    return list(arrays["feed_ids"])


def test_a_reordered_feed_order_of_the_imp_is_identical_and_named(twin, tmp_path, capsys):
    root = tmp_path / "copy"
    shutil.copytree(twin["root"], root)
    a, b = twin["a"], twin["b"]
    order = _reverse_feed_rows(root / "proc" / "imp" / b["imp"] / "agg_result.npz")
    code, out = _run(root, [a["psut"], a["uncprop"]], [b["psut"], b["uncprop"]], capsys)
    assert code == 0, out.out
    assert "imp     identical" in out.out
    assert "feed order differs" in out.out and str(order) in out.out
    assert out.out.rstrip().endswith("chains identical")


def test_a_changed_value_in_one_feeds_row_still_differs_after_the_alignment(
        twin, tmp_path, capsys):
    root = tmp_path / "copy"
    shutil.copytree(twin["root"], root)
    a, b = twin["a"], twin["b"]
    path = root / "proc" / "imp" / b["imp"] / "agg_result.npz"
    _reverse_feed_rows(path)
    with np.load(path, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    j = int(np.flatnonzero(np.isfinite(arrays["F"][0]))[0])
    arrays["F"][0, j] += 1.0
    np.savez(path, **arrays)
    code, out = _run(root, [a["psut"]], [b["psut"]], capsys)
    assert code == 1
    assert "imp     DIFFERENT" in out.out
    assert "agg_result.npz::F: first at index" in out.out
    assert "(1 of" in out.out
    assert "agg_result.npz::Sigma" not in out.out   # aligned rows agree


def test_a_stage_only_one_chain_has_is_skipped_and_is_no_difference(twin, capsys):
    a, b = twin["a"], twin["b"]
    code, out = _run(twin["root"], [a["psut"]], [b["psut"], b["uncprop"]], capsys)
    assert code == 0
    assert f"uncprop skipped   {b['uncprop']}: base has none" in out.out
    code, out = _run(twin["root"], [a["opt"]], [b["psut"]], capsys)
    assert code == 0
    assert "psut    skipped" in out.out


def test_a_leaf_names_its_ancestors_and_two_runs_of_one_stage_are_refused(twin):
    a, b = twin["a"], twin["b"]
    chain = lineage(str(twin["root"]), [a["psut"]])
    assert set(chain) == {"feed", "imp", "const", "opt", "psut"}
    assert sorted(chain["feed"]) == sorted(a["feed"].values())
    with pytest.raises(CompareError, match="two runs of stage 'opt'"):
        lineage(str(twin["root"]), [a["psut"], b["uncprop"]])


def test_an_unknown_run_exits_2_and_names_it(twin, capsys):
    code, out = _run(twin["root"], ["psut_20200101_000000_deadbeef"],
                     [twin["a"]["psut"]], capsys)
    assert code == 2
    assert "psut_20200101_000000_deadbeef" in out.err
