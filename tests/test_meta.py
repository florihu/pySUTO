"""
test_meta.py
============
DataFeedMetadata Excel parsing: the five pedigree scores, the optional sensitivities,
the scope mask and the ``row_penalty`` opt-in flag.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pysuto.meta import DataFeedMetadata

SCORES = [("Reliability", 3), ("Completeness", 2), ("Temporal_correlation", 4),
          ("Geographical_correlation", 1), ("Other_correlation", 2)]


def _write_meta(tmp_path, rows):
    """rows: list of (Meta_name, Value). Write a minimal sidecar xlsx."""
    df = pd.DataFrame(rows, columns=["Meta_name", "Value"])
    path = tmp_path / "meta.xlsx"
    df.to_excel(path, index=False)
    return str(path)


def _without(name):
    return [(k, v) for k, v in SCORES if k != name]


def test_row_penalty_true(tmp_path):
    path = _write_meta(tmp_path, [("Feed_name", "baci"), ("Row_penalty", "TRUE")] + SCORES)
    meta = DataFeedMetadata.from_excel(path)
    assert meta.row_penalty is True


def test_each_of_the_five_scores_is_read(tmp_path):
    meta = DataFeedMetadata.from_excel(_write_meta(tmp_path, [("Feed_name", "x")] + SCORES))
    assert (meta.reliability, meta.completeness, meta.temporal_correlation,
            meta.geographical_correlation, meta.other_correlation) == (3, 2, 4, 1, 2)


def test_score_names_are_case_insensitive_and_text_integers_are_read(tmp_path):
    rows = [("reliability", "3"), ("COMPLETENESS", 2.0), ("Temporal_correlation", 4),
            ("Geographical_correlation", 1), ("Other_correlation", 2)]
    meta = DataFeedMetadata.from_excel(_write_meta(tmp_path, rows))
    assert (meta.reliability, meta.completeness) == (3, 2)


@pytest.mark.parametrize("name", [k for k, _ in SCORES])
def test_an_absent_score_raises_instead_of_defaulting_to_the_best(tmp_path, name):
    """Regression (#280, U1a-S1-1): an absent score used to read as 1, the lowest σ."""
    path = _write_meta(tmp_path, _without(name))
    with pytest.raises(ValueError, match=name):
        DataFeedMetadata.from_excel(path)


@pytest.mark.parametrize("bad", ["good", 2.5, ""])
def test_a_score_that_is_not_an_integer_raises(tmp_path, bad):
    path = _write_meta(tmp_path, _without("Reliability") + [("Reliability", bad)])
    with pytest.raises(ValueError, match="Reliability"):
        DataFeedMetadata.from_excel(path)


def test_a_score_outside_one_to_four_raises_when_the_pedigree_is_built(tmp_path):
    path = _write_meta(tmp_path, _without("Reliability") + [("Reliability", 7)])
    meta = DataFeedMetadata.from_excel(path)
    with pytest.raises(ValueError, match="must be in"):
        meta.to_pedigree_scores()


def test_sensitivities_default_to_medium_and_are_read_per_indicator(tmp_path):
    meta = DataFeedMetadata.from_excel(_write_meta(tmp_path, SCORES))
    assert meta.completeness_sensitivity == "medium_sensitive"
    rows = SCORES + [("Completeness_sensitivity", "not_sensitive"),
                     ("geographical_sensitivity", "highly_sensitive")]
    meta = DataFeedMetadata.from_excel(_write_meta(tmp_path, rows))
    pedigree = meta.to_pedigree_scores()
    assert (pedigree.completeness_sensitivity, pedigree.geographical_sensitivity,
            pedigree.temporal_sensitivity) == (
        "not_sensitive", "highly_sensitive", "medium_sensitive")


def test_an_unknown_sensitivity_raises(tmp_path):
    path = _write_meta(tmp_path, SCORES + [("Other_sensitivity", "very")])
    with pytest.raises(ValueError, match="Other_sensitivity"):
        DataFeedMetadata.from_excel(path)


def test_the_scope_mask_is_read_and_blank_means_none(tmp_path):
    meta = DataFeedMetadata.from_excel(_write_meta(tmp_path, SCORES))
    assert meta.edge_scope_mask == ""
    rows = SCORES + [("Edge_scope_mask", " cross_region ")]
    assert DataFeedMetadata.from_excel(_write_meta(tmp_path, rows)).edge_scope_mask == \
        "cross_region"


def test_fill_computed_stamps_a_utc_timestamp_and_the_result_summary(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    meta = DataFeedMetadata(feed_name="x")
    meta.fill_computed(SimpleNamespace(cv_src=0.1, F_root=np.array([1.0, np.nan, 3.0]),
                                       sigma_root=np.array([0.5, np.nan, 1.5])))
    assert (meta.cv_total, meta.F_root_sum, meta.sigma_root_mean) == (0.1, 4.0, 1.0)
    assert pd.Timestamp(meta.computation_timestamp.rstrip("Z")).year >= 2026
    assert meta.computation_timestamp.endswith("Z")
