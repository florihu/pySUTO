"""
test_row_penalty.py
===================
Stage-1 ROW pedigree penalty: the feedpipe helper that turns a feed's
``row_penalty`` flag + the accounting structure into a per-root-edge cv
multiplier (and provenance summary).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pysuto.meta import DataFeedMetadata
from pysuto.unc import UncertaintyMixin
from pysuto.feedpipe import _build_row_penalty_scale, propagate_with_row_penalty


class _MixinWM(UncertaintyMixin):
    """Dense-W weight-matrix stand-in driven by the real UncertaintyMixin."""

    def __init__(self, W):
        self.weight_matrix = np.asarray(W, dtype=float)


def _acc():
    """Minimal acc stub: node_index has a 'region' column; edge_index has
    from_id/to_id. 3 edges — domestic, to-ROW, from-ROW."""
    class _Acc:
        pass

    a = _Acc()
    a.node_index = pd.DataFrame({"region": ["A", "A", "ROW"]}, index=[0, 1, 2])
    a.edge_index = pd.DataFrame({"from_id": [0, 0, 2], "to_id": [1, 2, 1]})
    return a


def test_off_returns_none():
    meta = DataFeedMetadata(feed_name="x", row_penalty=False)
    assert _build_row_penalty_scale(_acc(), meta) == (None, None, None)


def test_marks_row_edges_and_scales_by_ratio():
    meta = DataFeedMetadata(feed_name="x", geographical_correlation=1,
                            row_penalty=True)
    cv_scale, summary, mask = _build_row_penalty_scale(_acc(), meta)

    # edge 0 domestic untouched; edges 1 (->ROW) and 2 (ROW->) flagged
    np.testing.assert_array_equal(mask, [False, True, True])

    pg_norm = meta.to_pedigree_scores()
    ratio = (dataclasses.replace(pg_norm, geographical_correlation=4).total_cv()
             / pg_norm.total_cv())
    np.testing.assert_allclose(cv_scale, [1.0, ratio, ratio])

    assert summary["row_penalty"]["n_penalized"] == 2
    assert summary["row_penalty"]["dim"] == "geographical_correlation"


def test_propagate_inflates_row_edges_and_records_provenance():
    """End-to-end Stage-1: with row_penalty on, ROW edges' cv is inflated by the
    geo->4 ratio, non-ROW edges are untouched, and the result records which
    edges were penalised (only those this feed actually carries)."""
    acc = _acc()
    wm = _MixinWM(np.eye(3))                       # F_root = F_src, one source per root
    meta = DataFeedMetadata(feed_name="x", geographical_correlation=1,
                            row_penalty=True)
    F_src = np.array([10.0, 20.0, 30.0])

    result, summary = propagate_with_row_penalty(wm, F_src, meta, acc)

    # provenance: ROW edges (1,2) with non-zero flow flagged; domestic (0) not
    np.testing.assert_array_equal(result.row_penalized, [False, True, True])

    ratio = summary["row_penalty"]["ratio"]
    assert ratio > 1.0
    # ROW edge cv inflated by exactly `ratio` relative to the domestic edge
    assert result.cv_root[1] / result.cv_root[0] == pytest.approx(ratio)
    assert result.cv_root[2] / result.cv_root[0] == pytest.approx(ratio)
    # value share = (20+30)/(10+20+30)
    assert summary["row_penalty"]["value_share"] == pytest.approx(50.0 / 60.0)


def test_propagate_off_leaves_result_unpenalised():
    acc = _acc()
    wm = _MixinWM(np.eye(3))
    meta = DataFeedMetadata(feed_name="x", row_penalty=False)
    result, summary = propagate_with_row_penalty(wm, np.array([10.0, 20.0, 30.0]),
                                                 meta, acc)
    assert summary is None
    assert result.row_penalized is None
