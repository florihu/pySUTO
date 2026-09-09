"""Unit tests for the pure trade-share weight computation (ADR 0002 option D)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_trade_shares import compute_trade_share_weights  # noqa: E402

CONC = {"Cathode": 0.9975}


def _edge(rf, rt, icsg, baci, sf="Cathode", st="FU", t=2010):
    return {"root_sector_from": sf, "root_sector_to": st,
            "root_region_from": rf, "root_region_to": rt,
            "root_time": t, "icsg": icsg, "baci_mass": baci}


def test_domestic_residual_and_conc_conversion():
    # Consumer China; usage = Σ icsg = 7393. Imports: USA 2000, Chile 1000 (Mass).
    df = pd.DataFrame([
        _edge("China", "China", 7393 / 3, np.nan),   # domestic
        _edge("USA",   "China", 7393 / 3, 2000.0),
        _edge("Chile", "China", 7393 / 3, 1000.0),
    ])
    out = compute_trade_share_weights(df, CONC)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    # import legs use conc·Mass (NOT raw Mass)
    assert w["USA"]   == pytest.approx(0.9975 * 2000)
    assert w["Chile"] == pytest.approx(0.9975 * 1000)
    # domestic = usage − Σ conc·Mass
    assert w["China"] == pytest.approx(7393 - 0.9975 * 3000)
    assert len(out) == 3


def test_untraded_producer_in_covered_group_gets_zero():
    # Peru is a fan target but BACI shows no trade → weight 0, still emitted
    # (per-consumer-group complete fallback).
    df = pd.DataFrame([
        _edge("China", "China", 7393 / 4, np.nan),
        _edge("USA",   "China", 7393 / 4, 2000.0),
        _edge("Chile", "China", 7393 / 4, 1000.0),
        _edge("Peru",  "China", 7393 / 4, 0.0),
    ])
    out = compute_trade_share_weights(df, CONC)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["Peru"] == 0.0
    assert len(out) == 4


def test_import_free_group_in_observed_region_gets_domestic_fan():
    # No BACI imports, but BACI observes China → zero imports is a real
    # observation of domestic sourcing: all-domestic fan (ADR 0018).
    df = pd.DataFrame([
        _edge("China", "China", 100.0, np.nan),
        _edge("USA",   "China", 100.0, 0.0),
    ])
    out = compute_trade_share_weights(df, CONC, {"China"})
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["China"] == pytest.approx(200.0)   # residual = usage − 0
    assert w["USA"] == 0.0
    assert len(out) == 2


def test_import_free_group_in_trade_blind_region_emits_no_rows():
    # BACI never observes Taiwan (BACI 490 → ROW, ADR 0014), so zero imports
    # carries no information → fan stays undistributed (ADR 0008 holds).
    df = pd.DataFrame([
        _edge("Taiwan", "Taiwan", 100.0, np.nan),
        _edge("USA",    "Taiwan", 100.0, 0.0),
    ])
    out = compute_trade_share_weights(df, CONC, {"China"})
    assert out.empty


def test_withheld_group_does_not_get_the_domestic_fallback():
    # ADR 0020: an import-free group in an observed region normally gets the
    # all-domestic fan; when withheld it stays undistributed as before ADR 0018.
    df = pd.DataFrame([
        _edge("China", "China", 100.0, np.nan, sf="Matte", st="CFR"),
        _edge("USA",   "China", 100.0, 0.0,    sf="Matte", st="CFR"),
    ])
    key = {("Matte", "CFR", "China", 2010)}
    assert compute_trade_share_weights(df, {"Matte": 0.525}, {"China"},
                                       withhold_fallback=key).empty
    # ...and an unrelated group in the same region is untouched
    assert not compute_trade_share_weights(df, {"Matte": 0.525}, {"China"},
                                           withhold_fallback={("Matte", "CFR",
                                                               "Peru", 2010)}).empty


def test_withholding_never_touches_baci_covered_groups():
    # A group BACI covers keeps its trade-share weights even if withheld —
    # the gate only governs the import-free fallback.
    df = pd.DataFrame([
        _edge("China", "China", 100.0, np.nan, sf="Matte", st="CFR"),
        _edge("USA",   "China", 100.0, 50.0,   sf="Matte", st="CFR"),
    ])
    out = compute_trade_share_weights(df, {"Matte": 0.525}, {"China"},
                                      withhold_fallback={("Matte", "CFR",
                                                          "China", 2010)})
    assert len(out) == 2
    assert {r.root_region_from for r in out.itertuples()} == {"China", "USA"}


def test_uncovered_group_emits_no_rows_without_observability_info():
    # Default (no observability set) preserves pre-ADR-0018 behaviour.
    df = pd.DataFrame([
        _edge("China", "China", 100.0, np.nan),
        _edge("USA",   "China", 100.0, 0.0),
    ])
    assert compute_trade_share_weights(df, CONC).empty


def test_imports_exceeding_usage_clip_domestic_to_zero():
    # BACI transit/re-export > ICSG usage → domestic residual clipped to 0.
    df = pd.DataFrame([
        _edge("Switzerland", "Switzerland", 0.05 / 2, np.nan),
        _edge("Chile",       "Switzerland", 0.05 / 2, 300.0),
    ])
    out = compute_trade_share_weights(df, CONC)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["Switzerland"] == 0.0
    assert w["Chile"] == pytest.approx(0.9975 * 300)


def test_groups_are_independent_per_year():
    # Same route, two years, different BACI → different weights, no cross-talk.
    df = pd.DataFrame([
        _edge("China", "China", 1000.0, np.nan, t=2010),
        _edge("USA",   "China", 1000.0, 500.0,  t=2010),
        _edge("China", "China", 1000.0, np.nan, t=2011),
        _edge("USA",   "China", 1000.0, 900.0,  t=2011),
    ])
    out = compute_trade_share_weights(df, CONC)
    by = {(r.root_region_from, r.root_time): r.weight for r in out.itertuples()}
    assert by[("USA", 2010)] == pytest.approx(0.9975 * 500)
    assert by[("USA", 2011)] == pytest.approx(0.9975 * 900)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
