"""Unit tests for the pure trade-share weight computation (ADR 0002 option D)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_trade_shares import (compute_trade_share_weights,  # noqa: E402
                                stock_legs_take_the_cathode_trade)

# (sector, region) keyed; "*" is the region wildcard (ADR 0029 plumbing)
CONC = {("Cathode", "*"): 0.9975}


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
    assert compute_trade_share_weights(df, {("Matte", "*"): 0.525}, {"China"},
                                       withhold_fallback=key).empty
    # ...and an unrelated group in the same region is untouched
    assert not compute_trade_share_weights(df, {("Matte", "*"): 0.525}, {"China"},
                                           withhold_fallback={("Matte", "CFR",
                                                               "Peru", 2010)}).empty


def test_withholding_never_touches_baci_covered_groups():
    # A group BACI covers keeps its trade-share weights even if withheld —
    # the gate only governs the import-free fallback.
    df = pd.DataFrame([
        _edge("China", "China", 100.0, np.nan, sf="Matte", st="CFR"),
        _edge("USA",   "China", 100.0, 50.0,   sf="Matte", st="CFR"),
    ])
    out = compute_trade_share_weights(df, {("Matte", "*"): 0.525}, {"China"},
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


# ---- ADR 0024: domestic residual bounded by the region's own production ----

CONC_C = {("Concentrate", "*"): 0.275}


def _prod(rows):
    return pd.DataFrame(rows, columns=["root_sector_from", "root_region_to", "root_time", "production_cc"])


def _india(usage=821.0, chile_mass=1000.0, peru_mass=470.0):
    # ICSG smelter use 821 kt CC; BACI imports 1470 kt gross → 404 CC at 0.275;
    # residual 417 vs 30 mined.
    n = 3
    return pd.DataFrame([
        _edge("India", "India", usage / n, np.nan, sf="Concentrate", st="Smelting"),
        _edge("Chile", "India", usage / n, chile_mass, sf="Concentrate", st="Smelting"),
        _edge("Peru",  "India", usage / n, peru_mass,  sf="Concentrate", st="Smelting"),
    ])


def test_domestic_bounded_by_production_and_remainder_spread_on_imports():
    df = _india()
    prod = _prod([("Concentrate", "India", 2010, 30.0)])
    out = compute_trade_share_weights(df, CONC_C, production=prod)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["India"] == pytest.approx(30.0)                       # not 821 − 404.25 = 416.75
    imp_sum = 0.275 * 1470.0
    scale = (821.0 - 30.0) / imp_sum
    assert w["Chile"] == pytest.approx(0.275 * 1000.0 * scale)     # BACI proportions kept
    assert w["Peru"] == pytest.approx(0.275 * 470.0 * scale)
    assert w["Chile"] / w["Peru"] == pytest.approx(1000.0 / 470.0)
    assert sum(w.values()) == pytest.approx(821.0)                  # usage conserved


def test_bound_is_inert_when_residual_within_production():
    df = _india()
    prod = _prod([("Concentrate", "India", 2010, 5000.0)])
    free = compute_trade_share_weights(df, CONC_C)
    bound = compute_trade_share_weights(df, CONC_C, production=prod)
    pd.testing.assert_frame_equal(free, bound)


def test_bound_needs_a_production_row_and_an_import_leg():
    df = _india()
    # no production row for this group → unchanged
    other = _prod([("Concentrate", "Chile", 2010, 5000.0)])
    pd.testing.assert_frame_equal(compute_trade_share_weights(df, CONC_C),
                                  compute_trade_share_weights(df, CONC_C, production=other))
    # import-free fan (ADR 0018) → all-domestic form kept even with production 30
    df0 = pd.DataFrame([
        _edge("Poland", "Poland", 477.0, np.nan, sf="Concentrate", st="Smelting"),
        _edge("Chile",  "Poland", 0.0,   np.nan, sf="Concentrate", st="Smelting"),
    ])
    prod = _prod([("Concentrate", "Poland", 2010, 427.0)])
    out = compute_trade_share_weights(df0, CONC_C, {"Poland"}, production=prod)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["Poland"] == pytest.approx(477.0)


def test_zero_production_puts_all_usage_on_imports():
    df = _india(usage=1309.0, chile_mass=2000.0, peru_mass=1195.0)   # Japan-like: mines nothing
    prod = _prod([("Concentrate", "India", 2010, 0.0)])
    out = compute_trade_share_weights(df, CONC_C, production=prod)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["India"] == 0.0
    assert w["Chile"] + w["Peru"] == pytest.approx(1309.0)


def test_bound_needs_imports_able_to_carry_the_remainder():
    # United States 2000-like: usage 996, mine 886, imports 26 kt CC. The
    # remainder (996 − 26 − 886 = 84) would need the import legs to grow
    # fourfold → the gap is not BACI's; the fan keeps the ADR 0002 form.
    df = pd.DataFrame([
        _edge("USA",    "USA", 996 / 2, np.nan,        sf="Concentrate", st="Smelting"),
        _edge("Mexico", "USA", 996 / 2, 26.0 / 0.275, sf="Concentrate", st="Smelting"),
    ])
    prod = _prod([("Concentrate", "USA", 2010, 886.0)])
    free = compute_trade_share_weights(df, CONC_C)
    bound = compute_trade_share_weights(df, CONC_C, production=prod)
    pd.testing.assert_frame_equal(free, bound)
    w = {r.root_region_from: r.weight for r in bound.itertuples()}
    assert w["USA"] == pytest.approx(996 - 26.0)
    # India-like (imports 404 ≥ remainder 387) is still bounded
    out = compute_trade_share_weights(_india(), CONC_C, production=_prod([("Concentrate", "India", 2010, 30.0)]))
    assert {r.root_region_from: r.weight for r in out.itertuples()}["India"] == pytest.approx(30.0)


# ---- ADR 0027: a netting-exhausted import sum is not an observation --------

def _ledger(tmp_path, monkeypatch, rows):
    """Point the feed registry at a directory holding a reexport_ledger.csv."""
    import build_trade_shares as B

    d = tmp_path / "feed" / "baci" / "v9"
    d.mkdir(parents=True)
    if rows is not None:
        pd.DataFrame(rows).to_csv(d / "reexport_ledger.csv", index=False)

    class _Regs:
        feed = type("_Feed", (), {"resolve": staticmethod(lambda _id: d)})()

    monkeypatch.setattr(B, "_input_registries", lambda: _Regs())
    return B


def _edges_frame():
    return pd.DataFrame([
        _edge("China", "China", 50.0, np.nan, sf="Cathode", st="FU", t=2015),
        _edge("Chile", "China", 50.0, 0.0,    sf="Cathode", st="FU", t=2015),
        _edge("China", "China", 10.0, np.nan, sf="Cathode", st="USC", t=2015),
        _edge("Chile", "China", 10.0, 0.0,    sf="Cathode", st="USC", t=2015),
        _edge("Peru",  "Peru",  50.0, np.nan, sf="Cathode", st="FU", t=2015),
        _edge("Chile", "Peru",  50.0, 0.0,    sf="Cathode", st="FU", t=2015),
    ])


def test_exhausted_groups_are_derived_from_the_ledger(tmp_path, monkeypatch):
    """Every fan target of a flagged (product, region, year) is withheld —
    cathode fans into FU, USC and MESC, so one flag must reach all of them."""
    B = _ledger(tmp_path, monkeypatch, [
        {"sector": "Cathode", "region": "China", "time": 2015,
         "imports_exhausted": True},
        {"sector": "Cathode", "region": "Peru", "time": 2015,
         "imports_exhausted": False},
    ])
    got = B.imports_exhausted_groups(_edges_frame())
    assert got == {("Cathode", "FU", "China", "2015"),
                   ("Cathode", "USC", "China", "2015")}


def test_no_ledger_means_no_withholding(tmp_path, monkeypatch):
    """An un-netted feed has no ledger; behaviour must be unchanged."""
    B = _ledger(tmp_path, monkeypatch, None)
    assert B.imports_exhausted_groups(_edges_frame()) == set()


# ---- ADR 0027 / L.2: the production bound extended to Cathode and Anode ----

def test_cathode_domestic_residual_is_bounded_by_refined_output():
    """France 2015's real shape: ICSG cathode intake 185 kt CC, BACI imports
    146.7, no refinery. Without the bound the 38.3 kt gap is booked as domestic
    cathode France never refined — the cathode twin of the India case ADR 0024
    fixed, and the reason the re-export netting alone leaves France's phantom
    refinery standing (it moves from the export side to the use side)."""
    df = pd.DataFrame([
        _edge("France", "France", 92.5, np.nan),          # usage 185 over two legs
        _edge("Chile",  "France", 92.5, 146.7 / 0.9975),  # imports 146.7 kt CC
    ])
    prod = _prod([("Cathode", "France", 2010, 0.0)])
    out = compute_trade_share_weights(df, CONC, {"France"}, production=prod)
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    assert w["France"] == 0.0, "a region with no refinery supplies itself nothing"
    assert w["Chile"] == pytest.approx(185.0), "usage conserved onto the import leg"


def test_cathode_bound_still_refuses_to_fabricate_imports():
    """ADR 0024's guard is unchanged by the extension: where the import legs
    would have to grow several-fold to carry the remainder, the gap is
    ICSG-internal and spreading it would fabricate imports instead of domestic
    supply (the United States 1997–2003 case, in cathode)."""
    df = pd.DataFrame([
        _edge("France", "France", 200.0, np.nan),   # usage 400, imports ~100
        _edge("Chile",  "France", 200.0, 100.0),
    ])
    free = compute_trade_share_weights(df, CONC, {"France"})
    bound = compute_trade_share_weights(
        df, CONC, {"France"}, production=_prod([("Cathode", "France", 2010, 0.0)]))
    pd.testing.assert_frame_equal(free, bound)


def test_cathode_bound_is_inert_for_a_real_refiner():
    df = pd.DataFrame([
        _edge("Germany", "Germany", 200.0, np.nan),
        _edge("Chile",   "Germany", 200.0, 100.0),
    ])
    free = compute_trade_share_weights(df, CONC, {"Germany"})
    bound = compute_trade_share_weights(
        df, CONC, {"Germany"}, production=_prod([("Cathode", "Germany", 2010, 900.0)]))
    pd.testing.assert_frame_equal(free, bound)


def test_raw_production_emits_an_explicit_zero_for_a_non_producer(monkeypatch):
    """The bound reads a *missing* production row as "unknown" and declines to
    fire. A raw-ICSG lookup returns nothing for a region that mines nothing, so
    without an explicit zero the guard silently un-bounds exactly the
    phantom-mine regions it exists for (measured on Round L.2: Germany's
    concentrate cell went 0 → 38,731 z² and Korea's phantom mine 149 → 2,173 kt
    when this frame carried only the rows ICSG reports)."""
    import build_baci_reexport as R
    import build_trade_shares as B

    # ICSG reports Chile's mine and nothing for Germany, as in the real feed.
    monkeypatch.setattr(
        R, "icsg_intake_production",
        lambda sectors=None: ({}, {("Concentrate", "Chile", "2015"): 500.0}, set()))
    edges = pd.DataFrame([
        _edge("Chile",   "Germany", 100.0, 50.0, sf="Concentrate", st="Smelting", t=2015),
        _edge("Germany", "Germany", 100.0, np.nan, sf="Concentrate", st="Smelting", t=2015),
        _edge("Chile",   "Chile",   100.0, np.nan, sf="Concentrate", st="Smelting", t=2015),
    ])
    out = B.icsg_raw_production(["Concentrate"], edges)
    got = {r.root_region_to: r.production_cc for r in out.itertuples()}
    assert got["Germany"] == 0.0, "a non-producer must say zero, not nothing"
    assert got["Chile"] == pytest.approx(500.0)
    assert out["root_time"].tolist() == ["2015"] * len(out), "years are strings"


# ---- ADR 0029 plumbing: a leg is priced with its EXPORTER's concentration ----

def test_each_import_leg_uses_its_own_exporters_conc():
    """Two suppliers, two grades. The CC each contributes must use ITS grade.

    Before this, one scalar per sector priced every exporter alike, so a
    high-grade supplier's copper was understated and a low-grade one's
    overstated — and the domestic residual, being `usage − Σ conc·Mass`, is a
    mixed-unit difference, so normalisation cannot undo it.
    """
    conc = {("Concentrate", "*"): 0.275,
            ("Concentrate", "Chile"): 0.32,
            ("Concentrate", "Peru"): 0.24}
    df = pd.DataFrame([
        _edge("Japan", "Japan", 100.0, np.nan, sf="Concentrate", st="Smelting"),
        _edge("Chile", "Japan", 0.0, 100.0,    sf="Concentrate", st="Smelting"),
        _edge("Peru",  "Japan", 0.0, 100.0,    sf="Concentrate", st="Smelting"),
    ])
    out = compute_trade_share_weights(df, conc).set_index("root_region_from")
    assert out.loc["Chile", "weight"] == pytest.approx(32.0)
    assert out.loc["Peru", "weight"] == pytest.approx(24.0)
    # domestic residual is usage minus BOTH legs at their own grades
    assert out.loc["Japan", "weight"] == pytest.approx(100.0 - 32.0 - 24.0)


def test_exporter_without_its_own_row_falls_back_to_the_wildcard():
    conc = {("Concentrate", "*"): 0.275, ("Concentrate", "Chile"): 0.32}
    df = pd.DataFrame([
        _edge("Japan", "Japan", 100.0, np.nan, sf="Concentrate", st="Smelting"),
        _edge("Chile", "Japan", 0.0, 100.0,    sf="Concentrate", st="Smelting"),
        _edge("Mongolia", "Japan", 0.0, 100.0, sf="Concentrate", st="Smelting"),
    ])
    out = compute_trade_share_weights(df, conc).set_index("root_region_from")
    assert out.loc["Chile", "weight"] == pytest.approx(32.0)
    assert out.loc["Mongolia", "weight"] == pytest.approx(27.5)


def test_importer_region_does_not_price_the_leg():
    """The same cargo must cost the same CC wherever it lands."""
    conc = {("Concentrate", "*"): 0.275, ("Concentrate", "Chile"): 0.32,
            ("Concentrate", "Japan"): 0.20, ("Concentrate", "China"): 0.35}
    to_japan = compute_trade_share_weights(pd.DataFrame([
        _edge("Japan", "Japan", 100.0, np.nan, sf="Concentrate", st="Smelting"),
        _edge("Chile", "Japan", 0.0, 100.0,    sf="Concentrate", st="Smelting"),
    ]), conc).set_index("root_region_from")
    to_china = compute_trade_share_weights(pd.DataFrame([
        _edge("China", "China", 100.0, np.nan, sf="Concentrate", st="Smelting"),
        _edge("Chile", "China", 0.0, 100.0,    sf="Concentrate", st="Smelting"),
    ]), conc).set_index("root_region_from")
    assert to_japan.loc["Chile", "weight"] == pytest.approx(32.0)
    assert to_china.loc["Chile", "weight"] == pytest.approx(32.0)


# ---------------------------------------------------------------------------
# Signed stock-change legs (#40)
# ---------------------------------------------------------------------------

def _belgium_mesc(icsg_total):
    """Belgium's exchange stock change of one year, fanned over its origins."""
    return pd.DataFrame([
        _edge("Belgium", "Belgium", icsg_total / 3, np.nan, st="MESC", t=2000),
        _edge("Chile",   "Belgium", icsg_total / 3, 10.0,   st="MESC", t=2000),
        _edge("Peru",    "Belgium", icsg_total / 3, 2.0,    st="MESC", t=2000),
    ])


def test_a_draw_down_fans_like_a_build_of_the_same_size():
    """Regression: only usage > 0 was emitted, so every negative MESC/USC row
    reached the concordance unweighted and was not distributed (ADR 0008):
    150 of 150 negative stock changes never entered the account.  The weights
    stay non-negative; the sign travels with the value."""
    build = compute_trade_share_weights(_belgium_mesc(19.1), CONC)
    draw  = compute_trade_share_weights(_belgium_mesc(-19.1), CONC)
    assert len(draw) == 3
    w_build = {r.root_region_from: r.weight for r in build.itertuples()}
    w_draw  = {r.root_region_from: r.weight for r in draw.itertuples()}
    assert w_draw == pytest.approx(w_build)
    assert w_draw["Chile"] == pytest.approx(0.9975 * 10.0)
    assert (draw["weight"] >= 0).all()


def test_a_zero_stock_change_still_emits_no_rows():
    """A zero has no fan shape to give; an undistributed zero and the net-stock
    seed carry the same prior and sigma (ADR 0040)."""
    assert compute_trade_share_weights(_belgium_mesc(0.0), CONC).empty


def test_a_negative_flow_into_final_use_is_not_made_positive():
    """Only the stock-change legs are signed; a flow keeps its rule."""
    df = pd.DataFrame([_edge("China", "China", -5.0, np.nan),
                       _edge("USA",   "China", -5.0, 2000.0)])
    assert compute_trade_share_weights(df, CONC).empty


def _hub_mesc(change, imports):
    """The UAE's exchange stock change of 2003: a hub that refines no cathode."""
    n = 1 + len(imports)
    rows = [_edge("United Arab Emirates", "United Arab Emirates", change / n, np.nan,
                  st="MESC", t=2003)]
    for region, mass in imports:
        rows.append(_edge(region, "United Arab Emirates", change / n, mass, st="MESC", t=2003))
    return pd.DataFrame(rows)


def _shares(out):
    w = {r.root_region_from: r.weight for r in out.itertuples()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def test_a_hub_draw_down_never_lands_on_cathode_it_did_not_produce():
    """Regression (#40 round): imports of 5 kt could not carry a 59.9 kt draw-down,
    so ADR 0024's bound did not fire and the whole change went on the hub's own
    leg; the hub's cathode balance could not release it (z 58.6 on 48 use
    totals).  A stock change is drawn from the region's cathode supply in
    proportion, and a hub supplies itself nothing."""
    df = _hub_mesc(-59.9, [("Chile", 4.0), ("Zambia", 1.0)])
    prod = _prod([("Cathode", "United Arab Emirates", 2003, 0.0)])
    out = compute_trade_share_weights(df, CONC, {"United Arab Emirates"}, production=prod)
    s = _shares(out)
    assert s["United Arab Emirates"] == 0.0
    assert s["Chile"] == pytest.approx(0.8)
    assert s["Zambia"] == pytest.approx(0.2)


def test_a_hub_with_neither_imports_nor_production_gets_no_fan():
    """Nowhere to book the change: it stays undistributed and the net-stock
    seed prices the leg (ADR 0040)."""
    df = _hub_mesc(-59.9, [("Chile", 0.0)])
    prod = _prod([("Cathode", "United Arab Emirates", 2003, 0.0)])
    out = compute_trade_share_weights(df, CONC, {"United Arab Emirates"}, production=prod)
    assert out.empty


def test_a_producer_draws_its_stock_change_from_its_supply_mix():
    """Own production 20 kt CC, imports 4 kt Mass (3.99 kt CC): the change is
    drawn 20 : 3.99 — not imports first, which would put China's exchange
    draw-downs wholly on imported cathode although it refines three times what
    it imports."""
    df = _hub_mesc(-30.0, [("Chile", 4.0)])
    prod = _prod([("Cathode", "United Arab Emirates", 2003, 20.0)])
    out = compute_trade_share_weights(df, CONC, {"United Arab Emirates"}, production=prod)
    s = _shares(out)
    assert s["United Arab Emirates"] == pytest.approx(20.0 / (20.0 + 0.9975 * 4.0))
    assert s["Chile"] == pytest.approx(0.9975 * 4.0 / (20.0 + 0.9975 * 4.0))


def test_a_producer_without_imports_keeps_its_whole_stock_change():
    df = _hub_mesc(-30.0, [("Chile", 0.0)])
    prod = _prod([("Cathode", "United Arab Emirates", 2003, 20.0)])
    out = compute_trade_share_weights(df, CONC, {"United Arab Emirates"}, production=prod)
    assert _shares(out) == {"United Arab Emirates": 1.0, "Chile": 0.0}


def test_stock_legs_take_the_cathode_trade_into_the_region():
    """BACI books cathode trade on the FU legs only (100/0/0, #36): a stock fan
    reads its import pattern there, not from its own, now empty, BACI legs."""
    edges = pd.DataFrame([
        _edge("Chile",  "Belgium", 1.0,  500.0, st="FU",   t=2000),
        _edge("Chile",  "Belgium", -2.0, 0.0,   st="MESC", t=2000),
        _edge("Chile",  "Belgium", 0.5,  0.0,   st="USC",  t=2000),
        _edge("Peru",   "Belgium", 1.0,  80.0,  st="FU",   t=2000),
        _edge("Peru",   "Belgium", -2.0, 0.0,   st="MESC", t=2000),
        _edge("Peru",   "Belgium", 0.0,  0.0,   st="FU",   t=2001),
        _edge("Peru",   "Belgium", -1.0, 0.0,   st="MESC", t=2001),
    ])
    out = stock_legs_take_the_cathode_trade(edges)
    mass = {(r.root_region_from, r.root_sector_to, r.root_time): r.baci_mass for r in out.itertuples()}
    assert mass[("Chile", "MESC", 2000)] == 500.0
    assert mass[("Chile", "USC", 2000)] == 500.0
    assert mass[("Peru", "MESC", 2000)] == 80.0
    assert mass[("Peru", "MESC", 2001)] == 0.0
    assert mass[("Chile", "FU", 2000)] == 500.0                 # flows untouched
    assert out["icsg"].tolist() == edges["icsg"].tolist()
