"""
test_icsg_feed.py
=================
The ICSG feed's observed zeros (#76, narrowed by #78, extended by #96, #97, #31): in a
region ICSG covers that produces none of a product that year, by its raw ICSG production
(ADR 0024's bound), a weighted fan's domestic leg at trade-share weight 0 and the edge the
region's own mine or SX-EW output leaves on are covered at 0, priced, and reach Stage 2 as
covered.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from icsg_feed import (OBSERVED_ZERO_FLOOR_KT, icsg_coverage,    # noqa: E402
                       load_trade_shares, non_producer_mask, observed_zero_sigma,
                       production_zero_mask, raw_production, reported_elsewhere,
                       stamp_observed_zeros)
from pysuto.collector import FeedCollector                        # noqa: E402
from pysuto.conc import ConcordanceTable, WeightMatrix            # noqa: E402
from pysuto.unc import PropagationResult                          # noqa: E402
from test_collector import _MockAccStruct, _MockRegistry          # noqa: E402
from test_conc import maps, mock_acc_struct, raw_use_only         # noqa: E402,F401

DOMESTIC, IMPORT = 6, 7          # Concentrate(A) → Refining(A), Concentrate(B) → Refining(A)


def _legs(*pairs):
    return pd.DataFrame(pairs, columns=["src_id", "edge_id"])


# ---------------------------------------------------------------------------
# observed_zero_sigma
# ---------------------------------------------------------------------------

def test_a_small_fan_prices_its_zero_at_the_floor():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([10.0]), np.zeros(3))
    assert sigma[1] == pytest.approx(0.3)
    assert np.isnan(sigma[[0, 2]]).all()


def test_a_large_fan_prices_its_zero_at_one_percent_of_its_usage():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([100.0]), np.zeros(3))
    assert sigma[1] == pytest.approx(1.0)


def test_a_leg_another_fan_feeds_is_not_an_observed_zero():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([100.0]), np.array([0.0, 2.5, 0.0]))
    assert np.isnan(sigma).all()


def test_a_leg_several_fans_weigh_at_zero_is_priced_on_their_summed_usage():
    # an aggregate region (ROW) fans each member country's usage onto the same legs
    sigma = observed_zero_sigma(_legs((0, 1), (1, 1)), np.array([50.0, 400.0]), np.zeros(2))
    assert sigma[1] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# The brief's fan, from the trade-share weights to the collector
# ---------------------------------------------------------------------------

def _builder_weights(usage, imports):
    """The trade-share weights for a region using ``usage`` and importing ``imports``
    from B: the import leg carries the BACI mass and the domestic leg the residual
    max(0, usage − imports) (ADR 0002 option D)."""
    return max(0.0, usage - imports), imports


def _collected(raw_use_only, mock_acc_struct, maps, usage, w_domestic, w_import,
               producer=False, output_zero=()):
    """Fan ``usage`` over the toy legs with these weights, stamp the observed zeros
    (A produces none of the product unless ``producer``; ``output_zero`` lists the edges
    the production-edge mask allows), and load the result through the collector at
    Stage 2's min_flow_kt."""
    uw = pd.DataFrame([
        {"root_sector_from": "Concentrate", "root_sector_to": "Refining",
         "root_region_from": r, "root_region_to": "A", "root_time": 0, "weight": w}
        for r, w in (("A", w_domestic), ("B", w_import))])
    ct = ConcordanceTable.from_raw(raw_df=raw_use_only, acc_struct=mock_acc_struct,
                                   aux_col="Supply_Use", id="t", user_weights_df=uw,
                                   drop_unweighted_fans=True, **maps)
    F_src = np.full(ct.n_src_edges, usage)
    F_root = WeightMatrix(ct).apply(F_src)
    result = PropagationResult(F_root=F_root, sigma_root=0.05 * np.abs(F_root),
                               cv_root=np.full_like(F_root, 0.05), cv_src=0.05)
    production_zero = np.zeros(len(F_root), dtype=bool)
    production_zero[list(output_zero)] = True
    n = stamp_observed_zeros(ct, F_src, result, np.full(len(F_root), not producer),
                             production_zero)
    registry = _MockRegistry({"icsg": {"F_root": result.F_root, "sigma_root": result.sigma_root,
                                       "observed_zero": result.observed_zero}})
    coll = FeedCollector(registry, _MockAccStruct(len(F_root)), min_flow_kt=0.01)
    F, Sigma = coll.select(["icsg"])._load_arrays()
    return n, F[0], Sigma[0]


def test_a_fan_whose_imports_exceed_its_usage_feeds_its_domestic_leg_as_a_zero(
        raw_use_only, mock_acc_struct, maps):
    n, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 10.0,
                             *_builder_weights(usage=10.0, imports=12.0))
    assert n == 1
    assert F[DOMESTIC] == 0.0 and Sigma[DOMESTIC] == pytest.approx(0.3)
    assert F[IMPORT] == pytest.approx(10.0)


def test_a_larger_fan_prices_its_domestic_zero_at_one_percent(raw_use_only, mock_acc_struct, maps):
    _, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 100.0,
                             *_builder_weights(usage=100.0, imports=120.0))
    assert F[DOMESTIC] == 0.0 and Sigma[DOMESTIC] == pytest.approx(1.0)


def test_a_fan_with_no_positive_weight_stays_undistributed(raw_use_only, mock_acc_struct, maps):
    # weights present but all zero: ADR 0008 drops the fan, so not even a zero is fed
    uw = pd.DataFrame([
        {"root_sector_from": "Concentrate", "root_sector_to": "Refining",
         "root_region_from": r, "root_region_to": "A", "root_time": 0, "weight": 0.0}
        for r in ("A", "B")])
    ct = ConcordanceTable.from_raw(raw_df=raw_use_only, acc_struct=mock_acc_struct,
                                   aux_col="Supply_Use", id="t", user_weights_df=uw,
                                   drop_unweighted_fans=True, **maps)
    zeros = np.zeros(ct.n_root_edges)
    result = PropagationResult(F_root=zeros, sigma_root=zeros.copy(), cv_root=zeros, cv_src=0.05)
    assert ct.tab.empty and stamp_observed_zeros(
        ct, np.zeros(0), result, np.ones(ct.n_root_edges, dtype=bool),
        np.zeros(ct.n_root_edges, dtype=bool)) == 0


def test_a_producers_zero_weight_leg_stays_open_to_the_imputer(
        raw_use_only, mock_acc_struct, maps):
    """Regression (#78): Italy's, the UK's and South Africa's domestic semis legs were
    zero only because imports exceed usage; pinning them left the regions' own refined
    output nowhere to go, and common-set sum-row z² rose 29 %."""
    n, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 10.0,
                             *_builder_weights(usage=10.0, imports=12.0), producer=True)
    assert n == 0
    assert np.isnan(F[DOMESTIC]) and np.isnan(Sigma[DOMESTIC])   # not covered: imputable


OUTPUT = 4                       # Smelting(A) → Concentrate(A): a process's product edge


def test_an_output_zero_reaches_stage_2_covered_at_the_feeds_resolution(
        raw_use_only, mock_acc_struct, maps):
    """#97, #31: a non-producer's output edge is fed by no ICSG row, so it left Stage 1
    uncovered and the imputers wrote a mine there. Stamped, it is covered at 0 with no
    fan usage to price it on: σ is the floor, ICSG's reporting resolution."""
    n, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 10.0,
                             *_builder_weights(usage=10.0, imports=8.0), producer=True,
                             output_zero=(OUTPUT,))
    assert n == 1
    assert F[OUTPUT] == 0.0 and Sigma[OUTPUT] == pytest.approx(OBSERVED_ZERO_FLOOR_KT)
    assert F[DOMESTIC] == pytest.approx(2.0)   # the fan is untouched


def test_an_output_edge_the_feed_fills_is_not_an_observed_zero(
        raw_use_only, mock_acc_struct, maps):
    _, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 10.0,
                             *_builder_weights(usage=10.0, imports=8.0), producer=True,
                             output_zero=(IMPORT,))
    assert F[IMPORT] == pytest.approx(8.0) and Sigma[IMPORT] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# icsg_coverage: who can say zero
# ---------------------------------------------------------------------------

def _region_map(*pairs):
    return pd.DataFrame(pairs, columns=["src_region", "root_region"])


def _cov(*regions, years=("2018",)):
    return {(r, y) for r in regions for y in years}


def test_a_region_icsg_lists_in_any_table_is_covered_and_an_aggregate_never_is():
    """Any ICSG row, in any table, covers its region in every year: Germany's refining
    and semis rows make its missing mine row a 0. Zimbabwe and ICSG's "Others" fold into
    ROW, a sum of members that never says zero. Luxembourg is a root region ICSG reports
    only jointly with Belgium: no row maps to it, so it is not covered either (#97)."""
    raw = pd.DataFrame({"Region": ["Germany", "Germany", "Chile", "Others  4/", "Zimbabwe",
                                   "Belgium-Luxembourg"]})
    rmap = _region_map(("Germany", "Germany"), ("Chile", "Chile"), ("Others  4/", "ROW"),
                       ("Zimbabwe", "ROW"), ("Belgium-Luxembourg", "Belgium"),
                       ("Luxembourg", "Luxembourg"))
    assert icsg_coverage(raw, rmap, ["2017", "2018"], {}) == _cov(
        "Germany", "Chile", "Belgium", years=("2017", "2018"))


def test_a_region_year_icsg_reports_inside_its_own_aggregate_is_not_covered():
    """#67: ICSG reports Sweden inside its "Scandinavia" row until 2003. A missing
    Swedish row there is a reporting gap, not a zero."""
    raw = pd.DataFrame({"Region": ["Sweden", "Scandinavia"]})
    rmap = _region_map(("Sweden", "Sweden"), ("Scandinavia", "ROW"))
    assert icsg_coverage(raw, rmap, ["2003", "2004"],
                         {("Sweden", "2003"): "Scandinavia"}) == {("Sweden", "2004")}


# ---------------------------------------------------------------------------
# non_producer_mask: the fans' domestic legs (#78, #96)
# ---------------------------------------------------------------------------

def _edges(*rows):
    return pd.DataFrame(rows, columns=["edge_id", "root_sector_from", "root_region_to",
                                       "root_time"])


def _production(*rows):
    return pd.DataFrame(rows, columns=["root_sector_from", "root_region_to", "root_time",
                                       "production_cc"])


def test_only_a_non_producers_concentrate_and_cathode_legs_may_carry_an_observed_zero():
    """Germany mines no concentrate (#32) and France refines no cathode (#96), so their
    domestic legs at weight 0 are observations; Spain mines and Italy refines, so theirs
    are residuals of imports above usage. A missing row is 0 (Chile here). Anode legs
    are not stamped, whatever the region produces."""
    edges = _edges((0, "Concentrate", "Germany", "2018"), (1, "Concentrate", "Spain", "2018"),
                   (2, "Cathode", "Italy", "2018"), (3, "Concentrate", "Chile", "2018"),
                   (4, "Anode", "Belgium", "2018"), (5, "Cathode", "France", "2018"))
    prod = _production(("Concentrate", "Spain", "2018", 40.0),
                       ("Concentrate", "Germany", "2018", 0.0),
                       ("Cathode", "Italy", "2018", 50.0),
                       ("Cathode", "France", "2018", 0.0))
    covered = _cov("Germany", "Spain", "Italy", "Chile", "Belgium", "France")
    assert non_producer_mask(edges, prod, covered, 7).tolist() == [
        True, False, False, True, False, True, False]


def test_an_uncovered_or_aggregate_regions_zero_is_not_an_observation():
    """ROW sums the members ICSG lists and Luxembourg never appears in ICSG: a 0 there
    says nothing about their production."""
    edges = _edges((0, "Cathode", "ROW", "2018"), (1, "Concentrate", "Luxembourg", "2018"))
    prod = _production(("Cathode", "ROW", "2018", 0.0),
                       ("Concentrate", "Luxembourg", "2018", 0.0))
    covered = icsg_coverage(pd.DataFrame({"Region": ["Zimbabwe", "Germany"]}),
                            _region_map(("Zimbabwe", "ROW"), ("Germany", "Germany"),
                                        ("Luxembourg", "Luxembourg")), ["2018"], {})
    assert not non_producer_mask(edges, prod, covered, 2).any()


def test_the_cathode_test_reads_raw_icsg_production(monkeypatch):
    """#96: the feed's F_root carries no cathode production edge (it read 0 for anode in
    all 1,410 region-years while raw production was positive in 933), so the test reads
    the ADR 0024 bound's raw ICSG production. ICSG reports Italy's refinery and nothing
    for France; France's row must still say 0."""
    import build_baci_reexport as R

    monkeypatch.setattr(
        R, "icsg_intake_production",
        lambda sectors=None: ({}, {("Cathode", "Italy", "2018"): 50.0}, set()))
    edges = _edges((0, "Cathode", "Italy", "2018"), (1, "Cathode", "France", "2018"))
    prod = raw_production(edges)
    cathode = prod[prod["root_sector_from"] == "Cathode"]
    assert dict(zip(cathode["root_region_to"], cathode["production_cc"])) == {
        "Italy": 50.0, "France": 0.0}
    assert non_producer_mask(edges, prod, _cov("Italy", "France"), 2).tolist() == [
        False, True]


# ---------------------------------------------------------------------------
# production_zero_mask: the mine and SX-EW output edges (#97, #31)
# ---------------------------------------------------------------------------

CHAIN = (("CO", "Mining"), ("Mining", "Ore"), ("Ore", "MF"), ("Ore", "SXEW"),
         ("MF", "Concentrate"), ("MF", "Tailings"), ("SXEW", "CSXEW"), ("CSXEW", "CP"))


def _chain(*regions, t="2018"):
    """Every region's mine chain, both valuations, as the decoded edge frame."""
    rows = [(f, s, r, r, t, v) for r in regions for f, s in CHAIN for v in ("Mass", "CC")]
    out = pd.DataFrame(rows, columns=["root_sector_from", "root_sector_to",
                                      "root_region_from", "root_region_to", "root_time",
                                      "valuation"])
    out.insert(0, "edge_id", np.arange(len(out), dtype=np.int64))
    return out


def _stamped(edges, mask):
    hit = edges[mask[edges["edge_id"].to_numpy()]]
    return sorted(zip(hit["root_region_to"], hit["root_sector_from"], hit["root_sector_to"],
                      hit["valuation"]))


def test_a_covered_non_miner_says_zero_on_its_mine_and_sxew_output_only():
    """Germany mines nothing and runs no SX-EW plant (#97). The edges the feed books
    ICSG's Mining/Concentrate and SXEW/Cathode supply on carry the zero, in CC as ICSG
    reports it. The rest of the chain is not stamped twice: the hard balances and the
    efficiencies take CO→Mining, Ore, Tailings, CSXEW→CP and the Mass twins with it."""
    edges = _chain("Germany")
    prod = _production(("Concentrate", "Germany", "2018", 0.0),
                       ("CSXEW", "Germany", "2018", 0.0))
    mask = production_zero_mask(edges, prod, _cov("Germany"), len(edges), set())
    assert _stamped(edges, mask) == [("Germany", "MF", "Concentrate", "CC"),
                                     ("Germany", "SXEW", "CSXEW", "CC")]


def test_a_miner_and_row_say_no_output_zero():
    """Chile mines and runs SX-EW; ROW's 0 would be a sum of members, not a report."""
    edges = _chain("Chile", "ROW")
    prod = _production(("Concentrate", "Chile", "2018", 5000.0),
                       ("CSXEW", "Chile", "2018", 1500.0),
                       ("Concentrate", "ROW", "2018", 0.0), ("CSXEW", "ROW", "2018", 0.0))
    assert not production_zero_mask(edges, prod, _cov("Chile"), len(edges), set()).any()


def test_a_miner_without_sxew_says_zero_on_its_sxew_output_only():
    """Papua New Guinea mines concentrate and reports no SX-EW cathode."""
    edges = _chain("Papua New Guinea")
    prod = _production(("Concentrate", "Papua New Guinea", "2018", 100.0))
    mask = production_zero_mask(edges, prod, _cov("Papua New Guinea"), len(edges),
                                set())
    assert _stamped(edges, mask) == [("Papua New Guinea", "SXEW", "CSXEW", "CC")]


def test_the_output_test_reads_raw_icsg_mine_and_sxew_production(monkeypatch):
    """Chile's mine and SX-EW rows are in ICSG and Germany's are not. No edge of these
    chains leaves a Concentrate node, so the production rows must not depend on which
    edges the frame holds: a missing row reads as 0 and would stamp Chile."""
    import build_baci_reexport as R

    monkeypatch.setattr(
        R, "icsg_intake_production",
        lambda sectors=None: ({}, {("Concentrate", "Chile", "2018"): 5000.0,
                                   ("CSXEW", "Chile", "2018"): 1500.0}, set()))
    edges = _chain("Chile", "Germany")
    mask = production_zero_mask(edges, raw_production(edges), _cov("Chile", "Germany"),
                                len(edges), set())
    assert _stamped(edges, mask) == [("Germany", "MF", "Concentrate", "CC"),
                                     ("Germany", "SXEW", "CSXEW", "CC")]


def _ore_grade(*rows):
    return pd.DataFrame(rows, columns=["a_sector_from", "a_region_from", "a_time"])


def test_crude_ore_and_ore_grade_rows_report_mining_and_sxew_elsewhere():
    """Malaysia's crude ore is reported in 1995 (Mamut), not in 2000, and a zero ore mass
    reports nothing. An ore-grade row reports the output its from-sector names."""
    ore = pd.DataFrame({"Region": ["Malaysia", "Malaysia", "Korea"],
                        "Year": [1995, 2000, 1995], "Value": [3931.0, 0.0, 3.8]})
    rmap = _region_map(("Malaysia", "Malaysia"), ("Korea", "Korea, Rep."))
    grades = _ore_grade(("CO", "Spain", "2002"), ("SXEW", "Spain", "2009"),
                        ("MF", "Spain", "2009"))
    assert reported_elsewhere(ore, rmap, grades) == {
        ("Concentrate", "Malaysia", "1995"), ("Concentrate", "Korea, Rep.", "1995"),
        ("Concentrate", "Spain", "2002"), ("CSXEW", "Spain", "2009")}


def test_a_covered_non_miner_with_crude_ore_mining_keeps_no_mine_zero():
    """#97 guard: ICSG lists Malaysia but not its 1995 mine, which the crude-ore feed
    reports at 3.9 Mt of ore: ICSG's missing row is missing data, and a zero there would
    contradict the crude-ore feed (ADR 0019). No input reports SX-EW, so that zero
    stands."""
    edges = _chain("Malaysia", t="1995")
    prod = _production(("Concentrate", "Malaysia", "1995", 0.0),
                       ("CSXEW", "Malaysia", "1995", 0.0))
    ore = pd.DataFrame({"Region": ["Malaysia"], "Year": [1995], "Value": [3931.0]})
    elsewhere = reported_elsewhere(ore, _region_map(("Malaysia", "Malaysia")),
                                   _ore_grade())
    mask = production_zero_mask(edges, prod, _cov("Malaysia", years=("1995",)),
                                len(edges), elsewhere)
    assert _stamped(edges, mask) == [("Malaysia", "SXEW", "CSXEW", "CC")]


def test_an_sxew_ore_grade_row_spares_the_sxew_zero():
    edges = _chain("Spain", t="2009")
    prod = _production(("Concentrate", "Spain", "2009", 50.0))
    mask = production_zero_mask(edges, prod, _cov("Spain", years=("2009",)), len(edges),
                                {("CSXEW", "Spain", "2009")})
    assert not mask.any()


# ---------------------------------------------------------------------------
# Stale trade-share weights are refused (#53)
# ---------------------------------------------------------------------------

def test_the_feed_refuses_weights_it_cannot_check(tmp_path):
    """A trade_shares version that records no parents cannot be checked against
    the BACI feed run and inputs in force, so the ICSG build refuses it."""
    from pysuto.input_registry import InputRegistries

    out = tmp_path / "params" / "trade_shares" / "current"
    out.mkdir(parents=True)
    (out / "params.csv").write_text("weight\n1.0\n")
    inputs = InputRegistries.initialize(root=tmp_path)
    with pytest.raises(ValueError, match="records no parents.json"):
        load_trade_shares(inputs)
