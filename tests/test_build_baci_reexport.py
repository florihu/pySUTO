"""Unit tests for the feed-time re-export re-attribution (ADR 0027)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_baci_reexport import (  # noqa: E402
    cv_scale,
    label_row,
    net_one_hub,
    net_reexports,
    reexport_volumes,
    screened_origins,
)

T2015 = "2015"


def _leg(sector, a, z, cc, t=T2015):
    return {"sector": sector, "region_from": a, "region_to": z,
            "time": t, "cc": float(cc)}


def _trade(rows):
    return pd.DataFrame(rows)


def _vol(trade, intake=None, production=None, reported=None):
    v = reexport_volumes(trade, intake or {}, production or {}, reported or set())
    v["label"] = [label_row(r) for r in v.itertuples(index=False)]
    return v


def _row(vol, sector, region, t=T2015):
    m = vol[(vol.sector == sector) & (vol.region == region) & (vol.time == t)]
    assert len(m) == 1, f"expected one {sector}/{region} row, got {len(m)}"
    return m.iloc[0]


# ---------------------------------------------------------------------------
# The rule: T = min(X, I, max(R, E))
# ---------------------------------------------------------------------------

class TestReexportVolumes:

    def test_physical_hub_exports_fully_explained_by_imports(self):
        """Netherlands cathode 2015 shape: I 386, X 47, U 5, P 0 → T = X."""
        trade = _trade([_leg("Cathode", "Russia", "NL", 286.3),
                        _leg("Cathode", "Chile", "NL", 87.9),
                        _leg("Cathode", "NL", "Germany", 37.3),
                        _leg("Cathode", "NL", "Belgium", 10.0)])
        v = _row(_vol(trade, intake={("Cathode", "NL", T2015): 5.0},
                      production={("Cathode", "NL", T2015): 0.0},
                      reported={("Cathode", "NL")}), "Cathode", "NL")
        assert v.I == pytest.approx(374.2)
        assert v.X == pytest.approx(47.3)
        assert v.R == pytest.approx(369.2)
        assert v.E == pytest.approx(47.3)
        assert v["T"] == pytest.approx(47.3)          # capped by X
        assert v.label == "pass-through"

    def test_ceiling_term_catches_the_phantom_mine_case(self):
        """Korea concentrate: U > I so R = 0, but P = 0 so every export is a
        re-export. The surplus rule alone is blind to this."""
        trade = _trade([_leg("Concentrate", "Chile", "Korea", 491.9),
                        _leg("Concentrate", "Korea", "Japan", 16.2)])
        v = _row(_vol(trade, intake={("Concentrate", "Korea", T2015): 529.8},
                      production={("Concentrate", "Korea", T2015): 0.0},
                      reported={("Concentrate", "Korea")}),
                 "Concentrate", "Korea")
        assert v.R == 0.0
        assert v.E == pytest.approx(16.2)
        assert v.T_surplus == 0.0
        assert v["T"] == pytest.approx(16.2)

    def test_producer_with_a_thin_icsg_row_is_never_netted(self):
        """Kazakhstan cathode: E = 113 but I = 0, so the import cap gives T = 0.
        Its export/production gap is a coverage question, not a re-export."""
        trade = _trade([_leg("Cathode", "Kazakhstan", "China", 411.6)])
        v = _row(_vol(trade, intake={("Cathode", "Kazakhstan", T2015): 56.5},
                      production={("Cathode", "Kazakhstan", T2015): 298.5},
                      reported={("Cathode", "Kazakhstan")}),
                 "Cathode", "Kazakhstan")
        assert v.I == 0.0
        assert v.E == pytest.approx(113.1)
        assert v["T"] == 0.0
        assert v.label == "producer"

    def test_inbound_only_merchanting_moves_almost_nothing(self):
        """Zambia→Switzerland anode: X ≈ 0, so T ≈ 0 and the inbound tonnage
        stays where the QP already refuses it."""
        trade = _trade([_leg("Anode", "Zambia", "CH", 315.9),
                        _leg("Anode", "CH", "China", 0.005)])
        v = _row(_vol(trade, reported=set()), "Anode", "CH")
        assert v.R == pytest.approx(315.9)
        assert v["T"] == pytest.approx(0.005)
        assert v.resid == pytest.approx(315.895)
        assert v.label == "inbound-only/no-process"

    def test_row_is_never_a_hub(self):
        trade = _trade([_leg("Concentrate", "PNG", "ROW", 11223.0),
                        _leg("Concentrate", "ROW", "China", 500.0)])
        v = _row(_vol(trade, reported=set()), "Concentrate", "ROW")
        assert v.R > 0 and v.X > 0
        assert v["T"] == 0.0
        assert v.excluded_aggregate
        assert v.label == "excluded-aggregate"

    def test_icsg_gap_year_is_not_a_zero(self):
        """A region reporting the product in other years but not this one has a
        reporting gap; treating it as U = P = 0 would net real exports."""
        trade = _trade([_leg("Cathode", "Chile", "Sweden", 60.0),
                        _leg("Cathode", "Sweden", "Germany", 50.0)])
        gap = _row(_vol(trade, reported={("Cathode", "Sweden")}),
                   "Cathode", "Sweden")
        assert gap.icsg_gap and gap["T"] == 0.0 and gap.label == "icsg-gap"

        never = _row(_vol(trade, reported=set()), "Cathode", "Sweden")
        assert not never.icsg_gap and never["T"] == pytest.approx(50.0)

    def test_imports_exhausted_is_flagged(self):
        """T = I while the region still has use to cover: the fan must not read
        the resulting zero as observed domestic sourcing (ADR 0018)."""
        trade = _trade([_leg("Cathode", "Chile", "ZA", 23.4),
                        _leg("Cathode", "ZA", "China", 28.7)])
        v = _row(_vol(trade, intake={("Cathode", "ZA", T2015): 30.1},
                      production={("Cathode", "ZA", T2015): 0.0},
                      reported={("Cathode", "ZA")}), "Cathode", "ZA")
        assert v["T"] == pytest.approx(23.4) == pytest.approx(v.I)
        assert v.imports_exhausted

    def test_a_hub_needs_both_an_inflow_and_an_outflow(self):
        trade = _trade([_leg("Cathode", "Chile", "China", 100.0)])
        v = _vol(trade, reported=set())
        assert (v["T"] == 0.0).all()


# ---------------------------------------------------------------------------
# Screened origins
# ---------------------------------------------------------------------------

class TestScreenedOrigins:

    def test_flags_exports_no_production_plus_imports_explains(self):
        trade = _trade([_leg("Concentrate", "PNG", "Philippines", 11223.0),
                        _leg("Concentrate", "Chile", "Philippines", 100.0)])
        scr = screened_origins(trade, {("Concentrate", "PNG", T2015): 82.5,
                                       ("Concentrate", "Chile", T2015): 4000.0})
        assert ("Concentrate", "PNG", T2015) in scr
        assert ("Concentrate", "Chile", T2015) not in scr

    def test_groups_without_production_stay_unjudged(self):
        """Matches the Stage-2 screen: no covered production ⇒ no verdict."""
        trade = _trade([_leg("Concentrate", "PNG", "Philippines", 11223.0)])
        assert screened_origins(trade, {}) == set()

    def test_screened_origin_keeps_its_gross_leg(self):
        mat = {("PNG", "PH"): 1000.0, ("Chile", "PH"): 100.0,
               ("PH", "China"): 200.0}
        applied, _, _ = net_one_hub(mat, "PH", 200.0, screened_from={"PNG"})
        assert mat[("PNG", "PH")] == pytest.approx(1000.0)   # untouched
        assert applied == pytest.approx(100.0)               # capped by Chile
        assert mat[("Chile", "PH")] == pytest.approx(0.0)
        assert mat[("Chile", "China")] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# The re-allocation and its invariants
# ---------------------------------------------------------------------------

def _totals(mat):
    x, i = {}, {}
    for (a, z), c in mat.items():
        x[a] = x.get(a, 0.0) + c
        i[z] = i.get(z, 0.0) + c
    return x, i


class TestNetOneHub:

    def test_shares_split_by_origin_and_destination(self):
        mat = {("Russia", "NL"): 300.0, ("Chile", "NL"): 100.0,
               ("NL", "Germany"): 80.0, ("NL", "Belgium"): 20.0}
        applied, _, n_new = net_one_hub(mat, "NL", 100.0, set())
        assert applied == pytest.approx(100.0)
        # 100 × 0.75 (Russia) × 0.8 (Germany) = 60, etc.
        assert mat[("Russia", "Germany")] == pytest.approx(60.0)
        assert mat[("Russia", "Belgium")] == pytest.approx(15.0)
        assert mat[("Chile", "Germany")] == pytest.approx(20.0)
        assert mat[("Chile", "Belgium")] == pytest.approx(5.0)
        assert mat[("Russia", "NL")] == pytest.approx(225.0)
        assert mat[("NL", "Germany")] == pytest.approx(0.0)
        assert n_new == 4

    def test_invariants_hold(self):
        mat = {("Russia", "NL"): 300.0, ("Chile", "NL"): 100.0,
               ("NL", "Germany"): 80.0, ("NL", "Belgium"): 20.0}
        x0, i0 = _totals(dict(mat))
        applied, self_amt, _ = net_one_hub(mat, "NL", 100.0, set())
        x1, i1 = _totals(mat)
        assert self_amt == 0.0
        for a in ("Russia", "Chile"):
            assert x1[a] == pytest.approx(x0[a]), "origin exports unchanged"
        for z in ("Germany", "Belgium"):
            assert i1[z] == pytest.approx(i0[z]), "destination imports unchanged"
        assert x1["NL"] == pytest.approx(x0["NL"] - applied)
        assert i1["NL"] == pytest.approx(i0["NL"] - applied)
        assert sum(mat.values()) == pytest.approx(sum(x0.values()) - applied)

    def test_re_import_is_dropped_not_booked_as_domestic(self):
        """A→h→A is a returned good; it leaves both legs and creates nothing."""
        mat = {("Germany", "LU"): 100.0, ("LU", "Germany"): 100.0}
        applied, self_amt, n_new = net_one_hub(mat, "LU", 100.0, set())
        assert self_amt == pytest.approx(100.0)
        assert n_new == 0
        assert mat[("Germany", "LU")] == pytest.approx(0.0)
        assert mat[("LU", "Germany")] == pytest.approx(0.0)
        assert ("Germany", "Germany") not in mat
        assert applied == pytest.approx(100.0)

    def test_crumbs_fold_into_the_origins_kept_destinations(self):
        mat = {("Chile", "SG"): 100.0,
               ("SG", "China"): 99.0, ("SG", "Tiny"): 1.0}
        applied, _, n_new = net_one_hub(mat, "SG", 100.0, set(), min_new_leg=5.0)
        assert n_new == 1, "only the China pair clears the threshold"
        assert ("Chile", "Tiny") not in mat
        assert mat[("Chile", "China")] == pytest.approx(100.0), "crumb folded in"
        assert applied == pytest.approx(100.0)

    def test_existing_small_leg_counts_as_kept(self):
        mat = {("Chile", "SG"): 100.0, ("SG", "China"): 99.0,
               ("SG", "Tiny"): 1.0, ("Chile", "Tiny"): 0.5}
        net_one_hub(mat, "SG", 100.0, set(), min_new_leg=5.0)
        assert mat[("Chile", "Tiny")] > 0.5, "an observed route still receives"

    def test_origin_with_no_kept_destination_does_not_participate(self):
        mat = {("Chile", "SG"): 100.0, ("Tiny", "SG"): 0.2,
               ("SG", "China"): 100.0}
        applied, _, _ = net_one_hub(mat, "SG", 100.2, set(), min_new_leg=5.0)
        assert mat[("Tiny", "SG")] == pytest.approx(0.2), "left un-netted"
        # Tiny's 0.2 kt share cannot be placed, so the hub nets only Chile's
        # share and keeps the rest gross — conservative by design.
        assert applied == pytest.approx(100.0 * 100.0 / 100.2)
        assert mat[("SG", "China")] == pytest.approx(100.0 - applied)

    def test_no_leg_can_go_negative(self):
        mat = {("A", "H"): 10.0, ("B", "H"): 90.0,
               ("H", "X"): 5.0, ("H", "Y"): 95.0}
        net_one_hub(mat, "H", 100.0, set(), min_new_leg=0.0)
        assert min(mat.values()) >= -1e-9

    def test_a_hub_with_no_usable_inflow_moves_nothing(self):
        mat = {("PNG", "PH"): 500.0, ("PH", "China"): 100.0}
        applied, _, _ = net_one_hub(mat, "PH", 100.0, screened_from={"PNG"})
        assert applied == 0.0
        assert mat[("PH", "China")] == pytest.approx(100.0)


class TestNetReexports:

    def test_hub_to_hub_chain_lands_at_the_final_destination(self):
        """Chile → NL → Germany with Germany itself a hub onward to Poland."""
        trade = _trade([_leg("Cathode", "Chile", "NL", 100.0),
                        _leg("Cathode", "NL", "DE", 100.0),
                        _leg("Cathode", "DE", "PL", 100.0)])
        vol = _vol(trade, reported=set())
        new, ev = net_reexports(trade, vol, set())
        got = {(r.region_from, r.region_to): r.cc
               for r in new.itertuples(index=False) if r.cc > 1e-9}
        assert got == pytest.approx({("Chile", "PL"): 100.0})
        assert ev["T_applied"].sum() == pytest.approx(200.0)  # both hubs netted

    def test_idempotent(self):
        trade = _trade([_leg("Cathode", "Russia", "NL", 300.0),
                        _leg("Cathode", "Chile", "NL", 100.0),
                        _leg("Cathode", "NL", "DE", 80.0),
                        _leg("Cathode", "NL", "BE", 20.0)])
        new, _ = net_reexports(trade, _vol(trade, reported=set()), set())
        new = new[new.cc > 1e-9].reset_index(drop=True)
        again, ev2 = net_reexports(new, _vol(new, reported=set()), set())
        again = again[again.cc > 1e-9].reset_index(drop=True)
        assert ev2["T_applied"].sum() if len(ev2) else 0.0 == pytest.approx(0.0, abs=0.01)
        key = ["sector", "region_from", "region_to", "time"]
        pd.testing.assert_frame_equal(
            new.sort_values(key).reset_index(drop=True),
            again.sort_values(key).reset_index(drop=True), atol=1e-6)

    def test_years_are_independent(self):
        trade = _trade([_leg("Cathode", "Chile", "NL", 100.0, "2015"),
                        _leg("Cathode", "NL", "DE", 100.0, "2015"),
                        _leg("Cathode", "Chile", "NL", 50.0, "2016")])
        new, _ = net_reexports(trade, _vol(trade, reported=set()), set())
        y16 = new[(new.time == "2016") & (new.cc > 1e-9)]
        assert len(y16) == 1 and y16.iloc[0].region_to == "NL"


# ---------------------------------------------------------------------------
# σ of the inference
# ---------------------------------------------------------------------------

class TestCvScale:

    def test_new_leg_carries_root_two_times_the_pedigree(self):
        old = _trade([_leg("Cathode", "Chile", "NL", 100.0)])
        new = _trade([_leg("Cathode", "Chile", "NL", 0.0),
                      _leg("Cathode", "Chile", "DE", 100.0)])
        s = cv_scale(old, new).set_index("region_to")["cv_scale"]
        assert s["DE"] == pytest.approx(np.sqrt(2.0))

    def test_scale_is_floored_at_one_on_legs_that_grew(self):
        """Plain quadrature would make a leg that gained tonnage *more*
        certain than its pedigree — never allowed."""
        old = _trade([_leg("Cathode", "Chile", "DE", 100.0)])
        new = _trade([_leg("Cathode", "Chile", "DE", 150.0)])
        assert cv_scale(old, new)["cv_scale"].iloc[0] == 1.0

    def test_a_mostly_netted_leg_is_much_less_certain(self):
        old = _trade([_leg("Cathode", "Chile", "NL", 100.0)])
        new = _trade([_leg("Cathode", "Chile", "NL", 10.0)])
        got = cv_scale(old, new)["cv_scale"].iloc[0]
        assert got == pytest.approx(np.sqrt(100.0 ** 2 + 2 * 90.0 ** 2) / 10.0)
        assert got > 10.0

    def test_untouched_legs_are_absent(self):
        old = _trade([_leg("Cathode", "Chile", "DE", 100.0)])
        assert len(cv_scale(old, old.copy())) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
