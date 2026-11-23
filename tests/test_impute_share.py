"""
test_impute_share.py
====================
ShareImputer (Part B): PSC share-transfer (with 1995 boundary seed) and the
MESC/USC zero-mean seed held at 1 % of the co-located FU leg / 0.3 kt (ADR 0040).
"""
import numpy as np
import pandas as pd
import pytest

from pysuto.impute import ShareImputer, StockLeg, run_imputer
from pysuto.impute.share import Z95
from pysuto.relation.params import ParamRow

# The copper instances of the stock legs (a PSC transfer and a MESC seed).
_LEGS = [
    StockLeg("transfer", ("CP", "Process", "PSC", "SB"),
             (("CSXEW", "Flow", "CP", "Process"), ("CER", "Flow", "CP", "Process"))),
    StockLeg("seed", ("Cathode", "Flow", "MESC", "SB"), (("Cathode", "Flow", "FU", "SB"),)),
]


class _Acc:
    """Minimal acc: region A, years 1995/1996, Mass valuation.

    Nodes: 0 CP/Process, 1 PSC/SB, 2 CSXEW/Flow, 3 CER/Flow,
           4 Cathode/Flow, 5 MESC/SB, 6 FU/SB   (all region A)
    Edges: PSC net (CP→PSC) 1995[e0],1996[e1]; CSXEW→CP 1995[e2]/1996[e4];
           CER→CP 1995[e3]/1996[e5]; MESC net (Cathode→MESC) 1995[e6];
           Cathode→FU (MESC base) 1995[e7].
    """
    def __init__(self):
        self.node_index = pd.DataFrame(
            {"sector": ["CP", "PSC", "CSXEW", "CER", "Cathode", "MESC", "FU"],
             "entity": ["Process", "SB", "Flow", "Flow", "Flow", "SB", "SB"],
             "region": ["A"] * 7},
            index=list(range(7)),
        )
        self.edge_index = pd.DataFrame(
            {"edge_id":   [0, 1, 2, 3, 4, 5, 6, 7],
             "from_id":   [0, 0, 2, 3, 2, 3, 4, 4],
             "to_id":     [1, 1, 0, 0, 0, 0, 5, 6],
             "time":      ["1995", "1996", "1995", "1995", "1996", "1996", "1995", "1995"],
             "valuation": ["Mass"] * 8})
        self.n_edges = 8


def test_share_transfer_seeds_1995_and_seed_is_zero_mean():
    acc = _Acc()
    f = np.full(8, np.nan)
    sigma = np.full(8, np.nan)
    # PSC 1996 observed = 3.0; base 1996 = CSXEW60 + CER40 = 100 → share = 0.03
    f[1] = 3.0; f[4] = 60.0; f[5] = 40.0
    # PSC 1995 missing (e0); base 1995 = CSXEW50 + CER50 = 100
    f[2] = 50.0; f[3] = 50.0
    # MESC 1995 missing (e6); base Cathode→FU 1995 = 200 (e7)
    f[7] = 200.0

    imp = ShareImputer(
        p95={("PSC", "A"): 0.05, ("MESC", "A"): 0.10}, legs=_LEGS,
        scale_abs_floor=1.0,
        seed_cv=0.01, seed_abs_floor=0.3
    )
    run_imputer(imp, f, sigma, 0, acc)

    # PSC 1995 seeded by NOCB share (0.03 from 1996) × base(1995)=100
    assert f[0] == pytest.approx(0.03 * 100.0)
    assert sigma[0] == pytest.approx((0.05 / Z95) * 100.0, rel=1e-6)   # (p95/Z95)·base
    # MESC 1995: zero-mean, σ = max(seed_cv·FU, seed_abs_floor) = 0.01·200 (Round X;
    # the p95 band prices PSC only)
    assert f[6] == 0.0
    assert sigma[6] == pytest.approx(0.01 * 200.0, rel=1e-6)
    # observed PSC 1996 untouched
    assert f[1] == 3.0


def test_share_transfer_skips_when_no_base_or_no_observed_share():
    acc = _Acc()
    f = np.full(8, np.nan)
    sigma = np.full(8, np.nan)
    # No observed PSC anywhere, base present → transfer has no share to carry → skip
    f[2] = 50.0; f[3] = 50.0; f[7] = 200.0
    imp = ShareImputer(p95={("PSC", "A"): 0.05, ("MESC", "A"): 0.10}, legs=_LEGS,
                       scale_abs_floor=1.0,
                       seed_cv=0.01, seed_abs_floor=0.3)
    run_imputer(imp, f, sigma, 0, acc)
    assert np.isnan(f[0]) and np.isnan(f[1])     # PSC left for flat-prior
    assert f[6] == 0.0                            # MESC seed still fires (base present)


def test_seed_skipped_when_base_missing():
    acc = _Acc()
    f = np.full(8, np.nan)
    sigma = np.full(8, np.nan)
    f[1] = 3.0; f[4] = 60.0; f[5] = 40.0; f[2] = 50.0; f[3] = 50.0
    # e7 (MESC base) left NaN → MESC seed must skip (leave for flat-prior)
    imp = ShareImputer(p95={("PSC", "A"): 0.05, ("MESC", "A"): 0.10}, legs=_LEGS,
                       scale_abs_floor=1.0,
                       seed_cv=0.01, seed_abs_floor=0.3)
    run_imputer(imp, f, sigma, 0, acc)
    assert np.isnan(f[6])                          # no base → not seeded
    assert f[0] == pytest.approx(0.03 * 100.0)     # PSC transfer still works


def test_seed_sigma_is_one_percent_of_the_fu_leg_floored_at_0_3_kt():
    """Round X / ADR 0040: the MESC/USC seed is priced like a feed-backed level,
    σ = max(seed_cv·|FU leg|, seed_abs_floor), and needs no p95 entry."""
    acc = _Acc()
    f = np.full(8, np.nan)
    sigma = np.full(8, np.nan)
    f[7] = 200.0                       # Cathode→FU 1995 (the co-located base)
    imp = ShareImputer(p95={}, legs=_LEGS, scale_abs_floor=100.0, seed_cv=0.01,
                       seed_abs_floor=0.3)
    entries = run_imputer(imp, f, sigma, 0, acc)
    assert f[6] == 0.0
    assert sigma[6] == pytest.approx(0.01 * 200.0)     # 1 % of the leg, not p95·100
    (log,) = [x for x in entries if x["edge_id"] == 6]
    assert log["method"] == "share_seed" and log["param_id"] == "seed_mesc"
    assert log["source_edges"] == [7] and log["coefficients"] == [0.0]
    assert log["f_after"] == 0.0 and log["sigma_after"] == pytest.approx(2.0)
    assert isinstance(log["edge_id"], int) and isinstance(log["sigma_after"], float)

    # a tiny / no-trade pair sits on the floor
    f = np.full(8, np.nan); sigma = np.full(8, np.nan)
    f[7] = 5.0
    run_imputer(imp, f, sigma, 0, acc)
    assert sigma[6] == pytest.approx(0.3)

    # no base → no seed (left to the flat prior / Stage-3 no-data anchor)
    f = np.full(8, np.nan); sigma = np.full(8, np.nan)
    run_imputer(imp, f, sigma, 0, acc)
    assert np.isnan(f[6])


def test_seed_knobs_are_persisted_in_config():
    imp = ShareImputer(p95={}, legs=_LEGS, seed_cv=0.02, seed_abs_floor=0.5, scale_abs_floor=100.0)
    cfg = imp.config()
    assert cfg["seed_cv"] == 0.02 and cfg["seed_abs_floor"] == 0.5


# ---------------------------------------------------------------------------
# Non-copper labels: the labels are the caller's, nothing is parsed from param_id
# ---------------------------------------------------------------------------

_RELABEL = {"CP": "Mill", "PSC": "Hoard", "CSXEW": "Raw", "CER": "Scrap2",
            "Cathode": "Ingot", "MESC": "Vault", "FU": "Demand"}
_TOY_LEGS = [
    StockLeg("transfer", ("Mill", "Process", "Hoard", "SB"),
             (("Raw", "Flow", "Mill", "Process"), ("Scrap2", "Flow", "Mill", "Process"))),
    StockLeg("seed", ("Ingot", "Flow", "Vault", "SB"),
             (("Ingot", "Flow", "Demand", "SB"),)),
]


class _ToyAcc(_Acc):
    def __init__(self):
        super().__init__()
        self.node_index["sector"] = self.node_index["sector"].map(_RELABEL)


def _toy_f():
    f = np.full(8, np.nan)
    f[1] = 3.0; f[4] = 60.0; f[5] = 40.0; f[2] = 50.0; f[3] = 50.0; f[7] = 200.0
    return f


def test_relabelled_legs_give_the_same_fills_and_tags():
    f_c, s_c = _toy_f(), np.full(8, np.nan)
    f_t, s_t = _toy_f(), np.full(8, np.nan)
    kw = dict(scale_abs_floor=1.0, seed_cv=0.01, seed_abs_floor=0.3)
    run_imputer(ShareImputer({("PSC", "A"): 0.05}, _LEGS, **kw), f_c, s_c, 0, _Acc())
    e_t = run_imputer(ShareImputer({("Hoard", "A"): 0.05}, _TOY_LEGS, **kw), f_t, s_t, 0,
                      _ToyAcc())
    np.testing.assert_array_equal(f_t, f_c)
    np.testing.assert_array_equal(s_t, s_c)
    assert [x["param_id"] for x in e_t] == ["share_hoard_A", "seed_vault"]
    assert [x["description"] for x in e_t] == [
        "HOARD = share·(Raw+Scrap2)", "VAULT zero-mean, σ=max(cv·|Demand|, floor)"]


def test_writer_scope_is_the_declared_net_edges():
    imp = ShareImputer({}, _TOY_LEGS, scale_abs_floor=100.0, seed_cv=0.01, seed_abs_floor=0.3)
    scope = imp.writer_scope(_ToyAcc())
    assert np.flatnonzero(scope).tolist() == [0, 1, 6]


def test_p95_is_keyed_by_the_row_sector_not_the_param_id(monkeypatch):
    """The leg of a stock_bound row is its b_sector_to selector; a param_id that
    says 'mesc' on a Hoard row, or a row of an undeclared sector, changes nothing."""
    def row(pid, sector, region, vmax):
        sel = dict(ParamRow().selectors)
        sel.update(b_sector_to=sector, b_region_to=region)
        return ParamRow(param_id=pid, param_type="stock_bound", selectors=sel,
                        value_max=vmax)
    rows = [row("netstockboundmescx", "Hoard", "A", -0.07),
            row("netstockboundpscy", "Elsewhere", "A", 0.2),
            row("netstockboundhoard", "Hoard", "*", 0.3)]
    monkeypatch.setattr("pysuto.impute.share.read_params", lambda path, accepts: rows)
    imp = ShareImputer.from_stock_bounds_csv("unused.csv", _TOY_LEGS, scale_abs_floor=100.0, seed_cv=0.01, seed_abs_floor=0.3)
    assert imp._p95 == {("Hoard", "A"): 0.07}


def test_stock_leg_validates_its_declaration():
    net, base = ("A", "Flow", "S", "SB"), (("B", "Flow", "C", "SB"),)
    with pytest.raises(ValueError):
        StockLeg("share", net, base)
    with pytest.raises(ValueError):
        StockLeg("seed", net, base + base)
    with pytest.raises(ValueError):
        StockLeg("seed", ("A", "S"), base)
