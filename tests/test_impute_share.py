"""
test_impute_share.py
====================
ShareImputer — the net-stock seed: a stock-change leg no feed reports is held at 0
with σ = max(seed_cv·|co-located base|, seed_abs_floor) — the base is Cathode→FU for
MESC/USC (Round X, ADR 0040) and CP→Cathode for PSC (Round Y, ADR 0041).
"""
import numpy as np
import pandas as pd
import pytest

from pysuto.impute import ShareImputer


class _Acc:
    """Minimal acc: region A, years 1995/1996, Mass valuation.

    Nodes: 0 CP/Process, 1 PSC/SB, 2 CSXEW/Flow, 3 CER/Flow,
           4 Cathode/Flow, 5 MESC/SB, 6 FU/SB   (all region A)
    Edges: PSC net (CP→PSC) 1995[e0],1996[e1]; CSXEW→CP 1995[e2]/1996[e4];
           CER→CP 1995[e3]/1996[e5]; MESC net (Cathode→MESC) 1995[e6];
           Cathode→FU (MESC base) 1995[e7]; CP→Cathode (PSC base) 1995[e8]/1996[e9].
    """
    def __init__(self):
        self.node_index = pd.DataFrame(
            {"sector": ["CP", "PSC", "CSXEW", "CER", "Cathode", "MESC", "FU"],
             "entity": ["Process", "SB", "Flow", "Flow", "Flow", "SB", "SB"],
             "region": ["A"] * 7},
            index=list(range(7)),
        )
        self.edge_index = pd.DataFrame(
            {"edge_id":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             "from_id":   [0, 0, 2, 3, 2, 3, 4, 4, 0, 0],
             "to_id":     [1, 1, 0, 0, 0, 0, 5, 6, 4, 4],
             "time":      ["1995", "1996", "1995", "1995", "1996", "1996", "1995", "1995",
                           "1995", "1996"],
             "valuation": ["Mass"] * 10})
        self.n_edges = 10


def _blank():
    return np.full(10, np.nan), np.full(10, np.nan)


def test_psc_seed_is_zero_mean_one_percent_of_the_cathode_pool_product():
    """ADR 0041: an unreported PSC year is held at 0 with σ = 1 % of the same
    region's CP→Cathode; a reported year is untouched; the MESC seed is unchanged."""
    acc = _Acc()
    f, sigma = _blank()
    f[1] = 3.0                      # PSC 1996 reported
    f[8], f[9] = 100.0, 120.0       # CP→Cathode 1995 / 1996 (the PSC base)
    f[7] = 200.0                    # Cathode→FU 1995 (the MESC base)
    imp = ShareImputer(p95={}, seed_cv=0.01, seed_abs_floor=0.3)
    entries = imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)

    assert f[0] == 0.0 and sigma[0] == pytest.approx(0.01 * 100.0)
    (log,) = [x for x in entries if x["edge_id"] == 0]
    assert log["method"] == "share_seed" and log["param_id"] == "seed_psc"
    assert log["source_edges"] == [8] and log["coefficients"] == [0.0]
    assert f[1] == 3.0 and not any(x["edge_id"] == 1 for x in entries)
    assert f[6] == 0.0 and sigma[6] == pytest.approx(0.01 * 200.0)


def test_psc_seed_skipped_when_the_cathode_pool_product_is_missing():
    """No CP→Cathode prior → no PSC seed (left to the flat prior); the CSXEW/CER
    inputs are not the base. The MESC seed, whose base is present, still fires."""
    acc = _Acc()
    f, sigma = _blank()
    f[2] = 50.0; f[3] = 50.0; f[7] = 200.0
    imp = ShareImputer(p95={}, seed_cv=0.01, seed_abs_floor=0.3)
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert np.isnan(f[0]) and np.isnan(f[1])
    assert f[6] == 0.0


def test_seed_skipped_when_base_missing():
    acc = _Acc()
    f, sigma = _blank()
    f[1] = 3.0; f[8] = 100.0
    # e7 (MESC base) left NaN → MESC seed must skip (leave for flat-prior)
    imp = ShareImputer(p95={}, seed_cv=0.01, seed_abs_floor=0.3)
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert np.isnan(f[6])                          # no base → not seeded
    assert f[0] == 0.0                             # PSC 1995: base known → seeded


def test_seed_sigma_is_one_percent_of_the_fu_leg_floored_at_0_3_kt():
    """Round X / ADR 0040: the MESC/USC seed is priced like a feed-backed level,
    σ = max(seed_cv·|FU leg|, seed_abs_floor), and needs no p95 entry."""
    acc = _Acc()
    f, sigma = _blank()
    f[7] = 200.0                       # Cathode→FU 1995 (the co-located base)
    imp = ShareImputer(p95={}, seed_cv=0.01, seed_abs_floor=0.3)
    entries = imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert f[6] == 0.0
    assert sigma[6] == pytest.approx(0.01 * 200.0)     # 1 % of the leg, not p95·100
    (log,) = [x for x in entries if x["edge_id"] == 6]
    assert log["method"] == "share_seed" and log["param_id"] == "seed_mesc"
    assert log["source_edges"] == [7] and log["coefficients"] == [0.0]
    assert log["f_after"] == 0.0 and log["sigma_after"] == pytest.approx(2.0)
    assert isinstance(log["edge_id"], int) and isinstance(log["sigma_after"], float)

    # a tiny / no-trade pair sits on the floor
    f, sigma = _blank()
    f[7] = 5.0
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert sigma[6] == pytest.approx(0.3)

    # no base → no seed (left to the flat prior / Stage-3 no-data anchor)
    f, sigma = _blank()
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert np.isnan(f[6])


def test_seed_knobs_are_persisted_in_config():
    imp = ShareImputer(p95={}, seed_cv=0.02, seed_abs_floor=0.5)
    cfg = imp.config()
    assert cfg["seed_cv"] == 0.02 and cfg["seed_abs_floor"] == 0.5
