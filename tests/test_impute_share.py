"""
test_impute_share.py
====================
ShareImputer (Part B): PSC share-transfer (with 1995 boundary seed) and the
MESC/USC zero-mean seed held at 1 % of the co-located FU leg / 0.3 kt (ADR 0040).
"""
import numpy as np
import pandas as pd
import pytest

from pysuto.impute import ShareImputer
from pysuto.impute.share import Z95


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
    missing = np.isnan(f)

    imp = ShareImputer(
        p95={("psc", "A"): 0.05, ("mesc", "A"): 0.10},
        scale_abs_floor=1.0,
    )
    imp.impute(f, sigma, missing, step=0, acc_struct=acc)

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
    missing = np.isnan(f)
    imp = ShareImputer(p95={("psc", "A"): 0.05, ("mesc", "A"): 0.10}, scale_abs_floor=1.0)
    imp.impute(f, sigma, missing, step=0, acc_struct=acc)
    assert np.isnan(f[0]) and np.isnan(f[1])     # PSC left for flat-prior
    assert f[6] == 0.0                            # MESC seed still fires (base present)


def test_seed_skipped_when_base_missing():
    acc = _Acc()
    f = np.full(8, np.nan)
    sigma = np.full(8, np.nan)
    f[1] = 3.0; f[4] = 60.0; f[5] = 40.0; f[2] = 50.0; f[3] = 50.0
    # e7 (MESC base) left NaN → MESC seed must skip (leave for flat-prior)
    missing = np.isnan(f)
    imp = ShareImputer(p95={("psc", "A"): 0.05, ("mesc", "A"): 0.10}, scale_abs_floor=1.0)
    imp.impute(f, sigma, missing, step=0, acc_struct=acc)
    assert np.isnan(f[6])                          # no base → not seeded
    assert f[0] == pytest.approx(0.03 * 100.0)     # PSC transfer still works


def test_seed_sigma_is_one_percent_of_the_fu_leg_floored_at_0_3_kt():
    """Round X / ADR 0040: the MESC/USC seed is priced like a feed-backed level,
    σ = max(seed_cv·|FU leg|, seed_abs_floor), and needs no p95 entry."""
    acc = _Acc()
    f = np.full(8, np.nan)
    sigma = np.full(8, np.nan)
    f[7] = 200.0                       # Cathode→FU 1995 (the co-located base)
    missing = np.isnan(f)
    imp = ShareImputer(p95={}, scale_abs_floor=100.0, seed_cv=0.01, seed_abs_floor=0.3)
    imp.impute(f, sigma, missing, step=0, acc_struct=acc)
    assert f[6] == 0.0
    assert sigma[6] == pytest.approx(0.01 * 200.0)     # 1 % of the leg, not p95·100

    # a tiny / no-trade pair sits on the floor
    f = np.full(8, np.nan); sigma = np.full(8, np.nan)
    f[7] = 5.0
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert sigma[6] == pytest.approx(0.3)

    # no base → no seed (left to the flat prior / Stage-3 no-data anchor)
    f = np.full(8, np.nan); sigma = np.full(8, np.nan)
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert np.isnan(f[6])


def test_seed_knobs_are_persisted_in_config():
    imp = ShareImputer(p95={}, seed_cv=0.02, seed_abs_floor=0.5)
    cfg = imp.config()
    assert cfg["seed_cv"] == 0.02 and cfg["seed_abs_floor"] == 0.5
