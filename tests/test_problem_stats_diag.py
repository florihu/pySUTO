"""
test_problem_stats_diag.py
==========================
Tests for :class:`pysuto.diag.stats.ProblemStatsDiagnostic` — the problem one
Stage-4 period is handed: edge provenance and its totals, the chain profile, the
per-row prior breach / binding / on-flow flags, and a region's slice of the QP in
standard form.

Synthetic stubs only. Two regions A and B; 14 edges, twelve in 2020 and two in
2021 (e11, e12). The 2020 block's columns are ``cols_t = [0..10, 13]``:

    e   from → to                     val   provenance       f_agg   l
    0   Mining|A → Ore|A              CC    network            10    0
    1   Mining|A → Ore|A              Mass  feed              500    0
    2   Ore|A → Concentrate|A         CC    feed                9    0
    3   Ore|A → Concentrate|A         Mass  conc              250    0
    4   Concentrate|A → Smelting|B    CC    trade               4    0    leg
    5   Concentrate|A → Smelting|B    Mass  feed               13    0    leg
    6   Concentrate|B → Smelting|A    CC    flat_prior          0    0    leg
    7   Concentrate|B → Smelting|A    Mass  flat_prior          0    0    leg
    8   Cathode|A → USC|A             CC    net_stock_share     0  −1e6   signed
    9   Cathode|A → USC|A             Mass  net_stock_share     0  −1e6   signed
    10  Concentrate|B → Smelting|B    CC    feed                7    0
    13  Smelting|A → Cathode|A        CC    flat_prior          0    0

The 2020 rows (local r = global id; row 12 is a 2021 row outside the block):

    r   family            cone    entries                   c      prior resid  resid at a*
    0   mass balance      zero    e0 − e2                   0       1 (breach)   0
    1   stripping         zero    e1 − 2·e3                 0       0            0
    2   conc band         nonneg  e2 − 0.5·e3               0    −116            0  (on flow)
    3   eff band          nonneg  e13 − 0.9·e6              0       0            0  (zero flows)
    4   scrap-share band  nonneg  e4 − 0.1·e5               0     2.7 (breach) −0.5
    5   lower bound       nonneg  −e10 (+ stored 0 on e1)   0      −7           −7
    6   lower bound       nonneg  −e0                       0     −10        −1e-6  (binds, −tol)
    7   lower bound       nonneg  −e2                       0      −9        −2e-6  (slack)
    8   mass balance      zero    e8 − e9               −1e-6    1e-6 (= tol)    0
    9   soft ratio        free    e4 − 0.3·e5               0                  0.2
    10  scale total       free    e2 + e10                 20                  0.5
    11  scrap-share band  nonneg  e13 − 0.2·e6 (+ 0 on e2)  0       0            0  (zero flows)

``a*`` puts e0 at exactly the bind tolerance (1e-6, not flowing); the residuals
at ``a*`` are the stub's own (they are not recomputed).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_equal

from pysuto.diag.stats import ProblemStatsDiagnostic
from pysuto.diag.stats.problem import row_family


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_NODES = [("Mining", "A"), ("Ore", "A"), ("Concentrate", "A"), ("Smelting", "B"),
          ("Concentrate", "B"), ("Smelting", "A"), ("Cathode", "A"), ("USC", "A")]
#              from to  time    valuation
_EDGES = [(0, 1, "2020", "CC"), (0, 1, "2020", "Mass"),
          (1, 2, "2020", "CC"), (1, 2, "2020", "Mass"),
          (2, 3, "2020", "CC"), (2, 3, "2020", "Mass"),
          (4, 5, "2020", "CC"), (4, 5, "2020", "Mass"),
          (6, 7, "2020", "CC"), (6, 7, "2020", "Mass"),
          (4, 3, "2020", "CC"),
          (0, 1, "2021", "CC"), (0, 1, "2021", "Mass"),
          (5, 6, "2020", "CC")]
COLS_2020 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13])
F_AGG = np.array([10, 500, 9, 250, 4, 13, 0, 0, 0, 0, 7, 11, 510, 0], dtype=float)


def _acc():
    em = pd.DataFrame([{"edge_id": i, "from_id": a, "to_id": b, "time": t, "valuation": v,
                        "from_sector": _NODES[a][0], "from_region": _NODES[a][1],
                        "to_sector": _NODES[b][0], "to_region": _NODES[b][1]}
                       for i, (a, b, t, v) in enumerate(_EDGES)])
    return SimpleNamespace(edge_meta=em)


class _Log:
    def __init__(self, df):
        self._df = df

    def as_dataframe(self):
        return self._df.copy()


def _imp(extra_log=(), coverage_extra=()):
    log = pd.DataFrame({"edge_id": [0, 3, 4, 6, 7, 8, 9, 13],
                        "imputer_id": ["network", "conc", "trade", "flat_prior", "flat_prior",
                                       "net_stock_share", "net_stock_share", "flat_prior"]})
    if extra_log:
        extra = pd.DataFrame(list(extra_log), columns=["edge_id", "imputer_id"])
        log = pd.concat([log, extra], ignore_index=True)
    coverage = np.zeros(len(_EDGES), dtype=bool)
    coverage[[1, 2, 5, 10, 11, 12, *coverage_extra]] = True
    return SimpleNamespace(f_agg=F_AGG.copy(), coverage=coverage, imputation_log=_Log(log))


def _G2020() -> sp.csr_matrix:
    """The 2020 block over local columns (cols_t order), with two stored zeros."""
    j = {e: k for k, e in enumerate(COLS_2020)}
    trip = [(0, 0, 1), (0, 2, -1),
            (1, 1, 1), (1, 3, -2),
            (2, 2, 1), (2, 3, -0.5),
            (3, 13, 1), (3, 6, -0.9),
            (4, 4, 1), (4, 5, -0.1),
            (5, 10, -1), (5, 1, 0.0),                  # stored zero on e1
            (6, 0, -1),
            (7, 2, -1),
            (8, 8, 1), (8, 9, -1),
            (9, 4, 1), (9, 5, -0.3),
            (10, 2, 1), (10, 10, 1),
            (11, 13, 1), (11, 6, -0.2), (11, 2, 0.0)]  # stored zero on e2
    r, e, v = zip(*trip)
    G = sp.csr_matrix((np.array(v, dtype=float), (np.array(r), np.array([j[x] for x in e]))),
                      shape=(12, len(COLS_2020)))
    assert G.nnz == len(trip)                         # the explicit zeros are stored
    return G


def _row_meta():
    fam = [("massbal", "massbal", True, "zero", None),
           ("ratio", "strippingratio", True, "zero", None),
           ("ratio", "conc_upper", True, "nonneg", "conc"),
           ("ratio", "eff_upper", True, "nonneg", "eff"),
           ("ratio", "scrapshare_upper", True, "nonneg", "scrap_share"),
           ("bound", "lower", True, "nonneg", None),
           ("bound", "lower", True, "nonneg", None),
           ("bound", "lower", True, "nonneg", None),
           ("massbal", "massbal", True, "zero", None),
           ("ratio", "conc", False, "free", None),
           ("sum", "supplytotout", False, "free", None),
           ("ratio", "scrapshare_upper", True, "nonneg", "scrap_share"),
           ("massbal", "massbal", True, "zero", None)]          # row 12: 2021
    rm = pd.DataFrame(fam, columns=["type", "variant", "is_hard", "cone_type", "ratio_kind"])
    rm["time"] = ["2020"] * 12 + ["2021"]
    return rm


def _cr():
    c = np.zeros(13)                                  # stored over all rows
    c[8], c[10] = -1e-6, 20.0
    l = np.zeros(len(_EDGES))
    l[[8, 9]] = -1e6
    return SimpleNamespace(row_meta=_row_meta(), l=l,
                           timestep_data={"2020": {"c": c}},
                           G_blocks={"2020": _G2020()},
                           edge_cols_by_t={"2020": COLS_2020.copy()})


A_PRIOR = np.array([10, 500, 9, 250, 4, 13, 0, 0, 0, 0, 7, 0], dtype=float)    # cols_t order
F_OPT   = np.array([1e-6, 480, 9.5, 240, 3.5, 13, 0, 0, 0, 0, 7, 0], dtype=float)
RESID   = np.array([0, 0, 0, 0, -0.5, -7, -1e-6, -2e-6, 0, 0.2, 0.5, 0], dtype=float)


def _opt(cols=COLS_2020):
    return SimpleNamespace(F_opt=F_OPT.copy(), a_prior=A_PRIOR.copy(),
                           residuals=RESID.copy(), cols_t=np.asarray(cols),
                           rows_t=np.arange(12))


def _diag(year="2020", **kw):
    return ProblemStatsDiagnostic(kw.get("opt", _opt()), kw.get("cr", _cr()),
                                  kw.get("imp", _imp()), _acc(), year)


def test_year_is_a_time_label():
    with pytest.raises(TypeError, match="time label"):
        _diag(year=2020)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_feed_where_covered_else_the_imputer(self):
        p = _diag().provenance()
        assert p.tolist() == ["network", "feed", "feed", "conc", "trade", "feed",
                              "flat_prior", "flat_prior", "net_stock_share",
                              "net_stock_share", "feed", "feed", "feed", "flat_prior"]

    def test_raises_on_an_edge_both_covered_and_imputed(self):
        with pytest.raises(ValueError, match="both"):
            _diag(imp=_imp(coverage_extra=(0,))).provenance()

    def test_raises_on_an_edge_imputed_twice(self):
        with pytest.raises(ValueError, match="more than once"):
            _diag(imp=_imp(extra_log=[(3, "time_interp")])).provenance()

    def test_raises_on_an_edge_with_no_provenance(self):
        imp = _imp()
        imp.imputation_log = _Log(imp.imputation_log.as_dataframe().query("edge_id != 13"))
        with pytest.raises(ValueError, match="neither"):
            _diag(imp=imp).provenance()

    def test_totals_per_category_for_one_valuation_and_year(self):
        t = _diag().provenance_totals("CC")
        assert t.loc["feed"].tolist() == [2, 16.0]               # e2, e10
        assert t.loc["network"].tolist() == [1, 10.0]
        assert t.loc["trade"].tolist() == [1, 4.0]
        assert t.loc["flat_prior"].tolist() == [2, 0.0]          # e6, e13
        assert t.loc["net_stock_share"].tolist() == [1, 0.0]
        assert t.loc["eff"].tolist() == [0, 0.0]                 # a known imputer, no fills
        assert list(t.columns) == ["n_edges", "kt"]
        assert t.index[0] == "feed"

    def test_feed_edges_per_year_counts_every_year(self):
        s = _diag().feed_edges_by_year("CC")
        assert s.to_dict() == {"2020": 2, "2021": 1}


# ---------------------------------------------------------------------------
# Chain profile
# ---------------------------------------------------------------------------

class TestChainProfile:
    def test_pairs_in_chain_order_then_the_traded_commodities(self):
        p = _diag().chain_profile("CC")
        assert p.index.tolist() == ["Mining→Ore", "Ore→Concentrate", "Concentrate→Smelting",
                                    "Concentrate (trade)"]
        assert p["kind"].tolist() == ["domestic", "domestic", "domestic", "legs"]

    def test_each_bar_is_stacked_by_provenance(self):
        p = _diag().chain_profile("CC")
        assert p.loc["Mining→Ore", "network"] == pytest.approx(10.0)
        assert p.loc["Concentrate→Smelting", "feed"] == pytest.approx(7.0)    # region B
        assert p.loc["Concentrate (trade)", "trade"] == pytest.approx(4.0)
        assert p.loc["Concentrate (trade)", "flat_prior"] == pytest.approx(0.0)

    def test_drops_zero_flow_pairs_and_signed_edges(self):
        p = _diag().chain_profile("CC")
        assert "Smelting→Cathode" not in p.index                 # e13: zero flow
        assert "Cathode→USC" not in p.index                      # e8: l < 0


# ---------------------------------------------------------------------------
# Row activity
# ---------------------------------------------------------------------------

class TestRowActivity:
    def test_family_from_the_row_metadata(self):
        r = _diag().row_frame()
        assert r["family"].tolist() == [
            "mass balance", "stripping", "conc band", "eff band", "scrap-share band",
            "lower bound", "lower bound", "lower bound", "mass balance", "soft ratio",
            "scale total & scrap anchor", "scrap-share band"]
        assert_array_equal(r["row_id"].to_numpy(), np.arange(12))

    def test_unknown_family_raises(self):
        rm = _row_meta()
        rm.loc[2, "ratio_kind"] = "grade"
        with pytest.raises(ValueError, match="family"):
            row_family(rm)

    def test_prior_breach_is_strict_at_the_tolerance(self):
        r = _diag().row_frame()
        assert np.flatnonzero(r["prior_breach"].to_numpy()).tolist() == [0, 4]  # r8 at +tol

    def test_binding_on_nonneg_rows_down_to_minus_the_tolerance(self):
        r = _diag().row_frame()
        assert np.flatnonzero(r["binding"].to_numpy()).tolist() == [2, 3, 6, 11]

    def test_on_flow_ignores_stored_zeros_and_edges_at_the_tolerance(self):
        r = _diag().row_frame()
        assert np.flatnonzero(r["binding_on_flow"].to_numpy()).tolist() == [2]
        assert np.flatnonzero(r["binding_on_zero_flows"].to_numpy()).tolist() == [3, 6, 11]

    def test_family_summary(self):
        s = _diag().family_summary()
        assert s.index.tolist() == ["mass balance", "stripping", "conc band", "eff band",
                                    "scrap-share band", "lower bound", "soft ratio",
                                    "scale total & scrap anchor"]
        assert s["cone"].tolist() == ["zero", "zero", "nonneg", "nonneg", "nonneg",
                                      "nonneg", "free", "free"]
        assert s["rows"].tolist() == [2, 1, 1, 1, 2, 3, 1, 1]
        assert s.loc["mass balance", "prior_breach"] == 1
        assert s.loc["scrap-share band", "prior_breach"] == 1
        assert s.loc["scrap-share band", "binding"] == 1
        assert s.loc["scrap-share band", "binding_on_flow"] == 0
        assert s.loc["lower bound", "binding_on_zero_flows"] == 1
        assert pd.isna(s.loc["mass balance", "binding"])         # equalities always hold
        assert pd.isna(s.loc["soft ratio", "prior_breach"])      # soft rows: z², not breach

    def test_block_sizes_count_true_nonzeros(self):
        b = _diag().block_sizes()
        assert b["rows"] == 12 and b["edges"] == 12
        assert b["rows_by_cone"] == {"free": 2, "zero": 3, "nonneg": 7}
        assert b["nnz"] == 21 and b["nnz_stored"] == 23

    def test_columns_must_match_the_constraint_block(self):
        with pytest.raises(ValueError, match="cols_t"):
            _diag(opt=_opt(cols=COLS_2020[::-1])).row_frame()


# ---------------------------------------------------------------------------
# Region slice
# ---------------------------------------------------------------------------

class TestRegionSlice:
    def test_columns_domestic_plus_legs_with_a_prior_mass_then_cc(self):
        s = _diag().region_slice("A")
        assert_array_equal(s.edge_ids, [1, 3, 9, 5, 0, 2, 13, 8, 4])
        assert s.valuation.tolist() == ["Mass"] * 4 + ["CC"] * 5
        assert_array_equal(s.is_leg, [False, False, False, True,
                                      False, False, False, False, True])
        assert s.n_legs_omitted == 2                               # e6, e7: flat-prior legs

    def test_rows_by_cone_then_family_complete_first(self):
        s = _diag().region_slice("A")
        assert_array_equal(s.row_ids, [9, 10, 0, 8, 1, 2, 3, 4, 11, 6, 7])
        assert s.cone.tolist() == ["free"] * 2 + ["zero"] * 3 + ["nonneg"] * 6
        assert_array_equal(s.complete, [True, False, True, True, True, True, False, True,
                                        False, True, True])
        assert_array_equal(s.binding_on_flow, [r == 2 for r in s.row_ids])

    def test_a_row_reached_only_through_a_stored_zero_is_not_in_the_slice(self):
        assert 5 not in _diag().region_slice("A").row_ids

    def test_standard_form_with_the_slack_block(self):
        s = _diag().region_slice("A")
        assert s.n_edges == 9 and s.n_slack == 2
        A = s.A.toarray()
        assert A.shape == (11, 11)
        assert_array_equal(A[:, 9:], np.vstack([-np.eye(2), np.zeros((9, 2))]))
        assert s.A.nnz == 17 + 2                                   # no stored zeros
        pos = {e: k for k, e in enumerate(s.edge_ids)}
        assert A[2, pos[0]] == 1 and A[2, pos[2]] == -1            # r0: e0 − e2
        assert A[1, pos[2]] == 1 and np.count_nonzero(A[1, :9]) == 1   # r10 keeps only e2
