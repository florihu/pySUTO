"""test_constraints_application.py
===============================
The tests of ``tests/test_constraints.py`` that read the copper application (#304, #337):
the exploration generators of ``scripts/explo/``, the copper case's scale-total knobs and
its ``stock_bounds`` params table.
The methods are moved unchanged (the class helpers they use are borrowed from the library
test), only the path to the repository follows their new depth.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_constraints
import copper_case  # noqa: F401,E402  (puts scripts/ on sys.path, #189)
from test_constraints import (  # noqa: E402
    TestScrapLevelAnchor as _ScrapLevelAnchor,
    _make_scrap_acc, _scrap_f_and_cov)


class TestRatioBoundBuilderApplication:
    def _load_stock_bounds_generator(self):
        """Import scripts/explo/stock_bounds_study.py by path (it is import-safe:
        data reads live in functions, main() is guarded)."""
        import importlib.util, pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        mod_path = repo / "scripts" / "explo" / "stock_bounds_study.py"
        try:
            spec = importlib.util.spec_from_file_location("stock_bounds_study", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ImportError as e:                       # e.g. matplotlib absent in a minimal env
            pytest.skip(f"stock_bounds_study not importable: {e}")
        return mod

    @pytest.mark.parametrize("indicator,stock,denom", [
        ("PSC", "PSC", "Cathode"),
        ("USC", "USC", "FU"),
        ("MESC", "MESC", "FU"),
    ])
    def test_stock_bound_generator_orientation(self, indicator, stock, denom):
        """`_build_param_row` writes the stock edge on the b-side (numerator) and the
        FU/Cathode denominator on the a-side — the orientation RatioBoundBuilder needs.
        A straight a→a / b→b mapping inverts the bound (cathode-pool collapse)."""
        gen = self._load_stock_bounds_generator()
        row = gen._build_param_row(indicator, "Chile", 0.05, "test", "denom-label")
        assert row["b_sector_to"] == stock, f"{indicator}: stock edge must be the b-side numerator"
        assert row["a_sector_to"] == denom, f"{indicator}: denominator must be the a-side"
        assert float(row["value_min"]) < 0 < float(row["value_max"]), "two-sided ±p95 band in value_min/max"

    def test_scrap_share_generator_orientation(self):
        """scrap_share_study._build_param_row: Scrap numerator on the b-side,
        wildcard pooled denominator on the a-side, band in value_min/value_max,
        underscore-free param_id, CC on both sides."""
        import importlib.util, pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        mod_path = repo / "scripts" / "explo" / "scrap_share_study.py"
        try:
            spec = importlib.util.spec_from_file_location("scrap_share_study", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ImportError as e:
            pytest.skip(f"scrap_share_study not importable: {e}")
        row = mod._build_param_row("Smelting", "Rep. of Test", 0.1, 0.4,
                                   nominal=0.2, n_obs=12, own=True)
        assert row["b_sector_from"] == "Scrap" and row["b_entity_from"] == "SB", \
            "Scrap leg must be the b-side numerator"
        assert row["a_sector_from"] == "*" and row["a_sector_to"] == "Smelting", \
            "a-side must be the wildcard pooled denominator into the process"
        assert row["a_valuation"] == "CC" and row["b_valuation"] == "CC"
        assert float(row["value_min"]) == pytest.approx(0.1)
        assert float(row["value_max"]) == pytest.approx(0.4)
        assert "_" not in row["param_id"], "param_id fills the variant slot (underscore-free)"

    def test_cfr_share_uses_inferred_total_input(self):
        """scrap_share_study.share_frame: the CFR denominator is the anode
        entering ER ÷ eff_CFR (ADR 0021 removed the matte series), so the
        share is direct-to-refinery scrap over the total CFR input (ADR 0030)."""
        import importlib.util, pathlib
        import pandas as pd
        repo = pathlib.Path(__file__).resolve().parents[2]
        mod_path = repo / "scripts" / "explo" / "scrap_share_study.py"
        try:
            spec = importlib.util.spec_from_file_location("scrap_share_study", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ImportError as e:
            pytest.skip(f"scrap_share_study not importable: {e}")
        df = pd.DataFrame([
            dict(Region="Germany", Year=2020, Type="Scrap", Process="Refining", Supply_Use="Use", Value=20.0),
            dict(Region="Germany", Year=2020, Type="Anode", Process="Refining", Supply_Use="Use", Value=395.0),
            dict(Region="Germany", Year=2021, Type="Scrap", Process="Refining", Supply_Use="Use", Value=10.0),
            # no anode series in 2021 → no share observation, not a share of 1
            dict(Region="Chile", Year=2020, Type="Scrap", Process="Smelting", Supply_Use="Use", Value=5.0),
            dict(Region="Chile", Year=2020, Type="Concentrate", Process="Smelting", Supply_Use="Use", Value=95.0),
        ])
        fr = mod.share_frame(df, "CFR", eff={"effcfr": 0.9875})
        assert list(fr.index) == [("Germany", 2020)]
        assert fr["share"].iloc[0] == pytest.approx(20.0 / (395.0 / 0.9875))
        sm = mod.share_frame(df, "Smelting", eff={"effcfr": 0.9875})
        assert sm["share"].iloc[0] == pytest.approx(0.05)

    def test_stock_bound_params_contract(self):
        """The shipped stock_bounds params keep the correct orientation: stock edge
        (PSC/USC/MESC) on the b-side, FU/Cathode on the a-side, symmetric ±band in
        value_min/value_max. Skips when the (gitignored) params file is absent."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        p = repo / "data" / "input" / "params" / "stock_bounds" / "current" / "params.csv"
        if not p.exists():
            pytest.skip("stock_bounds/current/params.csv absent (gitignored data dir)")
        df = pd.read_csv(p, sep=None, engine="python", dtype=str)
        s = df[df["param_type"] == "stock_bound"]
        assert len(s) > 0, "no stock_bound rows in shipped params"
        assert s["b_sector_to"].isin(["PSC", "USC", "MESC"]).all(), "stock edge must be the b-side numerator"
        assert s["a_sector_to"].isin(["Cathode", "FU"]).all(), "denominator must be the a-side"
        vmin = pd.to_numeric(s["value_min"]); vmax = pd.to_numeric(s["value_max"])
        assert (vmin < 0).all() and (vmax > 0).all() and np.allclose(vmin, -vmax, rtol=1e-12), \
            "value_min/value_max must be a symmetric ±p95 band"


class TestScrapLevelAnchorApplication:
    _masks = staticmethod(_ScrapLevelAnchor._masks)
    _build = _ScrapLevelAnchor._build
    GROUP_KEYS = _ScrapLevelAnchor.GROUP_KEYS
    CV = _ScrapLevelAnchor.CV

    def test_every_row_is_feed_backed_so_the_adr0025_floor_applies(self):
        """Because the mask is coverage-gated every group is 100 % feed-backed,
        so ADR 0025's 0.3 kt floor applies unconditionally and the fabricated
        branch is never taken."""
        from copper import CASE
        from pysuto.constraints.anchors import scale_total_cv
        spec = CASE.const
        SCALE_TOTAL_ABS_FLOOR = spec.SCALE_TOTAL_ABS_FLOOR
        acc = _make_scrap_acc()
        f, cov = _scrap_f_and_cov(acc)
        absf = np.abs(f)
        seen = []

        def _resolver(key_map, edge_ids):
            ids = np.asarray(edge_ids, dtype=int)
            tot = absf[ids].sum()
            feed_share = absf[ids[cov[ids]]].sum() / tot
            seen.append(feed_share)
            return scale_total_cv(feed_share, abs(float(f[ids].sum())),
                                  anchor_cv=spec.ANCHOR_CV,
                                  fabricated_cv=spec.FABRICATED_ANCHOR_CV,
                                  feed_share_observed=spec.FEED_SHARE_OBSERVED,
                                  abs_floor=SCALE_TOTAL_ABS_FLOOR)

        smelt, cfr = self._masks(acc, cov)
        for mask, prefix in ((smelt, "smelterscrap"), (cfr, "directrefscrap")):
            block = self._build(acc, f, mask, prefix, resolver=_resolver)
            assert np.all(block.sigma_c >= SCALE_TOTAL_ABS_FLOOR - 1e-12)
        assert seen and all(s == pytest.approx(1.0) for s in seen)
