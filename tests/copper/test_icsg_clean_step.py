"""The ICSG clean step end to end, on a toy registry in ``tmp_path`` (#281).

``clean_icsg_flows`` is the step ``icsg_clean.py --to-feed`` runs: the order that ADR 0032
and #73-#85 make load-bearing (one territory, one label, before the nesting subtraction),
the registered inputs it reads (eff, icsg_aggregates, conc, the gross BACI version) and
the parents record that #63's guard and the feed's input version read. ``icsg_comb``'s
region renames decide which ICSG label meets which root region.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_clean import clean_icsg_flows, read_eff  # noqa: E402
from icsg_comb import clean_prices, main as comb_main, rename_regions  # noqa: E402
from pysuto.input_registry import InputRegistries  # noqa: E402

PARAM_COLS = ["param_id", "param_type", "description", "a_sector_from", "a_entity_from",
              "a_region_from", "a_sector_to", "a_entity_to", "a_region_to", "a_time",
              "a_valuation", "b_sector_from", "b_entity_from", "b_region_from",
              "b_sector_to", "b_entity_to", "b_region_to", "b_time", "b_valuation",
              "value_nominal", "unc_type", "value_min", "value_max"]
EFF = {"effsmelting": 0.965, "effcfr": 0.9875, "effer": 0.997}
_K = EFF["effcfr"] * EFF["effer"]


def _params(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PARAM_COLS).fillna("").to_csv(path, index=False, sep=";")


def _registry(root, eff=None):
    """baci v1 (one cathode leg), the icsg feed's concordance, conc, eff and an empty
    icsg_aggregates; registered, returned as the registries."""
    eff = EFF if eff is None else eff
    baci = root / "feed" / "baci" / "current"
    baci.mkdir(parents=True)
    pd.DataFrame([(2020, 740311, 276, 40, 1.0)],
                 columns=["Year", "Sector", "Region_from", "Region_to", "Value"]
                 ).to_csv(baci / "values.csv", index=False)
    with pd.ExcelWriter(baci / "conc.xlsx") as w:
        pd.DataFrame({"src_sector": [740311], "root_sector": ["Cathode"]}
                     ).to_excel(w, sheet_name="sector_map", index=False)
        pd.DataFrame({"src_region": [276, 40], "root_region": ["Germany", "Austria"]}
                     ).to_excel(w, sheet_name="region_map", index=False)
    icsg = root / "feed" / "icsg" / "current"
    icsg.mkdir(parents=True)
    with pd.ExcelWriter(icsg / "conc.xlsx") as w:
        pd.DataFrame({"src_region": ["Belgium", "Belgium-Luxembourg", "Austria"],
                      "root_region": ["Belgium", "Belgium", "Austria"]}
                     ).to_excel(w, sheet_name="region_map", index=False)
    _params(root / "params" / "conc" / "current" / "params.csv", [
        dict(param_id="concanodesupply", param_type="concentration", a_sector_from="Anode",
             a_entity_from="Flow", a_valuation="Mass", value_nominal=0.94,
             unc_type="min_max", value_min=0.9, value_max=0.98)])
    _params(root / "params" / "eff" / "current" / "params.csv", [
        dict(param_id=k, param_type="efficiency", value_nominal=v, unc_type="min_max",
             value_min=v - 0.01, value_max=v + 0.01) for k, v in eff.items()])
    agg = root / "params" / "icsg_aggregates" / "current"
    agg.mkdir(parents=True)
    (agg / "params.csv").write_text("src_region;root_region;year_from;year_to\n")
    return InputRegistries.initialize(root=root)


# the 2005 yearbook's "Congo" is DR Congo (#85): every clean step relabels it, so every
# fixture carries one such row
_CONGO_2005 = ("Congo Rep.", "Primary", 2004, 1.0, "Mining", "Supply", "kt", 2005)


def _raw(rows):
    cols = ["Region", "Type", "Year", "Value", "Process", "Supply_Use", "Unit",
            "Year_loaded_from_file"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])[cols]


def test_the_eff_divisors_are_the_registered_params_not_an_edited_current(tmp_path):
    """The model's efficiencies are the registered ``eff`` version in force, the file the
    registry hashes; an edit of ``current/`` waits for its registration (#93)."""
    _registry(tmp_path)
    (tmp_path / "params" / "eff" / "current" / "params.csv").write_text(
        (tmp_path / "params" / "eff" / "current" / "params.csv").read_text().replace(
            "0.965", "0.5"))
    eff = read_eff(tmp_path).set_index("Process")["Value_nominal"]
    assert eff.to_dict() == {"Smelting": 0.965, "CFR": 0.9875, "ER": 0.997}


def test_a_territory_labelled_twice_is_merged_before_the_nesting_and_the_parents_recorded(
        tmp_path):
    """Belgium's refined-secondary series carries the joint label, its smelter series
    the root one: the subtraction finds the smelter route only if the labels are merged
    first (ADR 0032). The parents file names every registered input the step read."""
    _registry(tmp_path)
    raw = _raw([("Belgium-Luxembourg", "Secondary", 2010, 158.0, "Refining", "Supply", "kt", 2014),
                ("Belgium", "Secondary", 2010, 150.6, "Smelting", "Supply", "kt", 2014),
                ("Austria", "Primary", 2010, 40.0, "Semi", "Supply", "kt", 2014),
                _CONGO_2005])
    clean_dir = tmp_path / "clean"
    out = clean_icsg_flows(raw, 1, root=str(tmp_path), clean_dir=str(clean_dir))

    df = pd.read_csv(out)
    assert set(df["Region"]) == {"Belgium", "Congo, D.R."}   # alias merged, Semi supply gone
    scrap = df[(df["Type"] == "Scrap") & (df["Process"] == "Refining")]["Value"].iloc[0]
    assert scrap == pytest.approx(round((158.0 - 150.6 * _K) / _K, 3))
    smelt = df[df["Process"] == "Smelting"]["Value"].iloc[0]
    assert smelt == pytest.approx(round(150.6 / EFF["effsmelting"], 3))
    assert (df[df["Region"] == "Belgium"]["Supply_Use"] == "Use").all()    # production -> input

    parents = json.loads((clean_dir / "icsg_flows_clean_parents.json").read_text())
    assert parents == {"inputs": {"feed": {"baci": 1},
                                  "params": {"conc": 1, "eff": 1, "icsg_aggregates": 1}}}


def test_a_netted_baci_version_never_reaches_the_clean_output(tmp_path):
    _registry(tmp_path)
    (tmp_path / "feed" / "baci" / "current" / "reexport_ledger.csv").write_text("sector\n")
    InputRegistries.initialize(root=tmp_path)                 # baci v2 carries the ledger
    raw = _raw([("Austria", "Secondary", 2010, 5.0, "Refining", "Supply", "kt", 2014),
                _CONGO_2005])
    with pytest.raises(ValueError, match="netted"):
        clean_icsg_flows(raw, 2, root=str(tmp_path), clean_dir=str(tmp_path / "clean"))


# ---------------------------------------------------------------------------
# icsg_comb: the region renames that decide the ICSG-to-root labels, and main
# ---------------------------------------------------------------------------

def test_region_renames_unify_the_labels_and_drop_what_is_no_country():
    s = pd.DataFrame({"Region": ["U.S.", "Others"], "Value": [1.0, 2.0]})
    f = pd.DataFrame({"Region": ["Russian Fed.", "Korean", "South", "Congo DR"],
                      "Value": [1.0, 2.0, 3.0, 4.0]})
    s, f = rename_regions(s, f)
    assert s["Region"].tolist() == ["United States"]
    assert f["Region"].tolist() == ["Russia", "South Korea", "Congo, D.R."]


def test_prices_take_their_unit_from_the_exchange_when_the_sheet_has_none():
    p = pd.DataFrame({"Exchange_type": ["LME Settlement", "COMEX 1M Settlement"], "Year": [2020, 2020],
                      "Value": [1.0, 2.0], "Value_type": ["avg", "avg"],
                      "Unit": [np.nan, np.nan], "Year_loaded_from_file": [2025, 2025]})
    out = clean_prices(p)
    assert list(out.columns) == ["Type", "Year", "Value", "Value_type", "Unit",
                                 "Year_loaded_from_file"]
    assert out["Unit"].tolist() == ["USD/t", "cents/lb"]


def test_the_combiner_writes_its_three_tables_to_out_dir(tmp_path):
    pd.DataFrame({"Exchange_type": ["LME"], "Year": [2020], "Value": [1.0],
                  "Value_type": ["avg"], "Unit": [np.nan]}
                 ).to_csv(tmp_path / "icsg_2025_prices.csv", index=False)
    pd.DataFrame({"Region": ["U.S."], "Type": ["Primary"], "Process": ["Refining"],
                  "Supply_Use": ["Supply"], "Year": [2020], "Value": [10.0], "Unit": ["kt"]}
                 ).to_csv(tmp_path / "icsg_2025_flows.csv", index=False)
    pd.DataFrame({"Region": ["U.S."], "Type": ["Producers"], "Stock_type": ["PSC"],
                  "Year": [2020], "Value": [3.0], "Unit": ["kt"]}
                 ).to_csv(tmp_path / "icsg_2025_stocks.csv", index=False)
    comb_main(vintages=(2025,), out_dir=str(tmp_path))
    flows = pd.read_csv(tmp_path / "icsg_flows.csv")
    assert flows["Region"].tolist() == ["United States"]
    assert (tmp_path / "icsg_stocks.csv").is_file() and (tmp_path / "icsg_prices.csv").is_file()
