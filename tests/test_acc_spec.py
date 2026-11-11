"""
test_acc_spec.py
================
The acc spec (#57): one in-memory declaration of an accounting structure, two readers of
it (``read_acc_spec_ods``, ``read_acc_spec_json``) and one constructor
(``AccountingStructure.from_spec``). The ODS fixtures here are the repo's first test of the
ODS path; they are written with a stdlib writer so the test needs no ODS library.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import pytest

from pysuto.acc import AccountingStructure, RuleSet, read_acc_spec_json, read_acc_spec_ods

REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# The toy spec, written once as a dict …
# =============================================================================

SPEC = {
    "sectors": [
        {"name": "Mining",   "entity": "Process"},
        {"name": "Ore",      "entity": "Flow"},
        {"name": "WR",       "entity": "SB",      "sub": "SEM"},
        {"name": "Smelting", "entity": "Process"},
        {"name": "Metal",    "entity": "Flow"},
        {"name": "Slag",     "entity": "SB",      "sub": "SEM"},
        {"name": "FU",       "entity": "SB",      "sub": "FD"},
    ],
    "regions":    ["Montara", "Portua", "Isla"],
    "valuations": ["Mass", "CC"],
    "times":      ["2001", "2002"],
    "supply":     {"Mining": ["Ore", "WR"], "Smelting": ["Metal", "Slag"]},
    "use":        {"Ore": ["Smelting"], "Metal": ["FU"]},
    "tradable":   ["Ore"],
    "accounting": [
        {"from": "Process", "to": "Flow",    "same_region": True},
        {"from": "Process", "to": "SB",      "same_region": True},
        {"from": "Flow",    "to": "Process", "same_region": False},
        {"from": "Flow",    "to": "SB",      "same_region": False},
    ],
    "bounds": [
        {"Sector_from": "Metal", "Region_from": "*", "Sector_to": "FU", "Region_to": "*",
         "Time": "2002", "Valuation": "*", "Lower": "-5", "Upper": "5"},
    ],
    "masks": [
        {"mask_name": "massbal_scope", "logic": "include",
         "sector_from": "*",      "sector_to": "*"},
        {"mask_name": "massbal_scope", "logic": "exclude",
         "sector_from": "Mining", "sector_to": "WR"},
    ],
}

REQUIRED = ("sectors", "regions", "valuations", "times",
            "supply", "use", "tradable", "accounting")


# =============================================================================
# … and once as the copper case's four files (literal sheets, not derived from SPEC)
# =============================================================================

ROOT_SHEETS = {
    "Sector_Entity": [["Sector_name", "Entity_name", "Entity_name_1"],
                      ["Mining",      "Process",     None],
                      ["Ore",         "Flow",        None],
                      ["WR",          "SB",          "SEM"],
                      ["Smelting",    "Process",     None],
                      ["Metal",       "Flow",        None],
                      ["Slag",        "SB",          "SEM"],
                      ["FU",          "SB",          "FD"]],
    "Region":        [["Region_name"], ["Montara"], ["Portua"], ["Isla"]],
    "Valuation":     [["Valuation_name"], ["Mass"], ["CC"]],
    "Time":          [["Time_name"], ["2001"], ["2002"]],
}

RULES_SHEETS = {
    "Supply":     [["P2FSB",    "Ore", "WR", "Metal", "Slag"],
                   ["Mining",   "1",   "1",  None,    None],
                   ["Smelting", None,  None, "1",     "1"]],
    "Use":        [["FSB2P",    "Smelting", "FU"],
                   ["Ore",      "1",        None],
                   ["WR",       None,       None],
                   ["Metal",    None,       "1"],
                   ["Slag",     None,       None]],
    "Trade":      [["Flow", "Traded"], ["Ore", "1"], ["Metal", None]],
    "Accounting": [["Entity_from", "Entity_to", "Same_reg"],
                   ["Process",     "Flow",      "1"],
                   ["Process",     "SB",        "1"],
                   ["Flow",        "Process",   None],
                   ["Flow",        "SB",        None]],
}

BOUNDS_SHEETS = {
    "Sheet1": [["Sector_from", "Region_from", "Sector_to", "Region_to", "Time", "Valuation",
                "Lower", "Upper"],
               ["Metal",       "*",           "FU",        "*",         "2002", "*",
                "-5",    "5"]],
}

MASKS_CSV = ("mask_name,logic,sector_from,sector_to\n"
             "massbal_scope,include,*,*\n"
             "massbal_scope,exclude,Mining,WR\n")


def _write_ods(path: Path, sheets: dict) -> None:
    """Minimal ODS: ``content.xml`` in a zip, one table per sheet; ``None`` = empty cell."""
    tables = []
    for name, rows in sheets.items():
        xml_rows = []
        for row in rows:
            cells = "".join(
                "<table:table-cell/>" if v is None else
                f"<table:table-cell><text:p>{escape(v)}</text:p></table:table-cell>"
                for v in row
            )
            xml_rows.append(f"<table:table-row>{cells}</table:table-row>")
        tables.append(f'<table:table table:name="{escape(name)}">'
                      f'{"".join(xml_rows)}</table:table>')
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content'
        ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
        ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        f'<office:body><office:spreadsheet>{"".join(tables)}'
        '</office:spreadsheet></office:body>'
        '</office:document-content>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        zf.writestr("content.xml", content)


def _write_ods_case(folder: Path) -> dict:
    """The toy as root.ods, rules.ods, bounds.ods and masks.csv in *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    _write_ods(folder / "root.ods",   ROOT_SHEETS)
    _write_ods(folder / "rules.ods",  RULES_SHEETS)
    _write_ods(folder / "bounds.ods", BOUNDS_SHEETS)
    (folder / "masks.csv").write_text(MASKS_CSV)
    return {"root":   str(folder / "root.ods"),
            "rules":  str(folder / "rules.ods"),
            "bounds": str(folder / "bounds.ods"),
            "masks":  str(folder / "masks.csv")}


def _write_json_case(folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "acc.json"
    path.write_text(json.dumps(SPEC, indent=2))
    return str(path)


def _assert_same_structure(a: AccountingStructure, b: AccountingStructure) -> None:
    pd.testing.assert_frame_equal(a.edge_index, b.edge_index)
    pd.testing.assert_frame_equal(a.node_index, b.node_index)
    assert a.masks.keys() == b.masks.keys()
    for name in a.masks:
        np.testing.assert_array_equal(a.masks[name], b.masks[name], err_msg=name)
    np.testing.assert_array_equal(a.bounds.l, b.bounds.l)
    np.testing.assert_array_equal(a.bounds.u, b.bounds.u)
    assert a.user_mask_names == b.user_mask_names


# =============================================================================
# 1. The two readers yield the spec
# =============================================================================

def test_the_ods_reader_yields_the_spec(tmp_path):
    """Matrices with "1" cells become adjacency lists; an empty row declares no pair."""
    p = _write_ods_case(tmp_path)
    assert read_acc_spec_ods(p["root"], p["rules"], p["bounds"], p["masks"]) == SPEC


def test_the_ods_reader_omits_the_optional_files_it_is_not_given(tmp_path):
    p = _write_ods_case(tmp_path)
    spec = read_acc_spec_ods(p["root"], p["rules"])
    assert "bounds" not in spec and "masks" not in spec
    assert spec == {k: v for k, v in SPEC.items() if k not in ("bounds", "masks")}


def test_the_json_reader_yields_the_spec(tmp_path):
    assert read_acc_spec_json(_write_json_case(tmp_path)) == SPEC


# =============================================================================
# 2. One constructor
# =============================================================================

def test_from_spec_builds_the_declared_structure():
    acc = AccountingStructure.from_spec(SPEC, "toy")
    assert acc.id == "toy"
    assert acc.n_nodes == 7 * 3                          # sectors × regions
    assert acc.time_range == ["2001", "2002"]
    assert acc.valuation  == ["Mass", "CC"]
    # 4 supply pairs and Metal→FU stay domestic (3 each); tradable Ore→Smelting crosses
    # (3 × 3); × 2 times × 2 valuations.
    assert acc.n_edges == (4 * 3 + 3 + 9) * 2 * 2
    assert int(acc.masks["cross_region"].sum()) == 6 * 2 * 2
    assert int(acc.masks["tradable"].sum())     == 9 * 2 * 2
    assert int(acc.masks["supply"].sum())       == 4 * 3 * 2 * 2
    sub = acc.node_index.drop_duplicates("sector").set_index("sector")["entity_sub"]
    assert sub["WR"] == "SEM" and sub["FU"] == "FD" and sub["Mining"] is None
    # The bound row reaches Metal→FU in 2002 only: 3 regions × 2 valuations.
    assert int((acc.bounds.l == -5.0).sum()) == 6 and int((acc.bounds.u == 5.0).sum()) == 6
    assert acc.user_mask_names == ["massbal_scope"]
    assert int((~acc.masks["massbal_scope"]).sum()) == 3 * 2 * 2   # Mining→WR excluded


def test_from_spec_without_the_optional_keys_has_default_bounds_and_no_user_masks():
    spec = {k: v for k, v in SPEC.items() if k not in ("bounds", "masks")}
    acc = AccountingStructure.from_spec(spec, "toy")
    assert np.all(acc.bounds.l == 0.0) and np.all(acc.bounds.u == 1e9)
    assert acc.user_mask_names == [] and "massbal_scope" not in acc.masks


def test_ruleset_reads_the_tables_it_is_built_from(tmp_path):
    p = _write_ods_case(tmp_path)
    by_file   = RuleSet(p["rules"])
    by_tables = RuleSet.from_tables(
        supply_pairs = {("Mining", "Ore"), ("Mining", "WR"), ("Smelting", "Metal"),
                        ("Smelting", "Slag")},
        use_pairs    = {("Ore", "Smelting"), ("Metal", "FU")},
        tradable     = {"Ore"},
        accounting   = {("Process", "Flow"): True,  ("Process", "SB"): True,
                        ("Flow", "Process"): False, ("Flow", "SB"):    False},
    )
    for attr in ("supply_pairs", "use_pairs", "tradable_flows", "accounting", "all_pairs"):
        assert getattr(by_file, attr) == getattr(by_tables, attr), attr
    assert not by_tables.requires_same_region("Ore",   "Smelting", "Flow", "Process")
    assert by_tables.requires_same_region("Metal", "FU",       "Flow", "SB")
    assert by_tables.requires_same_region("Mining", "Ore",     "Process", "Flow")


# =============================================================================
# 3. Three routes, one structure
# =============================================================================

def test_dict_json_and_ods_routes_build_the_same_structure(tmp_path):
    by_dict = AccountingStructure.from_spec(SPEC, "toy")
    by_json = AccountingStructure.from_spec(
        read_acc_spec_json(_write_json_case(tmp_path / "j")), "toy")
    p = _write_ods_case(tmp_path / "o")
    by_ods  = AccountingStructure(id="toy", root_file_path=p["root"],   # masks.csv found
                                  rules_file_path=p["rules"],            # beside root.ods
                                  bounds_file_path=p["bounds"])
    _assert_same_structure(by_json, by_dict)
    _assert_same_structure(by_ods,  by_dict)
    assert by_dict.user_mask_names == ["massbal_scope"]


# =============================================================================
# 4. Errors
# =============================================================================

@pytest.mark.parametrize("key", REQUIRED)
def test_a_spec_missing_a_required_key_raises_naming_it(key):
    spec = {k: v for k, v in SPEC.items() if k != key}
    with pytest.raises(KeyError, match=f"'{key}'"):
        AccountingStructure.from_spec(spec, "toy")


def test_an_accounting_pair_naming_an_unknown_entity_raises():
    spec = dict(SPEC, accounting=SPEC["accounting"]
                + [{"from": "Flow", "to": "Stock", "same_region": True}])
    with pytest.raises(ValueError, match="'Stock'"):
        AccountingStructure.from_spec(spec, "toy")


def test_a_spec_with_an_unknown_key_raises():
    """A misspelt optional key would otherwise drop its bounds or masks without a word."""
    spec = {("bound" if k == "bounds" else k): v for k, v in SPEC.items()}
    with pytest.raises(ValueError, match="'bound'"):
        AccountingStructure.from_spec(spec, "toy")


def test_a_label_that_is_not_a_string_raises():
    """Time labels are strings; JSON makes `2001` an int."""
    spec = dict(SPEC, times=[2001, 2002])
    with pytest.raises(TypeError, match="2001"):
        AccountingStructure.from_spec(spec, "toy")


# =============================================================================
# 5. scripts/acc.py picks the reader from what current/ holds
# =============================================================================

def _run_acc_script(monkeypatch, tmp_path: Path, input_root: Path, out: Path) -> None:
    script = REPO / "scripts" / "acc.py"
    spec = importlib.util.spec_from_file_location("_acc_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)                       # logs/ lands in the tmp dir
    monkeypatch.setattr(sys, "argv", ["acc.py", "--id", "toy",
                                      "--input-root", str(input_root), "--out", str(out)])
    mod.main()


def test_acc_script_builds_from_acc_json_what_it_builds_from_the_ods_files(tmp_path,
                                                                         monkeypatch):
    _write_ods_case(tmp_path / "ods" / "acc" / "toy" / "current")
    _write_json_case(tmp_path / "json" / "acc" / "toy" / "current")
    _run_acc_script(monkeypatch, tmp_path, tmp_path / "ods",  tmp_path / "out_ods")
    _run_acc_script(monkeypatch, tmp_path, tmp_path / "json", tmp_path / "out_json")
    for name in ("edge_index.csv", "edge_index_all.csv", "node_index.csv", "root_nodes.csv",
                 "bounds_l.npy", "bounds_u.npy", "masks_user.npz"):
        assert ((tmp_path / "out_json" / "toy" / name).read_bytes()
                == (tmp_path / "out_ods" / "toy" / name).read_bytes()), name


@pytest.mark.parametrize("ods_file", ["root.ods", "rules.ods", "bounds.ods", "masks.csv"])
def test_acc_script_refuses_a_current_dir_holding_acc_json_and_an_ods_file(tmp_path,
                                                                          monkeypatch,
                                                                          ods_file):
    """root.ods is the acceptance case; any of the four would be ignored beside acc.json."""
    _write_ods_case(tmp_path / "ods")
    current = tmp_path / "input" / "acc" / "toy" / "current"
    _write_json_case(current)
    (current / ods_file).write_bytes((tmp_path / "ods" / ods_file).read_bytes())
    with pytest.raises(ValueError, match=f"acc.json and \\['{ods_file}'\\]"):
        _run_acc_script(monkeypatch, tmp_path, tmp_path / "input", tmp_path / "out")
    assert not (tmp_path / "out" / "toy").exists()
