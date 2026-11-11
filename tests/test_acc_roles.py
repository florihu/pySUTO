"""
test_acc_roles.py
=================
Roles, not labels (#119, ADR 0034): the accounting structure binds a case's labels to the
framework's roles, keeps that binding and its rule-derived masks across ``save``/``load``,
and the library reads only roles. Also: the grep gate on label literals, acc compatibility
by spec comparison, and the copper ``root.ods`` amendment (only a ``role`` column added).
"""

from __future__ import annotations

import copy
import io
import json
import logging
import pickle
import re
import subprocess
import tokenize
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto.acc import (
    ALL_SPEC_KEYS, EDGE_UNIVERSE_KEYS, AccountingStructure, BoundaryRole, ValuationRole,
    Valuations, _ods_sheet_names, _read_ods, acc_spec_differences, read_acc_spec_dir,
    require_compatible_acc,
)
from pysuto.input_registry import InputRegistry
from pysuto.psut import classify_blocks

from test_acc_spec import ROOT_SHEETS, _write_ods, _write_ods_case

REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# One toy chain, written under any labels
# =============================================================================

def _spec(content="CC", gross="Mass", nature="Nature", secondary="SEM", demand="FD"):
    """Extraction, two processes, a secondary input, two residual outputs and final
    demand, in two regions; the valuation and boundary labels are the arguments."""
    return {
        "sectors": [
            {"name": "Ground",   "entity": "SB", "sub": nature,    "role": "nature"},
            {"name": "Mining",   "entity": "Process"},
            {"name": "Ore",      "entity": "Flow"},
            {"name": "WR",       "entity": "SB", "sub": secondary, "role": "secondary"},
            {"name": "Smelting", "entity": "Process"},
            {"name": "Scrap",    "entity": "SB", "sub": secondary, "role": "secondary"},
            {"name": "Metal",    "entity": "Flow"},
            {"name": "Slag",     "entity": "SB", "sub": secondary, "role": "secondary"},
            {"name": "FU",       "entity": "SB", "sub": demand,    "role": "final_demand"},
        ],
        "regions":    ["Montara", "Portua"],
        "valuations": [{"name": gross, "role": "gross"}, {"name": content, "role": "content"}],
        "times":      ["2001", "2002"],
        "supply":     {"Mining": ["Ore", "WR"], "Smelting": ["Metal", "Slag"]},
        "use":        {"Ground": ["Mining"], "Ore": ["Smelting"], "Scrap": ["Smelting"],
                       "Metal": ["FU"]},
        "tradable":   ["Ore", "Metal"],
        "accounting": [
            {"from": "Process", "to": "Flow",    "same_region": True},
            {"from": "Process", "to": "SB",      "same_region": True},
            {"from": "Flow",    "to": "Process", "same_region": False},
            {"from": "Flow",    "to": "SB",      "same_region": False},
            {"from": "SB",      "to": "Process", "same_region": True},
        ],
        "bounds": [
            {"Sector_from": "Metal", "Region_from": "*", "Sector_to": "FU", "Region_to": "*",
             "Time": "2002", "Valuation": content, "Lower": "-5", "Upper": "5"},
        ],
        "masks": [
            {"mask_name": "massbal_scope", "logic": "include",
             "sector_from": "*",      "sector_to": "*"},
            {"mask_name": "massbal_scope", "logic": "exclude",
             "sector_from": "Mining", "sector_to": "WR"},
        ],
    }


COPPER_LABELS = _spec()
OWN_LABELS    = _spec(content="Content", gross="Gross",
                      nature="Raw", secondary="Sec", demand="Demand")


def _assert_same(a: AccountingStructure, b: AccountingStructure) -> None:
    pd.testing.assert_frame_equal(a.edge_index, b.edge_index)
    assert a.masks.keys() == b.masks.keys()
    for name in a.masks:
        np.testing.assert_array_equal(a.masks[name], b.masks[name], err_msg=name)
    np.testing.assert_array_equal(a.bounds.l, b.bounds.l)
    np.testing.assert_array_equal(a.bounds.u, b.bounds.u)


# =============================================================================
# 1. The roles and their accessors
# =============================================================================

def test_the_role_vocabulary_is_the_glossarys():
    assert [r.value for r in BoundaryRole] == ["nature", "secondary", "final_demand"]
    assert [r.value for r in ValuationRole] == ["content", "gross"]


def test_a_spec_binds_its_labels_to_roles():
    acc = AccountingStructure.from_spec(OWN_LABELS, "own")
    assert acc.valuations == Valuations(content="Content", gross="Gross")
    assert acc.valuation == ["Gross", "Content"]                   # the axis keeps its order
    assert acc.roles == {"Ground": BoundaryRole.NATURE,   "WR":   BoundaryRole.SECONDARY,
                         "Scrap":  BoundaryRole.SECONDARY, "Slag": BoundaryRole.SECONDARY,
                         "FU":     BoundaryRole.FINAL_DEMAND}


# =============================================================================
# 2. Own labels: save/load round-trip, and the same structure as copper's labels
# =============================================================================

def test_own_labels_round_trip_through_save_and_load(tmp_path):
    acc = AccountingStructure.from_spec(OWN_LABELS, "own", storage_dir=str(tmp_path))
    acc.save()
    loaded = AccountingStructure.load("own", storage_dir=str(tmp_path))
    _assert_same(loaded, acc)
    # tradable was all-false after load() until the rule tables were persisted (#119)
    assert int(loaded.masks["tradable"].sum()) == int(acc.masks["tradable"].sum()) > 0
    assert loaded.roles == acc.roles and loaded.valuations == acc.valuations
    for attr in ("supply_pairs", "use_pairs", "tradable_flows", "accounting"):
        assert getattr(loaded.ruleset, attr) == getattr(acc.ruleset, attr), attr
    assert loaded.user_mask_names == acc.user_mask_names == ["massbal_scope"]


def test_own_labels_give_copper_labels_structure_edge_for_edge():
    own = AccountingStructure.from_spec(OWN_LABELS,    "own")
    cu  = AccountingStructure.from_spec(COPPER_LABELS, "cu")
    relabel = {"Content": "CC", "Gross": "Mass"}
    renamed = own.edge_index.assign(valuation=own.edge_index["valuation"].map(relabel))
    pd.testing.assert_frame_equal(renamed, cu.edge_index)
    pd.testing.assert_frame_equal(own.node_index[["sector", "entity", "region"]],
                                  cu.node_index[["sector", "entity", "region"]])
    for name in cu.masks:
        np.testing.assert_array_equal(own.masks[name], cu.masks[name], err_msg=name)
    np.testing.assert_array_equal(own.bounds.l, cu.bounds.l)
    np.testing.assert_array_equal(own.bounds.u, cu.bounds.u)
    assert own.roles == cu.roles
    # the library reads roles: the recipes and the PSUT blocks do not see the labels
    assert own.process_recipes == cu.process_recipes
    assert any(r.wastes for r in own.process_recipes)
    pd.testing.assert_series_equal(classify_blocks(own.edge_meta, own.roles),
                                   classify_blocks(cu.edge_meta, cu.roles))


# =============================================================================
# 3. Validation raises — on build and on load
# =============================================================================

def _with_sector(spec, name, **changes):
    spec = copy.deepcopy(spec)
    entry = next(s for s in spec["sectors"] if s["name"] == name)
    for key, value in changes.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return spec


def test_a_sector_declared_twice_raises():
    """sector → entity is 1:1."""
    spec = copy.deepcopy(COPPER_LABELS)
    spec["sectors"].append({"name": "Ore", "entity": "SB", "sub": "SEM", "role": "secondary"})
    with pytest.raises(ValueError, match=r"sector → entity must be 1:1.*'Ore'"):
        AccountingStructure.from_spec(spec, "toy")


def test_an_sb_sector_without_a_role_raises():
    with pytest.raises(ValueError, match="SB sector 'Slag' has no boundary role"):
        AccountingStructure.from_spec(_with_sector(COPPER_LABELS, "Slag", role=None), "toy")


def test_an_unknown_boundary_role_raises():
    with pytest.raises(ValueError, match="'Slag' has unknown boundary role 'residual'"):
        AccountingStructure.from_spec(_with_sector(COPPER_LABELS, "Slag", role="residual"),
                                      "toy")


def test_a_role_on_a_process_sector_raises():
    with pytest.raises(ValueError, match="only an SB sector carries a role"):
        AccountingStructure.from_spec(_with_sector(COPPER_LABELS, "Mining", role="nature"),
                                      "toy")


def test_an_unknown_valuation_role_raises():
    spec = copy.deepcopy(COPPER_LABELS)
    spec["valuations"][0]["role"] = "net"
    with pytest.raises(ValueError, match="valuation 'Mass' has unknown role 'net'"):
        AccountingStructure.from_spec(spec, "toy")


@pytest.mark.parametrize("roles", [("content", "content"), ("gross", "gross")])
def test_not_exactly_one_content_and_one_gross_valuation_raises(roles):
    spec = copy.deepcopy(COPPER_LABELS)
    for v, role in zip(spec["valuations"], roles):
        v["role"] = role
    with pytest.raises(ValueError, match="exactly one content and one gross"):
        AccountingStructure.from_spec(spec, "toy")


def test_a_valuation_without_a_role_raises_naming_the_key():
    spec = copy.deepcopy(COPPER_LABELS)
    del spec["valuations"][1]["role"]
    with pytest.raises(KeyError, match=r"valuations\[1\] is missing required key 'role'"):
        AccountingStructure.from_spec(spec, "toy")


def test_a_valuation_given_as_a_bare_label_raises():
    """The #57 format (``"valuations": ["Mass", "CC"]``) binds no role."""
    spec = dict(COPPER_LABELS, valuations=["Mass", "CC"])
    with pytest.raises(TypeError, match=r"valuations\[0\] must be a dict"):
        AccountingStructure.from_spec(spec, "toy")


def _saved(tmp_path) -> Path:
    AccountingStructure.from_spec(COPPER_LABELS, "cu", storage_dir=str(tmp_path)).save()
    return tmp_path / "cu" / "metadata.pkl"


def _rewrite(path: Path, edit) -> None:
    with open(path, "rb") as f:
        meta = pickle.load(f)
    edit(meta)
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def test_load_validates_the_persisted_roles(tmp_path):
    path = _saved(tmp_path)
    _rewrite(path, lambda meta: meta["roles"].pop("Slag"))
    with pytest.raises(ValueError, match="SB sector 'Slag' has no boundary role"):
        AccountingStructure.load("cu", storage_dir=str(tmp_path))


def test_load_refuses_a_structure_saved_before_roles(tmp_path):
    """A built acc from before #119 has neither roles nor rule tables: rebuild it."""
    def drop_the_119_keys(meta):
        for key in ("roles", "valuation_roles", "rule_tables"):
            del meta[key]

    _rewrite(_saved(tmp_path), drop_the_119_keys)
    with pytest.raises(ValueError, match="saved before roles.*scripts/acc.py --id cu"):
        AccountingStructure.load("cu", storage_dir=str(tmp_path))


def test_the_procedural_constructor_binds_no_roles_and_refuses_an_sb_node():
    nodes = pd.DataFrame({"sector": ["Mining", "Ore"], "entity": ["P", "F"],
                          "region": ["A", "A"]})
    acc = AccountingStructure(id="t", root={"nodes": nodes, "time_range": [0],
                                            "valuation": ["mass"]})
    assert acc.roles is None and acc.valuations is None
    sb = pd.DataFrame({"sector": ["Mining", "Waste"], "entity": ["Process", "SB"],
                       "region": ["A", "A"]})
    with pytest.raises(ValueError, match=r"binds no roles.*\['Waste'\]"):
        AccountingStructure(id="t", root={"nodes": sb, "time_range": [0],
                                          "valuation": ["mass"]})


# =============================================================================
# 4. Grep gate: the library compares no case label
# =============================================================================

# A string literal whose whole value is one of these labels (any quote or prefix); a
# docstring or comment that mentions one is not a literal of it.
_LABELS  = ("CC", "Mass", "SEM", "FD", "Nature", "ROW")
_LITERAL = re.compile(r"""^[rRuUfFbB]{0,2}(['"])(%s)\1$""" % "|".join(_LABELS))


def test_no_role_or_valuation_label_literal_in_the_library():
    root = REPO / "src" / "pysuto"
    hits = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == "acc.py" or rel.startswith("cases/"):
            continue
        source = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING and _LITERAL.match(tok.string):
                hits.append(f"{rel}:{tok.start[0]}: {tok.string}")
    assert hits == []


def test_the_gate_matches_a_label_literal_and_not_prose():
    assert _LITERAL.match('"CC"') and _LITERAL.match("'ROW'") and _LITERAL.match('f"SEM"')
    assert not _LITERAL.match('"kt CC"') and not _LITERAL.match('"""ROW region."""')


# =============================================================================
# 5. Acc compatibility compares specs, not versions
# =============================================================================

def acc_input_versions(root: Path, acc_id: str, specs: list) -> InputRegistry:
    """An acc input registry under *root* whose id *acc_id* has versions v1, v2, … =
    *specs*, each registered as its ``acc.json``. Shared with the lineage-check tests."""
    current = root / "acc" / acc_id / "current"
    current.mkdir(parents=True)
    for spec in specs:
        (current / "acc.json").write_text(json.dumps(spec))
        reg = InputRegistry("acc", root).initialize()
    return reg


ROLE_ONLY   = _with_sector(COPPER_LABELS, "WR", role="nature")
MASKS_ONLY  = dict(COPPER_LABELS, masks=COPPER_LABELS["masks"][:1])
BOUNDS_ONLY = dict(COPPER_LABELS, bounds=[dict(COPPER_LABELS["bounds"][0], Lower="-6")])
SECTORS     = dict(COPPER_LABELS, sectors=COPPER_LABELS["sectors"]
                   + [{"name": "Dust", "entity": "SB", "sub": "SEM", "role": "secondary"}])


def test_role_masks_and_bounds_differences_pass_the_edge_universe_check(tmp_path):
    reg = acc_input_versions(tmp_path, "toy",
                             [COPPER_LABELS, ROLE_ONLY, MASKS_ONLY, BOUNDS_ONLY])
    assert reg.latest_version("toy") == 4
    for v in (2, 3, 4):
        require_compatible_acc(reg, "toy", 1, v, EDGE_UNIVERSE_KEYS)       # Stage 2, 4, 6
    require_compatible_acc(reg, "toy", 1, 2, ALL_SPEC_KEYS)    # role never breaks Stage 3


def test_a_difference_outside_the_keys_is_logged(tmp_path, caplog):
    reg = acc_input_versions(tmp_path, "toy", [COPPER_LABELS, ROLE_ONLY, MASKS_ONLY])
    with caplog.at_level(logging.WARNING):
        require_compatible_acc(reg, "toy", 1, 2, EDGE_UNIVERSE_KEYS)       # role: silent
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    with caplog.at_level(logging.WARNING):
        require_compatible_acc(reg, "toy", 1, 3, EDGE_UNIVERSE_KEYS)
    assert "differ only in ['masks']" in caplog.text


def test_a_sector_difference_raises_naming_the_key(tmp_path):
    reg = acc_input_versions(tmp_path, "toy", [COPPER_LABELS, SECTORS])
    with pytest.raises(ValueError, match=r"toy v1 and v2; they differ in \['sectors'\]"):
        require_compatible_acc(reg, "toy", 1, 2, EDGE_UNIVERSE_KEYS)


def test_the_constraint_check_raises_on_a_bounds_difference(tmp_path):
    reg = acc_input_versions(tmp_path, "toy", [COPPER_LABELS, BOUNDS_ONLY])
    with pytest.raises(ValueError, match=r"differ in \['bounds'\]"):
        require_compatible_acc(reg, "toy", 1, 2, ALL_SPEC_KEYS)                # Stage 3


def test_the_ods_format_compares_role_free(tmp_path):
    """The copper case's format: a role-only edit of root.ods leaves the spec equal."""
    _write_ods_case(tmp_path / "a")
    sheets = copy.deepcopy(ROOT_SHEETS)
    sheets["Sector_Entity"][3][3] = "nature"                       # WR: secondary → nature
    _write_ods_case(tmp_path / "b")
    _write_ods(tmp_path / "b" / "root.ods", sheets)
    a, b = read_acc_spec_dir(tmp_path / "a"), read_acc_spec_dir(tmp_path / "b")
    assert a["sectors"] != b["sectors"]
    assert acc_spec_differences(a, b, ALL_SPEC_KEYS) == []


# =============================================================================
# 6. The copper root.ods gains only the role column (input acc/copper v5)
# =============================================================================

_COPPER_ROOT = REPO / "data" / "input" / "acc" / "copper" / "current" / "root.ods"
_V4_ROOT_BLOB = "dfeeac0cb616e99739c900820698c5bb3eefda0f"     # git blob of acc/copper v4


def test_the_copper_root_ods_gains_only_the_role_column(tmp_path):
    before = tmp_path / "root_v4.ods"
    before.write_bytes(subprocess.run(["git", "cat-file", "blob", _V4_ROOT_BLOB], cwd=REPO,
                                      capture_output=True, check=True).stdout)
    with zipfile.ZipFile(before) as zb, zipfile.ZipFile(_COPPER_ROOT) as za:
        assert [i.filename for i in zb.infolist()] == [i.filename for i in za.infolist()]
        assert [n for n in zb.namelist() if zb.read(n) != za.read(n)] == ["content.xml"]
    assert _ods_sheet_names(_COPPER_ROOT) == _ods_sheet_names(before)
    for sheet in _ods_sheet_names(before):
        old, new = _read_ods(before, sheet), _read_ods(_COPPER_ROOT, sheet)
        if sheet in ("Sector_Entity", "Valuation"):
            assert list(new.columns) == list(old.columns) + ["role"], sheet
            new = new.drop(columns="role")
        pd.testing.assert_frame_equal(new, old, obj=sheet)


def test_the_copper_root_ods_binds_its_labels_to_the_glossarys_roles():
    se = _read_ods(_COPPER_ROOT, "Sector_Entity")
    sb = se[se["Entity_name"] == "SB"]
    binding = dict(zip(sb["Entity_name_1"], sb["role"]))
    assert set(zip(sb["Entity_name_1"], sb["role"])) == set(binding.items())  # 1 role/label
    assert binding == {"Nature": "nature", "SEM": "secondary", "FD": "final_demand"}
    assert se.loc[se["Entity_name"] != "SB", "role"].isna().all()
    val = _read_ods(_COPPER_ROOT, "Valuation")
    assert dict(zip(val["Valuation_name"], val["role"])) == {"Mass": "gross", "CC": "content"}
