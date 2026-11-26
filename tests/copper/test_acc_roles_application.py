"""test_acc_roles_application.py
================================
The test of ``tests/test_acc_roles.py`` that reads the copper application (#337): the
copper ``root.ods`` binds its labels to the glossary's roles (input acc/copper v5). The
test is moved unchanged, only the path to the repository follows its new depth.
"""
from __future__ import annotations

from pathlib import Path

from pysuto.acc import _read_ods

REPO = Path(__file__).resolve().parents[2]

_COPPER_ROOT = REPO / "data" / "input" / "acc" / "copper" / "current" / "root.ods"


def test_the_copper_root_ods_binds_its_labels_to_the_glossarys_roles():
    se = _read_ods(_COPPER_ROOT, "Sector_Entity")
    sb = se[se["Entity_name"] == "SB"]
    binding = dict(zip(sb["Entity_name_1"], sb["role"]))
    assert set(zip(sb["Entity_name_1"], sb["role"])) == set(binding.items())  # 1 each
    assert binding == {"Nature": "nature", "SEM": "secondary", "FD": "final_demand"}
    assert se.loc[se["Entity_name"] != "SB", "role"].isna().all()
    val = _read_ods(_COPPER_ROOT, "Valuation")
    assert dict(zip(val["Valuation_name"], val["role"])) == {"Mass": "gross",
                                                             "CC": "content"}
