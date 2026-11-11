"""test_run_lineage.py — Stage 2's lineage checks (#53 item 2).

Stage 2 checks every feed run's acc against its own acc input entry, comparing the two
versions' per-file hashes, and can replay an earlier imp run's input versions
(``--pin-inputs-from``). Both checks live in :mod:`pysuto.input_registry`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pysuto.input_registry import (ACC_MASKS, InputRegistries, InputRegistry,
                                   check_feed_accs, input_pins)


def _acc_registry(root: Path) -> InputRegistry:
    """acc ``copper``: v1 → v2 adds masks.csv, v2 → v3 edits rules.ods."""
    current = root / "acc" / "copper" / "current"
    current.mkdir(parents=True)
    for name, body in (("root.ods", b"r"), ("bounds.ods", b"b"), ("rules.ods", b"u")):
        (current / name).write_bytes(body)
    InputRegistry("acc", root=root).initialize()
    (current / ACC_MASKS).write_text("mask_name,logic\n")
    InputRegistry("acc", root=root).initialize()
    (current / "rules.ods").write_bytes(b"u2")
    return InputRegistry("acc", root=root).initialize()


def _feed(version: int) -> dict:
    return {"parents": {"inputs": {"acc": {"id": "copper", "version": version},
                                   "feed": {"id": "icsg", "version": 1}}}}


class TestFeedAccCheck:

    def test_feeds_on_the_same_acc_version_pass_silently(self, tmp_path, caplog):
        reg = _acc_registry(tmp_path / "input")
        with caplog.at_level(logging.WARNING):
            check_feed_accs({"feed_a": _feed(2), "feed_b": _feed(2)}, reg,
                            reg.entry("copper", 2))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_feed_whose_acc_differs_only_in_masks_is_warned_about(self, tmp_path,
                                                                    caplog):
        """The final copper chain: imp_20261022_111102_1b49a2b3 (acc v4) aggregates
        feed_crude_ore_20260603_095651_558f7369 (acc v3); v3 → v4 is masks.csv only."""
        reg = _acc_registry(tmp_path / "input")
        with caplog.at_level(logging.WARNING):
            check_feed_accs({"feed_crude_ore_x": _feed(1)}, reg, reg.entry("copper", 2))
        assert "feed_crude_ore_x" in caplog.text and "masks.csv" in caplog.text

    def test_a_feed_whose_acc_differs_in_another_file_raises(self, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(ValueError, match=r"feed_icsg_x.*v3.*v2.*rules\.ods"):
            check_feed_accs({"feed_icsg_x": _feed(3)}, reg, reg.entry("copper", 2))

    def test_masks_and_another_file_raise_naming_only_the_structural_one(self, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(ValueError, match=r"differ in rules\.ods: rebuild"):
            check_feed_accs({"feed_icsg_x": _feed(1)}, reg, reg.entry("copper", 3))

    def test_a_feed_on_another_acc_id_raises(self, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        other = {"parents": {"inputs": {"acc": {"id": "toy", "version": 2}}}}
        with pytest.raises(ValueError, match=r"feed_x was built on acc toy v2"):
            check_feed_accs({"feed_x": other}, reg, reg.entry("copper", 2))

    def test_a_feed_that_records_no_acc_raises(self, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(KeyError, match="feed_x records no parents.inputs.acc"):
            check_feed_accs({"feed_x": {"parents": {"inputs": {}}}}, reg,
                            reg.entry("copper", 2))


def _imp_sidecar(inputs: dict) -> dict:
    return {"run_id": "imp_ref", "stage": "imp",
            "parents": {"inputs": inputs, "upstream": {"feed_run_ids": []}}}


class TestPinInputsFrom:

    @staticmethod
    def _inputs(root: Path) -> None:
        """params conc v1, v2 and eff v1; acc copper v1 → v3."""
        _acc_registry(root)
        conc = root / "params" / "conc" / "current"
        conc.mkdir(parents=True)
        (conc / "params.csv").write_text("v;1\n")
        eff = root / "params" / "eff" / "current"
        eff.mkdir(parents=True)
        (eff / "params.csv").write_text("e;1\n")
        InputRegistries.initialize(root=root)
        (conc / "params.csv").write_text("v;2\n")
        InputRegistries.initialize(root=root)

    def test_the_input_versions_an_imp_run_recorded_are_used(self, tmp_path):
        root = tmp_path / "data" / "input"
        self._inputs(root)
        pins = input_pins(_imp_sidecar({
            "acc":    {"id": "copper", "version": 2, "code": "x"},
            "params": [{"id": "conc", "version": 1}, {"id": "eff", "version": 1}],
        }), "copper")
        assert pins == {"acc": {"copper": 2}, "params": {"conc": 1, "eff": 1}}

        inputs = InputRegistries.initialize(root=root, pins=pins)
        assert inputs.params.entry("conc")["version"] == 1      # latest is v2
        assert inputs.acc.entry("copper")["version"] == 2       # latest is v3
        assert inputs.params.resolve("conc") == (root / "params" / "conc" / "v1").resolve()

    def test_a_recorded_version_that_no_longer_exists_raises(self, tmp_path):
        root = tmp_path / "data" / "input"
        self._inputs(root)
        pins = input_pins(_imp_sidecar({"acc":    {"id": "copper", "version": 2},
                                        "params": [{"id": "conc", "version": 5}]}),
                          "copper")
        with pytest.raises(KeyError, match="conc: version 5 not registered"):
            InputRegistries.initialize(root=root, pins=pins)

    def test_an_imp_run_built_on_another_acc_is_refused(self):
        sidecar = _imp_sidecar({"acc": {"id": "copper", "version": 4}, "params": []})
        with pytest.raises(ValueError,
                           match=r"imp_ref was built on acc \['copper'\], this run on 'toy'"):
            input_pins(sidecar, "toy")
