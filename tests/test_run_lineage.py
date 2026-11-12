"""test_run_lineage.py — Stage 2's lineage checks (#53 item 2, #119 item 6).

Stage 2 checks every feed run's acc against its own acc input entry, comparing the two
versions' specs on the keys that define the edge universe (a role never counts), and can
replay an earlier imp run's input versions (``--pin-inputs-from``). Both checks live in
:mod:`pysuto.input_registry`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pysuto.acc import check_feed_accs, read_acc_spec_dir
from pysuto.input_registry import InputRegistries, InputRegistry, input_pins

from test_acc_roles import (BOUNDS_ONLY, COPPER_LABELS, MASKS_ONLY, ROLE_ONLY, SECTORS,
                            acc_input_versions)
from test_acc_spec import _write_ods_case

_NO_MASKS = {k: v for k, v in COPPER_LABELS.items() if k != "masks"}
_NEW_RULE = dict(COPPER_LABELS, supply=dict(COPPER_LABELS["supply"], Mining=["Ore"]))


def _acc_registry(root: Path) -> InputRegistry:
    """acc ``copper``: v1 → v2 adds the masks, v2 → v3 edits the supply table."""
    return acc_input_versions(root, "copper", [_NO_MASKS, COPPER_LABELS, _NEW_RULE])


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

    def test_a_feed_whose_acc_differs_only_in_masks_is_logged(self, tmp_path,
                                                                    caplog):
        """The final copper chain: imp_20261022_111102_1b49a2b3 (acc v4) aggregates
        feed_crude_ore_20260603_095651_558f7369 (acc v3); v3 → v4 is masks.csv only."""
        reg = _acc_registry(tmp_path / "input")
        with caplog.at_level(logging.INFO):
            check_feed_accs({"feed_crude_ore_x": _feed(1)}, reg, reg.entry("copper", 2))
        assert "feed_crude_ore_x" in caplog.text and "differ in ['masks']" in caplog.text

    def test_a_feed_whose_acc_differs_in_an_edge_universe_key_raises(self, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(ValueError, match=r"feed_icsg_x.*v3.*v2.*\['supply'\]"):
            check_feed_accs({"feed_icsg_x": _feed(3)}, reg, reg.entry("copper", 2))

    def test_masks_and_another_key_raise_naming_only_the_structural_one(self, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(ValueError, match=r"differ in \['supply'\]: rebuild"):
            check_feed_accs({"feed_icsg_x": _feed(1)}, reg, reg.entry("copper", 3))

    def test_role_masks_and_bounds_differences_pass_and_a_sector_difference_raises(
            self, tmp_path):
        """#119: a feed reads no role, bound or mask, so the final chain's feed runs
        (acc v3/v4) stay valid against v5; a sector difference still raises."""
        specs = [COPPER_LABELS, ROLE_ONLY, MASKS_ONLY, BOUNDS_ONLY, SECTORS]
        reg = acc_input_versions(tmp_path / "input", "copper", specs)
        for v in (2, 3, 4):
            check_feed_accs({"feed_x": _feed(1)}, reg, reg.entry("copper", v))
        with pytest.raises(ValueError, match=r"\['sectors'\]: rebuild the feed"):
            check_feed_accs({"feed_x": _feed(1)}, reg, reg.entry("copper", 5))

    def test_an_acc_that_moved_from_the_ods_files_to_acc_json_keeps_its_feeds(
            self, tmp_path, caplog):
        """Every file differs, the structure does not: per-file hashes refused this."""
        root = tmp_path / "input"
        current = root / "acc" / "copper" / "current"
        _write_ods_case(current)
        reg = InputRegistry("acc", root=root).initialize()
        spec = read_acc_spec_dir(reg.resolve("copper", 1))
        for name in ("root.ods", "rules.ods", "bounds.ods", "masks.csv"):
            (current / name).unlink()
        (current / "acc.json").write_text(json.dumps(spec))
        reg = InputRegistry("acc", root=root).initialize()
        assert reg.latest_version("copper") == 2
        with caplog.at_level(logging.WARNING):
            check_feed_accs({"feed_x": _feed(1)}, reg, reg.entry("copper", 2))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

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
