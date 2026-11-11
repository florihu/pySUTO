"""test_run_lineage.py — Stage 2's lineage checks (#53 item 2).

Stage 2 checks every feed run's acc against its own acc input entry, comparing the two
versions' per-file hashes, and can replay an earlier imp run's input versions
(``--pin-inputs-from``). The driver is imported by path from a directory without a
config: importing it reads nothing (#58).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

from pysuto.input_registry import InputRegistries, InputRegistry

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def stage2(tmp_path, monkeypatch):
    """The Stage 2 driver as a module, imported from an empty working directory."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    spec = importlib.util.spec_from_file_location(
        "_lineage_stage2", REPO / "scripts" / "collect_aggregate_impute.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def _acc_registry(root: Path) -> InputRegistry:
    """acc ``copper``: v1 → v2 adds masks.csv, v2 → v3 edits rules.ods."""
    current = root / "acc" / "copper" / "current"
    current.mkdir(parents=True)
    for name, body in (("root.ods", b"r"), ("bounds.ods", b"b"), ("rules.ods", b"u")):
        (current / name).write_bytes(body)
    InputRegistry("acc", root=root).initialize()
    (current / "masks.csv").write_text("mask_name,logic\n")
    InputRegistry("acc", root=root).initialize()
    (current / "rules.ods").write_bytes(b"u2")
    return InputRegistry("acc", root=root).initialize()


def _feed(version: int) -> dict:
    return {"parents": {"inputs": {"acc": {"id": "copper", "version": version},
                                   "feed": {"id": "icsg", "version": 1}}}}


class TestFeedAccCheck:

    def test_feeds_on_the_same_acc_version_pass_silently(self, stage2, tmp_path, caplog):
        reg = _acc_registry(tmp_path / "input")
        log = logging.getLogger("t_lineage")
        with caplog.at_level(logging.WARNING):
            stage2.check_feed_accs({"feed_a": _feed(2), "feed_b": _feed(2)}, reg,
                                   reg.entry("copper", 2), log)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_feed_whose_acc_differs_only_in_masks_is_warned_about(self, stage2,
                                                                    tmp_path, caplog):
        """The final copper chain: imp_20261022_111102_1b49a2b3 (acc v4) aggregates
        feed_crude_ore_20260603_095651_558f7369 (acc v3); v3 → v4 is masks.csv only."""
        reg = _acc_registry(tmp_path / "input")
        with caplog.at_level(logging.WARNING):
            stage2.check_feed_accs({"feed_crude_ore_x": _feed(1)}, reg,
                                   reg.entry("copper", 2), logging.getLogger("t_lineage"))
        assert "feed_crude_ore_x" in caplog.text and "masks.csv" in caplog.text

    def test_a_feed_whose_acc_differs_in_another_file_raises(self, stage2, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(ValueError, match=r"feed_icsg_x.*v3.*v2.*rules\.ods"):
            stage2.check_feed_accs({"feed_icsg_x": _feed(3)}, reg,
                                   reg.entry("copper", 2), logging.getLogger("t_lineage"))

    def test_masks_and_another_file_raise_naming_only_the_structural_one(self, stage2,
                                                                         tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(ValueError, match=r"differ in rules\.ods: rebuild"):
            stage2.check_feed_accs({"feed_icsg_x": _feed(1)}, reg,
                                   reg.entry("copper", 3), logging.getLogger("t_lineage"))

    def test_a_feed_on_another_acc_id_raises(self, stage2, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        other = {"parents": {"inputs": {"acc": {"id": "toy", "version": 2}}}}
        with pytest.raises(ValueError, match=r"feed_x was built on acc toy v2"):
            stage2.check_feed_accs({"feed_x": other}, reg, reg.entry("copper", 2),
                                   logging.getLogger("t_lineage"))

    def test_a_feed_that_records_no_acc_raises(self, stage2, tmp_path):
        reg = _acc_registry(tmp_path / "input")
        with pytest.raises(KeyError, match="feed_x records no parents.inputs.acc"):
            stage2.check_feed_accs({"feed_x": {"parents": {"inputs": {}}}}, reg,
                                   reg.entry("copper", 2), logging.getLogger("t_lineage"))


def _imp_run(imp_dir: Path, inputs: dict) -> str:
    """A registered imp run whose sidecar records ``inputs`` as parents.inputs."""
    from pysuto.registry import ImputationRunRegistry
    from pysuto.registry._sidecar import build_sidecar
    reg = ImputationRunRegistry(storage_dir=str(imp_dir))
    run_dir = reg._make_run_dir("imp_ref")
    reg._write_sidecar(run_dir, build_sidecar(
        run_id="imp_ref", stage="imp", label="", created_at="2026-10-22T11:11:02Z",
        parents={"inputs": inputs, "upstream": {"feed_run_ids": []}}))
    reg._add_entry(run_id="imp_ref", label="", created_at="2026-10-22T11:11:02Z")
    reg.save()
    return "imp_ref"


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

    def test_the_input_versions_an_imp_run_recorded_are_used(self, stage2, tmp_path):
        root = tmp_path / "data" / "input"
        self._inputs(root)
        imp_id = _imp_run(tmp_path / "data" / "proc" / "imp", {
            "acc":    {"id": "copper", "version": 2, "code": "x"},
            "params": [{"id": "conc", "version": 1}, {"id": "eff", "version": 1}],
        })
        pins = stage2.input_pins(str(tmp_path / "data" / "proc" / "imp"), imp_id, "copper")
        assert pins == {"acc": {"copper": 2}, "params": {"conc": 1, "eff": 1}}

        inputs = InputRegistries.initialize(root=root, pins=pins)
        assert inputs.params.entry("conc")["version"] == 1      # latest is v2
        assert inputs.acc.entry("copper")["version"] == 2       # latest is v3
        assert inputs.params.resolve("conc") == (root / "params" / "conc" / "v1").resolve()

    def test_a_recorded_version_that_no_longer_exists_raises(self, stage2, tmp_path):
        root = tmp_path / "data" / "input"
        self._inputs(root)
        imp_id = _imp_run(tmp_path / "data" / "proc" / "imp", {
            "acc":    {"id": "copper", "version": 2},
            "params": [{"id": "conc", "version": 5}],
        })
        pins = stage2.input_pins(str(tmp_path / "data" / "proc" / "imp"), imp_id, "copper")
        with pytest.raises(KeyError, match="conc: version 5 not registered"):
            InputRegistries.initialize(root=root, pins=pins)

    def test_an_imp_run_built_on_another_acc_is_refused(self, stage2, tmp_path):
        imp_id = _imp_run(tmp_path / "imp", {"acc": {"id": "copper", "version": 4},
                                             "params": []})
        with pytest.raises(ValueError, match=r"built on acc \['copper'\], this run on 'toy'"):
            stage2.input_pins(str(tmp_path / "imp"), imp_id, "toy")

    def test_an_unknown_imp_run_raises(self, stage2, tmp_path):
        with pytest.raises(KeyError, match="imp_nope"):
            stage2.input_pins(str(tmp_path / "imp"), "imp_nope", "copper")
