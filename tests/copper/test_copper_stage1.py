"""test_copper_stage1.py — copper's Stage 1 hook (#109).

The hook's ordering and wiring are checked with stubs for the script runner and for
``run_feed``; no real data is read. The registry's sha256 rule, which lets a rebuild
leave every input version alone, is checked on a toy feed input.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from copper_case import copper, copper_feed
from pysuto.input_registry import InputRegistries


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(copper_feed, "_run_script",
                        lambda script, *args, data_dir:
                            log.append(("script", script, args)))

    def fake_run_feed(spec, acc_id, data_dir, run_log):
        log.append(("feed", spec.feed_id, data_dir))
        return SimpleNamespace(run_id=f"feed_{spec.feed_id}_x")

    monkeypatch.setattr(copper_feed, "run_feed", fake_run_feed)
    return log


def _icsg_parents(root, baci_version):
    d = root / "input" / "feed" / "icsg" / "current"
    d.mkdir(parents=True, exist_ok=True)
    (d / "parents.json").write_text(json.dumps({"inputs": {"feed": {"baci": baci_version}}}))


def test_the_hook_runs_the_load_bearing_order_on_the_given_data_dir(calls, tmp_path):
    _icsg_parents(tmp_path, 28)
    ids = copper_feed.run_stage1(str(tmp_path))
    d = str(tmp_path)
    assert calls == [
        ("script", "build_baci_reexport.py", ("--source-version", "28", "--data-dir", d)),
        ("feed", "baci", d),
        ("script", "build_trade_shares.py", ("--data-dir", d)),
        ("feed", "icsg", d),
        ("feed", "crude_ore", d),
    ]
    assert ids == {"baci": "feed_baci_x", "icsg": "feed_icsg_x",
                   "crude_ore": "feed_crude_ore_x"}


def test_copper_declares_the_hook_and_the_feed_step_s_child_reaches_it(calls, tmp_path):
    """The feed step runs in a child (#186): its entry point calls the hook of the case
    the reference names, in the order of the hook."""
    from pysuto._step import StepSpec, run_spec
    _icsg_parents(tmp_path, 28)
    assert copper.CASE.stage1 is not None
    out = run_spec(StepSpec(step="feed", case=copper.__file__, label="l", upstream=None,
                            data_dir=str(tmp_path), config=None))
    assert [c[1] for c in calls if c[0] == "feed"] == ["baci", "icsg", "crude_ore"]
    assert out == "feed_baci_x\nfeed_icsg_x\nfeed_crude_ore_x"


def test_a_feed_start_hands_the_case_reference_to_the_feed_child(tmp_path, monkeypatch,
                                                                 capsys):
    from pysuto.__main__ import main
    from pysuto.pipeline import Pipeline
    seen = {}

    def launch(self, step, spec):
        seen.update(step=step, spec=spec)
        return "feed_baci_x\nfeed_icsg_x\nfeed_crude_ore_x"

    monkeypatch.setattr(Pipeline, "_launch", launch)
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--case", copper.__file__, "--from", "feed", "--to", "feed",
                 "--data-dir", str(tmp_path / "d")]) == 0
    assert seen["step"] == "feed" and seen["spec"].case == copper.__file__
    assert "feed_baci_x feed_icsg_x feed_crude_ore_x" in capsys.readouterr().out


def test_a_failed_script_raises_with_its_log_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(copper_feed, "_SCRIPTS_DIR", tmp_path)
    (tmp_path / "boom.py").write_text("import sys\nprint('why')\nsys.exit(3)\n")
    with pytest.raises(RuntimeError, match="boom.py exited 3(.|\n)*why"):
        copper_feed._run_script("boom.py", data_dir=str(tmp_path))


def test_the_build_scripts_take_a_data_dir(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    import build_baci_reexport as R
    import build_trade_shares as T
    for mod, names in ((R, ("_INPUT_ROOT", "OUT_DIR")),
                       (T, ("_INPUT_ROOT", "OUT_DIR", "ACC_DIR", "FEED_REG"))):
        for n in names:                       # restore the module's paths afterwards
            monkeypatch.setattr(mod, n, getattr(mod, n))
        mod.set_data_dir(str(tmp_path))
        assert mod._INPUT_ROOT == str(tmp_path / "input")
        assert mod.OUT_DIR.startswith(str(tmp_path / "input"))
    assert T.FEED_REG == str(tmp_path / "proc" / "feed" / "registry")
    assert T.ACC_DIR == str(tmp_path / "proc" / "acc")


def test_the_hook_nets_the_baci_version_the_icsg_input_was_parsed_against(tmp_path):
    _icsg_parents(tmp_path, 28)
    assert copper_feed.baci_source_version(str(tmp_path)) == 28


def test_the_hook_refuses_an_icsg_input_that_records_no_baci_version(tmp_path):
    with pytest.raises(ValueError, match="records no BACI version"):
        copper_feed.baci_source_version(str(tmp_path))


def test_a_byte_identical_rebuild_of_a_feed_input_bumps_no_version(tmp_path):
    cur = tmp_path / "feed" / "baci" / "current"
    cur.mkdir(parents=True)
    (cur / "values.csv").write_text("a,b\n1,2\n")
    (cur / "conc.xlsx").write_bytes(b"x")
    (cur / "meta.xlsx").write_bytes(b"y")
    assert InputRegistries.initialize(root=tmp_path).feed.entry("baci")["version"] == 1
    (cur / "values.csv").write_text("a,b\n1,2\n")           # rewritten, same bytes
    assert InputRegistries.initialize(root=tmp_path).feed.entry("baci")["version"] == 1
    (cur / "values.csv").write_text("a,b\n1,3\n")
    assert InputRegistries.initialize(root=tmp_path).feed.entry("baci")["version"] == 2
