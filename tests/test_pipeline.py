"""test_pipeline.py — pysuto.pipeline: Pipeline and Chain (#103).

The stage drivers are replaced by one fake driver script, written to ``tmp_path`` under
the three real names, that speaks the same command-line protocol (upstream flags, label,
``--set``, ``--run-id-file``) and writes a sidecar into a fake ``data/proc``. Setting
``FAKE_FAIL=<stage>`` makes that stage exit 3 with "boom" on stderr.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from pysuto.pipeline import STEPS, Chain, Pipeline, StepFailed

FAKE_DRIVER = r'''
import argparse, json, os, sys, uuid
STAGE = {"collect_aggregate_impute.py": "imp", "constraints.py": "const",
         "run_solver.py": "opt"}[os.path.basename(__file__)]
p = argparse.ArgumentParser()
p.add_argument("--feed-id", action="append", dest="feed_ids")
p.add_argument("--imp-run-id")
p.add_argument("--const-run-id")
p.add_argument("--label")
if STAGE == "const":
    p.add_argument("pos_label")
p.add_argument("--set", action="append", default=[], dest="sets")
p.add_argument("--run-id-file")
a = p.parse_args()
print(f"{STAGE} starting", flush=True)
if os.environ.get("FAKE_FAIL") == STAGE:
    print("boom", file=sys.stderr)
    sys.exit(3)
rid = f"{STAGE}_20261021_120000_{uuid.uuid4().hex[:8]}"
up = {"imp": {"feed_run_ids": a.feed_ids}, "const": {"imp_run_id": a.imp_run_id},
      "opt": {"const_run_id": a.const_run_id}}[STAGE]
d = os.path.join("data", "proc", STAGE, rid)
os.makedirs(d)
label = a.pos_label if STAGE == "const" else a.label
with open(os.path.join(d, "metadata.json"), "w") as f:
    json.dump({"run_id": rid, "label": label, "parents": {"upstream": up},
               "stage_config": {"sets": a.sets}}, f)
with open(a.run_id_file, "w") as f:
    f.write(rid + "\n")
'''

FEEDS = ("feed_icsg_1", "feed_baci_2")


def _sidecar(repo, stage, rid):
    return json.loads((repo.root / "data" / "proc" / stage / rid / "metadata.json").read_text())


def _write_run(root, stage, upstream):
    rid = f"{stage}_20261020_000000_{uuid.uuid4().hex[:8]}"
    d = root / "data" / "proc" / stage / rid
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"run_id": rid,
                                                 "parents": {"upstream": upstream}}))
    return rid


@pytest.fixture
def repo(tmp_path):
    """A fake checkout: scripts/ with the three fake drivers and a base chain."""
    (tmp_path / "scripts").mkdir()
    for name in ("collect_aggregate_impute.py", "constraints.py", "run_solver.py"):
        (tmp_path / "scripts" / name).write_text(FAKE_DRIVER)
    imp   = _write_run(tmp_path, "imp", {"feed_run_ids": list(FEEDS)})
    const = _write_run(tmp_path, "const", {"imp_run_id": imp})
    opt   = _write_run(tmp_path, "opt", {"const_run_id": const})
    return SimpleNamespace(root=tmp_path,
                           base=Chain(feed_run_ids=FEEDS, imp=imp, const=const, opt=opt))


def _pipeline(repo, steps=STEPS):
    return Pipeline(steps=steps, base=repo.base, cwd=str(repo.root),
                    scripts_dir=str(repo.root / "scripts"), log_dir=str(repo.root / "logs"))


class TestChain:

    def test_from_opt_run_resolves_const_imp_and_feeds_through_the_sidecars(self, repo):
        got = Chain.from_opt_run(repo.base.opt, root=str(repo.root / "data" / "proc"))
        assert got == repo.base


class TestPipelineRun:

    def test_a_three_step_chain_threads_each_new_run_id_into_the_next_step(self, repo):
        chain = _pipeline(repo).run({}, label="t1")
        assert chain.feed_run_ids == FEEDS
        assert len({chain.imp, chain.const, chain.opt} & {repo.base.imp, repo.base.const,
                                                          repo.base.opt}) == 0
        up = lambda s, r: _sidecar(repo, s, r)["parents"]["upstream"]
        assert up("imp", chain.imp) == {"feed_run_ids": list(FEEDS)}
        assert up("const", chain.const) == {"imp_run_id": chain.imp}
        assert up("opt", chain.opt) == {"const_run_id": chain.const}
        assert _sidecar(repo, "const", chain.const)["label"] == "t1"

    def test_an_override_reaches_its_stage_only_as_a_repr(self, repo):
        chain = _pipeline(repo).run({"const.ANCHOR_CV": 0.02, "imp.TIME_METHOD": "linear",
                                     "opt.TIMESTEPS": ["1995"]}, label="t2")
        sets = lambda s, r: _sidecar(repo, s, r)["stage_config"]["sets"]
        assert sets("imp", chain.imp) == ["TIME_METHOD='linear'"]
        assert sets("const", chain.const) == ["ANCHOR_CV=0.02"]
        assert sets("opt", chain.opt) == ["TIMESTEPS=['1995']"]

    def test_a_later_slice_starts_from_the_base_chains_upstream_run(self, repo):
        chain = _pipeline(repo, steps=("const", "opt")).run({}, label="t3")
        assert chain.imp == repo.base.imp
        assert _sidecar(repo, "const", chain.const)["parents"]["upstream"] == {
            "imp_run_id": repo.base.imp}

    def test_a_failing_step_raises_with_its_output_tail_and_runs_nothing_after(
            self, repo, monkeypatch):
        monkeypatch.setenv("FAKE_FAIL", "const")
        with pytest.raises(StepFailed, match="boom") as exc:
            _pipeline(repo).run({}, label="t4")
        assert exc.value.step == "const" and exc.value.returncode == 3
        assert "const starting" in exc.value.tail
        assert len(list((repo.root / "data" / "proc" / "opt").iterdir())) == 1   # the base only


class TestPipelineRefusals:

    def test_steps_must_be_a_contiguous_slice_of_the_stages(self, repo):
        with pytest.raises(ValueError, match="contiguous"):
            _pipeline(repo, steps=("imp", "opt"))

    def test_a_key_of_an_unknown_stage_or_a_stage_outside_the_steps_is_refused(self, repo):
        with pytest.raises(KeyError, match="feed"):
            _pipeline(repo).run({"feed.P_PROP": 1.5, "const.P_PROP": 1.5}, label="x")
        with pytest.raises(KeyError, match="imp.Q_REG"):
            _pipeline(repo, steps=("const", "opt")).run({"imp.Q_REG": 2.0}, label="x")
        with pytest.raises(KeyError, match="no stage"):
            _pipeline(repo).run({"P_PROP": 1.5}, label="x")

    def test_a_tied_knob_set_on_one_side_only_or_to_two_values_is_refused(self, repo):
        p = _pipeline(repo)
        with pytest.raises(ValueError, match="imp.P_PROP.*const.P_PROP"):
            p.run({"imp.P_PROP": 1.5}, label="x")
        with pytest.raises(ValueError, match="share one value"):
            p.run({"imp.P_PROP": 1.5, "const.P_PROP": 1.4}, label="x")

    def test_a_tie_with_one_side_outside_the_steps_is_not_split(self, repo):
        chain = _pipeline(repo, steps=("const", "opt")).run({"const.P_PROP": 1.5},
                                                           label="x")
        assert _sidecar(repo, "const", chain.const)["stage_config"]["sets"] == [
            "P_PROP=1.5"]

    def test_nothing_runs_when_an_override_is_refused(self, repo):
        with pytest.raises(ValueError):
            _pipeline(repo).run({"imp.P_PROP": 1.5}, label="x")
        assert len(list((repo.root / "data" / "proc" / "imp").iterdir())) == 1
