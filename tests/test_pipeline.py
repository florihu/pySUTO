"""test_pipeline.py — pysuto.pipeline: Pipeline and Chain (#103), Axis and Sweep (#105).

The stage drivers are replaced by one fake driver script, written to ``tmp_path`` under
the three real names, that speaks the same command-line protocol (upstream flags, label,
``--set``, ``--run-id-file``) and writes a sidecar into a fake ``data/proc``. Setting
``FAKE_FAIL=<stage>`` makes that stage exit 3 with "boom" on stderr;
``FAKE_FAIL_SET=<pair>`` does the same for any stage given that ``--set`` pair. Like the
real drivers, the fake one writes its run under ``--imp-dir`` / ``--const-dir`` when given,
and exits 4 when its upstream run is not in the directory it was told to read.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import uuid
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from pysuto.pipeline import (GIT_LOCATION_VARS, STEPS, Axis, Chain, Pipeline, StepFailed,
                             Sweep)
from pysuto.registry import SweepRunRegistry

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
if STAGE in ("imp", "const"):
    p.add_argument("--imp-dir", default=os.path.join("data", "proc", "imp"))
if STAGE in ("const", "opt"):
    p.add_argument("--const-dir", default=os.path.join("data", "proc", "const"))
a = p.parse_args()
print(f"{STAGE} starting", flush=True)
if os.environ.get("FAKE_FAIL") == STAGE or os.environ.get("FAKE_FAIL_SET") in a.sets:
    print("boom", file=sys.stderr)
    sys.exit(3)
read = {"const": (a.imp_run_id, getattr(a, "imp_dir", None)),
        "opt": (a.const_run_id, getattr(a, "const_dir", None))}.get(STAGE)
if read and not os.path.isdir(os.path.join(read[1], read[0])):
    print(f"no upstream run {read[0]} in {read[1]}", file=sys.stderr)
    sys.exit(4)
rid = f"{STAGE}_20261021_120000_{uuid.uuid4().hex[:8]}"
up = {"imp": {"feed_run_ids": a.feed_ids}, "const": {"imp_run_id": a.imp_run_id},
      "opt": {"const_run_id": a.const_run_id}}[STAGE]
out = {"imp": getattr(a, "imp_dir", None), "const": getattr(a, "const_dir", None),
       "opt": os.path.join("data", "proc", "opt")}[STAGE]
d = os.path.join(out, rid)
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
    path = repo.root / "data" / "proc" / stage / rid / "metadata.json"
    return json.loads(path.read_text())


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


def _pipeline(repo, steps=STEPS, run_dirs=None):
    return Pipeline(steps=steps, base=repo.base, cwd=str(repo.root),
                    scripts_dir=str(repo.root / "scripts"), log_dir=str(repo.root / "logs"),
                    run_dirs=run_dirs)


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

    def test_a_prefix_slice_leaves_no_stale_later_run_in_the_chain(self, repo):
        chain = _pipeline(repo, steps=("imp", "const")).run({}, label="t5")
        assert chain.const is not None and chain.const != repo.base.const
        assert chain.opt is None

    def test_a_failing_step_raises_with_its_output_tail_and_runs_nothing_after(
            self, repo, monkeypatch):
        monkeypatch.setenv("FAKE_FAIL", "const")
        with pytest.raises(StepFailed, match="boom") as exc:
            _pipeline(repo).run({}, label="t4")
        assert exc.value.step == "const" and exc.value.returncode == 3
        assert "const starting" in exc.value.tail
        assert len(_runs(repo, "opt")) == 1                       # the base only


class TestPipelineRunDirs:

    def test_a_step_with_a_run_dir_writes_there_and_the_next_step_reads_it_there(
            self, repo, tmp_path_factory):
        scratch = tmp_path_factory.mktemp("scratch")
        dirs = {"imp": str(scratch / "imp"), "const": str(scratch / "const")}
        chain = _pipeline(repo, run_dirs=dirs).run({}, label="elsewhere")
        assert (scratch / "imp" / chain.imp).is_dir()
        assert (scratch / "const" / chain.const).is_dir()
        assert (repo.root / "data" / "proc" / "opt" / chain.opt).is_dir()
        assert len(_runs(repo, "imp")) == 1 and len(_runs(repo, "const")) == 1

    def test_a_step_reads_a_base_run_from_data_proc_whatever_its_stages_run_dir(
            self, repo, tmp_path_factory):
        """An opt-only pipeline's upstream is the base const, which lives in data/proc."""
        scratch = tmp_path_factory.mktemp("scratch")
        chain = _pipeline(repo, steps=("const", "opt"),
                          run_dirs={"const": str(scratch / "const")}).run({}, label="b")
        assert chain.imp == repo.base.imp and (scratch / "const" / chain.const).is_dir()

    @pytest.mark.parametrize("dirs", [{"opt": "x"}, {"imp": "x"}])
    def test_run_dirs_name_only_imp_or_const_steps_of_the_pipeline(self, repo, dirs):
        with pytest.raises(ValueError, match="run_dirs"):
            _pipeline(repo, steps=("const", "opt"), run_dirs=dirs)


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


# =============================================================================
# Axis and Sweep (#105)
# =============================================================================

class TestAxis:

    def test_a_linear_interval_maps_the_unit_interval_onto_lo_hi(self):
        ax = Axis.interval("const.ANCHOR_CV", 0.005, 0.02)
        assert ax.keys == ("const.ANCHOR_CV",)
        got = [ax.sample(u) for u in (0.0, 0.5, 1.0)]
        assert got == pytest.approx([0.005, 0.0125, 0.02], rel=1e-12)

    def test_a_log_interval_puts_the_geometric_mean_at_the_middle(self):
        ax = Axis.interval("const.SIGMA_C_ABS_FLOOR", 1e-6, 1e-3, log=True)
        got = [ax.sample(u) for u in (0.0, 0.5, 1.0)]
        assert got == pytest.approx([1e-6, math.sqrt(1e-9), 1e-3], rel=1e-10)
        assert all(type(v) is float for v in got)

    def test_a_list_axis_splits_the_unit_interval_into_equal_bins(self):
        ax = Axis("imp.TIME_WINDOW", [2, 3, 5])
        assert [ax.sample(u) for u in (0.0, 0.33, 0.34, 0.66, 0.67, 0.999, 1.0)] == \
            [2, 2, 3, 3, 5, 5, 5]

    def test_an_axis_needs_a_non_empty_list_or_a_distribution(self):
        with pytest.raises(ValueError):
            Axis("imp.Q_REG", [])
        with pytest.raises(TypeError):
            Axis("imp.Q_REG", 3.0)


class TestSweepPoints:

    def test_random_points_are_reproducible_from_the_seed(self, repo):
        axes = [Axis.interval("const.ANCHOR_CV", 0.005, 0.02),
                Axis("imp.TIME_WINDOW", [2, 3])]
        a = Sweep(_pipeline(repo), axes, n_points=8, seed=7).points()
        b = Sweep(_pipeline(repo), axes, n_points=8, seed=7).points()
        c = Sweep(_pipeline(repo), axes, n_points=8, seed=8).points()
        assert a == b and a != c and len(a) == 8
        assert all(0.005 <= p["const.ANCHOR_CV"] <= 0.02 for p in a)

    def test_the_hypercube_gives_every_list_level_the_same_number_of_points(self, repo):
        sweep = Sweep(_pipeline(repo), [Axis("imp.TIME_WINDOW", [2, 3, 5]),
                                        Axis.interval("const.ANCHOR_CV", 0.005, 0.02)],
                      n_points=12, seed=1)
        assert Counter(p["imp.TIME_WINDOW"] for p in sweep.points()) == {2: 4, 3: 4, 5: 4}

    def test_a_tied_axis_fans_one_value_to_every_key(self, repo):
        tied = Axis.interval(("imp.P_PROP", "const.P_PROP"), 1.1, 1.5)
        pts = Sweep(_pipeline(repo), [tied], n_points=4, seed=0).points()
        assert all(p["imp.P_PROP"] == p["const.P_PROP"] for p in pts)

    def test_a_grid_runs_every_combination(self, repo):
        sweep = Sweep(_pipeline(repo), [Axis("imp.TIME_WINDOW", [2, 3, 5]),
                                        Axis("const.STOCK_BOUNDS_MODE", ["soft", "hard"])],
                      mode="grid")
        got = {(p["imp.TIME_WINDOW"], p["const.STOCK_BOUNDS_MODE"]) for p in sweep.points()}
        assert got == {(w, m) for w in (2, 3, 5) for m in ("soft", "hard")}

    def test_sweep_points_skip_the_feasibility_probe_unless_told_otherwise(self, repo):
        ax = [Axis("const.ANCHOR_CV", [0.01, 0.02])]
        assert all(p["opt.FEASIBILITY_CHECK"] == "none"
                   for p in Sweep(_pipeline(repo), ax, mode="grid").points())
        fixed = Sweep(_pipeline(repo), ax, mode="grid",
                      fixed={"opt.FEASIBILITY_CHECK": "first"})
        assert all(p["opt.FEASIBILITY_CHECK"] == "first" for p in fixed.points())
        no_opt = Sweep(_pipeline(repo, steps=("imp", "const")), ax, mode="grid")
        assert all("opt.FEASIBILITY_CHECK" not in p for p in no_opt.points())

    @pytest.mark.parametrize("kw,err", [
        ({"mode": "grid", "n_points": 4}, ValueError),
        ({"mode": "grid", "seed": 1}, ValueError),
        ({"mode": "random", "seed": 1}, ValueError),
        ({"mode": "random", "n_points": 4}, ValueError),
        ({"mode": "lhs", "n_points": 4, "seed": 1}, ValueError),
    ])
    def test_each_mode_takes_only_its_own_arguments(self, repo, kw, err):
        with pytest.raises(err):
            Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01, 0.02])], **kw)

    def test_a_grid_refuses_a_distribution_axis(self, repo):
        with pytest.raises(ValueError, match="grid"):
            Sweep(_pipeline(repo), [Axis.interval("const.ANCHOR_CV", 0.005, 0.02)],
                  mode="grid")

    def test_bad_keys_are_refused_when_the_sweep_is_declared(self, repo):
        with pytest.raises(KeyError):
            Sweep(_pipeline(repo, steps=("const", "opt")), [Axis("imp.Q_REG", [2.0])],
                  mode="grid")
        with pytest.raises(ValueError, match="share one value|set together"):
            Sweep(_pipeline(repo), [Axis("imp.P_PROP", [1.2, 1.4]),
                                    Axis("const.P_PROP", [1.2, 1.4])], mode="grid")
        with pytest.raises(ValueError, match="more than one axis"):
            Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01]),
                                    Axis("const.ANCHOR_CV", [0.02])], mode="grid")


class TestSweepPlan:

    def test_a_sweep_over_a_later_step_plans_one_run_of_each_earlier_step(self, repo):
        sweep = Sweep(_pipeline(repo), [Axis("opt.TOL_GAP_REL", [1e-6, 1e-7, 1e-8])],
                      mode="grid")
        assert sweep.plan() == {"imp": 1, "const": 1, "opt": 3}

    def test_a_sweep_over_an_early_step_plans_every_later_step_per_point(self, repo):
        sweep = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01, 0.02]),
                                        Axis("opt.TOL_GAP_REL", [1e-6, 1e-8])], mode="grid")
        assert sweep.plan() == {"imp": 1, "const": 2, "opt": 4}

    def test_the_disk_need_follows_the_plan(self, repo, monkeypatch):
        """Three solver-only points write one imp, one const and three opt runs:
        0.08 + 0.85 + 3 × 0.08 + 3 GB reserve = 4.17 GB."""
        import pysuto.pipeline as pl
        sweep = Sweep(_pipeline(repo), [Axis("opt.TOL_GAP_REL", [1e-6, 1e-7, 1e-8])],
                      mode="grid")
        assert sweep.disk_need_gb() == pytest.approx(4.17, rel=1e-9)
        monkeypatch.setattr(pl.shutil, "disk_usage",
                            lambda _p: SimpleNamespace(total=0, used=0,
                                                       free=4.2 * 1024 ** 3))
        assert sweep.run("fits")


    def test_a_sweep_that_keeps_only_its_last_step_counts_only_those_runs_on_disk(
            self, repo, tmp_path_factory):
        """Two const points: 2 × 0.08 GB of opt runs + 3 GB reserve; imp and const go to
        the scratch store."""
        sweep = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01, 0.02])], mode="grid",
                      keep_intermediate=False,
                      scratch_dir=str(tmp_path_factory.mktemp("shm")))
        assert sweep.disk_need_gb() == pytest.approx(3.16, rel=1e-9)
        assert sweep.declaration()["keep_intermediate"] is False


def _runs(repo, stage):
    return sorted(p.name for p in (repo.root / "data" / "proc" / stage).iterdir())


def _sweeps(repo):
    return SweepRunRegistry(str(repo.root / "data" / "proc" / "sweep"))


def _one_point_sweep(repo):
    return Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01])], mode="grid")


def _git(repo, *args, cwd=None):
    """git in the fake checkout, never in the repository running the tests: a
    pre-commit hook exports GIT_DIR / GIT_INDEX_FILE, and an inherited one would
    point these commands at the real index."""
    env = {k: v for k, v in os.environ.items() if k not in GIT_LOCATION_VARS}
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          cwd=cwd or repo.root, check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


class TestSweepRun:

    def test_points_run_in_order_and_a_failed_point_does_not_stop_the_sweep(
            self, repo, monkeypatch):
        monkeypatch.setenv("FAKE_FAIL_SET", "ANCHOR_CV=0.02")
        sweep = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01, 0.02, 0.03])],
                      mode="grid")
        sid = sweep.run(label="s1")
        reg = SweepRunRegistry(str(repo.root / "data" / "proc" / "sweep"))
        pts = reg.load_manifest(sid)["points"]
        assert [p["status"] for p in pts] == ["done", "failed", "done"]
        assert "boom" in pts[1]["error"] and pts[1]["runs"]["opt"] is None
        assert pts[0]["overrides"] == {"const.ANCHOR_CV": 0.01,
                                       "opt.FEASIBILITY_CHECK": "none"}
        assert _sidecar(repo, "const", pts[2]["runs"]["const"])["label"] == f"{sid} p002"
        side = reg.load_sidecar(sid)
        summary = side["stage_summary"]
        assert summary["n_done"] == 2 and summary["n_failed"] == 1
        assert side["parents"]["upstream"]["opt_run_id"] == repo.base.opt
        assert side["stage_config"]["mode"] == "grid"

    def test_a_point_that_changes_only_a_later_step_reuses_the_earlier_runs(self, repo):
        sid = Sweep(_pipeline(repo), [Axis("opt.TOL_GAP_REL", [1e-6, 1e-8])],
                    mode="grid").run(label="s2")
        pts = _sweeps(repo).load_manifest(sid)["points"]
        assert pts[0]["runs"]["const"] == pts[1]["runs"]["const"]
        assert pts[0]["runs"]["opt"] != pts[1]["runs"]["opt"]
        assert len(_runs(repo, "const")) == 2 and len(_runs(repo, "opt")) == 3

    def test_resume_reruns_only_the_points_not_done(self, repo, monkeypatch):
        monkeypatch.setenv("FAKE_FAIL_SET", "ANCHOR_CV=0.02")
        sweep = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01, 0.02])], mode="grid")
        sid = sweep.run(label="s3")
        n_opt = len(_runs(repo, "opt"))
        monkeypatch.delenv("FAKE_FAIL_SET")
        assert sweep.run(label="s3", resume=sid) == sid
        pts = _sweeps(repo).load_manifest(sid)["points"]
        assert [p["status"] for p in pts] == ["done", "done"]
        assert len(_runs(repo, "opt")) == n_opt + 1

    def test_resume_refuses_a_sweep_declared_differently(self, repo):
        sid = _one_point_sweep(repo).run("s4")
        other = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.02])], mode="grid")
        with pytest.raises(ValueError, match="declared"):
            other.run("s4", resume=sid)

    def test_resume_refuses_another_base_chain(self, repo):
        sid = _one_point_sweep(repo).run("s7")
        other = dataclasses.replace(repo.base, opt=_write_run(
            repo.root, "opt", {"const_run_id": repo.base.const}))
        p = Pipeline(base=other, cwd=str(repo.root), scripts_dir=str(repo.root / "scripts"),
                     log_dir=str(repo.root / "logs"))
        with pytest.raises(ValueError, match="declared"):
            Sweep(p, [Axis("const.ANCHOR_CV", [0.01])], mode="grid").run("s7", resume=sid)

    def test_resume_refuses_points_drawn_differently(self, repo, monkeypatch):
        """Same declaration, other points (a scipy upgrade can redraw a hypercube):
        the remaining points would not be the sweep's."""
        sweep = Sweep(_pipeline(repo), [Axis.interval("const.ANCHOR_CV", 0.005, 0.02)],
                      n_points=2, seed=3)
        sid = sweep.run("s8")
        sweep._points[1]["const.ANCHOR_CV"] = 0.0123
        reg = SweepRunRegistry(str(repo.root / "data" / "proc" / "sweep"))
        man = reg.load_manifest(sid)
        man["points"][1]["status"] = "pending"
        reg.update(sid, manifest=man)
        with pytest.raises(ValueError, match="drawn"):
            sweep.run("s8", resume=sid)

    def test_resume_refuses_a_manifest_from_another_commit(self, repo):
        _git(repo, "init", "-q")
        _git(repo, "add", "scripts")
        _git(repo, "commit", "-q", "-m", "one")
        sweep = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01])], mode="grid")
        sid = sweep.run("s5")
        (repo.root / "scripts" / "extra.py").write_text("")
        _git(repo, "add", "scripts")
        _git(repo, "commit", "-q", "-m", "two")
        with pytest.raises(RuntimeError, match="commit"):
            sweep.run("s5", resume=sid)

    def test_the_commit_is_the_checkouts_even_inside_a_git_hook(self, repo, monkeypatch,
                                                                tmp_path_factory):
        """A hook exports GIT_DIR / GIT_INDEX_FILE of the repository being committed;
        the sweep must read its own checkout and leave that other index alone."""
        _git(repo, "init", "-q")
        _git(repo, "add", "scripts")
        _git(repo, "commit", "-q", "-m", "one")
        other = tmp_path_factory.mktemp("other")
        _git(repo, "init", "-q", cwd=other)
        (other / "f.txt").write_text("x")
        _git(repo, "add", "f.txt", cwd=other)
        index = other / ".git" / "index"
        before = index.read_bytes()
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_INDEX_FILE", str(index))
        sid = _one_point_sweep(repo).run("h")
        side = _sweeps(repo).load_sidecar(sid)
        monkeypatch.delenv("GIT_DIR")
        monkeypatch.delenv("GIT_INDEX_FILE")
        assert side["stage_config"]["commit"] == _git(repo, "rev-parse", "HEAD")
        assert index.read_bytes() == before

    def test_the_disk_guard_stops_a_sweep_before_anything_runs(self, repo, monkeypatch):
        import pysuto.pipeline as pl
        monkeypatch.setattr(pl.shutil, "disk_usage",
                            lambda _p: SimpleNamespace(total=0, used=0, free=4 * 1024 ** 3))
        sweep = Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01, 0.02])], mode="grid")
        with pytest.raises(RuntimeError, match="GB"):
            sweep.run("s6")
        assert not (repo.root / "data" / "proc" / "sweep").exists()


class TestSweepKeepsOnlyItsLastStep:

    @pytest.fixture
    def shm(self, tmp_path_factory):
        return tmp_path_factory.mktemp("shm")

    def _sweep(self, repo, shm, values=(0.01, 0.02)):
        return Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", list(values))], mode="grid",
                     keep_intermediate=False, scratch_dir=str(shm))

    def test_only_the_opt_runs_are_left_and_the_scratch_store_is_emptied(self, repo, shm):
        sid = self._sweep(repo, shm).run("k1")
        pts = _sweeps(repo).load_manifest(sid)["points"]
        assert [p["status"] for p in pts] == ["done", "done"]
        assert _runs(repo, "imp") == [repo.base.imp]
        assert _runs(repo, "const") == [repo.base.const]
        assert sorted(p["runs"]["opt"] for p in pts) == sorted(
            set(_runs(repo, "opt")) - {repo.base.opt})
        assert all(p["runs"]["const"] for p in pts)          # the lineage is recorded
        assert not (shm / sid).exists()

    def test_a_scratch_run_lives_until_the_last_point_that_reuses_it(self, repo, shm):
        """Both points vary only const, so the second reuses the first's imp run."""
        sid = self._sweep(repo, shm).run("k2")
        pts = _sweeps(repo).load_manifest(sid)["points"]
        assert pts[1]["reused"] == ["imp"]
        assert pts[0]["runs"]["imp"] == pts[1]["runs"]["imp"]

    def test_resume_reruns_a_step_whose_scratch_run_is_gone(self, repo, shm, monkeypatch):
        monkeypatch.setenv("FAKE_FAIL_SET", "ANCHOR_CV=0.02")
        sweep = self._sweep(repo, shm)
        sid = sweep.run("k3")
        first_imp = _sweeps(repo).load_manifest(sid)["points"][0]["runs"]["imp"]
        monkeypatch.delenv("FAKE_FAIL_SET")
        sweep.run("k3", resume=sid)
        pts = _sweeps(repo).load_manifest(sid)["points"]
        assert pts[1]["status"] == "done" and pts[1]["reused"] == []
        assert pts[1]["runs"]["imp"] != first_imp
        assert not (shm / sid).exists()

    def test_keeping_only_the_last_step_needs_a_scratch_dir(self, repo):
        with pytest.raises(ValueError, match="scratch_dir"):
            Sweep(_pipeline(repo), [Axis("const.ANCHOR_CV", [0.01])], mode="grid",
                  keep_intermediate=False)


def test_the_sweep_driver_dry_run_lists_the_points_and_writes_nothing(repo):
    import os
    import sys
    from pathlib import Path
    real = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(real / "scripts" / "sweep.py"), "--dry-run",
                        "--base", repo.base.opt], cwd=repo.root, capture_output=True,
                       text=True, env=dict(os.environ, PYTHONPATH=str(real / "src")),
                       timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "p003" in r.stdout and "plan:" in r.stdout and "GB" in r.stdout
    assert not (repo.root / "data" / "proc" / "sweep").exists()
    assert len(_runs(repo, "imp")) == 1
