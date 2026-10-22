"""
pipeline.py
===========
The pipeline (GLOSSARY.md "Pipeline"): the stage drivers composed as steps, so a
**chain** of runs can be produced with **knob** overrides from Python (#103).

Each step is a stage driver run as a subprocess — the module never imports
``scripts/`` (README §5, import direction) — through the driver's knob protocol
(:mod:`pysuto.knobs`): upstream run id(s), a label, ``--set KNOB=repr(value)`` per
override of that stage, and ``--run-id-file`` to read back the run it persisted. The
steps run one after another in one checkout, which the stage lock enforces (#101).

    base  = Chain.from_opt_run("opt_20261020_112726_2f451f31")
    chain = Pipeline(steps=("imp", "const", "opt"), base=base).run(
        {"imp.P_PROP": 1.5, "const.P_PROP": 1.5}, label="p = 1.5")

Override keys are ``<stage>.<KNOB>``, after scikit-learn's ``step__param``. Knobs that
must share one value across stages are listed in :data:`TIED`.

A :class:`Sweep` runs a set of chains that differ only in knob values and records them
as one resumable ``sweep`` run (#105), after scikit-learn's ``ParameterSampler`` and
``ParameterGrid``: a **random** sweep (the default) draws ``n_points`` from the axes'
ranges with a seeded, scrambled Latin hypercube; a **grid** sweep runs every
combination of the axes' listed values::

    Sweep(pipeline, [Axis.interval(("imp.P_PROP", "const.P_PROP"), 1.1, 1.5),
                     Axis.interval("const.ANCHOR_CV", 0.005, 0.02, log=True),
                     Axis("imp.TIME_WINDOW", [2, 3, 5])],
          n_points=30, seed=20261021).run(label="knob study")
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import scipy
from scipy.stats import loguniform, qmc, uniform

from pysuto.logging import get_logger
from pysuto.registry import SweepRunRegistry

_log = get_logger("pysuto.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]

# The stages a pipeline can run, in order. A feed step is #109.
STEPS = ("imp", "const", "opt")

# Knobs that must share one value whenever both stages run (the cross-stage Lp
# exponent: "P_PROP must match", collect_aggregate_impute.py and constraints.py).
TIED = (("imp.P_PROP", "const.P_PROP"),)

_SCRIPTS = {
    "imp":   "collect_aggregate_impute.py",
    "const": "constraints.py",
    "opt":   "run_solver.py",
}
_TAIL_LINES = 40
# The stage whose run each step reads; a step's driver takes ``--<stage>-dir`` for the
# directories it reads and writes outside ``data/proc`` (opt runs always land there).
_UPSTREAM  = {"imp": None, "const": "imp", "opt": "const"}
_DIR_STEPS = ("imp", "const")


@dataclass(frozen=True)
class Chain:
    """One lineage of runs through the pipeline (GLOSSARY.md "Chain")."""
    feed_run_ids: tuple
    imp:          Optional[str] = None
    const:        Optional[str] = None
    opt:          Optional[str] = None

    @classmethod
    def from_opt_run(cls, opt_run_id: str,
                     root: str = os.path.join("data", "proc")) -> "Chain":
        """Resolve an opt run's const, imp and feed runs through the sidecars'
        ``parents.upstream``."""
        const = _upstream(root, "opt", opt_run_id)["const_run_id"]
        imp   = _upstream(root, "const", const)["imp_run_id"]
        feeds = _upstream(root, "imp", imp)["feed_run_ids"]
        return cls(feed_run_ids=tuple(feeds), imp=imp, const=const, opt=opt_run_id)


def _upstream(root: str, stage: str, run_id: str) -> dict:
    with open(os.path.join(root, stage, run_id, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)["parents"]["upstream"]


class StepFailed(RuntimeError):
    """A step's driver exited non-zero; ``tail`` holds the end of its output."""

    def __init__(self, step: str, returncode: int, tail: str, log_path: str) -> None:
        self.step, self.returncode = step, returncode
        self.tail, self.log_path = tail, log_path
        super().__init__(f"step {step!r} exited {returncode} (log {log_path}):\n{tail}")


def split_key(key: str) -> tuple[str, str]:
    """``"const.ANCHOR_CV"`` → ``("const", "ANCHOR_CV")``; ``KeyError`` without a stage."""
    stage, sep, knob = key.partition(".")
    if not sep or not knob:
        raise KeyError(f"override key {key!r} names no stage; expected <stage>.<KNOB>")
    return stage, knob


class Pipeline:
    """Run a contiguous slice of :data:`STEPS` as one chain.

    Parameters
    ----------
    steps : the stages to run, a contiguous slice of :data:`STEPS`.
    base : the chain the first step's upstream comes from; stages outside ``steps``
        keep the base chain's runs in the result.
    cwd : the checkout the drivers run in (they resolve ``data/`` and ``logs/``
        relative to it); default the repository holding this module.
    scripts_dir : where the drivers live; default ``<cwd>/scripts``.
    log_dir : where each step's combined stdout/stderr is written; default
        ``<cwd>/logs/pipeline``.
    run_dirs : per imp or const step, the directory its run is written to and the next
        step reads it from, instead of ``data/proc/<stage>`` — a sweep's scratch store.
        A stage outside ``steps`` is always read from ``data/proc``: it is the base's.
    """

    def __init__(self, steps: tuple = STEPS, base: Optional[Chain] = None,
                 cwd: Optional[str] = None, scripts_dir: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 run_dirs: Optional[Mapping[str, str]] = None) -> None:
        steps = tuple(steps)
        if not _is_contiguous_slice(steps):
            raise ValueError(f"steps {steps} must be a contiguous slice of {STEPS}")
        if base is None:
            raise ValueError("a pipeline needs a base chain for its first step's upstream")
        self.steps       = steps
        self.base        = base
        self.cwd         = str(cwd or REPO_ROOT)
        self.scripts_dir = str(scripts_dir or os.path.join(self.cwd, "scripts"))
        self.log_dir     = str(log_dir or os.path.join(self.cwd, "logs", "pipeline"))
        self.run_dirs    = {k: str(v) for k, v in (run_dirs or {}).items()}
        bad = [k for k in self.run_dirs if k not in _DIR_STEPS or k not in steps]
        if bad:
            raise ValueError(f"run_dirs {bad}: only the pipeline's imp and const steps "
                             f"can write elsewhere than data/proc (steps {steps})")

    # ------------------------------------------------------------------
    def check_overrides(self, overrides: Mapping[str, object]) -> dict:
        """Group overrides by stage, refusing keys outside the steps and split ties."""
        by_stage: dict = {s: {} for s in self.steps}
        for key, value in overrides.items():
            stage, knob = split_key(key)
            if stage not in by_stage:
                raise KeyError(f"override {key!r}: stage {stage!r} is not among the "
                               f"pipeline's steps {self.steps}")
            by_stage[stage][knob] = value
        for tie in TIED:
            live = [k for k in tie if split_key(k)[0] in self.steps]
            given = [k for k in live if k in overrides]
            if given and len(given) != len(live):
                raise ValueError(f"tied knobs {' and '.join(live)} must be set together; "
                                 f"only {', '.join(given)} is")
            if len({repr(overrides[k]) for k in given}) > 1:
                got = ", ".join(f"{k}={overrides[k]!r}" for k in given)
                raise ValueError(f"tied knobs {' and '.join(live)} must share one value, "
                                 f"got {got}")
        return by_stage

    def run(self, overrides: Mapping[str, object], label: str) -> Chain:
        """Run every step with its overrides and return the new chain: the base's runs
        before the steps, the new runs, and no run after them."""
        by_stage = self.check_overrides(overrides)
        chain = replace(self.base, **{s: None for s in _after(self.steps)})
        for step in self.steps:
            rid = self.run_step(step, chain, by_stage[step], label)
            chain = replace(chain, **{step: rid})
        return chain

    def run_step(self, step: str, upstream: Chain, knobs: Mapping[str, object],
                 label: str) -> str:
        """Run one driver on ``upstream``'s runs with ``knobs`` set; return its run id."""
        os.makedirs(self.log_dir, exist_ok=True)
        slug = re.sub(r"[^\w.-]+", "_", label).strip("_") or "run"
        log_path = os.path.join(self.log_dir, f"{slug}_{step}.log")
        with tempfile.TemporaryDirectory() as tmp:
            rid_file = os.path.join(tmp, "run_id")
            cmd = self._command(step, upstream, knobs, label, rid_file)
            _log.info("step %s: %s", step, shlex.join(cmd))
            with open(log_path, "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=self.cwd, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise StepFailed(step, rc, _tail(log_path), log_path)
            with open(rid_file, encoding="utf-8") as f:
                run_id = f.read().strip()
        _log.info("step %s: %s", step, run_id)
        return run_id

    def _command(self, step, upstream, knobs, label, rid_file) -> list:
        cmd = [sys.executable, os.path.join(self.scripts_dir, _SCRIPTS[step])]
        if step == "imp":
            for feed in upstream.feed_run_ids:
                cmd += ["--feed-id", feed]
            cmd += ["--label", label]
        elif step == "const":
            cmd += ["--imp-run-id", upstream.imp, label]
        else:
            cmd += ["--const-run-id", upstream.const, "--label", label]
        for stage in (_UPSTREAM[step], step):
            if stage in self.run_dirs:
                cmd += [f"--{stage}-dir", self.run_dirs[stage]]
        for knob, value in knobs.items():
            cmd += ["--set", f"{knob}={value!r}"]
        return cmd + ["--run-id-file", rid_file]


def _after(steps: tuple) -> tuple:
    """The stages after a contiguous slice of :data:`STEPS`."""
    return STEPS[STEPS.index(steps[-1]) + 1:]


def _is_contiguous_slice(steps: tuple) -> bool:
    if not steps or steps[0] not in STEPS:
        return False
    i = STEPS.index(steps[0])
    return steps == STEPS[i:i + len(steps)]


def _tail(path: str, n: int = _TAIL_LINES) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


# =============================================================================
# Axis and Sweep (#105)
# =============================================================================

# Disk one run of each step writes (imp_20261019_165706_dd1ae143 80 MB,
# const_20261019_170151_13460e94 851 MB, opt_20261020_112726_2f451f31 74 MB) and the
# reserve a sweep leaves free; the pre-flight guard refuses a sweep that would not fit.
STEP_GB    = {"imp": 0.08, "const": 0.85, "opt": 0.08}
RESERVE_GB = 3.0
MODES      = ("random", "grid")


class Axis:
    """One knob, or a tied set of knobs that share one value, with its range
    (GLOSSARY.md "Axis").

    Parameters
    ----------
    keys : one ``<stage>.<KNOB>`` key or a tuple of keys set to the same value.
    values : a list of values — chosen from in a random sweep, the levels of a grid —
        or a frozen ``scipy.stats`` distribution (random sweeps only), drawn through
        its ``ppf``. :meth:`interval` builds the uniform or log-uniform one.
    """

    def __init__(self, keys, values) -> None:
        self.keys = (keys,) if isinstance(keys, str) else tuple(keys)
        if not self.keys:
            raise ValueError("an axis needs at least one key")
        if isinstance(values, (list, tuple)):
            if not values:
                raise ValueError(f"axis {self.keys}: empty value list")
            self.values = list(values)
        elif hasattr(values, "ppf"):
            self.values = values
        else:
            raise TypeError(f"axis {self.keys}: values must be a list or a frozen "
                            f"scipy.stats distribution, got {type(values).__name__}")

    @classmethod
    def interval(cls, keys, lo: float, hi: float, log: bool = False) -> "Axis":
        """Uniform on ``[lo, hi]``, or log-uniform when ``log`` (for a knob that spans
        decades, such as a σ floor)."""
        if not lo < hi:
            raise ValueError(f"interval needs lo < hi, got [{lo}, {hi}]")
        return cls(keys, loguniform(lo, hi) if log else uniform(lo, hi - lo))

    @property
    def is_list(self) -> bool:
        return isinstance(self.values, list)

    def sample(self, u: float):
        """The axis value at unit coordinate ``u`` ∈ [0, 1]: equal bins over a list,
        the quantile of a distribution."""
        if self.is_list:
            n = len(self.values)
            return _plain(self.values[min(int(u * n), n - 1)])
        return _plain(self.values.ppf(u))

    def describe(self) -> dict:
        """A JSON-safe statement of the axis, for the sweep's sidecar."""
        if self.is_list:
            return {"keys": list(self.keys), "values": [_plain(v) for v in self.values]}
        dist = self.values
        return {"keys": list(self.keys), "distribution": dist.dist.name,
                "args": [_plain(a) for a in dist.args],
                "kwds": {k: _plain(v) for k, v in dist.kwds.items()}}


def _plain(v):
    """numpy scalars → Python scalars, so values repr and serialise as literals."""
    return v.item() if isinstance(v, np.generic) else v


class Sweep:
    """A set of chains that share the pipeline's base and differ only in knob values,
    run one after another and recorded as one ``sweep`` run (GLOSSARY.md "Sweep").

    Parameters
    ----------
    pipeline : the steps every point runs and the base chain they start from.
    axes : the varied knobs; no key may appear on two axes.
    mode : ``"random"`` (default; needs ``n_points`` and ``seed``) or ``"grid"`` (list
        axes only; takes neither).
    fixed : overrides applied to every point under the axes' values. Unless ``fixed``
        or an axis sets it, a sweep whose steps include ``opt`` passes
        ``opt.FEASIBILITY_CHECK = "none"``: the probe is ≈ 7 of Stage 4's ≈ 11 min,
        and an infeasible period still fails its point (#104).
    keep_intermediate : ``False`` keeps only the last step's runs (#108): every earlier
        step writes to ``<scratch_dir>/<sweep id>/<stage>`` and its run is deleted as soon
        as no remaining point reuses it; the manifest still names it, so a point can be
        rebuilt from its overrides. Default ``True`` keeps every run in ``data/proc``.
    scratch_dir : where the dropped runs live meanwhile, a RAM-backed tmpfs such as
        ``/dev/shm/pysuto-sweep``; needed when ``keep_intermediate`` is ``False``.
    """

    def __init__(self, pipeline: Pipeline, axes: Sequence[Axis], mode: str = "random",
                 n_points: Optional[int] = None, seed: Optional[int] = None,
                 fixed: Optional[Mapping[str, object]] = None,
                 keep_intermediate: bool = True,
                 scratch_dir: Optional[str] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}; expected one of {MODES}")
        if mode == "random" and (n_points is None or seed is None):
            raise ValueError("a random sweep needs n_points and seed")
        if mode == "grid" and (n_points is not None or seed is not None):
            raise ValueError("a grid sweep runs every combination; it takes no n_points "
                             "or seed")
        if mode == "random" and (not isinstance(n_points, int) or n_points < 1):
            raise ValueError(f"n_points must be a positive int, got {n_points!r}")
        axes = list(axes)
        if not axes:
            raise ValueError("a sweep needs at least one axis")
        keys = [k for ax in axes for k in ax.keys]
        if len(set(keys)) != len(keys):
            raise ValueError(f"a key appears on more than one axis: {keys}")
        if mode == "grid" and not all(ax.is_list for ax in axes):
            raise ValueError("a grid sweep needs list axes; draw from a distribution in a "
                             "random sweep")
        if not keep_intermediate and scratch_dir is None:
            raise ValueError("a sweep that keeps only its last step's runs needs a "
                             "scratch_dir for the others")
        self.pipeline, self.axes, self.mode = pipeline, axes, mode
        self.n_points, self.seed = n_points, seed
        self.keep_intermediate, self.scratch_dir = keep_intermediate, scratch_dir
        base_fixed = ({"opt.FEASIBILITY_CHECK": "none"}
                      if "opt" in pipeline.steps and "opt.FEASIBILITY_CHECK" not in keys
                      else {})
        self.fixed = base_fixed | dict(fixed or {})
        self._points = self._draw()
        for pt in self._points:
            pipeline.check_overrides(pt)

    # ------------------------------------------------------------------
    def points(self) -> list:
        """Every point's overrides, drawn once at declaration (same seed ⇒ same points)."""
        return [dict(p) for p in self._points]

    def _draw(self) -> list:
        if self.mode == "grid":
            rows = itertools.product(*(ax.values for ax in self.axes))
            values = [[_plain(v) for v in row] for row in rows]
        else:
            u = qmc.LatinHypercube(d=len(self.axes), scramble=True,
                                   seed=self.seed).random(self.n_points)
            values = [[ax.sample(ui) for ax, ui in zip(self.axes, row)] for row in u]
        out = []
        for row in values:
            pt = dict(self.fixed)
            for ax, v in zip(self.axes, row):
                pt.update({k: v for k in ax.keys})
            out.append(pt)
        return out

    def _prefixes(self) -> list:
        """Per point and step, what decides that step's run: its own and every earlier
        step's overrides. Two points share a step's run exactly when these are equal."""
        out = []
        for pt in self._points:
            by_stage, pre, row = self.pipeline.check_overrides(pt), (), []
            for step in self.pipeline.steps:
                pre += (repr(sorted(by_stage[step].items())),)
                row.append(pre)
            out.append(row)
        return out

    def plan(self, indices: Optional[Sequence[int]] = None) -> dict:
        """How many distinct runs each step needs over the points ``indices`` (default
        all): a step is rerun only when its own or an earlier step's overrides differ,
        so a sweep over solver knobs plans one imp and one const run."""
        idx = range(len(self._points)) if indices is None else indices
        pre = self._prefixes()
        return {step: len({pre[i][k] for i in idx})
                for k, step in enumerate(self.pipeline.steps)}

    def _scratch_steps(self) -> tuple:
        return () if self.keep_intermediate else self.pipeline.steps[:-1]

    def disk_need_gb(self, indices: Optional[Sequence[int]] = None) -> float:
        """Disk the planned runs write to ``data/proc`` plus :data:`RESERVE_GB`."""
        return sum(STEP_GB[s] * n for s, n in self.plan(indices).items()
                   if s not in self._scratch_steps()) + RESERVE_GB

    def declaration(self) -> dict:
        """What defines the sweep's points, as recorded in its sidecar and manifest."""
        base = self.pipeline.base
        return {"base": {"feed_run_ids": list(base.feed_run_ids), "imp": base.imp,
                         "const": base.const, "opt": base.opt},
                "steps": list(self.pipeline.steps), "mode": self.mode,
                "n_points": len(self._points), "seed": self.seed,
                "axes": [ax.describe() for ax in self.axes],
                "fixed": {k: _plain(v) for k, v in self.fixed.items()},
                "keep_intermediate": self.keep_intermediate}

    # ------------------------------------------------------------------
    def run(self, label: str, resume: Optional[str] = None) -> str:
        """Run every point not yet done and return the sweep run id.

        A new sweep is registered before its first point runs, with every point
        ``pending``; the manifest is rewritten after each step. A point whose driver
        fails is recorded ``failed`` with the output tail and the sweep goes on;
        nothing is retried or loosened. Within a sweep a step whose upstream runs and
        overrides an earlier point already ran reuses that run. ``resume`` continues
        a sweep: it refuses a sweep declared differently or a checkout at another
        commit, and reruns every point not ``done``.
        """
        reg = SweepRunRegistry(os.path.join(self.pipeline.cwd, "data", "proc", "sweep"))
        decl = self.declaration()
        commit, dirty = _git_state(self.pipeline.cwd)
        if resume is None:
            self._check_disk(None)
            manifest = {"declaration": decl, "commit": commit, "points": [
                {"index": i, "overrides": {k: _plain(v) for k, v in pt.items()},
                 "status": "pending", "runs": {s: None for s in self.pipeline.steps},
                 "reused": [], "started_at": None, "finished_at": None, "error": None,
                 "log": None}
                for i, pt in enumerate(self._points)]}
            base = self.pipeline.base
            sid = reg.register(
                label=label, manifest=manifest,
                parents={"inputs": {}, "upstream": {
                    "feed_run_ids": list(base.feed_run_ids), "imp_run_id": base.imp,
                    "const_run_id": base.const, "opt_run_id": base.opt}},
                stage_config=decl | {"commit": commit, "dirty": dirty,
                                     "scipy": scipy.__version__},
                stage_summary=_summary(manifest, 0.0))
            manifest = reg.load_manifest(sid)
        else:
            sid, manifest = resume, reg.load_manifest(resume)
            if _canonical(manifest["declaration"]) != _canonical(decl):
                raise ValueError(f"sweep {sid} was declared differently; resume it with "
                                 f"the declaration in its sidecar")
            if manifest["commit"] != commit:
                raise RuntimeError(f"sweep {sid} ran at commit {manifest['commit']}, the "
                                   f"checkout is at {commit}; its points would not compare")
            drawn = [_canonical({k: _plain(v) for k, v in pt.items()})
                     for pt in self._points]
            if [_canonical(p["overrides"]) for p in manifest["points"]] != drawn:
                raise ValueError(f"sweep {sid}: this declaration's points were drawn "
                                 f"differently (another scipy?); resume it where it ran")
            if dirty:
                _log.warning("resuming sweep %s in a checkout with uncommitted changes to "
                             "tracked files; its points may not compare", sid)
            self._check_disk([p["index"] for p in manifest["points"]
                              if p["status"] != "done"])

        runner = self._runner(sid)
        memo = self._memo_from(manifest, runner)
        t0 = time.perf_counter()
        points = manifest["points"]
        for pos, pt in enumerate(points):
            if pt["status"] == "done":
                continue
            self._run_point(sid, pt, memo, reg, manifest, runner)
            self._release(pt, [q["index"] for q in points[pos + 1:]
                               if q["status"] != "done"], memo, runner)
        if self._scratch_steps():
            shutil.rmtree(os.path.join(self.scratch_dir, sid), ignore_errors=True)
        reg.update(sid, manifest=manifest,
                   stage_summary=_summary(manifest, time.perf_counter() - t0))
        return sid

    def _runner(self, sid: str) -> Pipeline:
        """The pipeline the points run: this one, its dropped steps sent to scratch."""
        if not self._scratch_steps():
            return self.pipeline
        p = self.pipeline
        return Pipeline(p.steps, p.base, p.cwd, p.scripts_dir, p.log_dir, run_dirs={
            s: os.path.join(self.scratch_dir, sid, s) for s in self._scratch_steps()})

    def _release(self, pt, remaining, memo, runner) -> None:
        """Delete the scratch runs of ``pt`` that no point in ``remaining`` will reuse."""
        pre = self._prefixes()
        for k, step in enumerate(self.pipeline.steps):
            rid = pt["runs"].get(step)
            if step not in runner.run_dirs or rid is None:
                continue
            if any(pre[j][k] == pre[pt["index"]][k] for j in remaining):
                continue
            shutil.rmtree(os.path.join(runner.run_dirs[step], rid), ignore_errors=True)
            for key in [m for m, v in memo.items() if v == rid]:
                del memo[key]

    def _run_point(self, sid, pt, memo, reg, manifest, runner) -> None:
        i = pt["index"]
        by_stage = self.pipeline.check_overrides(self._points[i])
        pt.update(status="running", started_at=_now(), error=None, log=None, reused=[],
                  runs={s: None for s in self.pipeline.steps})
        reg.update(sid, manifest=manifest)
        chain = self.pipeline.base
        try:
            for step in self.pipeline.steps:
                key = _memo_key(step, chain, by_stage[step])
                if key in memo:
                    rid = memo[key]
                    pt["reused"].append(step)
                else:
                    rid = runner.run_step(step, chain, by_stage[step], f"{sid} p{i:03d}")
                    memo[key] = rid
                chain = replace(chain, **{step: rid})
                pt["runs"][step] = rid
                reg.update(sid, manifest=manifest)
            pt["status"] = "done"
        except StepFailed as exc:
            pt.update(status="failed", error=exc.tail, log=exc.log_path)
            _log.warning("sweep %s point %d failed in step %s (log %s)",
                         sid, i, exc.step, exc.log_path)
        pt["finished_at"] = _now()
        reg.update(sid, manifest=manifest)

    def _memo_from(self, manifest, runner) -> dict:
        """Steps already run by any point of this sweep and still on hand, keyed for
        reuse (a released scratch run, or one a reboot cleared, is not)."""
        memo: dict = {}
        for pt in manifest["points"]:
            by_stage = self.pipeline.check_overrides(self._points[pt["index"]])
            chain = self.pipeline.base
            for step in self.pipeline.steps:
                rid = pt["runs"].get(step)
                if rid is None or (step in runner.run_dirs and not os.path.isdir(
                        os.path.join(runner.run_dirs[step], rid))):
                    break
                memo.setdefault(_memo_key(step, chain, by_stage[step]), rid)
                chain = replace(chain, **{step: rid})
        return memo

    def _check_disk(self, indices: Optional[Sequence[int]]) -> None:
        need = self.disk_need_gb(indices)
        free = shutil.disk_usage(self.pipeline.cwd).free / 1024 ** 3
        if free < need:
            raise RuntimeError(f"the sweep's planned runs {self.plan(indices)} need "
                               f"≈ {need:.2f} GB free ({RESERVE_GB} GB reserve included); "
                               f"{free:.2f} GB are free")
        if self._scratch_steps():
            # Points that share no run (a hypercube over imp knobs) hold one point's
            # dropped runs at a time; a grid whose points share runs can hold more.
            os.makedirs(self.scratch_dir, exist_ok=True)
            need = sum(STEP_GB[s] for s in self._scratch_steps())
            free = shutil.disk_usage(self.scratch_dir).free / 1024 ** 3
            if free < need:
                raise RuntimeError(f"the scratch store {self.scratch_dir} needs ≈ "
                                   f"{need:.2f} GB for one point's dropped runs; "
                                   f"{free:.2f} GB are free")


def _canonical(obj) -> str:
    """One JSON form for values read back from a manifest and values in memory (a tuple
    and the list it serialises to compare equal)."""
    return json.dumps(obj, sort_keys=True)


def _memo_key(step: str, chain: Chain, knobs: Mapping[str, object]) -> tuple:
    upstream = {"imp": tuple(chain.feed_run_ids), "const": chain.imp,
                "opt": chain.const}[step]
    return step, upstream, repr(sorted(knobs.items()))


def _summary(manifest: dict, wall_s: float) -> dict:
    status = [p["status"] for p in manifest["points"]]
    return {"n_points": len(status),
            **{f"n_{s}": status.count(s) for s in ("done", "failed", "pending", "running")},
            "wall_s_last_run": wall_s}


# Variables that point git at a repository other than the one at cwd. A git hook
# exports them, so a sweep started from one would otherwise read the hook's repository.
GIT_LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                     "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_PREFIX")


def _git_state(cwd: str) -> tuple:
    """(HEAD commit, dirty tracked files) of the checkout at ``cwd``, (None, None)
    outside git."""
    env = {k: v for k, v in os.environ.items() if k not in GIT_LOCATION_VARS}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                          text=True, env=env)
    if head.returncode != 0:
        return None, None
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=cwd, capture_output=True, text=True, env=env)
    return head.stdout.strip(), bool(status.stdout.strip())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
