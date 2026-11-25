"""
pipeline.py
===========
The pipeline (GLOSSARY.md "Pipeline"): the stages composed as steps, so a **chain** of
runs can be produced with **knob** overrides from Python (#103).

Each step runs in a fresh child process, ``python -m pysuto._step <spec.json>``
(:mod:`pysuto._step`, #186). The child resolves the case reference the pipeline holds,
calls the stage function of :mod:`pysuto.stages` and hands the run id back through a
file. The overrides cross as ``KNOB=VALUE`` literals, parsed and type-checked by the knob
protocol (:mod:`pysuto.knobs`), with the same names, types and precedence as the drivers'
``--set``. A child frees its memory when it exits, which ADR 0045 measured one process
does not do across points. A step's stdout and stderr go to
``<log_dir>/<label>_<step>.log``; the parent holds the stage lock of the data dir around
each child (#101). The module never imports ``scripts/`` or the cases (README §5, import
direction): the parent and the child resolve the case reference with the one resolver of
:mod:`pysuto.case` (#187).

    base  = Chain.from_opt_run("opt_20261020_112726_2f451f31")
    chain = Pipeline(steps=("imp", "const", "opt"), base=base, case="scripts/copper.py").run(
        {"imp.P_PROP": 1.5, "const.P_PROP": 1.5}, label="p = 1.5")

A chain can also start at Stage 1 and end at Stage 6 (#125): the steps are
:data:`ALL_STEPS`, ``python -m pysuto run --case <id> --from <stage> --to <stage>`` runs
them (:mod:`pysuto.__main__`), and a step past ``feed`` reads its upstream from the base
chain. The ``feed`` step is the case's Stage 1 hook (``CaseSpec.stage1``), run in its
child like every step; ``uncprop`` and
``psut`` take no knobs.

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
          n_points=30, seed=20261021).run(label="knob sweep")
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import scipy
from scipy.stats import loguniform, qmc, uniform

import pysuto
from pysuto._provenance import now_iso
from pysuto._step import StepSpec
from pysuto.case import case_reference, resolve_case
from pysuto.logging import get_logger
from pysuto.registry import SweepRunRegistry, stage_lock
from pysuto.registry.lock import HOLDER_ENV

_log = get_logger("pysuto.pipeline")

# The stages a pipeline sweeps by default, in order (a sweep varies imp, const and opt
# knobs only), and every step a pipeline can run (#125).
STEPS     = ("imp", "const", "opt")
ALL_STEPS = ("feed", "imp", "const", "opt", "uncprop", "psut")

# Knobs that must share one value whenever both stages run (the cross-stage Lp
# exponent: "P_PROP must match", the copper case's Stage 2 and Stage 3 sections).
TIED = (("imp.P_PROP", "const.P_PROP"),)

_TAIL_LINES = 40
# The stage whose run each step reads; the imp and const steps may read and write
# outside ``data/proc`` (opt runs always land there).
_UPSTREAM_ID = {"imp": lambda c: list(c.feed_run_ids), "const": lambda c: c.imp,
                "opt": lambda c: c.const, "uncprop": lambda c: c.opt,
                "psut": lambda c: c.opt}
_UPSTREAM  = {"imp": None, "const": "imp", "opt": "const", "uncprop": "opt",
              "psut": "opt"}
_DIR_STEPS = ("imp", "const")


@dataclass(frozen=True)
class Chain:
    """One lineage of runs through the pipeline (GLOSSARY.md "Chain")."""
    feed_run_ids: tuple
    imp:          Optional[str] = None
    const:        Optional[str] = None
    opt:          Optional[str] = None
    uncprop:      Optional[str] = None
    psut:         Optional[str] = None

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
    """A step's child exited non-zero; ``tail`` holds the end of its output."""

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
    """Run a contiguous slice of :data:`ALL_STEPS` as one chain.

    Parameters
    ----------
    steps : the stages to run, a contiguous slice of :data:`ALL_STEPS` (default
        :data:`STEPS`).
    base : the chain the first step's upstream comes from; stages outside ``steps``
        keep the base chain's runs in the result. A chain whose first step is ``feed``
        needs none.
    cwd : the directory ``data_dir``, ``config``, ``log_dir`` and ``run_dirs`` are
        relative to; default the working directory.
    case : the case reference every step runs (the ``--case`` string, required: an alias,
        a dotted module or a file path). The pipeline resolves it once with
        :func:`pysuto.case.resolve_case` to validate it, and each child resolves it again
        with the same function; a file path is made absolute against ``cwd`` here, and
        the children receive that form.
    data_dir : the data root of every step, holding ``proc/`` and ``input/`` (default
        ``data``, relative to ``cwd``); the child's file logs go to ``<data_dir>/logs``.
    config : a config whose ``pipeline.acc_id`` replaces the case's acc id in the imp and
        const steps, the only stages that read one.
    log_dir : where each step's stdout is written, with the traceback of a failed step;
        default ``<data_dir>/logs/pipeline`` (``data`` when ``data_dir`` is ``None``),
        relative to ``cwd``.
    run_dirs : per imp or const step, the directory its run is written to and the next
        step reads it from, instead of ``data/proc/<stage>`` — a sweep's scratch store.
        A stage outside ``steps`` is always read from ``data/proc``: it is the base's.

    The case's Stage 1 hook (``stage1``) is the ``feed`` step.
    """

    def __init__(self, steps: tuple = STEPS, base: Optional[Chain] = None,
                 cwd: Optional[str] = None, log_dir: Optional[str] = None,
                 run_dirs: Optional[Mapping[str, str]] = None, *, case: str,
                 data_dir: Optional[str] = None,
                 config: Optional[str] = None) -> None:
        steps = tuple(steps)
        if not _is_contiguous_slice(steps):
            raise ValueError(f"steps {steps} must be a contiguous slice of {ALL_STEPS}")
        if base is None and steps[0] == "feed":
            base = Chain(feed_run_ids=())
        if base is None:
            raise ValueError("a pipeline needs a base chain for its first step's upstream")
        case = case_reference(case, cwd)
        spec = resolve_case(case)
        if "feed" in steps and spec.stage1 is None:
            raise ValueError(f"the feed step is the case's Stage 1 hook, and case "
                             f"{spec.id!r} declares none")
        if "feed" in steps and data_dir is None:
            raise ValueError("the feed step writes into a data dir; pass data_dir")
        self.steps       = steps
        self.base        = base
        self.case        = case
        self.data_dir    = None if data_dir is None else str(data_dir)
        self.cwd         = str(cwd or os.getcwd())
        self.config      = None if config is None else self._under_cwd(config)
        self.log_dir     = str(log_dir or os.path.join(
            self.cwd, self.data_dir or "data", "logs", "pipeline"))
        self.run_dirs    = {k: self._under_cwd(v) for k, v in (run_dirs or {}).items()}
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
            if stage not in STEPS:
                raise KeyError(f"override {key!r}: the {stage} step takes no knobs")
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
            if step == "feed":
                chain = replace(chain, feed_run_ids=self._run_feed(label))
                continue
            rid = self.run_step(step, chain, by_stage[step], label)
            chain = replace(chain, **{step: rid})
        return chain

    def _under_cwd(self, path) -> str:
        """``path`` resolved against ``cwd`` (an absolute path stays as it is)."""
        return os.path.abspath(os.path.join(self.cwd, str(path)))

    def _data_root(self) -> str:
        return os.path.abspath(os.path.join(self.cwd, self.data_dir or "data"))

    def _run_feed(self, label: str) -> tuple:
        """The case's Stage 1 hook, in its child under the stage lock (the feed registry
        has none of its own); returns the feed run ids it registered."""
        run_ids = self.run_step("feed", None, {}, label).split("\n")
        return tuple(run_ids)

    def run_step(self, step: str, upstream: Optional[Chain], knobs: Mapping[str, object],
                 label: str) -> str:
        """Run one stage on ``upstream``'s runs with ``knobs`` set, in a child process
        under the data dir's stage lock; return its run id (the ``feed`` step: the feed
        run ids, one per line). A child that exits non-zero raises :class:`StepFailed`."""
        os.makedirs(self.log_dir, exist_ok=True)
        slug = re.sub(r"[^\w.-]+", "_", label).strip("_") or "run"
        log_path = os.path.join(self.log_dir, f"{slug}_{step}.log")
        proc = os.path.join(self._data_root(), "proc")
        os.makedirs(proc, exist_ok=True)
        spec = self._spec(step, upstream, knobs, label, log_path)
        _log.info("step %s: %s", step, label)
        with stage_lock(f"pipeline_{step}", root=proc):
            run_id = self._launch(step, spec)
        _log.info("step %s: %s", step, run_id.replace("\n", " "))
        return run_id

    def _spec(self, step, upstream, knobs, label, log_path) -> StepSpec:
        """What the child of ``step`` is told: knobs as ``KNOB=repr`` strings."""
        return StepSpec(
            step=step, case=self.case, label=label,
            upstream=None if step == "feed" else _UPSTREAM_ID[step](upstream),
            data_dir=self._data_root(),
            config=self.config if step in _DIR_STEPS else None,
            dirs={f"{s}_dir": self.run_dirs[s] for s in (_UPSTREAM.get(step), step)
                  if s in self.run_dirs},
            knobs=[f"{k}={v!r}" for k, v in knobs.items()], log_path=log_path)

    def _launch(self, step: str, spec: StepSpec) -> str:
        """Run ``spec`` in a fresh child, ``python -m pysuto._step``, from ``cwd``, with
        the directory of this process's ``pysuto`` first on its ``PYTHONPATH`` (a
        worktree or test parent never runs another checkout's code); return the run id
        the child wrote. Its stdout and stderr go to ``spec.log_path``. A child that
        exits non-zero, or exits 0 without a run id, raises :class:`StepFailed`."""
        env = _child_env()
        env[HOLDER_ENV] = str(os.getpid())          # run_step holds the stage lock here
        with tempfile.TemporaryDirectory() as tmp:
            spec = replace(spec, run_id_file=os.path.join(tmp, "run_id"))
            spec_path = os.path.join(tmp, "spec.json")
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(asdict(spec), f)
            cmd = [sys.executable, "-m", "pysuto._step", spec_path]
            with open(spec.log_path, "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=self.cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise StepFailed(step, rc, _tail(spec.log_path), spec.log_path)
            try:
                with open(spec.run_id_file, encoding="utf-8") as f:
                    run_id = f.read().strip()
            except FileNotFoundError:
                run_id = ""
            if not run_id:
                raise StepFailed(step, rc, "the child exited 0 and wrote no run id\n"
                                 + _tail(spec.log_path), spec.log_path)
            return run_id


def _child_env() -> dict:
    """This process's environment for a step child, with the directory of this ``pysuto``
    first on ``PYTHONPATH`` so a worktree or test parent never runs another checkout's
    code. A site-packages directory (an installed wheel) is left off: the child finds it
    anyway, and ahead of the stdlib on ``PYTHONPATH`` it would shadow it. Both sides are
    compared resolved, since site-packages is often reached through a symlink."""
    src = Path(pysuto.__file__).resolve().parents[1]
    sites = {Path(p).resolve() for p in (*site.getsitepackages(), site.getusersitepackages())}
    env = dict(os.environ)
    if src not in sites:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _after(steps: tuple) -> tuple:
    """The stages after a contiguous slice of :data:`ALL_STEPS`."""
    return ALL_STEPS[ALL_STEPS.index(steps[-1]) + 1:]


def _is_contiguous_slice(steps: tuple) -> bool:
    if not steps or steps[0] not in ALL_STEPS:
        return False
    i = ALL_STEPS.index(steps[0])
    return steps == ALL_STEPS[i:i + len(steps)]


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
    scratch_dir : where the dropped runs live meanwhile, such as ``default_scratch_dir()``
        (RAM-backed where the platform has ``/dev/shm``); needed when ``keep_intermediate``
        is ``False``. A run that survives its deletion raises.
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

    def scratch_need_gb(self) -> float:
        """Room one point's dropped runs take in the scratch store; 0 when every run is
        kept. Points that share no run (a hypercube over imp knobs) hold one point's
        dropped runs at a time; a grid whose points share runs can hold more."""
        return sum(STEP_GB[s] for s in self._scratch_steps())

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
        pl  = self.pipeline
        reg = SweepRunRegistry(os.path.join(pl.cwd, pl.data_dir or "data", "proc", "sweep"))
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
        reg.update(sid, manifest=manifest,
                   stage_summary=_summary(manifest, time.perf_counter() - t0))
        if self._scratch_steps():          # after the save, so --resume finds the manifest
            _remove_scratch(os.path.join(self.scratch_dir, sid))
        return sid

    def _runner(self, sid: str) -> Pipeline:
        """The pipeline the points run: this one, its dropped steps sent to scratch."""
        if not self._scratch_steps():
            return self.pipeline
        p = self.pipeline
        dirs = {**p.run_dirs, **{s: os.path.join(self.scratch_dir, sid, s)
                                 for s in self._scratch_steps()}}
        return Pipeline(steps=p.steps, base=p.base, cwd=p.cwd, log_dir=p.log_dir,
                        run_dirs=dirs, case=p.case,
                        data_dir=p.data_dir, config=p.config)

    def _release(self, pt, remaining, memo, runner) -> None:
        """Delete the scratch runs of ``pt`` that no point in ``remaining`` will reuse."""
        pre = self._prefixes()
        for k, step in enumerate(self.pipeline.steps):
            rid = pt["runs"].get(step)
            if step not in self._scratch_steps() or rid is None:
                continue
            if any(pre[j][k] == pre[pt["index"]][k] for j in remaining):
                continue
            _remove_scratch(os.path.join(runner.run_dirs[step], rid))
            for key in [m for m, v in memo.items() if v == rid]:
                del memo[key]

    def _run_point(self, sid, pt, memo, reg, manifest, runner) -> None:
        i = pt["index"]
        by_stage = self.pipeline.check_overrides(self._points[i])
        pt.update(status="running", started_at=now_iso(), error=None, log=None, reused=[],
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
        pt["finished_at"] = now_iso()
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
                if rid is None or (step in self._scratch_steps() and not os.path.isdir(
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
            os.makedirs(self.scratch_dir, exist_ok=True)
            need = self.scratch_need_gb()
            free = shutil.disk_usage(self.scratch_dir).free / 1024 ** 3
            if free < need:
                raise RuntimeError(f"the scratch store {self.scratch_dir} needs ≈ "
                                   f"{need:.2f} GB for one point's dropped runs; "
                                   f"{free:.2f} GB are free")


def default_scratch_dir() -> str:
    """A scratch store that exists on this platform: ``/dev/shm/pysuto-sweep`` (RAM-backed)
    where ``/dev/shm`` exists, else ``pysuto-sweep`` under ``tempfile.gettempdir()``."""
    if os.path.isdir("/dev/shm"):
        return "/dev/shm/pysuto-sweep"
    return os.path.join(tempfile.gettempdir(), "pysuto-sweep")


def _remove_scratch(path: str) -> None:
    """Delete a scratch directory; raise when anything of it is still there (Windows
    refuses to delete open files), never leave it behind silently."""
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise RuntimeError(f"the sweep could not delete its scratch run {path}: "
                           f"{exc}") from exc
    if os.path.exists(path):
        raise RuntimeError(f"the sweep's scratch run {path} is still there after its "
                           f"deletion")


def _canonical(obj) -> str:
    """One JSON form for values read back from a manifest and values in memory (a tuple
    and the list it serialises to compare equal)."""
    return json.dumps(obj, sort_keys=True)


def _memo_key(step: str, chain: Chain, knobs: Mapping[str, object]) -> tuple:
    up = _UPSTREAM[step]
    upstream = tuple(chain.feed_run_ids) if up is None else getattr(chain, up)
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

