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
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

from pysuto.logging import get_logger

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
        self.step, self.returncode, self.tail, self.log_path = step, returncode, tail, log_path
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
    python : the interpreter the drivers run under; default this one.
    """

    def __init__(self, steps: tuple = STEPS, base: Optional[Chain] = None,
                 cwd: Optional[str] = None, scripts_dir: Optional[str] = None,
                 log_dir: Optional[str] = None, python: str = sys.executable) -> None:
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
        self.python      = python

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
                raise ValueError(f"tied knobs {' and '.join(live)} must share one value, got "
                                 + ", ".join(f"{k}={overrides[k]!r}" for k in given))
        return by_stage

    def run(self, overrides: Mapping[str, object], label: str) -> Chain:
        """Run every step with its overrides and return the new chain."""
        by_stage = self.check_overrides(overrides)
        chain = self.base
        for step in self.steps:
            chain = replace(chain, **{step: self.run_step(step, chain, by_stage[step], label)})
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
        cmd = [self.python, os.path.join(self.scripts_dir, _SCRIPTS[step])]
        if step == "imp":
            for feed in upstream.feed_run_ids:
                cmd += ["--feed-id", feed]
            cmd += ["--label", label]
        elif step == "const":
            cmd += ["--imp-run-id", upstream.imp, label]
        else:
            cmd += ["--const-run-id", upstream.const, "--label", label]
        for knob, value in knobs.items():
            cmd += ["--set", f"{knob}={value!r}"]
        return cmd + ["--run-id-file", rid_file]


def _is_contiguous_slice(steps: tuple) -> bool:
    if not steps or steps[0] not in STEPS:
        return False
    i = STEPS.index(steps[0])
    return steps == STEPS[i:i + len(steps)]


def _tail(path: str, n: int = _TAIL_LINES) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])
