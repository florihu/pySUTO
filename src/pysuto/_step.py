"""
_step.py
========
The child of a pipeline step (#186): ``python -m pysuto._step <spec.json>`` runs one stage
of :mod:`pysuto.stages` on the case a reference names, then exits, so its memory returns
to the system (ADR 0045). Private: :class:`pysuto.pipeline.Pipeline` writes the spec and
launches it; nothing else calls it.

The spec is a :class:`StepSpec`, written as JSON. The run id (``feed``: the feed run ids,
one per line) goes to its ``run_id_file``. Knobs cross as ``KNOB=VALUE`` literals, parsed
and type-checked by :mod:`pysuto.knobs` against the defaults of the stage.

``run_spec`` resolves the case reference with :func:`pysuto.case.resolve_case`, the
function the pipeline validated it with (#187). Its imports are function-local so that the
pipeline can import :class:`StepSpec` without the stages.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from typing import List, Optional, Union


@dataclass
class StepSpec:
    """What the child of one pipeline step is told.

    ``upstream`` is the feed run ids (``imp``), the imp, const or opt run id of the stage
    before, or ``None`` (``feed``); ``dirs`` holds ``imp_dir`` and ``const_dir``;
    ``knobs`` are ``KNOB=repr(value)`` strings; ``log_path`` is where the parent sends
    the child's output.
    """
    step:        str
    case:        str
    label:       str
    upstream:    Optional[Union[str, List[str]]]
    data_dir:    str
    config:      Optional[str]
    dirs:        dict = field(default_factory=dict)
    knobs:       List[str] = field(default_factory=list)
    log_path:    str = ""
    run_id_file: str = ""


def run_spec(spec: StepSpec) -> str:
    """Run the step ``spec`` describes in this process; return its run id."""
    # lazy: pipeline imports StepSpec from here without the stages
    from pysuto import stages
    from pysuto.case import resolve_case
    from pysuto.config import load_config
    from pysuto.knobs import parse_overrides
    step, data_dir, upstream, pairs = spec.step, spec.data_dir, spec.upstream, spec.knobs
    case = resolve_case(spec.case)
    if step == "feed":
        return "\n".join(case.stage1(data_dir).values())
    if spec.config is not None:
        case = replace(case, acc_id=load_config(spec.config).pipeline.acc_id)
    if step in ("imp", "const"):
        defaults = getattr(case, step).knobs()
        knobs = parse_overrides(pairs, tuple(defaults), defaults)
        case = case.with_knobs(step, knobs)
        stage = stages.run_imp if step == "imp" else stages.run_const
        return stage(case, upstream, spec.label, knobs, data_dir=data_dir, **spec.dirs)
    if step == "opt":
        knobs = parse_overrides(pairs, stages.OPT_KNOBS, stages.opt_defaults())
        return stages.run_opt(case, upstream, spec.label, knobs, data_dir=data_dir,
                              **spec.dirs)
    if pairs:
        raise KeyError(f"the {step} step takes no knobs, got {pairs}")
    stage = stages.run_uncprop if step == "uncprop" else stages.run_psut
    return stage(case, upstream, spec.label, data_dir=data_dir)


def main(argv=None) -> int:
    # lazy: pipeline imports StepSpec from here without the stages
    from pysuto.knobs import write_run_id
    from pysuto.logging import start_run_log
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m pysuto._step <spec.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        spec = StepSpec(**json.load(f))
    start_run_log(spec.data_dir, f"step_{spec.step}")
    write_run_id(spec.run_id_file, run_spec(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
