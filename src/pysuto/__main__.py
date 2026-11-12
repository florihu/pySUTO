"""
__main__.py
===========
``python -m pysuto run`` (#125, #26 §3): one command runs a chain of a case through the
stage drivers, from a Stage 1 hook or a base chain to any later stage::

    python -m pysuto run --case toy --from feed --to psut --data-dir /tmp/toy
    python -m pysuto run --case copper --from imp --to psut --base opt_20261022_112855_928da10c

The stages are :data:`pysuto.pipeline.ALL_STEPS`. ``--from feed`` runs the case's Stage 1
hook (``CaseSpec.stage1``) and needs no base; any later ``--from`` takes its upstream
from ``--base <opt_run_id>`` (the base chain's parents, as ``Chain.from_opt_run``). Run it
from the repository root: the drivers' ``logs/`` and the default ``data`` are relative to
the working directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pysuto.cases import add_case_arg, case_from_args
from pysuto.pipeline import ALL_STEPS, REPO_ROOT, Chain, Pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pysuto")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a chain of a case through the stage drivers.")
    add_case_arg(run)
    run.add_argument("--from", dest="first", choices=ALL_STEPS, default="feed",
                     help="First stage to run (default feed).")
    run.add_argument("--to", dest="last", choices=ALL_STEPS, default="psut",
                     help="Last stage to run (default psut).")
    run.add_argument("--base", default=None, metavar="OPT_RUN_ID",
                     help="Opt run whose chain supplies the upstream of a --from past feed.")
    run.add_argument("--data-dir", default="data",
                     help="Data root holding proc/ and input/ (default 'data').")
    run.add_argument("--config", default=None,
                     help="Config passed to the imp and const steps only.")
    run.add_argument("--label", default=None,
                     help="Label of every run in the chain (default: the case id).")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    case = case_from_args(parser, args)
    first, last = ALL_STEPS.index(args.first), ALL_STEPS.index(args.last)
    if first > last:
        parser.error(f"--from {args.first} comes after --to {args.last}")
    if args.first == "feed" and args.base is not None:
        parser.error("--base supplies the upstream of a --from past feed; "
                     "--from feed has none")
    if args.first != "feed" and args.base is None:
        parser.error(f"--from {args.first} needs --base <opt_run_id> for its upstream")
    base = None
    if args.base is not None:
        base = Chain.from_opt_run(args.base, root=os.path.join(args.data_dir, "proc"))
    try:
        pipeline = Pipeline(steps=ALL_STEPS[first:last + 1], base=base, cwd=os.getcwd(),
                            scripts_dir=str(Path(REPO_ROOT) / "scripts"), case=case.id,
                            data_dir=args.data_dir, config=args.config,
                            stage1=case.stage1)
    except ValueError as exc:                   # e.g. copper's missing Stage 1 hook (#109)
        parser.error(str(exc))
    chain = pipeline.run({}, label=args.label or case.id)
    if chain.feed_run_ids:
        print(f"{'feed':8s}{' '.join(chain.feed_run_ids)}")
    for step in ("imp", "const", "opt", "uncprop", "psut"):
        print(f"{step:8s}{getattr(chain, step) or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
