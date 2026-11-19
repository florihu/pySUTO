"""
__main__.py
===========
``python -m pysuto run`` (#125, #26 §3): one command runs a chain of a case through the
six stages, each in a fresh child process (#186), from a Stage 1 hook or a base chain
to any later stage::

    python -m pysuto run --case toy --from feed --to psut --data-dir /tmp/toy
    python -m pysuto run --case scripts/copper.py --from imp --base <opt> --data-dir data

``--case`` is required: a bundled alias (``toy``, ``toy_coproduct``), a dotted module
(``mylab.zinc``) or a file path (``./zinc_case.py``), each naming a module that declares a
:class:`~pysuto.case.CaseSpec` ``CASE`` (:func:`pysuto.case.resolve_case`, #187).

The stages are :data:`pysuto.pipeline.ALL_STEPS`. ``--from feed`` runs the case's Stage 1
hook (``CaseSpec.stage1``) and needs no base; any later ``--from`` takes its upstream
from ``--base <opt_run_id>`` (the base chain's parents, as ``Chain.from_opt_run``). The
stage functions are :mod:`pysuto.stages`, called by the private child entry point
:mod:`pysuto._step` with the ``--case`` reference; the command needs no ``scripts/``
directory. ``--data-dir`` is required, and a relative one is relative to the working
directory.

Output: the command prints one line per stage to stdout, the stage name and the run
id(s) it produced (``feed`` lists its three feed runs), and nothing else; that is the
command's result, not a log. The log lines of the pipeline go to the logging handlers and
``<data-dir>/logs/``; each step's stdout and stderr are captured in
``<data-dir>/logs/pipeline/<label>_<step>.log``.

``python -m pysuto tutorial [DEST]`` (#188) copies the eleven tutorial notebooks, their
README and the params template out of the installed package into a writable folder
(default ``./pysuto-tutorial``, which must not exist or be empty); open
``00_world.ipynb`` there with the ``python3`` kernel (``pip install "pysuto[tutorial]"``).

``python -m pysuto compare --data-dir D --base <leaf> [<leaf>] --new <leaf> [<leaf>]``
(#157) compares two chains array by array through their leaf run ids and exits 1 on any
difference; it writes nothing (:mod:`pysuto.compare`).
"""

from __future__ import annotations

import argparse
import os
import sys

from pysuto.case import add_case_arg, case_from_args
from pysuto.compare import CompareError, compare_chains, format_report
from pysuto.logging import configure_file_logs
from pysuto.pipeline import ALL_STEPS, Chain, Pipeline
from pysuto.tutorial import copy_tutorial


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pysuto")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a chain of a case through the six stages.")
    add_case_arg(run, default=None)
    run.add_argument("--from", dest="first", choices=ALL_STEPS, default="feed",
                     help="First stage to run (default feed).")
    run.add_argument("--to", dest="last", choices=ALL_STEPS, default="psut",
                     help="Last stage to run (default psut).")
    run.add_argument("--base", default=None, metavar="OPT_RUN_ID",
                     help="Opt run whose chain supplies the upstream of a "
                          "--from past feed.")
    run.add_argument("--data-dir", required=True,
                     help="Data root with proc/ and input/; logs go to <data-dir>/logs.")
    run.add_argument("--config", default=None,
                     help="Config passed to the imp and const steps only.")
    run.add_argument("--label", default=None,
                     help="Label of every run in the chain (default: the case id).")
    tut = sub.add_parser("tutorial", help="Copy the tutorial notebooks into a folder.")
    tut.add_argument("dest", nargs="?", default="pysuto-tutorial",
                     help="Folder to write; must not exist or be empty "
                          "(default ./pysuto-tutorial).")
    cmp = sub.add_parser("compare", help="Compare two chains array by array; exit 1 on "
                                         "any difference; writes nothing.")
    cmp.add_argument("--base", nargs="+", required=True, metavar="LEAF",
                     help="Leaf run id(s) of the reference chain; name both psut and "
                          "uncprop to cover a chain whose opt has two children.")
    cmp.add_argument("--new", nargs="+", required=True, metavar="LEAF",
                     help="Leaf run id(s) of the chain to check.")
    cmp.add_argument("--data-dir", required=True,
                     help="Data root whose proc/ holds both chains (read only).")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "tutorial":
        try:
            written = copy_tutorial(args.dest)
        except FileExistsError as exc:
            parser.error(str(exc))
        print(f"{len(written)} files written to {args.dest}; open 00_world.ipynb")
        return 0
    if args.command == "compare":
        try:
            report = compare_chains(args.data_dir, args.base, args.new)
        except CompareError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(format_report(report))
        return 0 if report.identical else 1
    case = case_from_args(parser, args)
    first, last = ALL_STEPS.index(args.first), ALL_STEPS.index(args.last)
    if first > last:
        parser.error(f"--from {args.first} comes after --to {args.last}")
    if args.first == "feed" and args.base is not None:
        parser.error("--base supplies the upstream of a --from past feed; "
                     "--from feed has none")
    if args.first != "feed" and args.base is None:
        parser.error(f"--from {args.first} needs --base <opt_run_id> for its upstream")
    configure_file_logs(os.path.join(args.data_dir, "logs"))
    base = None
    if args.base is not None:
        base = Chain.from_opt_run(args.base, root=os.path.join(args.data_dir, "proc"))
    try:
        pipeline = Pipeline(steps=ALL_STEPS[first:last + 1], base=base, cwd=os.getcwd(),
                            case=args.case, data_dir=args.data_dir,
                            config=args.config)
    except ValueError as exc:                   # e.g. a case without a Stage 1 hook
        parser.error(str(exc))
    chain = pipeline.run({}, label=args.label or case.id)
    if chain.feed_run_ids:
        print(f"{'feed':8s}{' '.join(chain.feed_run_ids)}")
    for step in ("imp", "const", "opt", "uncprop", "psut"):
        print(f"{step:8s}{getattr(chain, step) or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
