"""
knobs.py
========
The knob surface of a stage driver (#100): the command-line protocol through which a
pipeline step overrides a driver's tuning constants without importing the driver.

A **knob** is an UPPER_CASE constant of a driver's ``# USER CONFIGURATION`` block that
the driver lists in its ``KNOBS`` tuple (GLOSSARY.md "Knob"). A driver adds two flags::

    --set KNOB=VALUE      (repeatable) VALUE is a Python literal, e.g. 1.5, "flows",
                          ("trade", "network"), None
    --run-id-file PATH    the driver writes the run id it persisted to PATH

and, before ``main()`` runs, parses the pairs against its ``KNOBS`` and its module
globals and applies them. Values are checked, never coerced: the type must be the
current constant's, with three allowances — an ``int`` for a ``float`` knob, ``None``
for any knob (the drivers' "off"), and any value for a knob whose default is ``None``.
"""

from __future__ import annotations

import ast
import os
from typing import Iterable, Mapping


def parse_overrides(pairs: Iterable[str], knobs: Iterable[str],
                    namespace: Mapping[str, object]) -> dict:
    """Parse ``KNOB=VALUE`` pairs into ``{KNOB: value}``.

    Parameters
    ----------
    pairs : the ``--set`` arguments, in command-line order (a later pair wins).
    knobs : the driver's ``KNOBS`` — the only names that may be overridden.
    namespace : the driver's module globals, read for each knob's current value.

    Raises
    ------
    ValueError
        A pair without ``=``, or a VALUE that is not a Python literal (a bare word is
        accepted for a ``str`` knob).
    KeyError
        A name outside ``knobs``.
    TypeError
        A value whose type is not the current constant's.
    """
    knobs = tuple(knobs)
    out: dict = {}
    for pair in pairs:
        name, sep, text = pair.partition("=")
        name = name.strip()
        if not sep:
            raise ValueError(f"--set expects KNOB=VALUE, got {pair!r}")
        if name not in knobs:
            raise KeyError(f"{name!r} is not a knob of this driver; its knobs are "
                           f"{', '.join(knobs)}")
        default = namespace[name]
        try:
            value = ast.literal_eval(text.strip())
        except (ValueError, SyntaxError):
            if not isinstance(default, str):
                raise ValueError(f"--set {name}: {text!r} is not a Python "
                                 f"literal") from None
            value = text.strip()
        _check_type(name, value, default)
        out[name] = value
    return out


def _check_type(name: str, value: object, default: object) -> None:
    if value is None or default is None:
        return
    if isinstance(default, bool) or isinstance(value, bool):
        ok = type(value) is type(default)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float))
    else:
        ok = type(value) is type(default)
    if not ok:
        raise TypeError(f"--set {name}: expected {type(default).__name__} like the "
                        f"default {default!r}, got {type(value).__name__} {value!r}")


def apply_overrides(namespace: dict, overrides: Mapping[str, object]) -> None:
    """Set each overridden knob in the driver's module globals."""
    for name, value in overrides.items():
        namespace[name] = value


def add_knob_args(parser) -> None:
    """Add ``--set KNOB=VALUE`` (repeatable, into ``knob_overrides``) and
    ``--run-id-file PATH`` to a driver's ``argparse`` parser."""
    parser.add_argument("--set", action="append", default=[], dest="knob_overrides",
                        metavar="KNOB=VALUE",
                        help="Override a knob of this driver (repeatable); VALUE is a "
                             "Python literal. See the driver's KNOBS.")
    parser.add_argument("--run-id-file", default=None, metavar="PATH",
                        help="Write the persisted run id to PATH (pysuto.pipeline).")


def write_run_id(path: str | None, run_id: str) -> None:
    """Write ``run_id`` and a newline to ``path``; no-op when ``path`` is None."""
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{run_id}\n")


def overrides_from_cli(parser, args, knobs: Iterable[str], namespace: dict) -> dict:
    """Parse ``args.knob_overrides`` against ``knobs`` and apply them to ``namespace``.

    A bad pair ends the driver through ``parser.error`` (exit 2, a one-line message,
    no traceback) before ``main()`` has loaded anything. Returns the applied
    overrides for the run's sidecar (``stage_config["overrides"]``).
    """
    try:
        overrides = parse_overrides(args.knob_overrides, knobs, namespace)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(exc.args[0] if exc.args else str(exc))
    apply_overrides(namespace, overrides)
    return overrides
