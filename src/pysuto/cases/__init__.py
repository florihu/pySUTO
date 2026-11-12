"""
cases
=====
Case modules (GLOSSARY.md "Case", #26 §3): ``pysuto.cases.<id>`` declares one metal's
case, wired through the six stages. A Stage 2 and 3 case is a
:class:`pysuto.case.CaseSpec` named ``CASE``.

* :mod:`pysuto.cases.copper` — the **copper case** (#120): the Stage 2 and 3 model of
  the copper application.
* :mod:`pysuto.cases.toy` — the **toy case** (#59): a synthetic three-region orium case
  generated from a planted ground truth.

Import direction: ``cases`` sits at the top of the stack, beside the stage drivers. It
imports the library; the library never imports it.
"""

from __future__ import annotations

import importlib
import pkgutil


def case_ids() -> list:
    """The ids of the case modules of this package, sorted."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__)
                  if not m.name.startswith("_"))


def load_case(case_id: str):
    """The :class:`pysuto.case.CaseSpec` ``CASE`` of ``pysuto.cases.<case_id>``.

    Raises
    ------
    ValueError
        ``case_id`` is not a case module of this package, or the module declares no
        ``CASE``; the message lists the known ids.
    """
    ids = case_ids()
    if case_id not in ids:
        raise ValueError(f"unknown case {case_id!r}; the known cases are {', '.join(ids)}")
    module = importlib.import_module(f"{__name__}.{case_id}")
    if not hasattr(module, "CASE"):
        raise ValueError(f"pysuto.cases.{case_id} declares no CASE")
    return module.CASE
