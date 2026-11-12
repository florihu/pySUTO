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
