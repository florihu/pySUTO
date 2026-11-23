"""
__init__.py
===========
relation: params-driven relations between edge sets (ADR 0011).

* :mod:`pysuto.relation.params`         — the one params reader: typed :class:`ParamRow`
                                          records, each consumer's declared ``param_type`` set.
* :mod:`pysuto.relation.param_relation` — the A→B edge resolver over an accounting structure.
* :mod:`pysuto.relation.relation`       — the :class:`Relation` record: edges, coefficients, sense, σ.
* :mod:`pysuto.relation.balance`        — node balances as relations (Stage 3 emits, Stage 2 solves).
* :mod:`pysuto.relation.ratio_relation` — pooled-eff, conc-twin and agg-bucket relations.
* :mod:`pysuto.relation.prior`          — ``solve_lone_unknown``: a relation's lone unknown against the prior.

Import direction: ``acc → relation → impute → constraints``.
"""

from __future__ import annotations
