"""
prior.py
========
The solve-for-prior adapter: a :class:`~pysuto.relation.relation.Relation` with exactly
one unknown edge yields that edge's value from the others (ADR 0011).

The relation is the same object Stage 3 emits into ``G`` (``constraints.emit``), so a
fill's coefficients are read off the emitted row: ``c_x = −G_x / G_e`` for every known
edge ``x`` and the unknown ``e``. The adapter returns value and provenance only; the σ of
a fill stays the imputer's own rule.

Public API
----------
Fill                                    A solved lone unknown: edge, value, sources, coefficients
solve_lone_unknown(rel, f, scope)       The lone unknown of ``rel`` against the prior ``f``
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Optional

import numpy as np

from .relation import SENSE_EQ, Relation


@dataclass(frozen=True)
class Fill:
    """The value of a relation's lone unknown edge.

    edge         The unknown edge id.
    value        ``(rhs − Σ_known g·f) / g_edge``.
    sources      The known edges of the relation, in the relation's edge order.
    coefficients ``−g_x / g_edge`` per source: ``value = Σ coefficients · f[sources]``
                 for a homogeneous relation (``rhs = 0``).
    edge_coefficient
                 ``g_edge``, the unknown's own coefficient (its sign says on which side
                 of a balance it sits).
    """

    edge:             int
    value:            float
    sources:          np.ndarray
    coefficients:     np.ndarray
    edge_coefficient: float


def solve_lone_unknown(rel: Relation, f: np.ndarray,
                       scope: Optional[np.ndarray] = None) -> Optional[Fill]:
    """Solve ``rel`` (an equality) for its one unknown edge, or ``None``.

    An edge is *unknown* when ``f`` is NaN there **and** it lies in ``scope`` (bool per
    edge; ``None`` = every edge). A NaN edge outside ``scope`` is not solved for and
    counts as 0, as a missing cross-region or stock-change leg does in a node balance
    (ADR 0010). ``None`` is returned unless exactly one edge is unknown.

    The known terms are summed in the relation's edge order, positive and negative
    coefficients apart (``P − N``, plain left-to-right addition on every Python: no
    pairwise summation, and not the built-in ``sum``, compensated since 3.12), so a
    ±1 balance reproduces ``Σ in − Σ out`` and a ratio twin ``f / r`` bit for bit.
    Raises ``ValueError`` on an inequality, which has no value to solve for.
    """
    if rel.sense != SENSE_EQ:
        raise ValueError(f"solve_lone_unknown: relation {rel.type}/{rel.variant} is "
                         f"{rel.sense!r}, not an equality")
    edges = rel.edges
    fe = f[edges]
    nan = np.isnan(fe)
    cand = nan if scope is None else nan & scope[edges]
    if np.count_nonzero(cand) != 1:
        return None
    k = int(np.argmax(cand))
    g = rel.coefficients
    known = ~nan
    g_k = g[known]
    terms = g_k * fe[known]
    pos = reduce(add, terms[g_k > 0].tolist(), 0.0)
    neg = reduce(add, (-terms[g_k < 0]).tolist(), 0.0)
    g_e = float(g[k])
    # "+ 0.0" folds a −0.0 (from 0 − 0 over a negative g_e) to +0.0
    value = (float(rel.rhs) - (pos - neg)) / g_e + 0.0
    return Fill(int(edges[k]), value, edges[known], -g_k / g_e, g_e)
