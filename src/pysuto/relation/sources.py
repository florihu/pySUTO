"""
sources.py
==========
The params relation source and its fill rules (ADR 0011): a params file read
as :class:`~pysuto.relation.relation.Relation` objects, and the rule that says which
direction of each relation a Stage 2 imputer may fill and where.

A ratio pair ``(a, b)`` with nominal ``r`` is the relation ``b − r · a = 0`` — the one
Stage 3 emits for a conc, ore-grade or per-pair efficiency row. ``forward`` fills ``b``
from ``a`` (``b = r·a``), ``backward`` fills ``a`` from ``b`` (``a = b/r``).

Public API
----------
PAIR_ACCEPTS, TWIN_ACCEPTS              The ``param_type`` sets of the two readings
ParamSource(relations)                  An ordered list of pair relations of one params file
ParamSource.from_params(path, acc, twin=)
                                        Resolve a params file against an accounting structure
Fill(source, rule, scope, id)           A cascade entry: a source, a direction and a scope
trade_scope(acc)                        Cross-region Flow→Process content edges (ADR 0010)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .param_relation import CONC_ACCEPTS, ParamRelationResolver, get_edge_cache
from .params import read_params
from .ratio_relation import ratio_relation

# The param_types of the per-pair reading (the ``agg`` flag partitions the rest of the shared
# params file: a pooled row is a set-scoped recipe relation owned by the NetworkImputer) and
# of the twin reading (a conc-family file only).
PAIR_ACCEPTS = frozenset({"concentration", "efficiency"})
TWIN_ACCEPTS = CONC_ACCEPTS

RULES = ("forward", "backward", "both")


class ParamSource:
    """The pair relations of one params file, in resolution order.

    Each relation is ``[b, a]`` at ``[1, −r]`` (:func:`~pysuto.relation.ratio_relation
    .ratio_relation`); ``source`` is the param id and ``meta`` carries the row's absolute
    ``sigma_param`` (ratio units, exactly as the row gives it), ``epsilon`` and
    ``description`` — what an imputer's σ rule reads. Nothing here is emitted into ``G``.
    """

    def __init__(self, relations: list, path: str = "") -> None:
        self.relations = relations
        self.path = str(path)

    def __len__(self) -> int:
        return len(self.relations)

    @classmethod
    def from_params(cls, path, acc, *, twin: bool = False) -> "ParamSource":
        """Resolve the params file at ``path`` against ``acc``.

        Per-pair reading (default): concentration and efficiency rows, ``agg=true`` rows
        skipped, the most specific row kept per ``(a, b)`` pair. Twin reading
        (``twin=True``): concentration rows only, the most specific row kept per CC edge
        ``b`` (its one Mass twin), non-positive nominals dropped — a traded product's CC
        is never seeded from a zero ratio. ``ValueError`` on a row with no nominal or σ.
        """
        resolver = ParamRelationResolver(acc)
        best: dict = {}  # key -> (pair, specificity)
        for row in read_params(path, TWIN_ACCEPTS if twin else PAIR_ACCEPTS):
            if row.agg and not twin:
                continue
            nominal = row.require_nominal()
            sigma_p = row.sigma()
            for rr in resolver.resolve_row(row):
                key = rr.b_edge_id if twin else (rr.a_edge_id, rr.b_edge_id)
                if key not in best or rr.specificity > best[key][1]:
                    pair = (rr.a_edge_id, rr.b_edge_id, nominal, sigma_p, row.epsilon,
                            row.param_id, row.description)
                    best[key] = (pair, rr.specificity)
        pairs = [v[0] for v in best.values()]
        if twin:
            pairs = [p for p in pairs if p[2] > 0.0]
        ei_exp = get_edge_cache(acc)[0]
        pos = pd.Index(ei_exp["edge_id"].to_numpy()).get_indexer([p[1] for p in pairs])
        times = ei_exp["time"].astype(str).to_numpy()[pos]
        vals = ei_exp["valuation"].astype(str).to_numpy()[pos]
        relations = []
        for (a, b, r, sig, eps, pid, desc), t, v in zip(pairs, times, vals):
            relations.append(ratio_relation(
                [b], [a], r, variant="conc" if twin else "pair", specs=(),
                time=t, valuation=v, sigma=sig / abs(r) if r else 0.0, source=pid,
                meta={"sigma_param": sig, "epsilon": eps, "description": desc},
            ))
        return cls(relations, path)


@dataclass(frozen=True)
class Fill:
    """A cascade entry: fill ``source``'s relations in ``rule`` direction inside ``scope``.

    source   The :class:`ParamSource`.
    rule     ``"forward"`` (``b`` from ``a``), ``"backward"`` (``a`` from ``b``) or
             ``"both"`` (whichever side is missing).
    scope    Bool ``(n_edges,)``: the edges the fill may write; ``None`` = every endpoint of
             the source's relations.
    id       The imputer id stamped on the log entries.
    """

    source: ParamSource
    rule:   str
    scope:  Optional[np.ndarray] = None
    id:     str = ""

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"Fill rule {self.rule!r} must be one of {RULES}")

    def allows(self, edge: int) -> bool:
        """Whether the fill may write ``edge``."""
        return True if self.scope is None else bool(self.scope[edge])

    def writer_scope(self, n_edges: int) -> np.ndarray:
        """The bool scope the fill declares to the ledger."""
        if self.scope is not None:
            return self.scope
        scope = np.zeros(n_edges, dtype=bool)
        if self.source.relations:
            scope[np.concatenate([r.edges for r in self.source.relations])] = True
        return scope


def trade_scope(acc) -> np.ndarray:
    """Bool ``(n_edges,)``: every cross-region ``Flow→Process`` edge of the content
    valuation — the trade seed's territory, the complement of the NetworkImputer's
    same-region one (ADR 0010)."""
    em = acc.edge_meta
    eligible = (
        (em["from_region"].to_numpy() != em["to_region"].to_numpy())
        & (em["from_entity"].to_numpy() == "Flow")
        & (em["to_entity"].to_numpy() == "Process")
        & (em["valuation"].to_numpy() == acc.valuation_roles.content)
    )
    scope = np.zeros(acc.n_edges, dtype=bool)
    scope[em["edge_id"].to_numpy()[eligible]] = True
    return scope
