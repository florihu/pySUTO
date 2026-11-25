"""
structure.py
============
The structural sources of relations: process recipes and node balances, derived once.

A **process recipe** is one Process node's content-valuation supply-use skeleton (edge
ids, no parameters); a **node balance** is the set of content edges entering and leaving
one node in one period. Both are read from the accounting structure alone (``edge_meta``,
the ``supply``/``use`` masks, the valuation and boundary roles), so they are derived here
once, cached per acc, and read by every consumer (the NetworkImputer, later the
constraint builders). ``pysuto.acc`` imports nothing from here or from ``impute``.

A recipe references its Process node's balance: its ``inputs`` are the node's incoming
use edges (any region), its ``products`` and ``wastes`` a subset of the outgoing supply
edges. Several Flow ``products`` are listed in full; whether a consumer can handle them
is the consumer's call (the NetworkImputer raises).

Public API
----------
ProcessRecipe                       One Process node's skeleton
NodeBalance                         Incoming / outgoing content edges of one node-period
process_recipes(acc)                Recipes of every (process, region, time), cached
node_balances(acc, valuation)       Balances of every Flow and Process node-period, cached
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

from pysuto.acc import BoundaryRole, boundary_role_values

_SECONDARY = BoundaryRole.SECONDARY.value  # the residual-output / secondary-input role

# id(acc) -> (weakref to acc, {key: derived}); the weakref guards against id() reuse
_cache: dict = {}


def _per_acc(acc) -> dict:
    hit = _cache.get(id(acc))
    if hit is None or hit[0]() is not acc:
        hit = (weakref.ref(acc, lambda _r, k=id(acc): _cache.pop(k, None)), {})
        _cache[id(acc)] = hit
    return hit[1]


@dataclass(frozen=True)
class ProcessRecipe:
    """One Process node's content supply-use skeleton (edge ids, no parameters).

    inputs    use-side content edge ids into the process (any region; incl. Scrap).
    products  supply content edge ids to Flow nodes (usually one; empty if none).
    wastes    supply content edge ids to SB nodes of boundary role ``secondary``.
    """

    process: str
    region: str
    time: str
    inputs: tuple[int, ...]
    products: tuple[int, ...]
    wastes: tuple[int, ...]


@dataclass(frozen=True)
class NodeBalance:
    """Content edges into (``incoming``) and out of (``outgoing``) one node-period."""

    node_id: int
    entity: str
    time: str
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]


def process_recipes(acc) -> tuple[ProcessRecipe, ...]:
    """Process recipes of ``acc``, one per ``(process, region, time)`` with a supply edge.

    Reads ``edge_meta``, the ``supply``/``use`` masks, the content valuation and the
    boundary roles — all of which survive the save/load round-trip. Cached per acc.
    """
    cache = _per_acc(acc)
    if "recipes" not in cache:
        cache["recipes"] = tuple(_derive_recipes(acc))
    return cache["recipes"]


def _derive_recipes(acc) -> list[ProcessRecipe]:
    em = acc.edge_meta
    supply = np.asarray(acc.masks["supply"], dtype=bool)
    use = np.asarray(acc.masks["use"], dtype=bool)
    is_cc = (em["valuation"].to_numpy() == acc.valuation_roles.content)

    sup = em[supply & is_cc & (em["from_entity"].to_numpy() == "Process")]
    use_e = em[use & is_cc & (em["to_entity"].to_numpy() == "Process")]

    recipes: list[ProcessRecipe] = []
    for (proc, region, time), g in sup.groupby(
        ["from_sector", "from_region", "time"], sort=False
    ):
        flow_out = g[g["to_entity"].to_numpy() == "Flow"]
        products = tuple(int(e) for e in flow_out["edge_id"])
        waste_rows = g[
            (g["to_entity"].to_numpy() == "SB")
            & (boundary_role_values(acc.boundary_roles, g["to_sector"]) == _SECONDARY)
        ]
        wastes = tuple(int(e) for e in waste_rows["edge_id"])

        ig = use_e[
            (use_e["to_sector"].to_numpy() == proc)
            & (use_e["to_region"].to_numpy() == region)
            & (use_e["time"].to_numpy() == time)
        ]
        inputs = tuple(int(e) for e in ig["edge_id"])
        recipes.append(
            ProcessRecipe(str(proc), str(region), str(time), inputs, products, wastes)
        )
    return recipes


def node_balances(acc, valuation: str) -> tuple[NodeBalance, ...]:
    """Per **(node, period)** of every Flow and Process node: incoming, outgoing edges.

    Only edges of ``valuation`` count, and only node-periods with both an incoming and an
    outgoing edge appear. Keyed by ``(node_id, time)``: ``node_index`` has no time axis,
    so grouping on the bare node id would pool all years onto one node. Sorted by
    ``(node_id, time)`` — a set iterates in string-hash order, which changes with every
    process. Cached per acc and valuation.
    """
    cache = _per_acc(acc)
    key = ("balances", valuation)
    if key not in cache:
        cache[key] = tuple(_derive_balances(acc, valuation))
    return cache[key]


def _derive_balances(acc, valuation: str) -> list[NodeBalance]:
    em = acc.edge_meta
    cc = em[em["valuation"].to_numpy() == valuation]
    entity = acc.node_index["entity"]
    by_to = {k: g["edge_id"].to_numpy() for k, g in cc.groupby(["to_id", "time"])}
    by_from = {k: g["edge_id"].to_numpy() for k, g in cc.groupby(["from_id", "time"])}
    balances: list[NodeBalance] = []
    for key in sorted(by_to.keys() & by_from.keys()):
        kind = entity.get(key[0])
        if kind not in ("Flow", "Process"):
            continue
        balances.append(NodeBalance(
            int(key[0]), str(kind), str(key[1]),
            tuple(int(e) for e in by_to[key]), tuple(int(e) for e in by_from[key]),
        ))
    return balances
