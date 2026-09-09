"""
flow_node_balance.py
====================
Prior-consistency check at Flow nodes, shared by Stage 2 and Stage 3.

A Flow node carries a **hard** mass balance (``cone_type = "zero"``): its
inflow (``Process→Flow``) must equal its outflow (``Flow→Process`` +
``Flow→SB``). Two *scale totals* (ADR 0006) sit on either side of that
balance — ``supplytotout`` on the inflow, ``usetotout`` on the outflow —
each anchored tightly at ``σ_c = ANCHOR_CV·|c|``.

When the two prior sums disagree, the balance cannot honour both anchors and
the solver must violate one. Because ``σ_c`` scales with ``|c|``, the *smaller*
side is the cheaper anchor to satisfy, so the solver collapses the node to the
smaller prior — even when that prior is the less trustworthy of the two. A
**contradicted flow node** is where this happens: the node's own priors, not
the reconciliation, are inconsistent.

Worked example (Round D, Russia Matte 1995): inflow prior 545.0 kt CC (ICSG
concentrate use) vs outflow prior 50.8 (ICSG refining feed, under-reported).
σ_c is 9.3 vs 0.86, so honouring the inflow costs z ≈ 574 while collapsing to
the outflow costs z ≈ 53 — Russia's smelting chain collapses.

This module only *detects* the contradiction. Callers decide the policy; see
ADR 0019 (the outflow side loses anchor authority via σ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .param_relation import expand_edge_index

# Outflow below this fraction of inflow ⇒ the node's priors are contradicted.
K_CLOSE = 0.5
# Nodes below this inflow (kt) are noise, not a contradiction worth pricing.
MIN_INFLOW_KT = 1.0


def flow_node_balance(acc, agg_result, valuation: str = "CC") -> pd.DataFrame:
    """Per (sector, region, time) Flow-node inflow vs outflow prior sums.

    Returns a frame with columns ``sector, region, time, inflow, outflow,
    ratio`` (``ratio = outflow/inflow``), one row per node with inflow > 0.
    Uses ``f_agg`` — the Stage-2 prior — not the posterior.
    """
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    f = np.nan_to_num(np.asarray(agg_result.f_agg, dtype=float))
    ei = ei[ei["valuation"].eq(valuation)].copy()
    ei["f"] = f[ei["edge_id"].to_numpy()]

    inflow = (ei[ei["entity_from"].eq("Process") & ei["entity_to"].eq("Flow")]
              .groupby(["sector_to", "region_to", "time"])["f"].sum())
    outflow = (ei[ei["entity_from"].eq("Flow")]
               .groupby(["sector_from", "region_from", "time"])["f"].sum())
    inflow.index.names = outflow.index.names = ("sector", "region", "time")

    out = pd.concat([inflow.rename("inflow"), outflow.rename("outflow")],
                    axis=1).fillna(0.0).reset_index()
    out = out[out["inflow"] > 0].copy()
    out["ratio"] = out["outflow"] / out["inflow"]
    return out


def contradicted_flow_nodes(
    acc,
    agg_result,
    sectors: tuple[str, ...] = ("Matte",),
    k: float = K_CLOSE,
    min_inflow: float = MIN_INFLOW_KT,
    valuation: str = "CC",
) -> tuple[set, pd.DataFrame]:
    """Flow nodes whose outflow prior cannot honour their inflow prior.

    Returns ``(keys, report)`` where ``keys`` is a set of
    ``(sector, region, time)`` and ``report`` is the failing rows of
    :func:`flow_node_balance`. ``sectors`` restricts which flow sectors are
    judged — the test itself is sector-generic, but acting on a node is a
    policy decision made per sector (ADR 0019 acts on Matte only).
    """
    bal = flow_node_balance(acc, agg_result, valuation=valuation)
    bad = bal[bal["sector"].isin(sectors)
              & (bal["inflow"] > min_inflow)
              & (bal["outflow"] < k * bal["inflow"])].copy()
    keys = set(map(tuple, bad[["sector", "region", "time"]].to_numpy()))
    return keys, bad.sort_values("ratio")
