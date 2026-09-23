"""
psut.py
=======
Stage-6 figures — the bipartite **supply-use Sankey** of a
:class:`~pysuto.psut.PSUT` and the **process-network Sankey** of a
:class:`~pysuto.piot.PIOT` (ADR 0037 §1), one renderer for both.

For a PSUT, processes and commodities are the nodes and the ``S`` and ``U``
rows the links; for a PIOT the nodes are processes and the ``Z`` edges carry
the commodity in their hover text. In both,
the boundary blocks attach as links to their own SB nodes — inputs ``E``/``O``
entering on the left, outputs ``P``/``W``/``Y`` leaving on the right. A use or
final-use flow that crosses a border — ``region_crossed`` after
``PSUT.aggregate(..., keep_crossings=True)``, or distinct regions in a
multi-region table — is drawn as its own **traded** link. One figure = one
(period, valuation); nodes are ``(region, sector)`` pairs, labelled by the
sector alone when the PSUT holds one region (after a World collapse).

:func:`sankey_tables` is the pure-pandas half (nodes + links, testable
without a browser); :func:`sankey_figure` turns it into a Plotly figure.
Plotly is imported inside that function, so neither this module nor the thin
``PSUT.sankey`` that delegates here imports a plotting library at module level.
Colours are the shared :data:`pysuto.diag._plot._C` palette, one fixed colour
per node kind and link class: the supply-use chain in blue, traded links red,
boundary inputs green, final use purple, residuals a neutral grey. The class
name also travels in the hover text and the title's colour key, so identity is
never colour alone. Node positions come from the link graph's longest-path
depth (:func:`_positions`) so that a residual or an input node sits beside the
process it belongs to, instead of Plotly's default, which pushes every sink to
the last column and drags the residual bands across the whole figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ._plot import _C

if TYPE_CHECKING:                                  # no data-layer import at runtime
    from pysuto.psut import _TidyView


@dataclass(frozen=True)
class SankeySpec:
    """What one supply-use Sankey shows.

    period, valuation
        ``None`` means the PSUT's latest period / primary valuation.
    title
        Figure title; ``None`` builds one from the PSUT's provenance and adds
        the colour key of the link classes.
    min_val
        Links below this value (after summing) are dropped, together with any
        node they leave unconnected — declutters a fine-grained table.
    split_traded
        Draw a border-crossing use / final-use flow as its own link.
    unit
        Label for the values in the hover text.
    layout
        ``"depth"`` places nodes by longest-path depth (residual and input
        nodes beside their process); ``"auto"`` leaves the layout to Plotly.
    """
    period:       Optional[str] = None
    valuation:    Optional[str] = None
    title:        Optional[str] = None
    min_val:      float = 0.0
    split_traded: bool = True
    unit:         str = "kt"
    layout:       str = "depth"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# One fixed colour per node kind and per link class (never cycled).
NODE_COLOR = {
    "process":      _C["navy"],
    "flow":         _C["blue"],
    "input":        _C["green"],     # SB nodes feeding a process: nature (E), secondary (O)
    "final_demand": _C["purple"],    # SB nodes receiving final use (Y)
    "residual":     _C["grey"],      # SB nodes receiving residuals / discards (P, W)
}
# The S/U chain shares one colour — position already tells supply from use —
# so the four chromatic classes (blue, red, green, purple) stay apart for every
# reader; residual is the neutral grey.
LINK_COLOR = {
    "supply":      _rgba(_C["blue"],   0.35),
    "use":         _rgba(_C["blue"],   0.35),
    "traded":      _rgba(_C["red"],    0.55),
    "boundary in": _rgba(_C["green"],  0.40),
    "final use":   _rgba(_C["purple"], 0.40),
    "residual":    _rgba(_C["grey"],   0.45),
}
_LINK_CLASS = {"S": "supply", "U": "use", "Z": "use", "E": "boundary in", "O": "boundary in",
               "Y": "final use", "P": "residual", "W": "residual"}
_SB_KIND    = {"E": "input", "O": "input", "Y": "final_demand", "P": "residual", "W": "residual"}
_KIND_RANK  = {"input": 0, "process": 1, "flow": 2, "final_demand": 3, "residual": 4}


def _node_kind(entity: str, block: str) -> str:
    if entity == "Process":
        return "process"
    if entity == "Flow":
        return "flow"
    return _SB_KIND[block]


def sankey_tables(view: "_TidyView", spec: Optional[SankeySpec] = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The nodes and links behind :func:`sankey_figure`, as two DataFrames,
    for a PSUT or a PIOT.

    ``nodes``: ``node, region, sector, entity, kind, label, color`` — ``node``
    is the 0-based index the links refer to. ``links``: ``source, target,
    value, block, traded, cls, color, hover`` (plus ``commodity`` for a PIOT,
    one link per commodity). Links below ``spec.min_val`` are dropped, and so
    is every node left without a link.
    """
    spec = spec or SankeySpec()
    period, valuation = view.resolve(spec.period, spec.valuation)
    tbl = view.table
    tbl = tbl[(tbl["period"] == period) & (tbl["valuation"] == valuation)]

    crossed = tbl["from_region"].to_numpy() != tbl["to_region"].to_numpy()
    if "region_crossed" in tbl.columns:
        crossed = crossed | tbl["region_crossed"].to_numpy(dtype=bool)
    traded = crossed if spec.split_traded else np.zeros(len(tbl), dtype=bool)

    has_commodity = "commodity" in tbl.columns
    key = ["block", "from_region", "from_sector", "from_entity",
           "to_region", "to_sector", "to_entity", "traded"] \
          + (["commodity"] if has_commodity else [])
    links = (tbl.assign(traded=traded)
                .groupby(key, sort=False)["val"].sum()
                .reset_index()
                .rename(columns={"val": "value"}))
    links = links[(links["value"] > 0) & (links["value"] >= spec.min_val)].reset_index(drop=True)

    ends = pd.concat([
        links[["from_region", "from_sector", "from_entity", "block"]]
            .set_axis(["region", "sector", "entity", "block"], axis=1),
        links[["to_region", "to_sector", "to_entity", "block"]]
            .set_axis(["region", "sector", "entity", "block"], axis=1),
    ], ignore_index=True)
    ends["kind"] = [_node_kind(e, b) for e, b in zip(ends["entity"], ends["block"])]
    nodes = (ends.drop_duplicates(["region", "sector"])
                 .assign(rank=lambda d: d["kind"].map(_KIND_RANK))
                 .sort_values(["rank", "region", "sector"], ignore_index=True)
                 .drop(columns=["block", "rank"]))
    multi = view.nodes["region"].nunique() > 1
    nodes.insert(0, "node", np.arange(len(nodes)))
    nodes["label"] = [f"{r} · {s}" if multi else s
                      for r, s in zip(nodes["region"], nodes["sector"])]
    nodes["color"] = nodes["kind"].map(NODE_COLOR)

    index = {(r, s): int(i) for i, r, s in zip(nodes["node"], nodes["region"], nodes["sector"])}
    links["source"] = [index[(r, s)] for r, s in zip(links["from_region"], links["from_sector"])]
    links["target"] = [index[(r, s)] for r, s in zip(links["to_region"], links["to_sector"])]
    links["cls"]    = links["block"].map(_LINK_CLASS).where(~links["traded"], "traded")
    links["color"]  = links["cls"].map(LINK_COLOR)
    links["hover"]  = links["block"] + " · " + links["cls"]
    out_cols = ["source", "target", "value", "block", "traded", "cls", "color", "hover"]
    if has_commodity:
        links["hover"] = links["hover"] + " · " + links["commodity"].astype(str)
        out_cols.append("commodity")
    return (nodes[["node", "region", "sector", "entity", "kind", "label", "color"]],
            links[out_cols])


def _positions(nodes: pd.DataFrame, links: pd.DataFrame):
    """Explicit node positions for Plotly: ``x`` from the longest-path depth
    of the link graph — a source sits just before its first target, a sink just
    after its last source, so a residual node lands beside the process that
    emits it and an input beside the process it feeds — and ``y`` by stacking
    each column top-down in kind order (inputs, processes, flows, final
    demand, residuals), heights scaled to the tallest column. Plotly's ``snap``
    arrangement then removes any overlap. ``None`` when the graph has a cycle,
    which leaves the layout to Plotly."""
    n   = len(nodes)
    src = links["source"].to_numpy()
    tgt = links["target"].to_numpy()
    val = links["value"].to_numpy(dtype=float)
    indeg  = np.bincount(tgt, minlength=n)
    outdeg = np.bincount(src, minlength=n)

    depth, remaining, seen = np.zeros(n, dtype=int), indeg.copy(), 0
    queue = [i for i in range(n) if indeg[i] == 0]
    while queue:
        i = queue.pop(0)
        seen += 1
        for j in tgt[src == i]:
            depth[j] = max(depth[j], depth[i] + 1)
            remaining[j] -= 1
            if remaining[j] == 0:
                queue.append(j)
    if seen < n:
        return None
    for i in range(n):
        if indeg[i] == 0 and outdeg[i]:
            depth[i] = depth[tgt[src == i]].min() - 1
        elif outdeg[i] == 0 and indeg[i]:
            depth[i] = depth[src[tgt == i]].max() + 1
    depth -= depth.min()
    x = 0.01 + 0.98 * depth / max(int(depth.max()), 1)

    size  = np.maximum(np.bincount(src, weights=val, minlength=n),
                       np.bincount(tgt, weights=val, minlength=n))
    rank  = nodes["kind"].map(_KIND_RANK).to_numpy()
    columns = [np.flatnonzero(depth == d) for d in np.unique(depth)]
    scale   = max(size[col].sum() for col in columns)          # the tallest column fills the canvas
    gap     = 0.02
    avail   = 0.96 - gap * (max(len(col) for col in columns) - 1)
    y       = np.zeros(n)
    for col in columns:
        col = col[np.lexsort((-size[col], rank[col]))]
        h   = avail * size[col] / scale
        cum = max(0.0, (1.0 - (h.sum() + gap * (len(col) - 1))) / 2)
        for i, hi in zip(col, h):
            y[i] = cum + hi / 2
            cum += hi + gap
    return x, np.clip(y, 0.02, 0.98)


def _default_title(view: "_TidyView", period: str, valuation: str, unit: str,
                   classes=None) -> str:
    where = ", ".join(str(step["region"]) for step in getattr(view, "aggregation", [])
                      if isinstance(step.get("region"), str))
    head = " · ".join(x for x in (getattr(view, "acc_id", "") or type(view).__name__,
                                  where, period, f"{valuation} [{unit}]") if x)
    run  = getattr(view, "opt_run_id", None) or "in-memory"
    key  = " · ".join(f'<span style="color:{LINK_COLOR[c].replace("rgba", "rgb").rsplit(",", 1)[0]})">{c}</span>'
                      for c in LINK_COLOR if classes is None or c in classes)
    return f"{head} — {getattr(view, '_figure_name', 'Sankey')}<br><sup>{run} · links: {key}</sup>"


def sankey_figure(view: "_TidyView", spec: Optional[SankeySpec] = None):
    """Plotly figure of the Sankey described by ``spec`` — the supply-use
    Sankey of a PSUT or the process-network Sankey of a PIOT.

    Boundary blocks are links to their SB nodes; border-crossing use /
    final-use flows are separate traded links. Hover text carries the value,
    the link class and, for a PIOT, the commodity.
    """
    import plotly.graph_objects as go            # lazy: no plotting import at module level

    spec = spec or SankeySpec()
    if spec.layout not in ("depth", "auto"):
        raise ValueError(f"layout must be 'depth' or 'auto', got {spec.layout!r}")
    nodes, links = sankey_tables(view, spec)
    period, valuation = view.resolve(spec.period, spec.valuation)
    title = spec.title or _default_title(view, period, valuation, spec.unit,
                                         classes=set(links["cls"]))

    node = dict(
        label=nodes["label"].tolist(),
        color=nodes["color"].tolist(),
        pad=12, thickness=14, line=dict(width=0),
        hovertemplate="%{label}<br>%{value:,.1f} " + spec.unit + "<extra></extra>",
    )
    pos = _positions(nodes, links) if spec.layout == "depth" else None
    if pos is not None:
        node["x"], node["y"] = pos[0].tolist(), pos[1].tolist()

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=node,
        link=dict(
            source=links["source"].tolist(),
            target=links["target"].tolist(),
            value=links["value"].tolist(),
            color=links["color"].tolist(),
            customdata=links["hover"].tolist(),
            hovertemplate=("%{source.label} → %{target.label}<br>%{value:,.1f} "
                           + spec.unit + " · %{customdata}<extra></extra>"),
        ),
    ))
    fig.update_layout(title=dict(text=title), font=dict(size=11),
                      margin=dict(l=16, r=16, t=64, b=16))
    return fig
