"""
piot.py
=======
Stage 6 — the physical input-output table (PIOT) as a construct family
(ADR 0037): a **derived view** of a :class:`~pysuto.psut.PSUT`, built by
``PSUT.to_piot(construct, level)`` and never persisted on its own.

Slice 5 (issue #25) ships the **market-share / industry-by-industry**
construct at level ``'Z'`` — the **process network**: processes are nodes,
commodity flows are edges. The market-share matrix ``D = S·q̂⁻¹`` gives each
producer ``p`` its share of commodity ``f``; every use row ``f → q`` of the
PSUT becomes edges ``p → q`` carrying ``f`` at ``D[p,f]·U[f,q]`` (a single
producer → one identity edge), every final-use row ``f → d`` becomes
``p → d`` at ``D[p,f]·Y[f,d]``, and the boundary rows ``E``/``O`` (into a
process) and ``P``/``W`` (out of a process) are carried as they are —
residual outputs are final outputs, never inter-process edges
(Altimiras-Martin 2014, fn. 3). ``region_crossed`` travels with the row it
came from. The product-by-product construct (``'suh_product'``), the
coefficient levels ``'A'``/``'L'`` and footprints belong to #20–22 and raise
``NotImplementedError`` here.

The tidy ``table`` has one row per directed edge of the process network:
``period, valuation, block, commodity_region, commodity, from_region,
from_sector, from_entity, to_region, to_sector, to_entity, [region_crossed],
val`` with ``block ∈ {Z, Y, E, O, P, W}`` (``Z`` = inter-process). ``Z()``
pivots it to the process-by-process transaction table, ``D()`` returns the
market shares, and ``sankey()`` draws the process-network Sankey through the
same renderer as the PSUT's supply-use Sankey.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .psut import PSUT, _TidyView

CONSTRUCTS = ("market_share", "suh_product")
LEVELS     = ("Z", "A", "L")

_TABLE_COLUMNS = [
    "period", "valuation", "block", "commodity_region", "commodity",
    "from_region", "from_sector", "from_entity",
    "to_region", "to_sector", "to_entity",
    "val",
]
_SHARE_COLUMNS = ["period", "valuation", "region", "process",
                  "commodity_region", "commodity", "share"]
_COMMODITY_KEY = ["period", "valuation", "commodity_region", "commodity"]


def market_shares(psut: PSUT) -> pd.DataFrame:
    """``D`` in tidy form: each producer's share of every commodity it
    supplies, per (period, valuation) — ``S[p,f] / Σ_p S[p,f]``. Commodities
    nobody supplies in a period have no row."""
    S = psut.table[psut.table["block"] == "S"]
    q = S.groupby(["period", "valuation", "to_region", "to_sector"])["val"].transform("sum")
    share = np.where(q > 0, S["val"].to_numpy() / np.where(q > 0, q, 1.0), np.nan)
    out = (S.assign(share=share)
             .loc[lambda d: d["share"].notna()]
             .rename(columns={"from_region": "region", "from_sector": "process",
                              "to_region": "commodity_region", "to_sector": "commodity"}))
    return out[_SHARE_COLUMNS].reset_index(drop=True)


def _allocate(rows: pd.DataFrame, shares: pd.DataFrame, block: str,
              crossed: bool) -> pd.DataFrame:
    """Re-attribute PSUT rows leaving a Flow node ``f`` to the producers of
    ``f``: one edge per producer, ``val · share``."""
    rows = rows.rename(columns={"from_region": "commodity_region", "from_sector": "commodity"})
    m = rows.merge(shares, on=_COMMODITY_KEY, how="left")
    unsupplied = m["share"].isna()
    lost = float(m.loc[unsupplied, "val"].sum())
    if lost > 1e-9 * max(float(rows["val"].sum()), 1.0):
        raise ValueError(
            f"{int(unsupplied.sum())} {block} row(s) carrying {lost:.4g} use a commodity "
            f"nobody supplies in that period — the PSUT is not flow-balanced"
        )
    m = m[~unsupplied]
    out = pd.DataFrame({
        "period":           m["period"].to_numpy(),
        "valuation":        m["valuation"].to_numpy(),
        "block":            block,
        "commodity_region": m["commodity_region"].to_numpy(),
        "commodity":        m["commodity"].to_numpy(),
        "from_region":      m["region"].to_numpy(),
        "from_sector":      m["process"].to_numpy(),
        "from_entity":      "Process",
        "to_region":        m["to_region"].to_numpy(),
        "to_sector":        m["to_sector"].to_numpy(),
        "to_entity":        m["to_entity"].to_numpy(),
        "val":              m["val"].to_numpy() * m["share"].to_numpy(),
    })
    if crossed:
        out["region_crossed"] = m["region_crossed"].to_numpy(dtype=bool)
    return out


def _carry(rows: pd.DataFrame, commodity_side: str, crossed: bool) -> pd.DataFrame:
    """Boundary rows kept as they are; the SB sector names the commodity."""
    out = rows.copy()
    out["commodity_region"] = out[f"{commodity_side}_region"].to_numpy()
    out["commodity"]        = out[f"{commodity_side}_sector"].to_numpy()
    cols = _TABLE_COLUMNS + (["region_crossed"] if crossed else [])
    return out[cols]


def market_share_network(psut: PSUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The process network of a PSUT under the market-share construct:
    ``(table, shares)`` — the tidy edge table and the tidy ``D``."""
    shares  = market_shares(psut)
    tbl     = psut.table
    crossed = "region_crossed" in tbl.columns
    by      = lambda b: tbl[tbl["block"] == b]
    parts = [
        _allocate(by("U"), shares, "Z", crossed),
        _allocate(by("Y"), shares, "Y", crossed),
        _carry(by("E"), "from", crossed),
        _carry(by("O"), "from", crossed),
        _carry(by("P"), "to", crossed),
        _carry(by("W"), "to", crossed),
    ]
    cols  = _TABLE_COLUMNS + (["region_crossed"] if crossed else [])
    order = [c for c in cols if c != "val"]
    table = (pd.concat([p[cols] for p in parts], ignore_index=True)
               .sort_values(order, ignore_index=True))
    return table, shares


class PIOT(_TidyView):
    """A physical input-output table derived from a PSUT by one construct.

    Attributes
    ----------
    table : DataFrame
        One row per directed edge of the process network (module docstring).
    shares : DataFrame
        The market shares ``D`` in tidy form (``market_shares``).
    nodes : DataFrame
        The node universe inherited from the PSUT; ``Z`` is labelled from its
        Process rows, ``D`` from its Process and Flow rows.
    construct, level
        How the table was built and how far (``'Z'`` only in Slice 5).
    valuations, primary, acc_id, opt_run_id, const_run_id, aggregation
        Carried over from the PSUT.
    """

    _figure_name = "process-network Sankey"

    def __init__(
        self,
        table: pd.DataFrame,
        nodes: pd.DataFrame,
        shares: pd.DataFrame,
        *,
        construct:    str,
        level:        str,
        valuations:   Sequence[str],
        primary:      str,
        acc_id:       str = "",
        opt_run_id:   Optional[str] = None,
        const_run_id: Optional[str] = None,
        aggregation:  Optional[list] = None,
    ) -> None:
        self.table        = table.reset_index(drop=True)
        self.nodes        = nodes.reset_index(drop=True)
        self.shares       = shares.reset_index(drop=True)
        self.construct    = construct
        self.level        = level
        self.valuations   = list(valuations)
        self.primary      = primary
        self.acc_id       = acc_id
        self.opt_run_id   = opt_run_id
        self.const_run_id = const_run_id
        self.aggregation  = list(aggregation or [])

    @classmethod
    def from_psut(cls, psut: PSUT, *, construct: str = "market_share",
                  level: str = "Z") -> "PIOT":
        if construct not in CONSTRUCTS:
            raise ValueError(f"construct must be one of {CONSTRUCTS}, got {construct!r}")
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if construct == "suh_product":
            raise NotImplementedError(
                "the product-by-product (Suh) construct is issue #20; Slice 5 builds 'market_share'"
            )
        if level != "Z":
            raise NotImplementedError(
                f"level {level!r} (technical coefficients / Leontief inverse) is issue #20; "
                "Slice 5 builds 'Z'"
            )
        table, shares = market_share_network(psut)
        return cls(
            table, psut.nodes, shares,
            construct    = construct,
            level        = level,
            valuations   = psut.valuations,
            primary      = psut.primary,
            acc_id       = psut.acc_id,
            opt_run_id   = psut.opt_run_id,
            const_run_id = psut.const_run_id,
            aggregation  = psut.aggregation,
        )

    # ------------------------------------------------------------------
    # Matrix views
    # ------------------------------------------------------------------

    def _labels(self, entity: str, name: str) -> pd.MultiIndex:
        n = self.nodes[self.nodes["entity"] == entity].sort_values(["region", "sector"])
        return pd.MultiIndex.from_arrays(
            [n["region"].to_numpy(), n["sector"].to_numpy()], names=["region", name]
        )

    def Z(self, period=None, valuation=None) -> pd.DataFrame:
        """Process-by-process transactions of one (period, valuation): rows
        the supplying process ``p``, columns the using process ``q``, summed
        over commodities and over the domestic/traded split."""
        period, valuation = self.resolve(period, valuation)
        procs = self._labels("Process", "process")
        tbl = self.table
        sub = tbl[(tbl["block"] == "Z") & (tbl["period"] == period)
                  & (tbl["valuation"] == valuation)]
        if sub.empty:
            return pd.DataFrame(0.0, index=procs, columns=procs)
        m = sub.pivot_table(index=["from_region", "from_sector"],
                            columns=["to_region", "to_sector"],
                            values="val", aggfunc="sum", fill_value=0.0)
        m = m.reindex(index=procs, columns=procs, fill_value=0.0)
        m.index.names = m.columns.names = procs.names
        return m.astype(float)

    def D(self, period=None, valuation=None) -> pd.DataFrame:
        """Market shares of one (period, valuation): rows ``(region, process)``,
        columns ``(region, flow)``; each column sums to 1 where the commodity
        is supplied at all."""
        period, valuation = self.resolve(period, valuation)
        procs = self._labels("Process", "process")
        flows = self._labels("Flow", "flow")
        sh = self.shares
        sub = sh[(sh["period"] == period) & (sh["valuation"] == valuation)]
        if sub.empty:
            return pd.DataFrame(0.0, index=procs, columns=flows)
        m = sub.pivot_table(index=["region", "process"],
                            columns=["commodity_region", "commodity"],
                            values="share", aggfunc="sum", fill_value=0.0)
        m = m.reindex(index=procs, columns=flows, fill_value=0.0)
        m.index.names   = procs.names
        m.columns.names = flows.names
        return m.astype(float)

    def A(self, period=None, valuation=None):
        raise NotImplementedError("technical coefficients A are issue #20 (level 'A')")

    def L(self, period=None, valuation=None):
        raise NotImplementedError("the Leontief inverse L is issue #20 (level 'L')")

    def footprint(self, *args, **kwargs):
        raise NotImplementedError("footprints are issue #21")

    def __repr__(self) -> str:
        p = self.periods()
        span = f"{p[0]}..{p[-1]} ({len(p)})" if p else "none"
        counts = (self.table[self.table["valuation"] == self.primary]["block"]
                    .value_counts().sort_index().to_dict())
        return (f"PIOT(construct={self.construct!r}, level={self.level!r}, "
                f"acc={self.acc_id!r}, valuations={self.valuations}, primary={self.primary!r}, "
                f"periods={span}, rows={len(self.table)}, blocks[{self.primary}]={counts})")
