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
came from. Levels ``'A'`` and ``'L'`` (issues #21/#22) add the industry-by-industry
calculus on that network — ``x = Z·1 + y``, ``A = Z·x̂⁻¹``, ``L = (I−A)⁻¹``
— and on top of it the consumption-based **footprints** and **embodied
bilateral trade** of a boundary extension (``E`` extraction, ``O`` secondary
input), the **ew-MFA headline indicators** (DE, imports, exports, PTB, DMI,
DMC, RMC) per region, **output multipliers** and a two-factor **SDA** across
periods. The product-by-product construct (``'suh_product'``, #20) raises
``NotImplementedError``; with one producer per commodity — copper — the two
constructs coincide.

The tidy ``table`` has one row per directed edge of the process network:
``period, valuation, block, commodity_region, commodity, from_region,
from_sector, from_entity, to_region, to_sector, to_entity, [region_crossed],
val`` with ``block ∈ {Z, Y, E, O, P, W}`` (``Z`` = inter-process). ``Z()``
pivots it to the process-by-process transaction table, ``D()`` returns the
market shares, and ``sankey()`` draws the process-network Sankey through the
same renderer as the PSUT's supply-use Sankey. The ``level`` a PIOT was built
with gates the calculus it serves; ``A`` and ``L`` are computed on demand and
cached per (period, valuation).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .psut import PSUT, _TidyView, boundary_labels

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
        How the table was built and how far (``'Z'``, ``'A'`` or ``'L'``).
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
        self._cache: dict = {}

    def _require(self, level: str, what: str) -> None:
        if LEVELS.index(self.level) < LEVELS.index(level):
            raise ValueError(
                f"{what} needs level {level!r}; this PIOT was built with level {self.level!r} "
                f"— use to_piot(level={level!r})"
            )

    def _slice(self, period: str, valuation: str) -> pd.DataFrame:
        t = self.table
        return t[(t["period"] == period) & (t["valuation"] == valuation)]

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

    # ------------------------------------------------------------------
    # Levels A and L — the industry-by-industry calculus (issues #21/#22)
    # ------------------------------------------------------------------

    def x(self, period=None, valuation=None) -> pd.Series:
        """Total output per process: ``x = Z·1 + y`` — what leaves each process
        into a use or a final use. Equals the PSUT's ``S`` row sum when the
        PSUT is flow-balanced."""
        period, valuation = self.resolve(period, valuation)
        procs = self._labels("Process", "process")
        t = self._slice(period, valuation)
        out = (t[t["block"].isin(["Z", "Y"])]
                 .groupby(["from_region", "from_sector"])["val"].sum())
        out.index = out.index.set_names(procs.names)
        return out.reindex(procs, fill_value=0.0).astype(float).rename("x")

    def y(self, period=None, valuation=None, by: Optional[str] = None):
        """Final demand attributed to each producing process: a Series over
        processes (``by=None``) or the process × final-demand-node matrix
        (``by='fd'``)."""
        period, valuation = self.resolve(period, valuation)
        procs = self._labels("Process", "process")
        fds   = boundary_labels(self.nodes, "Y")
        t = self._slice(period, valuation)
        t = t[t["block"] == "Y"]
        if t.empty:
            m = pd.DataFrame(0.0, index=procs, columns=fds)
        else:
            m = t.pivot_table(index=["from_region", "from_sector"],
                              columns=["to_region", "to_sector"],
                              values="val", aggfunc="sum", fill_value=0.0)
            m = m.reindex(index=procs, columns=fds, fill_value=0.0)
            m.index.names, m.columns.names = procs.names, fds.names
        m = m.astype(float)
        if by is None:
            return m.sum(axis=1).rename("y")
        if by == "fd":
            return m
        raise ValueError(f"by must be None or 'fd', got {by!r}")

    def A(self, period=None, valuation=None) -> pd.DataFrame:
        """Technical coefficients ``A = Z·x̂⁻¹``: input from process ``p`` per
        unit of process ``q``'s output (a process with no output has a zero
        column)."""
        self._require("A", "A()")
        period, valuation = self.resolve(period, valuation)
        key = ("A", period, valuation)
        if key not in self._cache:
            Z = self.Z(period, valuation)
            x = self.x(period, valuation).to_numpy()
            inv = np.where(x > 0, 1.0 / np.where(x > 0, x, 1.0), 0.0)
            self._cache[key] = pd.DataFrame(Z.to_numpy() * inv[None, :],
                                            index=Z.index, columns=Z.columns)
        return self._cache[key]

    def L(self, period=None, valuation=None) -> pd.DataFrame:
        """Total requirements ``L = (I − A)⁻¹``: output of ``p`` required,
        directly and indirectly, per unit of final demand for ``q``'s output.
        ``L·y = x`` exactly, except for output delivered to a process with no
        useful output of its own (its ``A`` column is zero — everything it
        takes in leaves as residual), which no final demand embodies."""
        self._require("L", "L()")
        period, valuation = self.resolve(period, valuation)
        key = ("L", period, valuation)
        if key not in self._cache:
            A = self.A(period, valuation)
            eye = np.eye(len(A))
            try:
                inv = np.linalg.inv(eye - A.to_numpy())
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"I − A is singular in {period}/{valuation}: the process network "
                    "has a closed cycle with no leakage to final demand"
                ) from exc
            self._cache[key] = pd.DataFrame(inv, index=A.index, columns=A.columns)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Footprints and embodied trade (issue #21)
    # ------------------------------------------------------------------

    def intensity(self, extension: str = "E", period=None, valuation=None) -> pd.Series:
        """Direct intensity of a boundary input per unit of process output:
        the ``extension`` block's inflow into ``p`` divided by ``x_p``
        (``'E'`` nature extraction, ``'O'`` secondary input)."""
        if extension not in ("E", "O"):
            raise ValueError(f"extension must be 'E' or 'O', got {extension!r}")
        period, valuation = self.resolve(period, valuation)
        x = self.x(period, valuation)
        t = self._slice(period, valuation)
        inflow = (t[t["block"] == extension]
                    .groupby(["to_region", "to_sector"])["val"].sum())
        inflow.index = inflow.index.set_names(x.index.names)
        inflow = inflow.reindex(x.index, fill_value=0.0).astype(float)
        xs = x.to_numpy()
        e = np.where(xs > 0, inflow.to_numpy() / np.where(xs > 0, xs, 1.0), 0.0)
        return pd.Series(e, index=x.index, name=f"intensity_{extension}")

    def footprint(self, extension: str = "E", period=None, valuation=None,
                  by: str = "fd"):
        """Consumption-based footprint ``e·L·ŷ``: the ``extension`` embodied
        in what each final-demand node takes (``by='fd'``), summed per
        consuming region (``by='region'``) or in total (``by=None``). Sums to
        the extension's total inflow, because ``L·y = x`` — less whatever was
        taken in by a process with no useful output (see :meth:`L`)."""
        self._require("L", "footprint()")
        period, valuation = self.resolve(period, valuation)
        e = self.intensity(extension, period, valuation).to_numpy()
        L = self.L(period, valuation).to_numpy()
        Y = self.y(period, valuation, by="fd")
        emb = pd.Series((e @ L) @ Y.to_numpy(), index=Y.columns,
                        name=f"footprint_{extension}")
        if by == "fd":
            return emb
        if by == "region":
            return emb.groupby(level="region").sum()
        if by is None:
            return float(emb.sum())
        raise ValueError(f"by must be 'fd', 'region' or None, got {by!r}")

    def embodied_trade(self, extension: str = "E", period=None, valuation=None
                       ) -> pd.DataFrame:
        """Embodied bilateral trade: rows the region where the ``extension``
        enters the system, columns the region whose final demand embodies it
        — ``Σ_{p∈r} e_p L[p,q] Y[q, d∈s]``. Row sums are each region's
        extraction, column sums its RMC."""
        self._require("L", "embodied_trade()")
        period, valuation = self.resolve(period, valuation)
        e = self.intensity(extension, period, valuation)
        L = self.L(period, valuation)
        Y = self.y(period, valuation, by="fd")
        M = pd.DataFrame(np.diag(e.to_numpy()) @ L.to_numpy() @ Y.to_numpy(),
                         index=L.index, columns=Y.columns)
        return M.groupby(level="region").sum().T.groupby(level="region").sum().T

    # ------------------------------------------------------------------
    # ew-MFA indicators, multipliers, SDA (issue #22)
    # ------------------------------------------------------------------

    def ew_mfa(self, period=None, valuation=None, extension: str = "E") -> pd.DataFrame:
        """Headline indicators per region, in the table's unit: ``DE`` (the
        ``extension`` entering the region's processes), ``imports`` and
        ``exports`` (network edges crossing into / out of the region —
        ``region_crossed`` rows of a collapsed table count as both), ``PTB =
        imports − exports``, ``DMI = DE + imports``, ``DMC = DMI − exports``,
        ``RMC`` (the footprint of the region's final demand)."""
        self._require("L", "ew_mfa()")
        period, valuation = self.resolve(period, valuation)
        t = self._slice(period, valuation)
        regions = sorted(self.nodes["region"].unique().tolist())
        de = (t[t["block"] == extension].groupby("to_region")["val"].sum()
                .reindex(regions, fill_value=0.0))
        flows = t[t["block"].isin(["Z", "Y"])]
        crossed = flows["from_region"].to_numpy() != flows["to_region"].to_numpy()
        if "region_crossed" in flows.columns:
            crossed = crossed | flows["region_crossed"].to_numpy(dtype=bool)
        traded  = flows[crossed]
        exports = traded.groupby("from_region")["val"].sum().reindex(regions, fill_value=0.0)
        imports = traded.groupby("to_region")["val"].sum().reindex(regions, fill_value=0.0)
        rmc = (self.footprint(extension, period, valuation, by="region")
                   .reindex(regions, fill_value=0.0))
        out = pd.DataFrame({"DE": de, "imports": imports, "exports": exports})
        out["PTB"] = out["imports"] - out["exports"]
        out["DMI"] = out["DE"] + out["imports"]
        out["DMC"] = out["DMI"] - out["exports"]
        out["RMC"] = rmc
        out.index.name = "region"
        return out.astype(float)

    def multipliers(self, period=None, valuation=None) -> pd.Series:
        """Output multipliers: the column sums of ``L`` — total system output
        per unit of final demand for each process's output."""
        return self.L(period, valuation).sum(axis=0).rename("output_multiplier")

    def sda(self, period_a, period_b, valuation=None) -> pd.DataFrame:
        """Two-factor structural decomposition of ``Δx = x_b − x_a`` into a
        technology term (``ΔL·y``) and a final-demand term (``L·Δy``), polar
        average of the two orderings so the terms sum to ``Δx`` exactly."""
        self._require("L", "sda()")
        period_a, valuation = self.resolve(period_a, valuation)
        period_b, _         = self.resolve(period_b, valuation)
        La, Lb = self.L(period_a, valuation).to_numpy(), self.L(period_b, valuation).to_numpy()
        ya, yb = self.y(period_a, valuation).to_numpy(), self.y(period_b, valuation).to_numpy()
        tech = 0.5 * ((Lb - La) @ ya + (Lb - La) @ yb)
        dem  = 0.5 * (La @ (yb - ya) + Lb @ (yb - ya))
        out = pd.DataFrame({"x_a": La @ ya, "x_b": Lb @ yb, "technology": tech,
                            "final_demand": dem}, index=self.L(period_a, valuation).index)
        out["dx"] = out["x_b"] - out["x_a"]
        return out

    def __repr__(self) -> str:
        p = self.periods()
        span = f"{p[0]}..{p[-1]} ({len(p)})" if p else "none"
        counts = (self.table[self.table["valuation"] == self.primary]["block"]
                    .value_counts().sort_index().to_dict())
        return (f"PIOT(construct={self.construct!r}, level={self.level!r}, "
                f"acc={self.acc_id!r}, valuations={self.valuations}, primary={self.primary!r}, "
                f"periods={span}, rows={len(self.table)}, blocks[{self.primary}]={counts})")
