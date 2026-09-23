"""
psut.py
=======
Stage 6 — the physical supply-use table (PSUT), tidy-canonical.

A :class:`PSUT` is the solved edge vector ``a*`` of one or more periods carried
as a **tidy edge table** — one row per directed edge per (period, valuation),
every row classified into a PSUT **block** — together with the node label
universe of the accounting structure it came from. The block matrices are
pivot *views* of that table, reindexed onto the node universe so their labels
are identical across periods (issues #18/#19, ADR 0037).

Blocks (PIOLab / ADR 0007), read off the endpoint entity kinds and, for the
boundary blocks, the ``SB`` node's ``entity_sub``::

    S  Process   → Flow         supply (useful output)
    U  Flow      → Process      use
    Y  Flow      → SB·FD        final use
    E  SB·Nature → Process      nature input
    O  SB·SEM    → Process      secondary input
    P  Process   → SB·SEM       residual output
    W  Process   → SB·Nature    discard to nature (empty in copper)

Every admissible edge lands in exactly one block; an edge outside this table
is an error, never a silent omission. The boundary labels ``Nature`` / ``SEM``
/ ``FD`` are read in one place, :func:`classify_blocks` — the site to switch
to ``acc.roles`` when issue #26 §1 (ADR 0034) lands.

The table carries **every** valuation of the structure; ``primary`` (copper:
``CC``) is the valuation the views default to, and :meth:`PSUT.realized_conc`
is the ratio of two valuations from the same table.

Construction contract
---------------------
* :meth:`PSUT.from_opt_run` — from a persisted optimiser run; resolves the
  accounting structure through provenance (opt sidecar → const sidecar →
  ``acc`` id) and loads every timestep of the run unless told otherwise.
* :meth:`PSUT.from_results` — in-memory handoff from
  ``dict[timestep, OptimResultsLike]``; a year-local ``F_opt`` (block path)
  is mapped to global edge ids through ``cols_t``.

Time is a column of the table, never aggregated: :meth:`PSUT.periods`,
:meth:`PSUT.latest` and :meth:`PSUT.at` select it. Region/sector
**aggregation** is a pushforward on the table (:meth:`PSUT.aggregate`, issue
#24): one concordance map per dimension applied to both facets, then a sum
over every key column — ``period``, ``valuation``, ``block`` and the entity
kinds never merge; ``keep_crossings=True`` stamps ``region_crossed`` before
the relabel so the realized traded share survives a World collapse.
Persistence lives in :class:`pysuto.registry.PsutRunRegistry`; the supply-use
Sankey is :meth:`PSUT.sankey`, a thin delegate to :mod:`pysuto.diag.psut`.
"""
from __future__ import annotations

from typing import Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .acc import AccountingStructure
from .logging import get_logger
from .registry import ConstraintRunRegistry, OptimRunRegistry

logger = get_logger("pysuto.psut")


class OptimResultsLike(Protocol):
    """The three attributes a solver result must expose to build a PSUT.

    Satisfied by :class:`pysuto.solver.OptimResult`; the PSUT never reads its
    other fields. ``cols_t`` holds the global edge id of every ``F_opt`` entry
    (year-local block path) or is ``None`` when ``F_opt`` spans every edge of
    the accounting structure (legacy full-vector path).
    """

    timestep: str
    F_opt:    np.ndarray
    cols_t:   Optional[np.ndarray]


# block ← (from_entity, from_entity_sub, to_entity, to_entity_sub); None = any.
# The boundary labels are the copper case's entity_sub values (ADR 0037 §2).
_BLOCK_RULES = (
    ("S", "Process", None,     "Flow",    None),
    ("U", "Flow",    None,     "Process", None),
    ("Y", "Flow",    None,     "SB",      "FD"),
    ("E", "SB",      "Nature", "Process", None),
    ("O", "SB",      "SEM",    "Process", None),
    ("P", "Process", None,     "SB",      "SEM"),
    ("W", "Process", None,     "SB",      "Nature"),
)
BLOCKS = tuple(rule[0] for rule in _BLOCK_RULES)

# (entity, entity_sub, level name) of each block's row and column labels. The
# level names are the framework's role words (CONTEXT "Role"); the labels are
# the case's sectors.
_ROW_COL = {
    "S": (("Process", None, "process"),   ("Flow", None, "flow")),
    "U": (("Flow", None, "flow"),         ("Process", None, "process")),
    "Y": (("Flow", None, "flow"),         ("SB", "FD", "final_demand")),
    "E": (("SB", "Nature", "nature"),     ("Process", None, "process")),
    "O": (("SB", "SEM", "secondary"),     ("Process", None, "process")),
    "P": (("Process", None, "process"),   ("SB", "SEM", "secondary")),
    "W": (("Process", None, "process"),   ("SB", "Nature", "nature")),
}

_TABLE_COLUMNS = [
    "period", "valuation", "block", "edge_id",
    "from_region", "from_sector", "from_entity",
    "to_region", "to_sector", "to_entity",
    "val",
]

_NODE_COLUMNS = ["sector", "entity", "region", "entity_sub"]


def _entity_sub(edge_meta: pd.DataFrame, col: str) -> np.ndarray:
    if col in edge_meta.columns:
        return edge_meta[col].to_numpy(dtype=object)
    return np.full(len(edge_meta), None, dtype=object)


def classify_blocks(edge_meta: pd.DataFrame) -> pd.Series:
    """Block label per row of ``edge_meta``; ``None`` for an edge no rule matches."""
    fe = edge_meta["from_entity"].to_numpy()
    te = edge_meta["to_entity"].to_numpy()
    fs = _entity_sub(edge_meta, "from_entity_sub")
    ts = _entity_sub(edge_meta, "to_entity_sub")
    block = pd.Series([None] * len(edge_meta), index=edge_meta.index, dtype=object)
    for label, f_ent, f_sub, t_ent, t_sub in _BLOCK_RULES:
        m = (fe == f_ent) & (te == t_ent)
        if f_sub is not None:
            m &= (fs == f_sub)
        if t_sub is not None:
            m &= (ts == t_sub)
        block[m] = label
    return block


def _unclassified_report(edge_meta: pd.DataFrame, block: pd.Series) -> str:
    bad  = edge_meta[block.isna().to_numpy()]
    cols = [c for c in ("from_entity", "from_entity_sub", "to_entity", "to_entity_sub")
            if c in bad.columns]
    counts = bad.groupby(cols, dropna=False).size()
    return "; ".join(f"{tuple(k) if isinstance(k, tuple) else (k,)}: {n} edge(s)"
                     for k, n in counts.items())


def _period_values(res: OptimResultsLike, positions: np.ndarray,
                   n_edges: int) -> np.ndarray:
    """Solved values of the edges at global ``positions`` for one result."""
    F = np.asarray(res.F_opt, dtype=float)
    if res.cols_t is None:                       # legacy: full-length vector
        if F.shape[0] != n_edges:
            raise ValueError(
                f"timestep {res.timestep!r}: F_opt has {F.shape[0]} entries but "
                f"the accounting structure has {n_edges} edges and cols_t is None"
            )
        return F[positions]
    cols = np.asarray(res.cols_t)
    if cols.shape[0] != F.shape[0]:
        raise ValueError(
            f"timestep {res.timestep!r}: F_opt ({F.shape[0]}) and cols_t "
            f"({cols.shape[0]}) differ in length"
        )
    where = pd.Index(cols).get_indexer(positions)
    if (where < 0).any():
        raise ValueError(
            f"timestep {res.timestep!r}: {int((where < 0).sum())} of its "
            f"{len(positions)} edges are not covered by cols_t"
        )
    return F[where]


class PSUT:
    """Tidy physical supply-use table over one or more periods, every valuation.

    Attributes
    ----------
    table : DataFrame
        One row per (period, edge): ``period, valuation, block, edge_id,
        from_region, from_sector, from_entity, to_region, to_sector, to_entity,
        val``. ``edge_id`` is the global edge id of the accounting structure.
    nodes : DataFrame
        The node universe (``sector, entity, region, entity_sub``) the block
        views are labelled from.
    valuations, primary
        Every valuation in the table, and the one the views default to.
    acc_id, opt_run_id, const_run_id
        Provenance; the run ids are ``None`` for an in-memory handoff.
    aggregation : list
        The :meth:`aggregate` calls applied so far, oldest first; empty for
        the structure's own resolution. After an aggregation the table has no
        ``edge_id`` and may carry ``region_crossed``.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        nodes: pd.DataFrame,
        *,
        valuations:   Sequence[str],
        primary:      str,
        acc_id:       str = "",
        opt_run_id:   Optional[str] = None,
        const_run_id: Optional[str] = None,
        aggregation:  Optional[list] = None,
    ) -> None:
        if primary not in valuations:
            raise ValueError(f"primary {primary!r} is not one of {list(valuations)}")
        self.table        = table.reset_index(drop=True)
        self.nodes        = nodes.reset_index(drop=True)
        self.valuations   = list(valuations)
        self.primary      = primary
        self.acc_id       = acc_id
        self.opt_run_id   = opt_run_id
        self.const_run_id = const_run_id
        self.aggregation  = list(aggregation or [])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_results(
        cls,
        results: Mapping[str, OptimResultsLike],
        acc: AccountingStructure,
        *,
        primary: str = "CC",
    ) -> "PSUT":
        """Build from ``{timestep: result}`` (the solver's output shape).

        Each result's ``F_opt`` is placed on global edge ids — through
        ``cols_t`` on the block path, positionally on the legacy path — and
        every edge of that timestep, in every valuation, becomes one row
        carrying its block. An edge no block rule matches raises.
        """
        if primary not in acc.valuation:
            raise ValueError(
                f"primary {primary!r} is not one of {acc.valuation} "
                f"(accounting structure {acc.id!r})"
            )
        if not results:
            raise ValueError("results is empty — nothing to materialise")

        meta  = acc.edge_meta
        block = classify_blocks(meta)
        if block.isna().any():
            raise ValueError(
                "edges outside the PSUT block taxonomy — (from_entity, from_entity_sub, "
                f"to_entity, to_entity_sub): {_unclassified_report(meta, block)}"
            )
        time = meta["time"].to_numpy()
        keep = ["edge_id", "valuation",
                "from_region", "from_sector", "from_entity",
                "to_region", "to_sector", "to_entity"]

        frames = []
        for t, res in results.items():
            t = str(t)
            if str(res.timestep) != t:
                raise ValueError(
                    f"results key {t!r} holds a result for timestep {res.timestep!r}"
                )
            positions = np.flatnonzero(time == t)
            frame = meta.iloc[positions][keep].copy()
            frame.insert(0, "period", t)
            frame.insert(2, "block", block.iloc[positions].to_numpy())
            frame["val"] = _period_values(res, positions, acc.n_edges)
            frames.append(frame[_TABLE_COLUMNS])

        table = (pd.concat(frames, ignore_index=True)
                   .sort_values(["period", "valuation", "block", "edge_id"],
                                ignore_index=True))
        nodes = acc.node_index.copy()
        if "entity_sub" not in nodes.columns:
            nodes["entity_sub"] = None
        nodes = nodes[_NODE_COLUMNS]

        counts = (table[table["valuation"] == primary]["block"]
                    .value_counts().sort_index().to_dict())
        logger.info(
            "PSUT %s: %d timestep(s) × %s, %d rows; per period in %s: %s",
            acc.id, len(frames), list(acc.valuation), len(table), primary,
            ", ".join(f"{b}={n // len(frames)}" for b, n in counts.items()),
        )
        return cls(table, nodes, valuations=list(acc.valuation), primary=primary,
                   acc_id=acc.id)

    @classmethod
    def from_opt_run(
        cls,
        opt_run_id: str,
        timesteps: Optional[Sequence] = None,
        *,
        primary:   str = "CC",
        opt_dir:   str = "data/proc/opt",
        const_dir: str = "data/proc/const",
        acc_dir:   str = "data/proc/acc",
    ) -> "PSUT":
        """Build from a persisted optimiser run.

        The accounting structure is found through provenance: the opt sidecar
        names its ``const_run_id``, whose sidecar names the ``acc`` id. All
        timesteps of the run are loaded unless ``timesteps`` lists a subset.
        """
        opt_reg = OptimRunRegistry(opt_dir)
        sidecar = opt_reg.load_sidecar(opt_run_id)
        const_run_id = (sidecar.get("parents", {})
                               .get("upstream", {})
                               .get("const_run_id"))
        if not const_run_id:
            raise KeyError(
                f"OptimRun {opt_run_id!r} sidecar has no parents.upstream.const_run_id"
            )
        const_sidecar = ConstraintRunRegistry(const_dir).load_sidecar(const_run_id)
        acc_id = (const_sidecar.get("parents", {})
                               .get("inputs", {})
                               .get("acc", {})
                               .get("id"))
        if not acc_id:
            raise KeyError(
                f"ConstraintRun {const_run_id!r} sidecar has no parents.inputs.acc.id"
            )
        logger.info("Loading OptimRun %r → ConstraintRun %r → acc %r",
                    opt_run_id, const_run_id, acc_id)
        acc     = AccountingStructure.load(acc_id, storage_dir=acc_dir)
        results = opt_reg.load(opt_run_id)
        if timesteps is not None:
            wanted  = [str(t) for t in timesteps]
            missing = [t for t in wanted if t not in results]
            if missing:
                raise KeyError(
                    f"timesteps {missing} are not in OptimRun {opt_run_id!r}: "
                    f"{list(results)}"
                )
            results = {t: results[t] for t in wanted}

        psut = cls.from_results(results, acc, primary=primary)
        psut.opt_run_id   = opt_run_id
        psut.const_run_id = const_run_id
        return psut

    # ------------------------------------------------------------------
    # Time selection — period is carried, never aggregated
    # ------------------------------------------------------------------

    def periods(self) -> list:
        """Sorted period labels present in the table."""
        return sorted(self.table["period"].unique().tolist())

    def latest(self) -> str:
        return self.periods()[-1]

    def at(self, period) -> "PSUT":
        """The rows of one period, as a new PSUT."""
        period = str(period)
        if period not in self.periods():
            raise KeyError(f"period {period!r} not in PSUT: {self.periods()}")
        return self._with_table(self.table[self.table["period"] == period])

    def _with_table(self, table: pd.DataFrame) -> "PSUT":
        return PSUT(
            table, self.nodes,
            valuations   = self.valuations,
            primary      = self.primary,
            acc_id       = self.acc_id,
            opt_run_id   = self.opt_run_id,
            const_run_id = self.const_run_id,
            aggregation  = self.aggregation,
        )

    def resolve(self, period=None, valuation=None) -> tuple:
        """``(period, valuation)`` with the defaults applied — the latest period,
        the primary valuation — and both validated against the table."""
        period = self.latest() if period is None else str(period)
        if period not in self.periods():
            raise KeyError(f"period {period!r} not in PSUT: {self.periods()}")
        valuation = self.primary if valuation is None else valuation
        if valuation not in self.valuations:
            raise KeyError(f"valuation {valuation!r} not in PSUT: {self.valuations}")
        return period, valuation

    # ------------------------------------------------------------------
    # Aggregation — a pushforward on the table (issue #24)
    # ------------------------------------------------------------------

    def aggregate(self, region=None, sector=None, *,
                  keep_crossings: bool = False) -> "PSUT":
        """Region/sector aggregation as a pushforward on the tidy table.

        ``region`` and ``sector`` are concordance maps ``{old_label: new_label}``
        applied to **both** the ``from_`` and the ``to_`` facet; a label absent
        from a map keeps its name, a key naming no existing label raises.
        ``region`` may also be one string, which collapses every region to that
        label (``'World'``). A sector map may only merge sectors of one entity
        kind (and boundary category), so a sector still names one kind.

        Rows are then summed over every key column: ``period``, ``valuation``,
        ``block`` and the entity kinds stay in the key, so nothing merges across
        them and time is never aggregated. ``edge_id`` does not survive.

        ``keep_crossings=True`` stamps ``region_crossed`` — ``from_region !=
        to_region`` at the resolution *before* this relabel, OR-ed with any
        earlier stamp — into the key, so the realized traded share survives a
        region collapse (CONTEXT "Traded (realized) vs tradable"). Returns a
        new PSUT; the call is appended to :attr:`aggregation`.
        """
        if region is None and sector is None and not keep_crossings:
            raise ValueError(
                "nothing to aggregate: give region=, sector= or keep_crossings=True"
            )
        tbl   = self.table.copy()
        nodes = self.nodes.copy()

        if keep_crossings:
            crossed = tbl["from_region"].to_numpy() != tbl["to_region"].to_numpy()
            if "region_crossed" in tbl.columns:
                crossed |= tbl["region_crossed"].to_numpy(dtype=bool)
            tbl["region_crossed"] = crossed

        region_map = self._coerce_map(region, "region", nodes["region"])
        sector_map = self._coerce_map(sector, "sector", nodes["sector"])
        for col in ("from_region", "to_region"):
            tbl[col] = tbl[col].map(lambda x: region_map.get(x, x))
        for col in ("from_sector", "to_sector"):
            tbl[col] = tbl[col].map(lambda x: sector_map.get(x, x))
        nodes["region"] = nodes["region"].map(lambda x: region_map.get(x, x))
        nodes["sector"] = nodes["sector"].map(lambda x: sector_map.get(x, x))
        nodes = nodes.drop_duplicates(ignore_index=True)

        kinds = nodes[["sector", "entity", "entity_sub"]].drop_duplicates()
        clash = kinds[kinds.duplicated("sector", keep=False)].sort_values("sector")
        if not clash.empty:
            raise ValueError(
                "sector map merges different entity kinds — a sector must name one kind: "
                + "; ".join(f"{r.sector}: {r.entity}/{r.entity_sub}" for r in clash.itertuples())
            )

        key = [c for c in tbl.columns if c not in ("edge_id", "val")]
        out = (tbl.groupby(key, sort=False, dropna=False)["val"].sum()
                  .reset_index()
                  .sort_values(key, ignore_index=True))
        step = {
            "region":         region if (region is None or isinstance(region, str)) else dict(region),
            "sector":         None if sector is None else dict(sector),
            "keep_crossings": bool(keep_crossings),
        }
        return PSUT(
            out, nodes,
            valuations   = self.valuations,
            primary      = self.primary,
            acc_id       = self.acc_id,
            opt_run_id   = self.opt_run_id,
            const_run_id = self.const_run_id,
            aggregation  = self.aggregation + [step],
        )

    @staticmethod
    def _coerce_map(spec, dim: str, labels: pd.Series) -> dict:
        if spec is None:
            return {}
        present = set(labels.unique().tolist())
        if isinstance(spec, str):
            if dim != "region":
                raise TypeError(
                    "sector= must be a mapping {old: new}; a one-label collapse "
                    "would merge entity kinds"
                )
            return {lab: spec for lab in present}
        spec = dict(spec)
        unknown = sorted(set(spec) - present)
        if unknown:
            raise KeyError(f"{dim} map names labels that do not exist: {unknown}")
        return spec

    # ------------------------------------------------------------------
    # Block views — pivots of the table on the node universe
    # ------------------------------------------------------------------

    def block(self, name: str, period=None, valuation=None) -> pd.DataFrame:
        """One block of one (period, valuation) as a labelled matrix, rows and
        columns reindexed on the node universe. ``period=None`` means
        :meth:`latest`; ``valuation=None`` means :attr:`primary`."""
        if name not in _ROW_COL:
            raise KeyError(f"unknown block {name!r}; blocks are {BLOCKS}")
        period, valuation = self.resolve(period, valuation)
        (r_ent, r_sub, r_name), (c_ent, c_sub, c_name) = _ROW_COL[name]
        row_idx = self._labels(r_ent, r_sub, r_name)
        col_idx = self._labels(c_ent, c_sub, c_name)
        tbl = self.table
        sub = tbl[(tbl["block"] == name) & (tbl["period"] == period)
                  & (tbl["valuation"] == valuation)]
        if sub.empty:
            return pd.DataFrame(0.0, index=row_idx, columns=col_idx)
        m = sub.pivot_table(
            index=["from_region", "from_sector"], columns=["to_region", "to_sector"],
            values="val", aggfunc="sum", fill_value=0.0,
        )
        m = m.reindex(index=row_idx, columns=col_idx, fill_value=0.0)
        m.index.names   = row_idx.names
        m.columns.names = col_idx.names
        return m.astype(float)

    def S(self, period=None, valuation=None) -> pd.DataFrame:
        """Supply: rows ``(region, process)``, columns ``(region, flow)``."""
        return self.block("S", period, valuation)

    def U(self, period=None, valuation=None) -> pd.DataFrame:
        """Use: rows ``(region, flow)``, columns ``(region, process)``."""
        return self.block("U", period, valuation)

    def Y(self, period=None, valuation=None) -> pd.DataFrame:
        """Final use: rows ``(region, flow)``, columns ``(region, final_demand)``."""
        return self.block("Y", period, valuation)

    def E(self, period=None, valuation=None) -> pd.DataFrame:
        """Nature input: rows ``(region, nature)``, columns ``(region, process)``."""
        return self.block("E", period, valuation)

    def O(self, period=None, valuation=None) -> pd.DataFrame:
        """Secondary input: rows ``(region, secondary)``, columns ``(region, process)``."""
        return self.block("O", period, valuation)

    def P(self, period=None, valuation=None) -> pd.DataFrame:
        """Residual output: rows ``(region, process)``, columns ``(region, secondary)``."""
        return self.block("P", period, valuation)

    def W(self, period=None, valuation=None) -> pd.DataFrame:
        """Discard to nature: rows ``(region, process)``, columns ``(region, nature)``."""
        return self.block("W", period, valuation)

    def _labels(self, entity: str, sub: Optional[str], name: str) -> pd.MultiIndex:
        n = self.nodes[self.nodes["entity"] == entity]
        if sub is not None:
            n = n[n["entity_sub"] == sub]
        n = n.sort_values(["region", "sector"])
        return pd.MultiIndex.from_arrays(
            [n["region"].to_numpy(), n["sector"].to_numpy()], names=["region", name]
        )

    # ------------------------------------------------------------------
    # Realized concentration — the ratio of two valuations, per edge
    # ------------------------------------------------------------------

    def realized_conc(self, period=None, *, numerator: Optional[str] = None,
                      denominator: Optional[str] = None) -> pd.DataFrame:
        """Per-edge ratio of two valuations: the *ex-post* concentration
        ``CC/Mass`` of the materialised flows (CONTEXT "Realized conc"), to be
        read against the parametric ``conc``. One row per (period, edge
        template) with ``val_<numerator>``, ``val_<denominator>`` and ``conc``
        (NaN where the denominator is zero). ``period=None`` means every
        period; the numerator defaults to :attr:`primary`, the denominator to
        the only other valuation. The ratio of two near-zero flows is noise
        (the solver's concentration rows are soft, and a tiny flow can carry
        ``CC > Mass`` at negligible cost) — filter on ``val_<denominator>``
        before reading grades."""
        num = self.primary if numerator is None else numerator
        if denominator is None:
            others = [v for v in self.valuations if v != num]
            if len(others) != 1:
                raise ValueError(
                    f"denominator= is required: the PSUT holds {self.valuations}"
                )
            den = others[0]
        else:
            den = denominator
        for v in (num, den):
            if v not in self.valuations:
                raise KeyError(f"valuation {v!r} not in PSUT: {self.valuations}")
        if num == den:
            raise ValueError("numerator and denominator are the same valuation")

        sub = self.table
        if period is not None:
            period = str(period)
            if period not in self.periods():
                raise KeyError(f"period {period!r} not in PSUT: {self.periods()}")
            sub = sub[sub["period"] == period]
        key = [c for c in sub.columns if c not in ("valuation", "edge_id", "val")]
        a = sub[sub["valuation"] == num].set_index(key)["val"].rename(f"val_{num}")
        b = sub[sub["valuation"] == den].set_index(key)["val"].rename(f"val_{den}")
        out = pd.concat([a, b], axis=1).reset_index()
        n = out[f"val_{num}"].to_numpy(dtype=float)
        d = out[f"val_{den}"].to_numpy(dtype=float)
        nz = d != 0
        out["conc"] = np.where(nz, n / np.where(nz, d, 1.0), np.nan)
        return out

    # ------------------------------------------------------------------
    # Figures — thin delegates; the plotting code lives in diag (ADR 0037 §1)
    # ------------------------------------------------------------------

    def sankey(self, period=None, valuation=None, **spec):
        """The bipartite supply-use Sankey of one (period, valuation) as a
        Plotly figure: processes and commodities as nodes, the S/U rows as
        links, the boundary blocks as their own nodes, and every
        border-crossing use / final-use flow (``region_crossed`` after
        ``aggregate(..., keep_crossings=True)``, or distinct regions) as a
        separate traded link. Further keywords fill
        :class:`pysuto.diag.psut.SankeySpec` (``title``, ``min_val``,
        ``split_traded``, ``unit``). Plotly is imported here, not at module
        level; the tables behind the figure are
        :func:`pysuto.diag.psut.sankey_tables`."""
        from .diag.psut import SankeySpec, sankey_figure   # lazy: consumer layer
        return sankey_figure(self, SankeySpec(period=period, valuation=valuation, **spec))

    def __repr__(self) -> str:
        p = self.periods()
        span = f"{p[0]}..{p[-1]} ({len(p)})" if p else "none"
        counts = (self.table[self.table["valuation"] == self.primary]["block"]
                    .value_counts().sort_index().to_dict())
        agg = f", aggregation={len(self.aggregation)} step(s)" if self.aggregation else ""
        return (f"PSUT(acc={self.acc_id!r}, valuations={self.valuations}, "
                f"primary={self.primary!r}, periods={span}, rows={len(self.table)}, "
                f"blocks[{self.primary}]={counts}{agg})")
