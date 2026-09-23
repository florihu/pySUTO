"""
psut.py
=======
Stage 6 — the physical supply-use table (PSUT), tidy-canonical.

A :class:`PSUT` is the solved edge vector ``a*`` of one or more periods carried
as a **tidy edge table** — one row per directed edge per (period, valuation),
every row classified into a PSUT **block** — together with the node label
universe of the accounting structure it came from. The block matrices ``S``
(Process→Flow, supply) and ``U`` (Flow→Process, use) are pivot *views* of that
table, reindexed onto the node universe so their labels are identical across
periods (issue #18, ADR 0037).

Slice 1 (issue #18) materialises the ``S`` and ``U`` blocks for one valuation.
The boundary blocks ``Y``/``E``/``O``/``P`` (#19), region/sector aggregation
(#24) and ``to_piot`` (#25) extend the same table.

Construction contract
---------------------
* :meth:`PSUT.from_opt_run` — from a persisted optimiser run; resolves the
  accounting structure through provenance (opt sidecar → const sidecar →
  ``acc`` id) and loads every timestep of the run unless told otherwise.
* :meth:`PSUT.from_results` — in-memory handoff from
  ``dict[timestep, OptimResultsLike]``; a year-local ``F_opt`` (block path)
  is mapped to global edge ids through ``cols_t``.

Time is a column of the table, never aggregated: :meth:`PSUT.periods`,
:meth:`PSUT.latest` and :meth:`PSUT.at` select it. Persistence lives in
:class:`pysuto.registry.PsutRunRegistry`.
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


# block ← (from_entity, to_entity). Slice 1 carries the two core blocks; #19
# adds the boundary blocks, which also need the SB node's entity_sub.
_BLOCK_OF_ENTITY_PAIR = {
    ("Process", "Flow"): "S",
    ("Flow", "Process"): "U",
}

_TABLE_COLUMNS = [
    "period", "valuation", "block", "edge_id",
    "from_region", "from_sector", "from_entity",
    "to_region", "to_sector", "to_entity",
    "val",
]

_NODE_COLUMNS = ["sector", "entity", "region", "entity_sub"]


def classify_blocks(edge_meta: pd.DataFrame) -> pd.Series:
    """Block label (``S``/``U``) per row of ``edge_meta``; ``None`` for every
    edge outside the blocks this slice materialises."""
    fe = edge_meta["from_entity"].to_numpy()
    te = edge_meta["to_entity"].to_numpy()
    block = pd.Series([None] * len(edge_meta), index=edge_meta.index, dtype=object)
    for (f, t), label in _BLOCK_OF_ENTITY_PAIR.items():
        block[(fe == f) & (te == t)] = label
    return block


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
            f"{len(positions)} supply/use edges are not covered by cols_t"
        )
    return F[where]


class PSUT:
    """Tidy physical supply-use table of one valuation over one or more periods.

    Attributes
    ----------
    table : DataFrame
        One row per (period, edge): ``period, valuation, block, edge_id,
        from_region, from_sector, from_entity, to_region, to_sector, to_entity,
        val``. ``edge_id`` is the global edge id of the accounting structure.
    nodes : DataFrame
        The node universe (``sector, entity, region, entity_sub``) the block
        views are labelled from.
    valuation, acc_id, opt_run_id, const_run_id
        Provenance; the run ids are ``None`` for an in-memory handoff.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        nodes: pd.DataFrame,
        *,
        valuation:    str,
        acc_id:       str = "",
        opt_run_id:   Optional[str] = None,
        const_run_id: Optional[str] = None,
    ) -> None:
        self.table        = table.reset_index(drop=True)
        self.nodes        = nodes.reset_index(drop=True)
        self.valuation    = valuation
        self.acc_id       = acc_id
        self.opt_run_id   = opt_run_id
        self.const_run_id = const_run_id

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_results(
        cls,
        results: Mapping[str, OptimResultsLike],
        acc: AccountingStructure,
        *,
        valuation: str = "CC",
    ) -> "PSUT":
        """Build from ``{timestep: result}`` (the solver's output shape).

        Each result's ``F_opt`` is placed on global edge ids — through
        ``cols_t`` on the block path, positionally on the legacy path — and
        every supply/use edge of ``valuation`` in that timestep becomes one
        row. Any other edge (boundary blocks, the other valuation) is left
        out of this slice.
        """
        if valuation not in acc.valuation:
            raise ValueError(
                f"valuation {valuation!r} is not one of {acc.valuation} "
                f"(accounting structure {acc.id!r})"
            )
        if not results:
            raise ValueError("results is empty — nothing to materialise")

        meta     = acc.edge_meta
        block    = classify_blocks(meta)
        in_scope = block.notna().to_numpy() & (meta["valuation"].to_numpy() == valuation)
        time     = meta["time"].to_numpy()
        keep     = ["edge_id", "valuation",
                    "from_region", "from_sector", "from_entity",
                    "to_region", "to_sector", "to_entity"]

        frames = []
        for t, res in results.items():
            t = str(t)
            if str(res.timestep) != t:
                raise ValueError(
                    f"results key {t!r} holds a result for timestep {res.timestep!r}"
                )
            positions = np.flatnonzero(in_scope & (time == t))
            frame = meta.iloc[positions][keep].copy()
            frame.insert(0, "period", t)
            frame.insert(2, "block", block.iloc[positions].to_numpy())
            frame["val"] = _period_values(res, positions, acc.n_edges)
            frames.append(frame[_TABLE_COLUMNS])

        table = (pd.concat(frames, ignore_index=True)
                   .sort_values(["period", "block", "edge_id"], ignore_index=True))
        nodes = acc.node_index.copy()
        if "entity_sub" not in nodes.columns:
            nodes["entity_sub"] = None
        nodes = nodes[_NODE_COLUMNS]

        counts = table["block"].value_counts().to_dict()
        logger.info(
            "PSUT %s/%s: %d timestep(s), %d rows (%s) of %d in-scope edges per period",
            acc.id, valuation, len(frames), len(table),
            ", ".join(f"{b}={n}" for b, n in sorted(counts.items())),
            int(in_scope.sum()) // max(len(acc.time_range), 1),
        )
        return cls(table, nodes, valuation=valuation, acc_id=acc.id)

    @classmethod
    def from_opt_run(
        cls,
        opt_run_id: str,
        timesteps: Optional[Sequence] = None,
        valuation: str = "CC",
        *,
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

        psut = cls.from_results(results, acc, valuation=valuation)
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
            valuation    = self.valuation,
            acc_id       = self.acc_id,
            opt_run_id   = self.opt_run_id,
            const_run_id = self.const_run_id,
        )

    # ------------------------------------------------------------------
    # Block views — pivots of the table on the node universe
    # ------------------------------------------------------------------

    def S(self, period=None) -> pd.DataFrame:
        """Supply block of one period: rows ``(region, process)``, columns
        ``(region, flow)``. ``period=None`` means :meth:`latest`."""
        return self._block_view("S", period, ("Process", "process"), ("Flow", "flow"))

    def U(self, period=None) -> pd.DataFrame:
        """Use block of one period: rows ``(region, flow)``, columns
        ``(region, process)``. ``period=None`` means :meth:`latest`."""
        return self._block_view("U", period, ("Flow", "flow"), ("Process", "process"))

    def _labels(self, entity: str, name: str) -> pd.MultiIndex:
        n = (self.nodes[self.nodes["entity"] == entity]
                 .sort_values(["region", "sector"]))
        return pd.MultiIndex.from_arrays(
            [n["region"].to_numpy(), n["sector"].to_numpy()], names=["region", name]
        )

    def _block_view(self, block, period, rows, cols) -> pd.DataFrame:
        period  = self.latest() if period is None else str(period)
        if period not in self.periods():
            raise KeyError(f"period {period!r} not in PSUT: {self.periods()}")
        row_idx = self._labels(*rows)
        col_idx = self._labels(*cols)
        sub = self.table[(self.table["block"] == block) & (self.table["period"] == period)]
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

    def __repr__(self) -> str:
        p = self.periods()
        span = f"{p[0]}..{p[-1]} ({len(p)})" if p else "none"
        counts = self.table["block"].value_counts().sort_index().to_dict()
        return (f"PSUT(acc={self.acc_id!r}, valuation={self.valuation!r}, "
                f"periods={span}, rows={len(self.table)}, blocks={counts})")
