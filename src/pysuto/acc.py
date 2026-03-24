"""
acc.py
======
AccountingStructure — defines the admissible node/edge space for a
physical accounting system, plus per-edge lower/upper bounds used by
the linear optimiser.

Bounds design
-------------
Default bounds:  l_i = 0,  u_i = +∞  (all flows are non-negative)

Exceptions are stored in an Excel file with one row per exception rule.
A rule matches by (sector_from, sector_to, valuation); region and time
are optional filters.  Unspecified filter columns are treated as "all".

The resulting bound vectors l and u (length = n_edges) are built once
during __init__ and stored as numpy arrays aligned to edge_index order.
ConstraintBuilder can then access acc.l and acc.u directly.

Excel format  (sheet name: "bounds",  columns):
    sector_from   : str   — matches from-node sector   ("*" = any)
    sector_to     : str   — matches to-node sector     ("*" = any)
    valuation     : str   — matches edge valuation      ("*" = any)
    region_from   : str   — optional, ("*" = any)
    region_to     : str   — optional, ("*" = any)
    time          : str   — optional, ("*" = any)
    lower         : float — new lower bound  (default overridden for matched edges)
    upper         : float — new upper bound  (default overridden for matched edges)

Rows are applied top-to-bottom; later rows override earlier ones for
the same edge.  This lets you set a broad exception then refine it.

Example rows
------------
sector_from  sector_to  valuation   lower   upper
PSC          *          *           -1e9    1e9     # stock can be negative
WA           WA         mass        0       0       # self-loops forbidden
"""

from __future__ import annotations

import inspect
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import sparse

try:
    from .logging import get_logger          # package import
except ImportError:
    import logging                           # fallback for standalone use
    def get_logger(name):
        return logging.getLogger(name)


# ── sentinel for "no bound" ──────────────────────────────────────────────────
_INF = np.inf


# =============================================================================
# EdgeBounds
# =============================================================================

class EdgeBounds:
    """
    Per-edge lower and upper bound vectors, aligned to an edge_index.

    Attributes
    ----------
    l : np.ndarray, shape (n_edges,)
        Lower bounds.  Default 0 unless overridden.
    u : np.ndarray, shape (n_edges,)
        Upper bounds.  Default +inf unless overridden.
    n_exceptions : int
        Number of edges whose bounds differ from the default.
    """

    DEFAULT_LOWER: float = 0.0
    DEFAULT_UPPER: float = _INF

    def __init__(self,
                 edge_index:  pd.DataFrame,
                 node_index:  pd.DataFrame,
                 bounds_df:   Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        edge_index : DataFrame with columns edge_id, from_id, to_id, time, valuation
        node_index : DataFrame indexed by node_id with columns sector, entity, region
        bounds_df  : parsed bounds Excel sheet (None → all-default bounds)
        """
        self.n_edges = len(edge_index)
        self.l = np.full(self.n_edges, self.DEFAULT_LOWER, dtype=float)
        self.u = np.full(self.n_edges, self.DEFAULT_UPPER, dtype=float)

        if bounds_df is not None and len(bounds_df) > 0:
            self._apply_rules(edge_index, node_index, bounds_df)

        # count exceptions (edges that differ from both defaults)
        non_default = (self.l != self.DEFAULT_LOWER) | (self.u != self.DEFAULT_UPPER)
        self.n_exceptions = int(non_default.sum())

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------

    def _apply_rules(self,
                     edge_index: pd.DataFrame,
                     node_index: pd.DataFrame,
                     bounds_df:  pd.DataFrame) -> None:
        """
        Apply each bounds rule row in order.  Later rows override earlier
        ones for the same edge (allows broad rule + narrow refinement).
        """
        # Build a fast lookup: edge_id → (sector_from, sector_to,
        #                                  region_from, region_to,
        #                                  time, valuation)
        ni = node_index.loc[edge_index["from_id"].values]
        nj = node_index.loc[edge_index["to_id"].values]

        # Align to integer positions (edge_index may not be 0-based)
        pos = np.arange(self.n_edges)

        sf = ni["sector"].values
        rf = ni["region"].values
        st = nj["sector"].values
        rt = nj["region"].values
        tm = edge_index["time"].values.astype(str)
        vl = edge_index["valuation"].values.astype(str)

        required_cols = {"sector_from", "sector_to", "valuation", "lower", "upper"}
        missing = required_cols - set(bounds_df.columns)
        if missing:
            raise ValueError(
                f"Bounds Excel sheet is missing required columns: {missing}\n"
                f"Required: sector_from, sector_to, valuation, lower, upper\n"
                f"Optional: region_from, region_to, time"
            )

        for _, rule in bounds_df.iterrows():
            mask = np.ones(self.n_edges, dtype=bool)

            def _match(arr, val):
                v = str(val).strip()
                return np.ones(self.n_edges, dtype=bool) if v in ("*", "", "nan") \
                       else arr == v

            mask &= _match(sf, rule["sector_from"])
            mask &= _match(st, rule["sector_to"])
            mask &= _match(vl, rule["valuation"])

            if "region_from" in rule.index:
                mask &= _match(rf, rule["region_from"])
            if "region_to" in rule.index:
                mask &= _match(rt, rule["region_to"])
            if "time" in rule.index:
                mask &= _match(tm, rule["time"])

            n_matched = mask.sum()
            if n_matched == 0:
                continue          # rule matched nothing — not an error,
                                  # but worth logging at caller level

            if pd.notna(rule["lower"]):
                self.l[mask] = float(rule["lower"])
            if pd.notna(rule["upper"]):
                self.u[mask] = float(rule["upper"])

    # ------------------------------------------------------------------
    # Validation helper (used by ConstraintBuilder)
    # ------------------------------------------------------------------

    def validate(self,
                 F: np.ndarray,
                 edge_index: pd.DataFrame,
                 raise_on_violation: bool = True,
                 ) -> pd.DataFrame:
        """
        Check that all values in F satisfy l ≤ F ≤ u.

        Parameters
        ----------
        F : np.ndarray, shape (n_edges,)
        edge_index : the matching edge_index DataFrame (for diagnostics)
        raise_on_violation : if True, raise ValueError on any violation

        Returns
        -------
        DataFrame of violated edges (empty if all OK).
        Columns: edge_id, from_id, to_id, time, valuation, lower, value, upper
        """
        lo_viol = F < self.l
        hi_viol = F > self.u
        any_viol = lo_viol | hi_viol

        if not any_viol.any():
            return pd.DataFrame()

        viol_idx = np.where(any_viol)[0]
        report = edge_index.iloc[viol_idx].copy()
        report["lower"] = self.l[viol_idx]
        report["value"] = F[viol_idx]
        report["upper"] = self.u[viol_idx]
        report["violation"] = np.where(
            lo_viol[viol_idx], "below_lower",
            np.where(hi_viol[viol_idx], "above_upper", "")
        )

        if raise_on_violation:
            _cols = ['edge_id', 'from_id', 'to_id', 'time', 'valuation',
                     'lower', 'value', 'upper', 'violation']
            raise ValueError(
                f"{len(viol_idx)} edge(s) violate bounds.\n"
                f"{report[_cols].to_string(index=False)}"
            )
        return report

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        return {"l": self.l, "u": self.u}

    def summary(self) -> str:
        neg_lower   = (self.l < 0).sum()
        finite_upper = np.isfinite(self.u).sum()
        return (
            f"EdgeBounds: {self.n_edges} edges | "
            f"{self.n_exceptions} exceptions | "
            f"{neg_lower} negative-lower | "
            f"{finite_upper} finite-upper"
        )


# =============================================================================
# AccountingStructure  (extended)
# =============================================================================

class AccountingStructure:
    """
    Defines the admissible node/edge space for a physical accounting system,
    and holds per-edge lower/upper bounds for the linear optimiser.

    New parameters
    --------------
    bounds_file_path : str, optional
        Path to an Excel file (see module docstring for column spec).
        If None, all edges receive default bounds (l=0, u=+inf).

    New attributes
    --------------
    bounds : EdgeBounds
        Per-edge bound vectors  .l  and  .u  aligned to edge_index.
    """

    def __init__(self, id, root=None, rules=None, **kwargs):
        self.rules       = rules or []
        self.id          = id or "accounting_structure"
        self.storage_dir = kwargs.get('storage_dir', r'data/proc/acc')
        self.logger      = get_logger(self.__class__.__name__)

        if kwargs.get('init_from_file', False):
            assert 'root_file_path' in kwargs
            self.syst_desc_from_file(kwargs['root_file_path'])
        else:
            assert root is not None, \
                "Root definition must be provided if not initializing from file"
            self.root_nodes = root['nodes'].copy()
            self.time_range = list(dict.fromkeys(root['time_range']))
            self.valuation  = list(dict.fromkeys(root['valuation']))

        self._classify_rules()
        self._build_node_index()
        self._build_dim_indices()
        self.A_nodes        = self._build_node_mask()
        self.A              = self._build_tensor()
        self.edge_index     = self._build_edge_index()
        self.edge_index_all = self._build_edge_index(admissible_only=False)
        self.n_edges_all    = len(self.edge_index_all)

        # ── NEW: bounds ──────────────────────────────────────────────
        bounds_path = kwargs.get('bounds_file_path', None)
        self.bounds = self._load_bounds(bounds_path)
        self.logger.info(self.bounds.summary())
        # ─────────────────────────────────────────────────────────────

        self.logger.info(
            f"AccountingStructure '{self.id}' initialised: "
            f"{self.n_nodes} nodes, {self.n_edges} admissible edges, "
            f"A shape={self.A.shape}, nnz={self.A.nnz}"
        )

    # ------------------------------------------------------------------
    # Bounds loading  (NEW)
    # ------------------------------------------------------------------

    def _load_bounds(self, path: Optional[str]) -> EdgeBounds:
        """
        Load exception bounds from Excel and build EdgeBounds.
        If path is None or file is absent, returns all-default bounds.
        """
        bounds_df = None
        if path is not None:
            if not os.path.exists(path):
                self.logger.warning(
                    f"Bounds file not found: '{path}'. Using default bounds (l=0, u=+inf)."
                )
            else:
                self.logger.info(f"Loading edge bounds from: {path}")
                bounds_df = pd.read_excel(path, sheet_name="bounds")
                self.logger.info(
                    f"  Loaded {len(bounds_df)} bound exception rule(s)."
                )
                self._log_unmatched_rules(bounds_df)

        return EdgeBounds(
            edge_index=self.edge_index,
            node_index=self.node_index,
            bounds_df=bounds_df,
        )

    def _log_unmatched_rules(self, bounds_df: pd.DataFrame) -> None:
        """Warn if a rule's sector_from/sector_to never appear in node_index."""
        known_sectors = set(self.node_index["sector"].unique())
        for i, row in bounds_df.iterrows():
            for col in ("sector_from", "sector_to"):
                v = str(row.get(col, "*")).strip()
                if v not in ("*", "", "nan") and v not in known_sectors:
                    self.logger.warning(
                        f"Bounds rule row {i}: '{col}={v}' "
                        f"does not match any node sector. Rule may be a no-op."
                    )

    def reload_bounds(self, path: str) -> None:
        """
        Hot-reload the bounds from a (new) Excel file without
        rebuilding the full structure.
        """
        self.logger.info(f"Reloading edge bounds from: {path}")
        bounds_df = pd.read_excel(path, sheet_name="bounds")
        self.bounds = EdgeBounds(
            edge_index=self.edge_index,
            node_index=self.node_index,
            bounds_df=bounds_df,
        )
        self.logger.info(self.bounds.summary())

    # ------------------------------------------------------------------
    # Convenience accessor for ConstraintBuilder
    # ------------------------------------------------------------------

    @property
    def l(self) -> np.ndarray:
        """Lower bound vector, shape (n_edges,)."""
        return self.bounds.l

    @property
    def u(self) -> np.ndarray:
        """Upper bound vector, shape (n_edges,)."""
        return self.bounds.u

    # ------------------------------------------------------------------
    # Rule classification  (unchanged)
    # ------------------------------------------------------------------

    def _classify_rules(self):
        self._node_rules = []
        self._edge_rules = []
        for rule in self.rules:
            sig    = inspect.signature(rule)
            n_args = sum(
                1 for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            (self._edge_rules if n_args > 2 else self._node_rules).append(rule)

    def _node_pair_admissible(self, ni_dict, nj_dict) -> bool:
        for rule in self._node_rules:
            if not rule(ni_dict, nj_dict):
                return False
        return True

    def _edge_admissible(self, ni_dict, nj_dict, time, valuation) -> bool:
        if not self._node_pair_admissible(ni_dict, nj_dict):
            return False
        for rule in self._edge_rules:
            if not rule(ni_dict, nj_dict, time, valuation):
                return False
        return True

    # ------------------------------------------------------------------
    # Excel initialisation  (unchanged)
    # ------------------------------------------------------------------

    def syst_desc_from_file(self, path):
        self.logger.info(f"Initialising structure from file: {path}")
        file = pd.read_excel(path)
        assert {'sector_entity', 'region', 'time_range', 'valuation'}.issubset(
            file.columns
        ), "Excel file must contain: sector_entity, region, time_range, valuation"

        sector_entity = file['sector_entity'].str.split('_', expand=True)
        sector_entity.columns = ['sector', 'entity']
        region = file['region'].unique()
        self.root_nodes = pd.DataFrame(
            [(s, e, r)
             for s, e in zip(sector_entity['sector'], sector_entity['entity'])
             for r in region],
            columns=['sector', 'entity', 'region']
        )
        self.time_range = list(dict.fromkeys(file['time_range'].tolist()))
        self.valuation  = list(dict.fromkeys(file['valuation'].tolist()))

    # ------------------------------------------------------------------
    # Node index  (unchanged)
    # ------------------------------------------------------------------

    def _build_node_index(self):
        self.node_to_id = {}
        self.id_to_node = {}
        node_id = 0
        for _, node in self.root_nodes.iterrows():
            key = (node['sector'], node['entity'], node['region'])
            if key not in self.node_to_id:
                self.node_to_id[key] = node_id
                self.id_to_node[node_id] = key
                node_id += 1

        self.n_nodes  = node_id
        self.n_timeps = len(self.time_range)
        self.n_valtns = len(self.valuation)
        self.node_index = pd.DataFrame.from_dict(
            self.id_to_node, orient='index', columns=['sector', 'entity', 'region']
        )
        self.logger.info(f"Total nodes: {self.n_nodes}")

    def _build_dim_indices(self):
        self.time_to_idx = {t: i for i, t in enumerate(self.time_range)}
        self.val_to_idx  = {v: i for i, v in enumerate(self.valuation)}

    # ------------------------------------------------------------------
    # Node-only mask  (unchanged)
    # ------------------------------------------------------------------

    def _build_node_mask(self) -> sparse.COO:
        self.logger.info("Building 2-D node admissibility mask...")
        c0, c1 = [], []
        keys = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(['sector', 'entity', 'region'], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(['sector', 'entity', 'region'], nj))
                if self._node_pair_admissible(ni_dict, nj_dict):
                    c0.append(i)
                    c1.append(j)

        data = np.ones(len(c0), dtype=np.int8)
        return sparse.COO(
            coords=np.array([c0, c1], dtype=np.int64),
            data=data,
            shape=(self.n_nodes, self.n_nodes),
        )

    # ------------------------------------------------------------------
    # Full 4-D admissibility tensor  (unchanged)
    # ------------------------------------------------------------------

    def _build_tensor(self) -> sparse.COO:
        self.logger.info("Building 4-D admissibility tensor...")
        c0, c1, c2, c3 = [], [], [], []
        keys = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(['sector', 'entity', 'region'], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(['sector', 'entity', 'region'], nj))

                if not self._node_pair_admissible(ni_dict, nj_dict):
                    continue

                for t_idx, t in enumerate(self.time_range):
                    for v_idx, v in enumerate(self.valuation):
                        if self._edge_rules and not self._edge_admissible(
                            ni_dict, nj_dict, t, v
                        ):
                            continue
                        c0.append(i);   c1.append(j)
                        c2.append(t_idx); c3.append(v_idx)

        data = np.ones(len(c0), dtype=np.int8)
        return sparse.COO(
            coords=np.array([c0, c1, c2, c3], dtype=np.int64),
            data=data,
            shape=(self.n_nodes, self.n_nodes, self.n_timeps, self.n_valtns),
        )

    # ------------------------------------------------------------------
    # Edge index DataFrame  (unchanged)
    # ------------------------------------------------------------------

    def _build_edge_index(self, admissible_only=True) -> pd.DataFrame:
        rows, cols, times, vals, edge_ids = [], [], [], [], []
        edge_id = 0
        keys    = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(['sector', 'entity', 'region'], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(['sector', 'entity', 'region'], nj))

                if admissible_only and not self._node_pair_admissible(ni_dict, nj_dict):
                    continue

                for t in self.time_range:
                    for v in self.valuation:
                        if admissible_only and self._edge_rules:
                            if not self._edge_admissible(ni_dict, nj_dict, t, v):
                                continue
                        rows.append(i);     cols.append(j)
                        times.append(t);    vals.append(v)
                        edge_ids.append(edge_id)
                        edge_id += 1

        if admissible_only:
            self.n_edges = len(edge_ids)
        return pd.DataFrame({
            "edge_id":   edge_ids,
            "from_id":   rows,
            "to_id":     cols,
            "time":      times,
            "valuation": vals,
        })

    # ------------------------------------------------------------------
    # compress  (unchanged)
    # ------------------------------------------------------------------

    def compress(self, F) -> np.ndarray:
        if isinstance(F, sparse.SparseArray):
            F = F.todense()
        ei     = self.edge_index
        t_idxs = ei["time"].map(self.time_to_idx).values.astype(int)
        v_idxs = ei["valuation"].map(self.val_to_idx).values.astype(int)
        return F[
            ei["from_id"].values.astype(int),
            ei["to_id"].values.astype(int),
            t_idxs,
            v_idxs,
        ]

    # ------------------------------------------------------------------
    # Lookup helpers  (unchanged)
    # ------------------------------------------------------------------

    def decode_node(self, node_id: int):
        return self.id_to_node[node_id]

    def encode_node(self, sector, entity, region) -> int:
        return self.node_to_id[(sector, entity, region)]

    def decode_edge(self, edge_id: int):
        row = self.edge_index.iloc[edge_id]
        return (
            self.decode_node(int(row['from_id'])),
            self.decode_node(int(row['to_id'])),
            row['time'],
            row['valuation'],
        )

    # ------------------------------------------------------------------
    # Save / Load  (extended to persist bounds)
    # ------------------------------------------------------------------

    def save(self, **kwargs):
        path = os.path.join(self.storage_dir, self.id)
        os.makedirs(path, exist_ok=True)

        sparse.save_npz(os.path.join(path, "A.npz"),       self.A)
        sparse.save_npz(os.path.join(path, "A_nodes.npz"), self.A_nodes)

        self.edge_index.to_csv(    os.path.join(path, "edge_index.csv"),     index=False)
        self.edge_index_all.to_csv(os.path.join(path, "edge_index_all.csv"), index=False)
        self.node_index.to_csv(    os.path.join(path, "node_index.csv"),     index=False)
        self.root_nodes.to_csv(    os.path.join(path, "root_nodes.csv"),     index=False)

        # ── NEW: persist bounds vectors ──────────────────────────────
        np.save(os.path.join(path, "bounds_l.npy"), self.bounds.l)
        np.save(os.path.join(path, "bounds_u.npy"), self.bounds.u)
        # ─────────────────────────────────────────────────────────────

        meta = {
            "time_range": self.time_range,
            "valuation":  self.valuation,
            "node_to_id": self.node_to_id,
            "id_to_node": self.id_to_node,
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self.logger.info(f"Structure saved to {path}")

    @classmethod
    def load(cls, id, **kwargs):
        obj             = cls.__new__(cls)
        obj.id          = id
        obj.storage_dir = kwargs.get('storage_dir', r'data/proc/acc')
        path            = os.path.join(obj.storage_dir, id)
        assert os.path.isdir(path), f"Directory '{id}' not found in {obj.storage_dir}"

        obj.A         = sparse.load_npz(os.path.join(path, "A.npz"))
        obj.A_nodes   = sparse.load_npz(os.path.join(path, "A_nodes.npz"))

        obj.edge_index      = pd.read_csv(os.path.join(path, "edge_index.csv"))
        obj.edge_index_all  = pd.read_csv(os.path.join(path, "edge_index_all.csv"))
        obj.root_nodes      = pd.read_csv(os.path.join(path, "root_nodes.csv"))
        obj.node_index      = pd.read_csv(os.path.join(path, "node_index.csv"))

        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)

        obj.time_range  = meta["time_range"]
        obj.valuation   = meta["valuation"]
        obj.node_to_id  = meta["node_to_id"]
        obj.id_to_node  = meta["id_to_node"]
        obj.n_nodes     = len(obj.id_to_node)
        obj.n_timeps    = len(obj.time_range)
        obj.n_valtns    = len(obj.valuation)
        obj.n_edges     = len(obj.edge_index)
        obj.n_edges_all = len(obj.edge_index_all)
        obj.time_to_idx = {t: i for i, t in enumerate(obj.time_range)}
        obj.val_to_idx  = {v: i for i, v in enumerate(obj.valuation)}
        obj.rules       = []
        obj._node_rules = []
        obj._edge_rules = []
        obj.logger      = get_logger(cls.__name__)

        # ── NEW: restore bounds (prefer saved .npy, fall back to default) ──
        l_path = os.path.join(path, "bounds_l.npy")
        u_path = os.path.join(path, "bounds_u.npy")
        bounds = EdgeBounds.__new__(EdgeBounds)
        bounds.n_edges = obj.n_edges
        if os.path.exists(l_path) and os.path.exists(u_path):
            bounds.l = np.load(l_path)
            bounds.u = np.load(u_path)
        else:
            bounds.l = np.zeros(obj.n_edges,  dtype=float)
            bounds.u = np.full(obj.n_edges, _INF, dtype=float)
        non_default = (bounds.l != EdgeBounds.DEFAULT_LOWER) | \
                      (bounds.u != EdgeBounds.DEFAULT_UPPER)
        bounds.n_exceptions = int(non_default.sum())
        obj.bounds = bounds
        # ──────────────────────────────────────────────────────────────
        return obj
