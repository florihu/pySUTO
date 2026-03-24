import inspect
import os
import pickle
import numpy as np
import pandas as pd
import sparse
from pysuto.logging import get_logger


class AccountingStructure:
    """
    Defines the admissible node/edge space for a physical accounting system.

    Admissibility tensor
    --------------------
    ``A`` is a 4-D COO tensor (``sparse.COO``) of shape::

        (n_nodes, n_nodes, n_timeps, n_valtns)

    A[i, j, t, v] = 1  iff the edge from node i to node j at time t
                        with valuation v is admissible under all rules.

    Node-level rules constrain (i, j) only and are broadcast across all
    (t, v) slices.  Edge-level rules additionally constrain (t, v).

    The 2-D node-only mask is kept separately as ``A_nodes`` (``sparse.COO``,
    shape ``(n_nodes, n_nodes)``) for fast node-pair lookups independent of
    time and valuation.

    Rules
    -----
    Node rules  (common case) — signature ``rule(ni, nj) -> bool``
        Keys in ni/nj: ``sector``, ``entity``, ``region``.

    Edge rules  (edge-case) — signature ``rule(ni, nj, time, valuation) -> bool``
        Auto-detected by counting required positional parameters (> 2).

    Rules short-circuit: the first False stops further evaluation.

    Time / valuation deduplication
    --------------------------------
    Duplicate values in ``time_range`` or ``valuation`` are silently removed.
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
        self.A_nodes = self._build_node_mask()
        self.A       = self._build_tensor()
        self.edge_index     = self._build_edge_index()
        self.edge_index_all = self._build_edge_index(admissible_only=False)
        self.n_edges_all    = len(self.edge_index_all)

        self.logger.info(
            f"AccountingStructure '{self.id}' initialised: "
            f"{self.n_nodes} nodes, {self.n_edges} admissible edges, "
            f"A shape={self.A.shape}, nnz={self.A.nnz}"
        )

    # ------------------------------------------------------------------
    # Rule classification
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
    # Excel initialisation
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
    # Node index
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
        """Integer axis-position maps: time value → int, valuation → int."""
        self.time_to_idx = {t: i for i, t in enumerate(self.time_range)}
        self.val_to_idx  = {v: i for i, v in enumerate(self.valuation)}

    # ------------------------------------------------------------------
    # Node-only mask  —  sparse.COO  (n_nodes, n_nodes)
    # ------------------------------------------------------------------

    def _build_node_mask(self) -> sparse.COO:
        """
        2-D COO mask over node pairs, time/valuation agnostic.
        Reflects node-level rules only; used for fast pair lookup and
        as the structural basis of the full 4-D tensor.
        """
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
    # Full 4-D admissibility tensor  —  sparse.COO  (n, n, n_t, n_v)
    # ------------------------------------------------------------------

    def _build_tensor(self) -> sparse.COO:
        """
        Build A[i, j, t_idx, v_idx] = 1 for every admissible edge.

        When no edge rules exist, the result is equivalent to broadcasting
        A_nodes across the time and valuation axes.  Edge rules add
        fine-grained (t, v) filtering on top.
        """
        self.logger.info("Building 4-D admissibility tensor...")
        c0, c1, c2, c3 = [], [], [], []
        keys = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(['sector', 'entity', 'region'], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(['sector', 'entity', 'region'], nj))

                if not self._node_pair_admissible(ni_dict, nj_dict):
                    continue          # short-circuit: skip all (t, v) for this pair

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
    # Edge index DataFrame
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
    # Compress: 4-D flow tensor → 1-D edge vector
    # ------------------------------------------------------------------

    def compress(self, F) -> np.ndarray:
        """
        Extract admissible-edge values from a full 4-D flow tensor.

        Parameters
        ----------
        F : array-like, shape (n_nodes, n_nodes, n_timeps, n_valtns)
            Dense ndarray or sparse.SparseArray.

        Returns
        -------
        np.ndarray, shape (n_edges,)
            One scalar per admissible edge, in edge_index order.
        """
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
    # Lookup helpers
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
    # Save / Load
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
        return obj


# =============================================================================
# __main__ smoke test
# =============================================================================

if __name__ == "__main__":

    root_def = {
        "nodes": pd.DataFrame({
            "sector": ["Mining", "Smelting", "Mining"],
            "entity": ["P",      "F",        "P"],
            "region": ["A",      "B",        "B"],
        }),
        "time_range": [0, 1, 0],
        "valuation":  ["mass", "copper_c", "mass"],
    }

    def no_p_to_p(ni, nj):
        return not (ni["entity"] == "P" and nj["entity"] == "P")

    def only_time_0(ni, nj, time, valuation):
        return time == 0

    acc = AccountingStructure(
        id="test_structure", root=root_def,
        rules=[no_p_to_p, only_time_0]
    )

    print("n_nodes:        ", acc.n_nodes)
    print("n_edges:        ", acc.n_edges)
    print("A shape:        ", acc.A.shape)
    print("A.nnz:          ", acc.A.nnz)
    print("A_nodes shape:  ", acc.A_nodes.shape)
    print("A_nodes.nnz:    ", acc.A_nodes.nnz)
    print("time_range:     ", acc.time_range)
    print("valuation:      ", acc.valuation)