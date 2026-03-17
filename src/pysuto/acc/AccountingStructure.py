
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import os

from pysuto.logging import get_logger


import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pysuto.logging import get_logger

class AccountingStructure:
    def __init__(self, id, root=None, rules=None, **kwargs):
        self.rules = rules or []
        self.id = id or "accounting_structure"
        self.storage_dir = kwargs.get('storage_dir', r'data/proc/acc')
        self.logger = get_logger(self.__class__.__name__)

        # Initialize nodes
        if kwargs.get('init_from_file', False):
            assert 'root_file_path' in kwargs
            self.syst_desc_from_file(kwargs['root_file_path'])
        else:
            assert root is not None, "Root definition must be provided if not initializing from file"
            self.root_nodes = root['nodes'].copy()
            self.time_range = root['time_range']
            self.valuation = root['valuation']

        # Build integer node IDs (3-tuple)
        self._build_node_index()

        # Build admissibility mask (nodes only)
        self.A = self._build_mask()

        # Build edge index (admissible only, edges include time & valuation)
        self.edge_index = self._build_edge_index()

        # Full edge index
        self.edge_index_all = self._build_edge_index(admissible_only=False)

        self.logger.info(f"AccountingStructure initialized with {len(self.root_nodes)} root nodes, time range: {self.time_range}, valuation: {self.valuation}")

    # ---------------------------
    # Excel initialization
    # ---------------------------
    def syst_desc_from_file(self, path):
        self.logger.info(f"Initializing structure from file: {path}")
        file = pd.read_excel(path)
        assert set(['sector_entity', 'region', 'time_range', 'valuation']).issubset(file.columns), \
            "Excel file must contain columns: 'sector_entity', 'region', 'time_range', 'valuation'"

        sector_entity = file['sector_entity'].str.split('_', expand=True)
        sector_entity.columns = ['sector', 'entity']
        region = file['region'].unique()
        root_nodes_ = pd.DataFrame(
            [(s, e, r) for s, e in zip(sector_entity['sector'], sector_entity['entity']) for r in region],
            columns=['sector', 'entity', 'region']
        )
        self.root_nodes = root_nodes_
        self.time_range = file['time_range'].unique().tolist()
        self.valuation = file['valuation'].unique().tolist()
        self.logger.info(f"Loaded {len(self.root_nodes)} root nodes, time range: {self.time_range}, valuation: {self.valuation}")

    # ---------------------------
    # Node index (3-tuple)
    # ---------------------------
    def _build_node_index(self):
        self.logger.info("Building node index (3-tuple nodes)...")
        self.node_to_id = {}
        self.id_to_node = {}
        node_id = 0
        for _, node in self.root_nodes.iterrows():
            key = (node['sector'], node['entity'], node['region'])
            if key not in self.node_to_id:
                self.node_to_id[key] = node_id
                self.id_to_node[node_id] = key
                node_id += 1

        self.n_nodes = node_id
        self.n_timeps = len(self.time_range)
        self.n_valtns = len(self.valuation)
        self.node_index = pd.DataFrame.from_dict(self.id_to_node, orient='index', columns=['sector', 'entity', 'region'])
        self.logger.info(f"Total nodes (3-tuple): {self.n_nodes}")

    # ---------------------------
    # Admissibility mask (nodes only)
    # ---------------------------
    def _build_mask(self):
        self.logger.info("Building admissibility mask (nodes only)...")
        A = np.ones((self.n_nodes, self.n_nodes), dtype=int)
        keys = list(self.id_to_node.values())
        for i, ni in enumerate(keys):
            for j, nj in enumerate(keys):
                for rule in self.rules:
                    ni_dict = dict(zip(['sector','entity','region'], ni))
                    nj_dict = dict(zip(['sector','entity','region'], nj))
                    if not rule(ni_dict, nj_dict):
                        A[i, j] = 0
                        break
        self.logger.info("Admissibility mask built.")
        return sp.csr_matrix(A)

    # ---------------------------
    # Edge index (edges include time & valuation)
    # ---------------------------
    def _build_edge_index(self, admissible_only=True):
        rows, cols, times, vals, edge_ids = [], [], [], [], []
        edge_id = 0
        for i, ni in enumerate(self.id_to_node.values()):
            for j, nj in enumerate(self.id_to_node.values()):
                if admissible_only:
                    mask_pass = all(rule(dict(zip(['sector','entity','region'], ni)),
                                        dict(zip(['sector','entity','region'], nj))) 
                                    for rule in self.rules)
                    if not mask_pass:
                        continue
                for t in self.time_range:
                    for v in self.valuation:
                        rows.append(i)
                        cols.append(j)
                        times.append(t)
                        vals.append(v)
                        edge_ids.append(edge_id)
                        edge_id += 1
        
        
        
        self.n_edges = len(edge_ids)
        
        return pd.DataFrame({
            "edge_id": edge_ids,
            "from_id": rows,
            "to_id": cols,
            "time": times,
            "valuation": vals
        })

    # ---------------------------
    # Compress F_root
    # ---------------------------
    def compress(self, F_root):
        rows = self.edge_index["from_id"].values
        cols = self.edge_index["to_id"].values
        return F_root[rows, cols].A1 if sp.issparse(F_root) else F_root[rows, cols]

    # ---------------------------
    # Lookup
    # ---------------------------
    def decode_node(self, node_id):
        return self.id_to_node[node_id]

    def encode_node(self, sector, entity, region):
        return self.node_to_id[(sector, entity, region)]

    def decode_edge(self, edge_id):
        row = self.edge_index.iloc[edge_id]
        return (self.decode_node(row['from_id']), self.decode_node(row['to_id']), row['time'], row['valuation'])

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, **kwargs):
        path = os.path.join(self.storage_dir, self.id)
        os.makedirs(path, exist_ok=True)
        sp.save_npz(os.path.join(path, "A.npz"), self.A)
        self.edge_index.to_csv(os.path.join(path, "edge_index.csv"), index=False)
        self.edge_index_all.to_csv(os.path.join(path, "edge_index_all.csv"), index=False)
        self.node_index.to_csv(os.path.join(path, "node_index.csv"), index=False)
        self.root_nodes.to_csv(os.path.join(path, "root_nodes.csv"), index=False)
        meta = {"time_range": self.time_range, "valuation": self.valuation,
                "node_to_id": self.node_to_id, "id_to_node": self.id_to_node}
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self.logger.info(f"Structure saved to {path}")

    @classmethod
    def load(cls, id, **kwargs):
        obj = cls.__new__(cls)
        obj.id = id
        obj.storage_dir = kwargs.get('storage_dir', r'data/proc/acc')
        path = os.path.join(obj.storage_dir, id)
        assert os.path.isdir(path), f"Directory {id} not found"
        obj.A = sp.load_npz(os.path.join(path, "A.npz"))
        obj.edge_index = pd.read_csv(os.path.join(path, "edge_index.csv"))
        obj.edge_index_all = pd.read_csv(os.path.join(path, "edge_index_all.csv"))
        obj.root_nodes = pd.read_csv(os.path.join(path, "root_nodes.csv"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)
        obj.time_range = meta["time_range"]
        obj.valuation = meta["valuation"]
        obj.node_to_id = meta["node_to_id"]
        obj.id_to_node = meta["id_to_node"]
        obj.n_nodes = len(obj.id_to_node)
        obj.rules = []
        return obj

if __name__ == "__main__":

    acc_id = "test_structure"
    # -------------------------
    # Define root nodes
    # -------------------------
    root_def = {
        "nodes": pd.DataFrame({
            "sector": ["Mining", "Smelting", "Mining"],
            "entity": ["P", "F", "P" ],
            "region": ["A", "B", "B"]
        }),
        "time_range": [0, 1, 0],              # 2 time steps
        "valuation": ["mass", "copper_c", "mass"]     # 2 valuation types
    }

    # -------------------------
    # Example rule: no P -> P
    # -------------------------
    def no_p_to_p(ni, nj):
        return not (ni["entity"] == "P" and nj["entity"] == "P")
    
    def no_v_to_v(ni, nj):
        return not ni["valuation"] == nj["valuation"]
    
    def no_same_time(ni, nj):
        return not ni["time"] == nj["time"]

    # -------------------------
    # Initialize AccountingStructure
    # -------------------------
    structure = AccountingStructure(id= acc_id, root=root_def, rules=[no_p_to_p, no_v_to_v, no_same_time])

    print("Number of nodes:", structure.n_nodes)
    print("Admissibility mask shape:", structure.A.shape)
    print("Number of admissible edges:", len(structure.edge_index))

    # -------------------------
    # Create random F_root
    # -------------------------
    np.random.seed(42)
    F_root = sp.random(structure.n_nodes, structure.n_nodes, density=0.2, format='csr')

    # -------------------------
    # Save and load structure
    # -------------------------
    structure.save()
    structure_loaded = AccountingStructure.load(id=acc_id)

    print("Loaded nodes:", structure_loaded.root_nodes)
    print("Loaded admissible edges:", len(structure_loaded.edge_index))
    print("Loaded compressed shape:", structure_loaded.A.shape)
