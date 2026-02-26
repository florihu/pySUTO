
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import os

import logging


class AccountingStructure:
    def __init__(self, id, root=None, rules=None, **kwargs):
        self.rules = rules or []
        self.id = id or "accounting_structure"
        self.storage_dir = kwargs.get('storage_dir', r'data/proc/acc')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize nodes
        if kwargs.get('init_from_file', False):
            assert 'root_file_path' in kwargs
            self.syst_desc_from_file(kwargs['root_file_path'])
           
        else:
            assert root is not None
            self.root_nodes = root['nodes'].copy()
            self.time_range = root['time_range']
            self.valuation = root['valuation']

        # Build integer node IDs
        self._build_node_index()

        # Build admissibility mask
        self.A = self._build_mask()

        # Build edge index (admissible only)
        self.edge_index = self._build_edge_index()

        # General edge index (all combinations)
        self.edge_index_all = self._build_edge_index(admissible_only=False)

        self.logger.info(f"AccountingStructure initialized with {len(self.root_nodes)} root nodes, time range: {self.time_range}, valuation: {self.valuation}")

    # ---------------------------
    # Excel initialization
    # ---------------------------
    def syst_desc_from_file(self, path):

        self.logger.info(f"Initializing structure from file: {path}")

        file = pd.read_excel(path)

        # assert excel file has sheet with columns: 'sector_entity', 'region', 'time_range', 'valuation'
        assert set(['sector_entity', 'region', 'time_range', 'valuation']).issubset(file.columns), \
            "Excel file must contain columns: 'sector_entity', 'region', 'time_range', 'valuation'"

        sector_entity = file['sector_entity'].str.split('_', expand=True)
        sector_entity.columns = ['sector', 'entity']
        region = file['region'].unique()
        root_nodes_ = pd.DataFrame(
            [(s, e, r) for s, e in zip(sector_entity['sector'], sector_entity['entity'])
             for r in region],
            columns=['sector', 'entity', 'region']
        )
        self.root_nodes = root_nodes_
        self.time_range = file['time_range'].unique().tolist()
        self.valuation = file['valuation'].unique().tolist()

        self.logger.info(f"Loaded {len(self.root_nodes)} root nodes, time range: {self.time_range}, valuation: {self.valuation}")

    # ---------------------------
    # Node index with time & valuation
    # ---------------------------
    def _build_node_index(self):
        self.logger.info("Building node index...")
        self.node_to_id = {}
        self.id_to_node = {}
        node_id = 0
        for _, node in self.root_nodes.iterrows():
            for t in self.time_range:
                for v in self.valuation:
                    key = (node['sector'], node['entity'], node['region'], t, v)
                    self.node_to_id[key] = node_id
                    self.id_to_node[node_id] = key
                    node_id += 1
        self.n_nodes = node_id

        self.node_index = pd.DataFrame.from_dict(self.id_to_node, orient='index', columns=['sector', 'entity', 'region', 'time', 'valuation'])
        self.logger.info(f"Total nodes (with time & valuation): {self.n_nodes}")


    # ---------------------------
    # Admissibility mask
    # ---------------------------
    def _build_mask(self):
        self.logger.info("Building admissibility mask...")
        A = np.ones((self.n_nodes, self.n_nodes), dtype=int)
        keys = list(self.id_to_node.values())
        for i, ni in enumerate(keys):
            for j, nj in enumerate(keys):
                for rule in self.rules:
                    ni_dict = dict(zip(['sector','entity','region','time','valuation'], ni))
                    nj_dict = dict(zip(['sector','entity','region','time','valuation'], nj))
                    if not rule(ni_dict, nj_dict):
                        A[i, j] = 0
                        break
        self.logger.info("Admissibility mask built.")
        return sp.csr_matrix(A)
        

    # ---------------------------
    # Edge index
    # ---------------------------
    def _build_edge_index(self, admissible_only=True):
        self.logger.info(f"Building edge index (admissible_only={admissible_only})...")
        if admissible_only:
            mask = self.A
        else:
            mask = sp.csr_matrix(np.ones((self.n_nodes, self.n_nodes), dtype=int))
        rows, cols = mask.nonzero()
        edge_ids = np.arange(len(rows))
        self.logger.info(f"Total edges in index: {len(edge_ids)}")
        return pd.DataFrame({
            "edge_id": edge_ids,
            "from_id": rows,
            "to_id": cols
        })
    

    # ---------------------------
    # Compress F_root to 1D vector
    # ---------------------------
    def compress(self, F_root):
        F_masked = F_root.multiply(self.A)
        rows = self.edge_index["from_id"].values
        cols = self.edge_index["to_id"].values
        return F_masked[rows, cols].A1

    # ---------------------------
    # Lookup
    # ---------------------------
    def decode_node(self, node_id):
        return self.id_to_node[node_id]

    def encode_node(self, sector, entity, region, time, valuation):
        return self.node_to_id[(sector, entity, region, time, valuation)]

    def decode_edge(self, edge_id):
        from_id, to_id = self.edge_index.iloc[edge_id][['from_id','to_id']]
        return self.decode_node(from_id), self.decode_node(to_id)

    # ---------------------------
    # Save/Load methods
    # ---------------------------
    def save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        path = os.path.join(self.storage_dir, self.id)
        # Save sparse matrix
        sp.save_npz(os.path.join(path, "A.npz"), self.A)
        # Save DataFrames and dicts
        self.edge_index.to_csv(os.path.join(path, "edge_index.csv"), index=False)
        self.node_index.to_csv(os.path.join(path, "node_index.csv"), index=False)
        self.edge_index_all.to_csv(os.path.join(path, "edge_index_all.csv"), index=False)
        self.root_nodes.to_csv(os.path.join(path, "root_nodes.csv"), index=False)
        meta = {
            "time_range": self.time_range,
            "valuation": self.valuation,
            "node_to_id": self.node_to_id,
            "id_to_node": self.id_to_node,
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self.logger.info(f"Structure saved to {path}")

    @classmethod
    def load(cls, path):
        # Create empty instance
        obj = cls.__new__(cls)
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
        obj.rules = []  # rules are not saved; can be re-added if needed
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
    structure.save("test_structure/")
    structure_loaded = AccountingStructure.load("test_structure/")

    print("Loaded nodes:", structure_loaded.root_nodes)
    print("Loaded admissible edges:", len(structure_loaded.edge_index))
    print("Loaded compressed shape:", structure_loaded.A.shape)
