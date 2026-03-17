import pandas as pd
import numpy as np
import pickle
from typing import Optional
import os
import numpy as np
import pandas as pd
import sparse
from typing import Optional

from pysuto.logging import get_logger

l = get_logger("WeightMatrix")


class WeightMatrix:
    """
    Edge-based concordance matrix:

    W[src_edge, root_edge]

    - src_edge: external classification
    - root_edge: AccountingStructure edge space
    """

    def __init__(self,
                 tab: pd.DataFrame,
                 acc_struct,
                 prop_shares: Optional[pd.DataFrame] = None,
                 id: str = "ct_test"):

        self.id = id
        self.tab = tab.copy()
        self.acc_struct = acc_struct
        self.prop_shares = prop_shares
        self.acc_used = acc_struct.id

        self.src_cols = [
            'src_sector_from','src_entity_from','src_region_from',
            'src_sector_to','src_entity_to','src_region_to',
            'src_time','src_valuation'
        ]

        self.root_cols = [
            'root_sector_from','root_entity_from','root_region_from',
            'root_sector_to','root_entity_to','root_region_to',
            'root_time','root_valuation'
        ]

        self.odir = "data/proc/conc"

        # Pipeline
        self._validate_table()
        self._impute_weights()
        self._validate_weight_sums()
        self._build_src_index()
        self._build_weight_matrix()

    # =====================================================
    # VALIDATION
    # =====================================================
    def _validate_table(self):

        required_cols = self.src_cols + self.root_cols + ['weight_value','weight_mode']
        missing = [c for c in required_cols if c not in self.tab.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        valid_modes = {"user", "uniform", "manufacturing_shares"}
        invalid_modes = set(self.tab['weight_mode'].unique()) - valid_modes
        if invalid_modes:
            raise ValueError(f"Invalid weight_mode(s): {invalid_modes}")

        # Validate root edges only
        edge_df = self.acc_struct.edge_index

        for _, r in self.tab.iterrows():
            root_from = (r['root_sector_from'], r['root_entity_from'], r['root_region_from'])
            root_to   = (r['root_sector_to'],   r['root_entity_to'],   r['root_region_to'])

            if root_from not in self.acc_struct.node_to_id:
                raise ValueError(f"Invalid root node: {root_from}")
            if root_to not in self.acc_struct.node_to_id:
                raise ValueError(f"Invalid root node: {root_to}")

            from_id = self.acc_struct.node_to_id[root_from]
            to_id   = self.acc_struct.node_to_id[root_to]

            exists = (
                (edge_df['from_id'] == from_id) &
                (edge_df['to_id'] == to_id) &
                (edge_df['time'] == r['root_time']) &
                (edge_df['valuation'] == r['root_valuation'])
            ).any()

            if not exists:
                raise ValueError(f"Invalid root edge: {root_from} → {root_to}")

    # =====================================================
    # WEIGHT IMPUTATION
    # =====================================================
    def _impute_weights(self):

        mask_uniform = self.tab['weight_mode'] == 'uniform'
        if mask_uniform.any():
            grp = self.tab[mask_uniform].groupby(self.src_cols).size().rename('n')
            self.tab = self.tab.merge(grp.reset_index(), on=self.src_cols, how='left')
            self.tab.loc[mask_uniform, 'weight_value'] = 1.0 / self.tab.loc[mask_uniform, 'n']
            self.tab.drop(columns=['n'], inplace=True)

        mask_user = self.tab['weight_mode'] == 'user'
        if mask_user.any():
            if self.tab.loc[mask_user, 'weight_value'].isnull().any():
                raise ValueError("User weights must be provided")

    # =====================================================
    # VALIDATE SUMS
    # =====================================================
    def _validate_weight_sums(self):
        sums = self.tab.groupby(self.src_cols)['weight_value'].sum()
        if not np.allclose(sums.values, 1.0):
            raise ValueError(f"Weights do not sum to 1:\n{sums}")

    # =====================================================
    # SOURCE EDGE INDEX (NEW)
    # =====================================================
    def _build_src_index(self):

        self.src_edges = (
            self.tab[self.src_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.src_edge_to_id = {
            tuple(row): i for i, row in self.src_edges.iterrows()
        }

        self.n_src_edges = len(self.src_edges)
        self.n_root_edges = len(self.acc_struct.edge_index)

    # =====================================================
    # ROOT EDGE LOOKUP
    # =====================================================
    def _get_root_edge_id(self, r):

        from_tuple = (r['root_sector_from'], r['root_entity_from'], r['root_region_from'])
        to_tuple   = (r['root_sector_to'],   r['root_entity_to'],   r['root_region_to'])

        from_id = self.acc_struct.node_to_id[from_tuple]
        to_id   = self.acc_struct.node_to_id[to_tuple]

        edge = self.acc_struct.edge_index[
            (self.acc_struct.edge_index['from_id'] == from_id) &
            (self.acc_struct.edge_index['to_id'] == to_id) &
            (self.acc_struct.edge_index['time'] == r['root_time']) &
            (self.acc_struct.edge_index['valuation'] == r['root_valuation'])
        ]

        return int(edge.iloc[0]['edge_id'])

    # =====================================================
    # BUILD MATRIX
    # =====================================================
    def _build_weight_matrix(self):

        coords = [[], []]
        data = []

        for _, r in self.tab.iterrows():

            # SRC edge
            src_key = tuple(r[c] for c in self.src_cols)
            src_id = self.src_edge_to_id[src_key]

            # ROOT edge
            root_id = self._get_root_edge_id(r)

            coords[0].append(src_id)
            coords[1].append(root_id)
            data.append(float(r['weight_value']))

        shape = (self.n_src_edges, self.n_root_edges)

        self.weight_matrix = sparse.COO(coords, data, shape=shape)

    
    # =====================================================
    # APPLY
    # =====================================================
    def apply(self, F_src):
        """
        F_src: vector over src_edge

        Returns:
        vector over root_edge
        """
        return self.weight_matrix.T @ F_src
    

    # =====================================================
    # FACTORY: BUILD FROM RAW DATA
    # =====================================================
    @classmethod
    def from_raw(cls,
                raw_df: pd.DataFrame,
                acc_struct,
                correspondence_df: Optional[pd.DataFrame] = None,
                var_map_df: Optional[pd.DataFrame] = None,
                id: str = "ct_from_raw"):
        """
        Create WeightMatrix directly from raw data + correspondence tables
        """

        # -----------------------------
        # DEFAULT COLUMN MAPPING
        # -----------------------------
        raw_to_src_map = {
            "Process": "src_sector_from",
            "Type": "src_entity_from",
            "Region": "src_region_from",
            "Year": "src_time",
            "Unit": "src_valuation"
        }

        df = raw_df.copy()

        # -----------------------------
        # MAP RAW → SRC
        # -----------------------------
        for raw_col, src_col in raw_to_src_map.items():
            if raw_col in df.columns:
                df[src_col] = df[raw_col]
            else:
                df[src_col] = np.nan

        # Default "to" side (can be refined later)
        df["src_sector_to"] = "Market"
        df["src_entity_to"] = "X"
        df["src_region_to"] = df["src_region_from"]

        # -----------------------------
        # VARIABLE / VALUATION MAPPING
        # -----------------------------
        if var_map_df is not None:
            val_map = dict(zip(var_map_df['raw_val'], var_map_df['root_valuation']))
            df["src_valuation"] = df["src_valuation"].map(val_map).fillna(df["src_valuation"])

        # -----------------------------
        # CORRESPONDENCE (SRC → ROOT)
        # -----------------------------
        if correspondence_df is not None:

            # sector mapping
            if {"src_sector", "root_sector"}.issubset(correspondence_df.columns):
                sec_map = dict(zip(correspondence_df["src_sector"], correspondence_df["root_sector"]))
                df["root_sector_from"] = df["src_sector_from"].map(sec_map).fillna(df["src_sector_from"])
                df["root_sector_to"]   = df["src_sector_to"].map(sec_map).fillna(df["src_sector_to"])
            else:
                df["root_sector_from"] = df["src_sector_from"]
                df["root_sector_to"]   = df["src_sector_to"]

            # region mapping
            if {"src_region", "root_region"}.issubset(correspondence_df.columns):
                reg_map = dict(zip(correspondence_df["src_region"], correspondence_df["root_region"]))
                df["root_region_from"] = df["src_region_from"].map(reg_map).fillna(df["src_region_from"])
                df["root_region_to"]   = df["src_region_to"].map(reg_map).fillna(df["src_region_to"])
            else:
                df["root_region_from"] = df["src_region_from"]
                df["root_region_to"]   = df["src_region_to"]

            # entity (1:1 assumption)
            df["root_entity_from"] = df["src_entity_from"]
            df["root_entity_to"]   = df["src_entity_to"]

        else:
            # identity fallback
            df["root_sector_from"] = df["src_sector_from"]
            df["root_sector_to"]   = df["src_sector_to"]
            df["root_region_from"] = df["src_region_from"]
            df["root_region_to"]   = df["src_region_to"]
            df["root_entity_from"] = df["src_entity_from"]
            df["root_entity_to"]   = df["src_entity_to"]

        # -----------------------------
        # TIME & VALUATION
        # -----------------------------
        df["root_time"] = df["src_time"]
        df["root_valuation"] = df["src_valuation"]

        # -----------------------------
        # BUILD TAB
        # -----------------------------
        tab = pd.DataFrame({
            # SRC
            "src_sector_from": df["src_sector_from"],
            "src_entity_from": df["src_entity_from"],
            "src_region_from": df["src_region_from"],
            "src_sector_to":   df["src_sector_to"],
            "src_entity_to":   df["src_entity_to"],
            "src_region_to":   df["src_region_to"],
            "src_time":        df["src_time"],
            "src_valuation":   df["src_valuation"],

            # ROOT
            "root_sector_from": df["root_sector_from"],
            "root_entity_from": df["root_entity_from"],
            "root_region_from": df["root_region_from"],
            "root_sector_to":   df["root_sector_to"],
            "root_entity_to":   df["root_entity_to"],
            "root_region_to":   df["root_region_to"],
            "root_time":        df["root_time"],
            "root_valuation":   df["root_valuation"],

            # weights
            "weight_value": np.nan,
            "weight_mode": "uniform"
        })

        # -----------------------------
        # RETURN INSTANCE
        # -----------------------------
        return cls(tab=tab, acc_struct=acc_struct, id=id)
        

    # =====================================================
    # SAVE / LOAD
    # =====================================================
    def save(self, **kwargs):
        os.makedirs(self.odir, exist_ok=True)
        path = kwargs.get('path', os.path.join(self.odir, f"{self.id}.pkl"))
        with open(path, "wb") as f:
            pickle.dump({
                "id": self.id,
                "acc_id": self.acc_used,
                "tab": self.tab,
                "weight_matrix": self.weight_matrix,
                "src_edges": self.src_edges
            }, f)

    @classmethod
    def load(cls, id, acc):
        path = os.path.join("data/proc/conc", f"{id}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls.__new__(cls)
        obj.id = data['id']
        obj.tab = data['tab']
        obj.acc_used = data['acc_id']
        obj.weight_matrix = data['weight_matrix']
        obj.src_edges = data['src_edges']
        obj.acc_struct = acc
        return obj