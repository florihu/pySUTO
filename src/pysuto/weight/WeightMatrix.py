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
                aux_cols: Optional[str] = None,
                id: str = "ct_from_raw"):
        """
        Create WeightMatrix directly from raw data + correspondence tables.

        New features:
        - aux_cols: optional column in raw_df that differentiates how raw columns
            are mapped to src columns (e.g. 'Supply_Use' with values 'Supply'|'Use').
        - var_map_df can contain:
            * column mapping instructions: dict with keys (raw_col, aux_val) -> src_col
                or DataFrame with columns ['raw_var','aux_val','src_var'] or
                DataFrame with ['raw_var','src_var'] (global map).
            * valuation value mapping: DataFrame with ['raw_val','root_valuation']
                or dict accessible via key 'value_map'.
        """
        # -----------------------------
        # DEFAULT RAW -> SRC column mapping (fallback)
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
        # Build mapping tables from var_map_df
        # -----------------------------
        # col_map: (raw_col, aux_val) -> src_col  OR raw_col -> src_col
        col_map = {}
        # val_map: raw_val (Unit) -> root_valuation
        val_map = {}

        if var_map_df is not None:
            # dict form (user provided)
            if isinstance(var_map_df, dict):
                # tuple keys -> column mapping
                for k, v in var_map_df.items():
                    if isinstance(k, tuple):
                        col_map[k] = v
                    elif isinstance(k, str):
                        # simple raw_col -> src_col mapping
                        col_map[k] = v
                # optional value map under a reserved key
                if "value_map" in var_map_df and isinstance(var_map_df["value_map"], dict):
                    val_map = var_map_df["value_map"]

            # DataFrame form
            elif isinstance(var_map_df, pd.DataFrame):
                cols = set(var_map_df.columns)
                # column mapping with aux_val
                if {"raw_var", "aux_val", "src_var"}.issubset(cols):
                    for _, r in var_map_df.iterrows():
                        col_map[(r["raw_var"], r["aux_val"])] = r["src_var"]
                # column mapping without aux_val (global)
                if {"raw_var", "src_var"}.issubset(cols) and "aux_val" not in cols:
                    for _, r in var_map_df.iterrows():
                        col_map[r["raw_var"]] = r["src_var"]
                # value mapping for valuations
                if {"raw_val", "root_valuation"}.issubset(cols):
                    val_map = dict(zip(var_map_df["raw_val"], var_map_df["root_valuation"]))

        # -----------------------------
        # MAP RAW → SRC (column-level mapping, row-wise if aux_cols provided)
        # -----------------------------
        # initialize canonical src columns to NaN (so we guarantee their existence)
        canonical_src_cols = [
            "src_sector_from", "src_entity_from", "src_region_from",
            "src_sector_to",   "src_entity_to",   "src_region_to",
            "src_time", "src_valuation"
        ]
        for c in canonical_src_cols:
            df[c] = np.nan

        # For each raw column, determine for each row which src column it should fill,
        # based on (raw_col, aux_val) mapping, otherwise raw_col->src fallback mapping.
        for raw_col in raw_to_src_map.keys():
            if raw_col not in df.columns:
                continue

            # pick target src column per-row
            if aux_cols is not None and aux_cols in df.columns:
                # compute vector of target column names
                targets = df.apply(
                    lambda row: col_map.get((raw_col, row[aux_cols]),
                                            col_map.get(raw_col, raw_to_src_map[raw_col])),
                    axis=1
                )
            else:
                # no aux column: global mapping or default
                target_col_global = col_map.get(raw_col, raw_to_src_map[raw_col])
                targets = pd.Series([target_col_global] * len(df), index=df.index)

            # assign raw values into the chosen target columns
            for tgt in targets.unique():
                # ensure tgt is one of canonical_src_cols; if not, still assign (keeps flexible)
                df.loc[targets == tgt, tgt] = df.loc[targets == tgt, raw_col]

        # -----------------------------
        # Default "to" side for any remaining missing values (same logic as before)
        # -----------------------------
        # If src_sector_to wasn't filled by mapping, set to "Market"
        df["src_sector_to"] = df["src_sector_to"].fillna("Market")
        df["src_entity_to"] = df["src_entity_to"].fillna("X")
        # If src_region_to is still NaN, default to src_region_from
        df["src_region_to"] = df["src_region_to"].fillna(df["src_region_from"])

        # -----------------------------
        # VARIABLE / VALUATION VALUE MAPPING (Unit -> root_valuation)
        # -----------------------------
        # src_valuation currently holds the raw 'Unit' value (unless remapped above)
        if val_map:
            df["src_valuation"] = df["src_valuation"].map(val_map).fillna(df["src_valuation"])

        # -----------------------------
        # CORRESPONDENCE (SRC → ROOT) — identical to previous implementation
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
        # TIME & VALUATION (propagate)
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