import pandas as pd
import numpy as np
import sparse  # pip install sparse
import pickle
from typing import Optional

class ConcordanceTableOptimized:
    """
    Concordance Table that maps source nodes to root nodes in an AccountingStructure,
    and builds a multi-dimensional sparse weight tensor including time & valuation.
    """

    def __init__(self,
                 tab: pd.DataFrame,
                 acc_struct,
                 prop_shares: Optional[pd.DataFrame] = None,
                 id: str = "concordance_table"):
        """
        Parameters
        ----------
        tab : pd.DataFrame
            Concordance table. Must contain columns:
            ['data_sector_from','data_entity_from','data_region_from','data_time','data_valuation',
             'root_sector_from','root_entity_from','root_region_from','root_time','root_valuation',
             'weight_value','weight_mode']
        acc_struct : AccountingStructure
            The system's accounting structure defining all root nodes.
        prop_shares : pd.DataFrame, optional
            Table of shares for manufacturing_shares mode. Columns: ['Region_from','Region_to','Share']
        id : str
            Identifier for this concordance table.
        """
        self.id = id
        self.tab = tab.copy()
        self.acc_struct = acc_struct
        self.prop_shares = prop_shares

        # Validate the concordance table
        self._validate_table()

        # Impute weights according to weight_mode
        self._impute_weights()

        # Validate that weights sum to 1 per source node
        self._validate_weight_sums()

        # Build multi-dimensional sparse tensor
        self._build_weight_tensor()

    # ---------------------------
    # Step 1: Validation
    # ---------------------------
    def _validate_table(self):
        # Required columns
        required_cols = [
            'data_sector_from','data_entity_from','data_region_from','data_time','data_valuation',
            'root_sector_from','root_entity_from','root_region_from','root_time','root_valuation',
            'weight_value','weight_mode'
        ]
        missing = [c for c in required_cols if c not in self.tab.columns]
        if missing:
            raise ValueError(f"Missing columns in concordance table: {missing}")

        # Validate root nodes exist in accounting structure
        root_nodes = self.acc_struct.root_nodes
        # Check root entities
        missing_entities = set(self.tab['root_entity_from']).difference(set(root_nodes['entity']))
        if missing_entities:
            raise ValueError(f"Root entities not found in accounting structure: {missing_entities}")

        # Validate weight_mode values
        valid_modes = {"user", "uniform", "manufacturing_shares"}
        invalid_modes = set(self.tab['weight_mode'].unique()) - valid_modes
        if invalid_modes:
            raise ValueError(f"Invalid weight_mode(s) found: {invalid_modes}")

    # ---------------------------
    # Step 2: Impute Weights
    # ---------------------------
    def _impute_weights(self):
        src_cols = ['data_sector_from','data_entity_from','data_region_from','data_time','data_valuation']

        # ----- Uniform mode -----
        mask_uniform = self.tab['weight_mode'] == 'uniform'
        if mask_uniform.any():
            grp_counts = self.tab[mask_uniform].groupby(src_cols).size().rename('n_targets')
            self.tab = self.tab.merge(grp_counts.reset_index(), how='left', on=src_cols)
            self.tab.loc[mask_uniform, 'weight_value'] = 1.0 / self.tab.loc[mask_uniform, 'n_targets']
            self.tab.drop(columns=['n_targets'], inplace=True)

        # ----- Manufacturing shares mode -----
        mask_manu = self.tab['weight_mode'] == 'manufacturing_shares'
        if mask_manu.any():
            if self.prop_shares is None:
                raise ValueError("prop_shares must be provided for manufacturing_shares mode")
            ps = self.prop_shares.rename(columns={
                'Region_from': 'data_region_from',
                'Region_to': 'root_region_from',
                'Share': 'prop_share'
            })
            self.tab = self.tab.merge(ps[['data_region_from','root_region_from','prop_share']],
                                      on=['data_region_from','root_region_from'], how='left')

            # Normalize per source node
            total_share = self.tab.groupby(src_cols)['prop_share'].transform('sum')
            self.tab.loc[mask_manu, 'weight_value'] = self.tab.loc[mask_manu, 'prop_share']/total_share
            self.tab.drop(columns=['prop_share'], inplace=True)

        # ----- User mode -----
        mask_user = self.tab['weight_mode'] == 'user'
        if mask_user.any():
            if self.tab.loc[mask_user, 'weight_value'].isnull().any():
                raise ValueError("User mode weights must be provided and non-null")

    # ---------------------------
    # Step 3: Validate weight sums
    # ---------------------------
    def _validate_weight_sums(self, rtol=1e-6, atol=1e-9):
        src_cols = ['data_sector_from','data_entity_from','data_region_from','data_time','data_valuation']
        sums = self.tab.groupby(src_cols)['weight_value'].sum()
        bad = sums[~np.isclose(sums,1.0,rtol=rtol,atol=atol)]
        if not bad.empty:
            raise ValueError(f"Weights do not sum to 1 for sources:\n{bad}")

    # ---------------------------
    # Step 4: Build multi-dimensional sparse tensor
    # ---------------------------
    def _build_weight_tensor(self):
        src_labels = ['data_sector_from','data_entity_from','data_region_from','data_time','data_valuation']
        root_labels = ['root_sector_from','root_entity_from','root_region_from','root_time','root_valuation']

        # Build unique value lists for each axis
        src_unique = [self.tab[c].unique() for c in src_labels]
        root_unique = [self.acc_struct.root_nodes[c.replace('root_','')].unique() for c in root_labels]

        # Build label -> index mappings
        self.src_index = {c: {v:i for i,v in enumerate(vals)} for c,vals in zip(src_labels, src_unique)}
        self.root_index = {c: {v:i for i,v in enumerate(vals)} for c,vals in zip(root_labels, root_unique)}

        # Initialize coordinates and data
        coords = [[] for _ in range(len(src_labels)+len(root_labels))]
        data = []

        for _, row in self.tab.iterrows():
            for i, c in enumerate(src_labels):
                coords[i].append(self.src_index[c][row[c]])
            for j, c in enumerate(root_labels):
                coords[len(src_labels)+j].append(self.root_index[c][row[c]])
            data.append(float(row['weight_value']))

        shape = [len(u) for u in src_unique + root_unique]
        self.weight_tensor = sparse.COO(np.array(coords), np.array(data), shape=shape)

    # ---------------------------
    # Step 5: Save / Load tensor
    # ---------------------------
    def store_tensor(self, path):
        """Store sparse tensor to file using pickle"""
        with open(path,'wb') as f:
            pickle.dump({
                'tensor': self.weight_tensor,
                'src_index': self.src_index,
                'root_index': self.root_index
            }, f)

    @staticmethod
    def load_tensor(path):
        """Load sparse tensor from file"""
        with open(path,'rb') as f:
            d = pickle.load(f)
        obj = ConcordanceTableOptimized.__new__(ConcordanceTableOptimized)
        obj.weight_tensor = d['tensor']
        obj.src_index = d['src_index']
        obj.root_index = d['root_index']
        return obj

    # ---------------------------
    # Step 6: Slice tensor for computation
    # ---------------------------
    def get_dense_slice(self,
                        time_idx=None,
                        valuation_idx=None):
        """
        Returns a dense numpy slice of the weight tensor
        for specific time and valuation indices.
        If None, uses all indices along that axis.
        """
        s = []
        for axis, c in enumerate(['data_sector_from','data_entity_from','data_region_from','data_time','data_valuation']):
            if c == 'data_time' and time_idx is not None:
                s.append(time_idx)
            elif c == 'data_valuation' and valuation_idx is not None:
                s.append(valuation_idx)
            else:
                s.append(slice(None))
        for _ in range(5):  # root axes
            s.append(slice(None))
        return self.weight_tensor[tuple(s)].todense()