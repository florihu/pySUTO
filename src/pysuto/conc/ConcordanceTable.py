import pandas as pd
import numpy as np
import sparse  # pip install sparse
import pickle
from typing import Optional
import os
import numpy as np
import pandas as pd
import sparse
from typing import Optional

from pysuto.logging import get_logger

l = get_logger("ConcordanceTable")


class ConcordanceTable:
    """
    Edge-based concordance tensor:

    W[src_edge, time, valuation, root_edge]

    Uses AccountingStructure for:
    - node validation
    - edge validation
    - edge indexing
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

        # Define columns
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

        #output dir
        self.odir = "data/proc/conc"

        # Pipeline
        self._validate_table()
        self._impute_weights()
        self._validate_weight_sums()
        self._build_indices()
        self._build_weight_tensor()

    # =====================================================
    # VALIDATION
    # =====================================================
    def _validate_table(self):

        required_cols = self.src_cols + self.root_cols + ['weight_value','weight_mode']

        missing = [c for c in required_cols if c not in self.tab.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # ---- Validate weight modes ----
        valid_modes = {"user", "uniform", "manufacturing_shares"}
        invalid_modes = set(self.tab['weight_mode'].unique()) - valid_modes
        if invalid_modes:
            raise ValueError(f"Invalid weight_mode(s): {invalid_modes}")

        # ---- Validate root nodes + edges ----
        missing_nodes = []
        missing_edges = []

        edge_df = self.acc_struct.edge_index

        for _, r in self.tab.iterrows():

            root_from = (
                r['root_sector_from'],
                r['root_entity_from'],
                r['root_region_from'],
                r['root_time'],
                r['root_valuation']
            )

            root_to = (
                r['root_sector_to'],
                r['root_entity_to'],
                r['root_region_to'],
                r['root_time'],
                r['root_valuation']
            )

            if root_from not in self.acc_struct.node_to_id:
                missing_nodes.append(("from", root_from))
                continue

            if root_to not in self.acc_struct.node_to_id:
                missing_nodes.append(("to", root_to))
                continue

            from_id = self.acc_struct.node_to_id[root_from]
            to_id   = self.acc_struct.node_to_id[root_to]

            exists = (
                (edge_df['from_id'] == from_id) &
                (edge_df['to_id'] == to_id)
            ).any()

            if not exists:
                missing_edges.append((root_from, root_to))

        if missing_nodes:
            raise ValueError(f"Invalid root nodes:\n{missing_nodes}")

        if missing_edges:
            raise ValueError(f"Invalid root edges (not admissible):\n{missing_edges}")

    # =====================================================
    # WEIGHT IMPUTATION
    # =====================================================
    def _impute_weights(self):

        # ----- Uniform -----
        mask_uniform = self.tab['weight_mode'] == 'uniform'
        if mask_uniform.any():
            grp = self.tab[mask_uniform].groupby(self.src_cols).size().rename('n')
            self.tab = self.tab.merge(grp.reset_index(), on=self.src_cols, how='left')
            self.tab.loc[mask_uniform, 'weight_value'] = 1.0 / self.tab.loc[mask_uniform, 'n']
            self.tab.drop(columns=['n'], inplace=True)

        # ----- Manufacturing shares -----
        mask_manu = self.tab['weight_mode'] == 'manufacturing_shares'
        if mask_manu.any():

            if self.prop_shares is None:
                raise ValueError("prop_shares required for manufacturing_shares")

            ps = self.prop_shares.rename(columns={
                'Region_from': 'src_region_from',
                'Region_to': 'root_region_from',
                'Share': 'prop_share'
            })

            self.tab = self.tab.merge(
                ps[['src_region_from','root_region_from','prop_share']],
                on=['src_region_from','root_region_from'],
                how='left'
            )

            total = self.tab.groupby(self.src_cols)['prop_share'].transform('sum')
            self.tab.loc[mask_manu, 'weight_value'] = self.tab.loc[mask_manu, 'prop_share'] / total

            self.tab.drop(columns=['prop_share'], inplace=True)

        # ----- User -----
        mask_user = self.tab['weight_mode'] == 'user'
        if mask_user.any():
            if self.tab.loc[mask_user, 'weight_value'].isnull().any():
                raise ValueError("User weights must be provided")

    # =====================================================
    # VALIDATE SUMS
    # =====================================================
    def _validate_weight_sums(self, rtol=1e-6, atol=1e-9):

        sums = self.tab.groupby(self.src_cols)['weight_value'].sum()
        bad = sums[~np.isclose(sums, 1.0, rtol=rtol, atol=atol)]

        if not bad.empty:
            raise ValueError(f"Weights do not sum to 1:\n{bad}")

    # =====================================================
    # BUILD INDICES
    # =====================================================
    def _build_indices(self):

        self.time_vals = sorted(self.tab['src_time'].unique())
        self.val_vals  = sorted(self.tab['src_valuation'].unique())

        self.time_index = {v:i for i,v in enumerate(self.time_vals)}
        self.val_index  = {v:i for i,v in enumerate(self.val_vals)}

    # =====================================================
    # EDGE LOOKUP
    # =====================================================
    def _get_edge_id(self, from_tuple, to_tuple):

        from_id = self.acc_struct.node_to_id[from_tuple]
        to_id   = self.acc_struct.node_to_id[to_tuple]

        edge = self.acc_struct.edge_index[
            (self.acc_struct.edge_index['from_id'] == from_id) &
            (self.acc_struct.edge_index['to_id'] == to_id)
        ]

        if edge.empty:
            raise ValueError(f"Edge not found: {from_tuple} → {to_tuple}")

        return int(edge.iloc[0]['edge_id'])

    # =====================================================
    # BUILD TENSOR
    # =====================================================
    def _build_weight_tensor(self):

        coords = [[], [], [], []]
        data = []

        for _, r in self.tab.iterrows():

            # SRC edge
            src_from = (
                r['src_sector_from'],
                r['src_entity_from'],
                r['src_region_from'],
                r['src_time'],
                r['src_valuation']
            )

            src_to = (
                r['src_sector_to'],
                r['src_entity_to'],
                r['src_region_to'],
                r['src_time'],
                r['src_valuation']
            )

            src_edge = self._get_edge_id(src_from, src_to)

            # ROOT edge
            root_from = (
                r['root_sector_from'],
                r['root_entity_from'],
                r['root_region_from'],
                r['root_time'],
                r['root_valuation']
            )

            root_to = (
                r['root_sector_to'],
                r['root_entity_to'],
                r['root_region_to'],
                r['root_time'],
                r['root_valuation']
            )

            root_edge = self._get_edge_id(root_from, root_to)

            coords[0].append(src_edge)
            coords[1].append(self.time_index[r['src_time']])
            coords[2].append(self.val_index[r['src_valuation']])
            coords[3].append(root_edge)

            data.append(float(r['weight_value']))

        shape = (
            len(self.acc_struct.edge_index),
            len(self.time_vals),
            len(self.val_vals),
            len(self.acc_struct.edge_index)
        )

        self.weight_tensor = sparse.COO(coords, data, shape=shape)

    # =====================================================
    # APPLY
    # =====================================================
    def apply(self, F_src):
        """
        Transform source flow matrix → root flow tensor

        Returns:
        (time, valuation, root_edge)
        """

        v_src = self.acc_struct.compress(F_src)  # edge vector

        result = sparse.tensordot(
            self.weight_tensor,
            v_src,
            axes=([0], [0])
        )

        return result  # shape: (time, valuation, root_edge)
    
    # =====================================================
    # Store and Load
    # =====================================================
    def save(self, **kwargs):
        os.makedirs(self.odir, exist_ok=True)

        if 'id' in kwargs:
            self.id = kwargs['id']

        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = os.path.join(self.odir, f"{self.id}.pkl")

        with open(path, "wb") as f:
            pickle.dump({
                "id": self.id,
                "tab": self.tab,
                "weight_tensor": self.weight_tensor
            }, f)
        l.info(f"ConcordanceTable saved to {path}")


    @classmethod
    def load(cls, id, acc):
        path = os.path.join("data/proc/conc", f"{id}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.id = data['id']
        obj.tab = data['tab']
        obj.weight_tensor = data['weight_tensor']
        obj.acc_struct = acc
        l.info(f"ConcordanceTable loaded from {path}")
        return obj


    
    


