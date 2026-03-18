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
                sector_map_df: pd.DataFrame,
                region_map_df: pd.DataFrame,
                time_map_df: pd.DataFrame,
                valuation_map_df: pd.DataFrame,
                var_map_df: pd.DataFrame,
                aux_col: Optional[str] = None,
                id: str = "ct_from_raw"):
        """
        Create WeightMatrix from raw data by:
        1) mapping raw columns -> src columns using var_map_df (with optional aux column),
        2) mapping src sector/region -> root sector/region using sector_map_df/region_map_df,
        3) merging (joining) the mapped src combinations against acc_struct.edge_index
            (expanded with node attributes) to produce one output row per matching root edge,
        4) taking root entities/from-to from the matched edge nodes and setting src entities
            to the same implied values (1:1).
        """

        # -----------------------------
        # 0) defensive copy
        # -----------------------------
        df = raw_df.copy()

        # -----------------------------
        # 1) Initialize canonical SRC columns
        # - create all expected src_* columns so mapping assignments work cleanly
        # -----------------------------
        canonical_src_cols = [
            "src_sector_from", "src_entity_from", "src_region_from",
            "src_sector_to",   "src_entity_to",   "src_region_to",
            "src_time", "src_valuation"
        ]
        for c in canonical_src_cols:
            df[c] = np.nan

        # -----------------------------
        # 2) Build the column mapping (col_map)
        # - var_map_df is expected to contain rows with columns:
        #     raw_var, src_var, aux_col (optional)
        #   where aux_col differentiates mappings like 'Supply'/'Use'.
        # - produce a dict: (raw_var, aux_val) -> src_var  and raw_var -> src_var for global rules.
        # -----------------------------
        col_map = {}
        for _, r in var_map_df.iterrows():
            raw_var = r["raw_var"]
            src_var = r["src_var"]
            aux_val = r.get(aux_col) if aux_col in r.index else None
            if aux_val is not None and not (pd.isna(aux_val)):
                col_map[(raw_var, aux_val)] = src_var
            else:
                col_map[raw_var] = src_var

        # -----------------------------
        # 3) Perform raw -> src mapping
        # - For each raw column, choose the target src column per-row:
        #     * if aux_cols provided, mapping can differ row-by-row using (raw_col, aux_val)
        #     * otherwise use the global mapping for raw_col (col_map[raw_col]) if present
        # - Assign raw values into chosen src columns (vectorized via group assignment)
        # -----------------------------
        for raw_col in raw_df.columns:
            if aux_col and aux_col in df.columns:
                # row-wise selection of the target src column using the auxiliary column
                targets = df.apply(
                    lambda row: col_map.get((raw_col, row[aux_col]), col_map.get(raw_col, raw_col)),
                    axis=1
                )
            else:
                # same target for all rows
                targets = pd.Series([col_map.get(raw_col, raw_col)] * len(df), index=df.index)

            # write values into the chosen src columns
            for tgt in targets.unique():
                if tgt in canonical_src_cols:
                    df.loc[targets == tgt, tgt] = df.loc[targets == tgt, raw_col]

        # NOTE: per your instruction there is NO default "to-side" assignment here.
        # If the var_map_df does not assign e.g. src_sector_to, it remains NaN and
        # must be covered by the mapping rules (or will later produce no merges and raise).

        # -----------------------------
        # 4) Map source sector/region -> root sector/region using provided maps
        # - sector_map_df: columns ['src_sector','root_sector']
        # - region_map_df: columns ['src_region','root_region']
        # -----------------------------
        sec_map = dict(zip(sector_map_df["src_sector"], sector_map_df["root_sector"]))
        reg_map = dict(zip(region_map_df["src_region"], region_map_df["root_region"]))
        t_map = dict(zip(time_map_df["src_time"], time_map_df["root_time"]))
        v_map = dict(zip(valuation_map_df["src_valuation"], valuation_map_df["root_valuation"]))

        df["root_time"] = df["src_time"].map(t_map)
        df["root_valuation"] = df["src_valuation"].map(v_map)

        df["root_sector_from"] = df["src_sector_from"].map(sec_map)
        df["root_sector_to"]   = df["src_sector_to"].map(sec_map)
        df["root_region_from"] = df["src_region_from"].map(reg_map)
        df["root_region_to"]   = df["src_region_to"].map(reg_map)

        # Early sanity: if any root sector/region is still NaN, it's a mapping omission -> fail fast
        #missing_root_sector = df[["root_sector_from", "root_sector_to"]].isna().any(axis=1)
        #if missing_root_sector.any():
        #    rows = df.index[missing_root_sector].tolist()
        #    raise ValueError(f"Missing root sector mapping for rows: {rows}")

        #missing_root_region = df[["root_region_from", "root_region_to"]].isna().any(axis=1)
        #if missing_root_region.any():
        #    rows = df.index[missing_root_region].tolist()
        #    raise ValueError(f"Missing root region mapping for rows: {rows}")

        # -----------------------------
        # 5) Expand acc_struct.edge_index with node attributes
        # - acc_struct.node_to_id is dict: (sector, entity, region) -> node_id
        # - create a node table (node_id, sector, entity, region)
        # - join edge_index with node table twice to get 'from_' and 'to_' node attributes
        # Result: edges_expanded with columns:
        #   edge_id, from_id, to_id, time, valuation, from_sector, from_entity, from_region, to_sector, to_entity, to_region
        # -----------------------------
        node_rows = [
            {"node_id": nid, "sector": t[0], "entity": t[1], "region": t[2]}
            for t, nid in acc_struct.node_to_id.items()
        ]
        node_df = pd.DataFrame(node_rows)

        # prepare renamed node tables for left/right join
        from_nodes = node_df.rename(columns={
            "node_id": "from_id", "sector": "from_sector", "entity": "from_entity", "region": "from_region"
        })
        to_nodes = node_df.rename(columns={
            "node_id": "to_id", "sector": "to_sector", "entity": "to_entity", "region": "to_region"
        })

        edges_expanded = (
            acc_struct.edge_index
            .merge(from_nodes, on="from_id", how="left")
            .merge(to_nodes, on="to_id", how="left")
        )

        # -----------------------------
        # 6) Merge mapped source rows with expanded edges using auxiliary column
        # - aux_col specifies which raw column differentiates mapping (e.g., 'Supply_Use')
        # - For each aux_val (e.g., 'Supply' or 'Use'), select the relevant edge side:
        #   - Supply: match root_sector_from/root_region_from → from_sector/from_region
        #   - Use:    match root_sector_to/root_region_to → to_sector/to_region
        # - Always match on root_time → edge.time and root_valuation → edge.valuation
        # - Left merge ensures all raw rows are kept; rows with no matching edge will have NaNs
        # -----------------------------
        merged_list = []

        if aux_col is None:
            # no auxiliary distinction → merge all rows against both sides? (simpler: use from side)
            merged = df.merge(
                edges_expanded,
                left_on=["root_sector_from", "root_region_from", "root_time", "root_valuation"],
                right_on=["from_sector", "from_region", "time", "valuation"],
                how="left",
                indicator=True,
                suffixes=("", "_edge")
            )
        else:
            # split by each auxiliary value
            for aux_val in df[aux_col].unique():
                sub_df = df[df[aux_col] == aux_val]

                if aux_val.lower() == "supply":
                    # match the 'from' side of edges
                    left_cols = ["root_sector_from", "root_region_from", "root_time", "root_valuation"]
                    right_cols = ["from_sector", "from_region", "time", "valuation"]
                elif aux_val.lower() == "use":
                    # match the 'to' side of edges
                    left_cols = ["root_sector_to", "root_region_to", "root_time", "root_valuation"]
                    right_cols = ["to_sector", "to_region", "time", "valuation"]
                else:
                    raise ValueError(f"Unknown auxiliary value '{aux_val}' in column '{aux_col}'")

                merged_sub = sub_df.merge(
                    edges_expanded,
                    left_on=left_cols,
                    right_on=right_cols,
                    how="left",
                    indicator=True,
                    suffixes=("", "_edge")
                )

                merged_list.append(merged_sub)

        # combine all subsets after conditional merge
        merged = pd.concat(merged_list, axis=0, ignore_index=True)

        # -----------------------------
        # 7) Detect unmapped rows (rows that did not match any accounting edge)
        # - Tracks the original raw row index to correctly identify rows with no matching edge
        # - Works regardless of auxiliary column splitting
        # -----------------------------
        # Ensure original row index is preserved
        df["_raw_idx"] = df.index

        # After merging with auxiliary logic
        # `_merge` column added by merge indicates whether the merge was successful
        unmatched_mask = merged["_merge"] == "left_only"

        if unmatched_mask.any():
            # Find unique original row indices that had no match
            bad_raw_indices = merged.loc[unmatched_mask, "_raw_idx"].unique().tolist()
            
            # Optional: collect example diagnostics to help debugging
            # Show which root labels caused the mismatch
            diagnostics_cols = []
            if aux_col is not None:
                # include auxiliary column in diagnostics
                diagnostics_cols = [aux_col]
            diagnostics_cols += [
                "root_sector_from", "root_region_from",
                "root_sector_to", "root_region_to",
                "root_time", "root_valuation"
            ]
            diagnostics = merged.loc[unmatched_mask, diagnostics_cols].drop_duplicates().to_dict(orient="records")
            
            raise ValueError(
                f"Some mapped source rows did not match any root edge. "
                f"Original raw row indices: {bad_raw_indices}; example diagnostics: {diagnostics[:5]}"
            )    


        # -----------------------------
        # 8) Populate root and implied src entity columns using matched node attributes
        # - from_entity/to_entity come from the matched edge's node attributes
        # - src_entity_from/to are implied 1:1 from root entities
        # -----------------------------
        merged["root_entity_from"] = merged["from_entity"]
        merged["root_entity_to"]   = merged["to_entity"]

        # Imply source entities from root entities (1:1 relationship)
        merged["src_entity_from"] = merged["root_entity_from"]
        merged["src_entity_to"]   = merged["root_entity_to"]

        # -----------------------------
        # 9) Build final tab DataFrame (one row per matched root edge)
        # - select the columns WeightMatrix expects in its pipeline
        # - weight_value: initially NaN, weight_mode: 'uniform' (imputation step will fill if needed)
        # -----------------------------
        tab = pd.DataFrame({
            # SRC
            "src_sector_from": merged["src_sector_from"],
            "src_entity_from": merged["src_entity_from"],
            "src_region_from": merged["src_region_from"],
            "src_sector_to":   merged["src_sector_to"],
            "src_entity_to":   merged["src_entity_to"],
            "src_region_to":   merged["src_region_to"],
            "src_time":        merged["src_time"],
            "src_valuation":   merged["src_valuation"],

            # ROOT (from/to entity taken from matched nodes)
            "root_sector_from": merged["root_sector_from"],
            "root_entity_from": merged["root_entity_from"],
            "root_region_from": merged["root_region_from"],
            "root_sector_to":   merged["root_sector_to"],
            "root_entity_to":   merged["root_entity_to"],
            "root_region_to":   merged["root_region_to"],
            "root_time":        merged["root_time"],
            "root_valuation":   merged["root_valuation"],

            # weights (to be imputed/validated downstream)
            "weight_value": np.nan,
            "weight_mode": "uniform"
        }).reset_index(drop=True)


        # -----------------------------
        # Verify no NaNs in indexing columns
        # -----------------------------
        index_cols = [
            "src_sector_from", "src_entity_from", "src_region_from",
            "src_sector_to", "src_entity_to", "src_region_to",
            "src_time", "src_valuation",
            "root_sector_from", "root_entity_from", "root_region_from",
            "root_sector_to", "root_entity_to", "root_region_to",
            "root_time", "root_valuation"
        ]

        # Check for any NaNs in these columns
        nan_mask = tab[index_cols].isna()
        if nan_mask.any().any():
            # Identify columns that contain NaNs
            cols_with_nan = nan_mask.any()[nan_mask.any()].index.tolist()
            # Optionally, show how many NaNs per column
            nan_counts = tab[cols_with_nan].isna().sum().to_dict()
            raise ValueError(
                f"NaNs detected in WeightMatrix indexing columns! Columns with NaNs: {cols_with_nan}\n"
                f"Counts per column: {nan_counts}"
    )
        # -----------------------------
        # 10) Return instance (rest of WeightMatrix pipeline will validate / impute / build matrix)
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