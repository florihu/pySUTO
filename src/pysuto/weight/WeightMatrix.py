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
        Create WeightMatrix from raw data.

        Workflow:
        1) Map raw columns -> src columns using var_map_df (aux_col may alter per-row mapping).
        2) Map src sector/region/time/valuation -> root equivalents using provided maps.
        3) Expand acc_struct.edge_index with node attributes (from/to sector/entity/region).
        4) **Edge-driven** merge: for each aux category (Supply/Use) merge EDGES -> mapped rows
            so that a supply row (which specifies 'from' side) expands to all matching 'to' sides.
        5) Fail if any original raw row had zero matched edges (diagnostics included).
        6) Fill any remaining src/root indexing columns from matched edge attributes so that
            no indexing column is NaN.
        7) Build final tab and return WeightMatrix instance.

        Notes:
        - sector_map_df, region_map_df, time_map_df, valuation_map_df and var_map_df are required.
        - var_map_df must contain columns: raw_var, src_var, and optionally aux_col (the name of the
        aux column value to switch mapping).
        """

        # Defensive copy
        df = raw_df.copy()

        # 1) Initialize canonical SRC columns (guarantee existence)
        canonical_src_cols = [
            "src_sector_from", "src_entity_from", "src_region_from",
            "src_sector_to",   "src_entity_to",   "src_region_to",
            "src_time", "src_valuation"
        ]
        for c in canonical_src_cols:
            df[c] = np.nan

        # 2) Build col_map from var_map_df
        #    var_map_df rows expected: raw_var, src_var, (optional) aux_col
        #    Create mapping: (raw_var, aux_val) -> src_var  and raw_var -> src_var (global)
        col_map = {}
        for _, r in var_map_df.iterrows():
            raw_var = r["raw_var"]
            src_var = r["src_var"]
            # if var_map_df used the literal column name 'aux_col' to denote the aux category value,
            # read it from that column (user-provided DF should use that column name)
            aux_val = r.get(aux_col) if aux_col in r.index else None
            if aux_val is not None and not (pd.isna(aux_val)):
                col_map[(raw_var, aux_val)] = src_var
            else:
                col_map[raw_var] = src_var

        # 3) Map raw -> src (per-row if aux_col is provided)
        #    For each raw column determine target src column (may differ by aux_val).
        #    Assign the raw value into the chosen src column.
        for raw_col in raw_df.columns:
            if aux_col and aux_col in df.columns:
                targets = df.apply(
                    lambda row: col_map.get((raw_col, row[aux_col]),
                                            col_map.get(raw_col, raw_col)),
                    axis=1
                )
            else:
                targets = pd.Series([col_map.get(raw_col, raw_col)] * len(df), index=df.index)

            for tgt in targets.unique():
                if tgt in canonical_src_cols:
                    df.loc[targets == tgt, tgt] = df.loc[targets == tgt, raw_col]

        # 4) Map source -> root for time & valuation & sector & region
        #    Use the provided mapping dataframes to translate src values into root values.
        t_map = dict(zip(time_map_df["src_time"], time_map_df["root_time"]))
        v_map = dict(zip(valuation_map_df["src_valuation"], valuation_map_df["root_valuation"]))
        sec_map = dict(zip(sector_map_df["src_sector"], sector_map_df["root_sector"]))
        reg_map = dict(zip(region_map_df["src_region"], region_map_df["root_region"]))

        df["root_time"] = df["src_time"].map(t_map)
        df["root_valuation"] = df["src_valuation"].map(v_map)

        df["root_sector_from"] = df["src_sector_from"].map(sec_map)
        df["root_sector_to"]   = df["src_sector_to"].map(sec_map)
        df["root_region_from"] = df["src_region_from"].map(reg_map)
        df["root_region_to"]   = df["src_region_to"].map(reg_map)

        # 5) Prepare expanded edges (edge rows with node attributes)
        node_rows = [
            {"node_id": nid, "sector": t[0], "entity": t[1], "region": t[2]}
            for t, nid in acc_struct.node_to_id.items()
        ]
        node_df = pd.DataFrame(node_rows)

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

        # 6) Edge-driven merge (edges -> df) depending on auxiliary side
        #    We're merging EDGES onto the mapped rows so that a Supply row which defines
        #    only the 'from' side will expand across all matching 'to' sides in the edges.
        merged_list = []

        # preserve original row indices for diagnostics
        df["_raw_idx"] = df.index

        # Helper to perform an inner join of edges -> sub_df on the chosen side columns
        def _merge_edges_to_rows(edges, rows, side):
            """
            edges: edges_expanded
            rows: subset of df (contains _raw_idx)
            side: "from" or "to" - decides which edge columns to match
            returns merged_sub (edges columns + rows columns) via inner join
            """
            if side == "from":
                left_on = ["from_sector", "from_region", "time", "valuation"]
                right_on = ["root_sector_from", "root_region_from", "root_time", "root_valuation"]
            else:
                left_on = ["to_sector", "to_region", "time", "valuation"]
                right_on = ["root_sector_to", "root_region_to", "root_time", "root_valuation"]

            # edges on left, rows on right: this expands rows by matching edges
            merged_sub = edges.merge(rows, left_on=left_on, right_on=right_on, how="inner", suffixes=("_edge", ""))
            return merged_sub

        if aux_col is None:
            # no auxiliary: default to matching on 'from' side (i.e., treat as Supply)
            merged_sub = _merge_edges_to_rows(edges_expanded, df, side="from")
            merged_list.append(merged_sub)
        else:
            # iterate over unique auxiliary values present in the mapped DataFrame
            for aux_val in df[aux_col].dropna().unique():
                sub_df = df[df[aux_col] == aux_val]
                if str(aux_val).lower() == "supply":
                    merged_sub = _merge_edges_to_rows(edges_expanded, sub_df, side="from")
                elif str(aux_val).lower() == "use":
                    merged_sub = _merge_edges_to_rows(edges_expanded, sub_df, side="to")
                else:
                    raise ValueError(f"Unknown auxiliary value '{aux_val}' in column '{aux_col}'")
                merged_list.append(merged_sub)

        # concat results (each merged_sub contains edge attributes and original mapped columns)
        if len(merged_list) == 0:
            # no merged subsets -> nothing matched
            raise ValueError("No auxiliary groups produced merges; check var_map_df / aux_col logic.")
        merged = pd.concat(merged_list, axis=0, ignore_index=True)

        # 7) Detect unmapped original rows
        #    If any original raw row index (_raw_idx) is missing from merged, it means
        #    that particular input row didn't match any accounting edge → fail.
        all_raw_idxs = set(df["_raw_idx"].unique().tolist())
        matched_raw_idxs = set(merged["_raw_idx"].unique().tolist())
        unmatched_raw_idxs = sorted(list(all_raw_idxs - matched_raw_idxs))
        if unmatched_raw_idxs:
            # include diagnostics: show original mapping columns for the offending inputs
            diag_cols = [aux_col] if aux_col is not None else []
            diag_cols += [
                "src_sector_from", "src_region_from", "src_sector_to", "src_region_to",
                "root_sector_from", "root_region_from", "root_sector_to", "root_region_to",
                "root_time", "root_valuation"
            ]
            diagnostics = df.loc[unmatched_raw_idxs, diag_cols].to_dict(orient="index")
            raise ValueError(
                f"Some mapped source rows did not match any root edge. Original raw row indices: {unmatched_raw_idxs}\n"
                f"Example diagnostics: {list(diagnostics.items())[:5]}"
            )

        # 8) Populate root/from-to attributes from matched edges where necessary
        #    After merging edges->rows, edges provide 'from_*' and 'to_*' columns.
        #    Ensure both root_from and root_to columns are filled (take from edge if missing),
        #    then ensure src_* side columns are filled (use edge attributes if user mapping left them empty).
        merged["root_sector_from"] = merged["root_sector_from"].fillna(merged["from_sector"])
        merged["root_region_from"] = merged["root_region_from"].fillna(merged["from_region"])
        merged["root_sector_to"]   = merged["root_sector_to"].fillna(merged["to_sector"])
        merged["root_region_to"]   = merged["root_region_to"].fillna(merged["to_region"])

        # Fill src-side columns where they were left NaN by mapping rules:
        # - For supply rows (which were matched on 'from'), src_sector_to/src_region_to are often NaN:
        #   fill them from the matched edge's to_sector/to_region so the source edge is complete.
        merged["src_sector_from"] = merged["src_sector_from"].fillna(merged["from_sector"])
        merged["src_region_from"] = merged["src_region_from"].fillna(merged["from_region"])
        merged["src_sector_to"]   = merged["src_sector_to"].fillna(merged["to_sector"])
        merged["src_region_to"]   = merged["src_region_to"].fillna(merged["to_region"])

        # 9) Populate entities: entities come from the matched nodes on the edge.
        merged["root_entity_from"] = merged["from_entity"]
        merged["root_entity_to"]   = merged["to_entity"]

        # Source entities are implied 1:1 from root entities
        merged["src_entity_from"] = merged["root_entity_from"]
        merged["src_entity_to"]   = merged["root_entity_to"]

        # Also ensure time & valuation are present (they were used for matching so they should be present)
        merged["root_time"] = merged["root_time"].fillna(merged["time"])
        merged["root_valuation"] = merged["root_valuation"].fillna(merged["valuation"])
        merged["src_time"] = merged["src_time"].fillna(merged["root_time"])
        merged["src_valuation"] = merged["src_valuation"].fillna(merged["root_valuation"])

        # 10) Build final tab DataFrame (one row per matched root edge)
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

            # ROOT
            "root_sector_from": merged["root_sector_from"],
            "root_entity_from": merged["root_entity_from"],
            "root_region_from": merged["root_region_from"],
            "root_sector_to":   merged["root_sector_to"],
            "root_entity_to":   merged["root_entity_to"],
            "root_region_to":   merged["root_region_to"],
            "root_time":        merged["root_time"],
            "root_valuation":   merged["root_valuation"],

            # weights
            "weight_value": np.nan,
            "weight_mode": "uniform"
        }).reset_index(drop=True)

        # 11) Final safety check: no NaNs in indexing columns (fail with diagnostics if any)
        index_cols = [
            "src_sector_from", "src_entity_from", "src_region_from",
            "src_sector_to", "src_entity_to", "src_region_to",
            "src_time", "src_valuation",
            "root_sector_from", "root_entity_from", "root_region_from",
            "root_sector_to", "root_entity_to", "root_region_to",
            "root_time", "root_valuation"
        ]

        nan_mask = tab[index_cols].isna()
        if nan_mask.any().any():
            cols_with_nan = nan_mask.any()[nan_mask.any()].index.tolist()
            nan_counts = tab[cols_with_nan].isna().sum().to_dict()
            # Provide sample rows with NaNs for debugging
            sample_problem_rows = tab[tab[cols_with_nan].isna().any(axis=1)].head(10).to_dict(orient="records")
            raise ValueError(
                f"NaNs detected in WeightMatrix indexing columns! Columns: {cols_with_nan}\n"
                f"Counts: {nan_counts}\n"
                f"Sample problematic rows: {sample_problem_rows}"
            )

        # 12) Return instance: the rest of the WeightMatrix pipeline handles weights and building sparse matrix
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