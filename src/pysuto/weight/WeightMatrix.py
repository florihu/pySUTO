import pandas as pd
import numpy as np
import pickle
from typing import Optional
import os
import numpy as np
import pandas as pd
import sparse
from typing import Optional
import warnings

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
                allow_unmatched_edges: bool = True,
                id: str = "ct_from_raw"):


        # --------------------------------------------------
        # 0) Copy raw data & preserve original index
        # --------------------------------------------------
        # This keeps a pointer to the original row for diagnostics
        df = raw_df.copy().reset_index(drop=False).rename(columns={"index": "_raw_idx"})

        # --------------------------------------------------
        # 1) Initialize SRC columns
        # --------------------------------------------------
        # Guarantee these exist for raw→src mapping
        src_cols = [
            "src_sector_from", "src_entity_from", "src_region_from",
            "src_sector_to",   "src_entity_to",   "src_region_to",
            "src_time", "src_valuation"
        ]
        for c in src_cols:
            df[c] = pd.Series([pd.NA] * len(df), dtype="object")

        # --------------------------------------------------
        # 2) RAW → SRC mapping (vectorized, aux-aware)
        # --------------------------------------------------
        # Map each raw variable to the corresponding src column, conditional on auxiliary category
        for _, r in var_map_df.iterrows():
            raw_var = r["raw_var"]
            src_var = r["src_var"]
            aux_val = r[aux_col] if (aux_col and aux_col in r.index) else None

            if aux_val is not None and not pd.isna(aux_val):
                mask = df[aux_col] == aux_val
            else:
                mask = pd.Series(True, index=df.index)

            df.loc[mask, src_var] = df.loc[mask, raw_var].astype("object")

        # --------------------------------------------------
        # 3) Build expanded edge structure
        # --------------------------------------------------
        # Each accounting node has sector/entity/region; edges are joined with node attributes
        node_rows = [
            {"node_id": nid, "sector": t[0], "entity": t[1], "region": t[2]}
            for t, nid in acc_struct.node_to_id.items()
        ]
        node_df = pd.DataFrame(node_rows)

        from_nodes = node_df.rename(columns={
            "node_id": "from_id", "sector": "from_sector",
            "entity": "from_entity", "region": "from_region"
        })
        to_nodes = node_df.rename(columns={
            "node_id": "to_id", "sector": "to_sector",
            "entity": "to_entity", "region": "to_region"
        })

        edges = (
            acc_struct.edge_index
            .merge(from_nodes, on="from_id", how="left")
            .merge(to_nodes, on="to_id", how="left")
        )

        # --------------------------------------------------
        # 4) Prepare mapping tables (multi-mapping safe)
        # --------------------------------------------------
        def prep_map(df_map, src_col, root_col):
            return df_map[[src_col, root_col]].drop_duplicates()

        sec_map = prep_map(sector_map_df, "src_sector", "root_sector")
        reg_map = prep_map(region_map_df, "src_region", "root_region")
        t_map   = prep_map(time_map_df, "src_time", "root_time")
        v_map   = prep_map(valuation_map_df, "src_valuation", "root_valuation")

        # --------------------------------------------------
        # 5) Process Supply / Use separately
        # --------------------------------------------------
        merged_parts = []
        work_parts = []

        def process_side(df_part, side):
            df_work = df_part.copy()

            # Map sector & region per side
            if side == "from":
                df_work = df_work.merge(
                    sec_map.rename(columns={"src_sector": "src_sector_from",
                                            "root_sector": "root_sector_from"}),
                    on="src_sector_from", how="left"
                )
                df_work = df_work.merge(
                    reg_map.rename(columns={"src_region": "src_region_from",
                                            "root_region": "root_region_from"}),
                    on="src_region_from", how="left"
                )
                edge_keys = ["from_sector", "from_region", "time", "valuation"]
                df_keys   = ["root_sector_from", "root_region_from", "root_time", "root_valuation"]

            else:  # USE side
                df_work = df_work.merge(
                    sec_map.rename(columns={"src_sector": "src_sector_to",
                                            "root_sector": "root_sector_to"}),
                    on="src_sector_to", how="left"
                )
                df_work = df_work.merge(
                    reg_map.rename(columns={"src_region": "src_region_to",
                                            "root_region": "root_region_to"}),
                    on="src_region_to", how="left"
                )
                edge_keys = ["to_sector", "to_region", "time", "valuation"]
                df_keys   = ["root_sector_to", "root_region_to", "root_time", "root_valuation"]

            # Merge time & valuation root
            df_work = df_work.merge(t_map, on="src_time", how="left")
            df_work = df_work.merge(v_map, on="src_valuation", how="left")

            # Fail fast if any root mapping missing (mapping issue)
            if df_work[df_keys].isna().any().any():
                raise ValueError(f"Missing src→root mapping in {side}-side rows")

            # --------------------------------------------------
            # 5b) PRE-FILTER infeasible rows
            # --------------------------------------------------
            # Keep only rows that have at least one edge candidate
            # (prevents large inner merges that will always produce zero matches)
            if side == "from":
                edge_match_cols = ["from_sector", "from_region"]
                root_cols = ["root_sector_from", "root_region_from"]
            else:
                edge_match_cols = ["to_sector", "to_region"]
                root_cols = ["root_sector_to", "root_region_to"]

            # quick existence check
            feasible_mask = df_work[root_cols[0]].isin(edges[edge_match_cols[0]]) & \
                            df_work[root_cols[1]].isin(edges[edge_match_cols[1]])

            df_work_filtered = df_work[feasible_mask]
            infeasible_rows = df_work[~feasible_mask]["_raw_idx"].tolist()

            # log warning for pre-filtered rows
            if infeasible_rows:
                msg = (
                    f"{len(infeasible_rows)} rows cannot possibly match any accounting edge "
                    f"(pre-filtered). Examples: {infeasible_rows[:5]}"
                )
                warnings.warn(msg)

            # --------------------------------------------------
            # 6) Merge with edges (only feasible rows)
            # --------------------------------------------------
            merged = edges.merge(
                df_work_filtered,
                left_on=edge_keys,
                right_on=df_keys,
                how="inner",
                suffixes=("_edge", "")
            )

            return merged, df_work_filtered, infeasible_rows

        # run both sides
        unmatched_edge = []

        if aux_col:
            for val in df[aux_col].dropna().unique():
                if str(val).lower() == "supply":
                    m, w, unf = process_side(df[df[aux_col] == val], "from")
                elif str(val).lower() == "use":
                    m, w, unf = process_side(df[df[aux_col] == val], "to")
                else:
                    raise ValueError(f"Unknown {aux_col}: {val}")

                merged_parts.append(m)
                work_parts.append(w)
                unmatched_edge.extend(unf)
        else:
            m, w, unf = process_side(df, "from")
            merged_parts.append(m)
            work_parts.append(w)
            unmatched_edge.extend(unf)

        merged = pd.concat(merged_parts, ignore_index=True)
        df_work_all = pd.concat(work_parts, ignore_index=False)

        # --------------------------------------------------
        # 7) Final unmatched detection
        # --------------------------------------------------
        all_raw = set(df["_raw_idx"])
        matched = set(merged["_raw_idx"])
        unmatched = sorted(list(all_raw - matched - set(unmatched_edge)))

        if unmatched:
            msg = (
                f"{len(unmatched)} rows did not match any edge after merging. "
                f"Examples: {unmatched[:5]}"
            )
            if allow_unmatched_edges:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

        # --------------------------------------------------
        # 8) Populate all remaining fields from edges
        # --------------------------------------------------
        merged["src_sector_from"] = merged["src_sector_from"].fillna(merged["from_sector"])
        merged["src_region_from"] = merged["src_region_from"].fillna(merged["from_region"])
        merged["src_sector_to"]   = merged["src_sector_to"].fillna(merged["to_sector"])
        merged["src_region_to"]   = merged["src_region_to"].fillna(merged["to_region"])

        merged["root_sector_from"] = merged["from_sector"]
        merged["root_region_from"] = merged["from_region"]
        merged["root_sector_to"]   = merged["to_sector"]
        merged["root_region_to"]   = merged["to_region"]

        merged["root_entity_from"] = merged["from_entity"]
        merged["root_entity_to"]   = merged["to_entity"]

        merged["src_entity_from"] = merged["root_entity_from"]
        merged["src_entity_to"]   = merged["root_entity_to"]

        # --------------------------------------------------
        # 9) Build final WeightMatrix table
        # --------------------------------------------------
        tab = pd.DataFrame({
            "src_sector_from": merged["src_sector_from"],
            "src_entity_from": merged["src_entity_from"],
            "src_region_from": merged["src_region_from"],
            "src_sector_to":   merged["src_sector_to"],
            "src_entity_to":   merged["src_entity_to"],
            "src_region_to":   merged["src_region_to"],
            "src_time":        merged["src_time"],
            "src_valuation":   merged["src_valuation"],

            "root_sector_from": merged["root_sector_from"],
            "root_entity_from": merged["root_entity_from"],
            "root_region_from": merged["root_region_from"],
            "root_sector_to":   merged["root_sector_to"],
            "root_entity_to":   merged["root_entity_to"],
            "root_region_to":   merged["root_region_to"],
            "root_time":        merged["root_time"],
            "root_valuation":   merged["root_valuation"],

            "weight_value": np.nan,
            "weight_mode": "uniform",
            # include edge id
            "edge_id": merged["edge_id"]
        }).reset_index(drop=True)

        # --------------------------------------------------
        # 10) Return instance + diagnostics for unmatched edges
        # --------------------------------------------------
        instance = cls(tab=tab, acc_struct=acc_struct, id=id)
        instance.unmatched_raw_rows = unmatched_edge  # rows pre-filtered as impossible
        return instance

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