from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp



from pysuto.feed import DataFeed

# =============================================================================
# CONCORDANCE TABLE
# =============================================================================

class ConcordanceTable(DataFeed):
    """
    Stores and validates the raw src→root edge mapping table.

    Owns:
      - tab          : DataFrame with src_* / root_* / weight_value / weight_mode
      - src_edges    : deduplicated src-edge index
      - src_edge_to_id
    """

    SRC_COLS = [
        'src_sector_from', 'src_entity_from', 'src_region_from',
        'src_sector_to',   'src_entity_to',   'src_region_to',
        'src_time',        'src_valuation',
    ]
    ROOT_COLS = [
        'root_sector_from', 'root_entity_from', 'root_region_from',
        'root_sector_to',   'root_entity_to',   'root_region_to',
        'root_time',        'root_valuation',
    ]

    def __init__(self,
                 tab: pd.DataFrame,
                 acc_struct,
                 id: str = "ct_test",
                 odir: str = "data/proc/conc",
                 _skip_validation: bool = False):

        super().__init__(id=id, acc_struct=acc_struct, odir=odir)

        self.tab = tab.copy()

        # Pipeline — validation can be skipped when from_raw already
        # guarantees correctness via the edge-join construction.
        if not _skip_validation:
            self._validate_table()
        self._impute_weights()
        self._validate_weight_sums()
        self._build_src_index()

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def _validate_table(self):
        required = self.SRC_COLS + self.ROOT_COLS + ['weight_value', 'weight_mode']
        missing = [c for c in required if c not in self.tab.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        valid_modes = {"user", "uniform", "manufacturing_shares"}
        invalid = set(self.tab['weight_mode'].unique()) - valid_modes
        if invalid:
            raise ValueError(f"Invalid weight_mode(s): {invalid}")

        # --- Node validation (vectorised set lookup) -------------------------
        valid_nodes = set(self.acc_struct.node_to_id.keys())

        bad_from = self.tab[
            ~self.tab.apply(
                lambda r: (r['root_sector_from'], r['root_entity_from'], r['root_region_from'])
                          in valid_nodes, axis=1)
        ]
        if not bad_from.empty:
            ex = tuple(bad_from.iloc[0][['root_sector_from','root_entity_from','root_region_from']])
            raise ValueError(f"Invalid root node (from): {ex} — {len(bad_from)} bad rows")

        bad_to = self.tab[
            ~self.tab.apply(
                lambda r: (r['root_sector_to'], r['root_entity_to'], r['root_region_to'])
                          in valid_nodes, axis=1)
        ]
        if not bad_to.empty:
            ex = tuple(bad_to.iloc[0][['root_sector_to','root_entity_to','root_region_to']])
            raise ValueError(f"Invalid root node (to): {ex} — {len(bad_to)} bad rows")

        # --- Edge validation (single vectorised merge) -----------------------
        # Build from_id / to_id columns, deduplicate, then merge once against
        # edge_index — replaces one full 854k-row scan per concordance row.
        node_to_id = self.acc_struct.node_to_id
        tab_check = self.tab[['root_sector_from','root_entity_from','root_region_from',
                               'root_sector_to',  'root_entity_to',  'root_region_to',
                               'root_time',        'root_valuation']].drop_duplicates().copy()

        tab_check['from_id'] = tab_check.apply(
            lambda r: node_to_id[(r['root_sector_from'], r['root_entity_from'], r['root_region_from'])],
            axis=1)
        tab_check['to_id'] = tab_check.apply(
            lambda r: node_to_id[(r['root_sector_to'], r['root_entity_to'], r['root_region_to'])],
            axis=1)

        edge_keys = self.acc_struct.edge_index[['from_id','to_id','time','valuation']].copy()
        edge_keys['time'] = edge_keys['time'].astype(str)
        edge_keys = edge_keys.drop_duplicates()

        tab_keys = (tab_check[['from_id','to_id','root_time','root_valuation']]
                    .rename(columns={'root_time': 'time', 'root_valuation': 'valuation'}))

        matched = tab_keys.merge(edge_keys, on=['from_id','to_id','time','valuation'],
                                 how='left', indicator=True)
        bad = matched[matched['_merge'] != 'both']
        if not bad.empty:
            ex = bad.iloc[0]
            raise ValueError(
                f"Invalid root edge: from_id={ex['from_id']} → to_id={ex['to_id']} "
                f"time={ex['time']} val={ex['valuation']} — {len(bad)} bad combos")

    # ------------------------------------------------------------------ #
    # Weight imputation                                                    #
    # ------------------------------------------------------------------ #

    def _impute_weights(self):
        # Validate user weights are populated
        mask_user = self.tab['weight_mode'] == 'user'
        if mask_user.any() and self.tab.loc[mask_user, 'weight_value'].isnull().any():
            raise ValueError("weight_mode='user' rows must have weight_value set")

        # Uniform rows contribute equally (weight 1.0) before normalization.
        # Normalizing the whole group by its sum handles three cases uniformly:
        #   B  all-uniform  : all get 1/N
        #   A  all-user     : user values divided by their total
        #   C  mixed        : user values + 1.0 per uniform row, then normalized
        self.tab.loc[self.tab['weight_mode'] == 'uniform', 'weight_value'] = 1.0
        grp_sum = self.tab.groupby(self.SRC_COLS)['weight_value'].transform('sum')
        self.tab['weight_value'] = self.tab['weight_value'] / grp_sum

    def _validate_weight_sums(self):
        sums = self.tab.groupby(self.SRC_COLS)['weight_value'].sum()
        if not np.allclose(sums.values, 1.0):
            raise ValueError(f"Weights do not sum to 1:\n{sums}")

    # ------------------------------------------------------------------ #
    # Src-edge index                                                       #
    # ------------------------------------------------------------------ #

    def _build_src_index(self):
        self.src_edges = (
            self.tab[self.SRC_COLS]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # to_numpy is ~10× faster than iterrows for building the lookup dict
        arr = self.src_edges[self.SRC_COLS].to_numpy()
        self.src_edge_to_id = {tuple(row): i for i, row in enumerate(arr)}
        self.n_src_edges  = len(self.src_edges)
        self.n_root_edges = len(self.acc_struct.edge_index)

    # ------------------------------------------------------------------ #
    # from_raw factory                                                     #
    # ------------------------------------------------------------------ #
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
                id: str = "ct_from_raw",
                user_weights_df: Optional[pd.DataFrame] = None):


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
        # Normalise time to str — acc_struct stores int64 but all mapping
        # tables and raw data use string year labels ("1995", "1996", …).
        # Must happen before any merge on the "time" / "root_time" key.
        edges["time"] = edges["time"].astype(str)
        # --------------------------------------------------
        # 4) Prepare mapping tables (multi-mapping safe)
        # --------------------------------------------------
        def prep_map(df_map, src_col, root_col):
            return df_map[[src_col, root_col]].drop_duplicates()

        sec_map = prep_map(sector_map_df, "src_sector", "root_sector")
        reg_map = prep_map(region_map_df, "src_region", "root_region")
        t_map   = prep_map(time_map_df, "src_time", "root_time")
        t_map["root_time"] = t_map["root_time"].astype(str)
        v_map   = prep_map(valuation_map_df, "src_valuation", "root_valuation")

        # --------------------------------------------------
        # 5) Process Supply / Use separately
        # --------------------------------------------------
        merged_parts = []
        work_parts = []

        def process_side(df_part, side):
            df_work = df_part.copy()

            # --------------------------------------------------
            # Map BOTH sides (from and to) to root classifications
            # so the edge join can be constrained on both endpoints.
            # --------------------------------------------------

            # from-side
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

            # to-side (skip merge when column is entirely absent/NaN — dtype mismatch guard)
            if df_work["src_sector_to"].notna().any():
                df_work = df_work.merge(
                    sec_map.rename(columns={"src_sector": "src_sector_to",
                                            "root_sector": "root_sector_to"}),
                    on="src_sector_to", how="left"
                )
            else:
                df_work["root_sector_to"] = np.nan
            if df_work["src_region_to"].notna().any():
                df_work = df_work.merge(
                    reg_map.rename(columns={"src_region": "src_region_to",
                                            "root_region": "root_region_to"}),
                    on="src_region_to", how="left"
                )
            else:
                df_work["root_region_to"] = np.nan

            # time & valuation
            df_work = df_work.merge(t_map, on="src_time",      how="left")
            df_work = df_work.merge(v_map, on="src_valuation", how="left")

            # Which root columns are the authoritative join keys for this side,
            # and which are constraints from the opposite side?
            if side == "from":
                # Primary key: from-side (what was explicitly classified in raw data)
                # Constraint:  to-side may be NA if not provided — skip if so
                edge_keys = ["from_sector", "from_region", "time", "valuation"]
                df_keys   = ["root_sector_from", "root_region_from", "root_time", "root_valuation"]
                primary_root_cols   = ["root_sector_from", "root_region_from"]
                primary_edge_cols   = ["from_sector",      "from_region"]
                secondary_root_cols = ["root_sector_to",   "root_region_to"]
                secondary_edge_cols = ["to_sector",        "to_region"]
            else:
                edge_keys = ["to_sector", "to_region", "time", "valuation"]
                df_keys   = ["root_sector_to", "root_region_to", "root_time", "root_valuation"]
                primary_root_cols   = ["root_sector_to",   "root_region_to"]
                primary_edge_cols   = ["to_sector",        "to_region"]
                secondary_root_cols = ["root_sector_from", "root_region_from"]
                secondary_edge_cols = ["from_sector",      "from_region"]

            # Fail fast if primary-side root mapping is missing
            if df_work[df_keys].isna().any().any():
                raise ValueError(f"Missing src→root mapping on primary side ({side}) rows")

            # --------------------------------------------------
            # 5b) PRE-FILTER on primary side (joint sector+region pair)
            # --------------------------------------------------
            valid_pairs = (
                edges[[primary_edge_cols[0], primary_edge_cols[1]]]
                .drop_duplicates()
                .rename(columns={
                    primary_edge_cols[0]: primary_root_cols[0],
                    primary_edge_cols[1]: primary_root_cols[1],
                })
            )
            feasible_mask = (
                df_work[primary_root_cols]
                .merge(valid_pairs, on=primary_root_cols, how="left", indicator=True)
                ["_merge"].eq("both").values
            )
            df_work_filtered = df_work[feasible_mask]

            # --------------------------------------------------
            # 6) Merge with edges on primary-side keys
            # --------------------------------------------------
            merged = edges.merge(
                df_work_filtered,
                left_on=edge_keys,
                right_on=df_keys,
                how="inner",
                suffixes=("_edge", "")
            )

            # --------------------------------------------------
            # 6b) Filter on secondary-side root classification
            # --------------------------------------------------
            # The raw data may classify the opposite endpoint (e.g. Type=Ore in a Use row).
            # After the edge join we must drop any matched edge whose opposite-side
            # attributes don't agree with the root classification derived from that
            # raw column — otherwise edges are accepted purely because the primary
            # endpoint matched, regardless of what the other endpoint actually is.
            for src_col, edge_col in zip(secondary_root_cols, secondary_edge_cols):
                has_constraint = merged[src_col].notna()
                mismatch       = merged[src_col] != merged[edge_col]
                merged = merged[~(has_constraint & mismatch)]

            return merged, df_work_filtered

        if aux_col:
            for val in df[aux_col].dropna().unique():
                if str(val).lower() == "supply":
                    m, w = process_side(df[df[aux_col] == val], "from")
                elif str(val).lower() == "use":
                    m, w = process_side(df[df[aux_col] == val], "to")
                else:
                    raise ValueError(f"Unknown {aux_col}: {val}")

                merged_parts.append(m)
                work_parts.append(w)
        else:
            m, w = process_side(df, "from")
            merged_parts.append(m)
            work_parts.append(w)

        merged      = pd.concat(merged_parts, ignore_index=True)
        df_work_all = pd.concat(work_parts,   ignore_index=False)

        # --------------------------------------------------
        # 7) Unmatched detection
        # --------------------------------------------------
        # A raw row is unmatched if its _raw_idx never appears in the merged result —
        # regardless of whether it was dropped by the pre-filter or the inner join.
        unmatched = sorted(set(df["_raw_idx"]) - set(merged["_raw_idx"]))

        if unmatched:
            msg = (
                f"{len(unmatched)} raw rows are not associated with any edge. "
                f"Examples: {unmatched[:5]}"
            )
            if allow_unmatched_edges:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

        # --------------------------------------------------
        # 8) Populate all remaining fields from edges
        # --------------------------------------------------
        # root_* always come from the matched edge attributes
        merged["root_sector_from"] = merged["from_sector"]
        merged["root_region_from"] = merged["from_region"]
        merged["root_entity_from"] = merged["from_entity"]
        merged["root_sector_to"]   = merged["to_sector"]
        merged["root_region_to"]   = merged["to_region"]
        merged["root_entity_to"]   = merged["to_entity"]

        # src_* fall back to the edge attribute if the raw data did not classify
        # the opposite endpoint (e.g. Supply rows don't classify the to-side)
        merged["src_sector_from"] = merged["src_sector_from"].fillna(merged["from_sector"])
        merged["src_region_from"] = merged["src_region_from"].fillna(merged["from_region"])
        merged["src_sector_to"]   = merged["src_sector_to"].fillna("<any>")
        merged["src_region_to"]   = merged["src_region_to"].fillna("<any>")

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
            "edge_id": merged["edge_id"],
            "raw_id" : merged["_raw_idx"]
        }).reset_index(drop=True)

        # --------------------------------------------------
        # 10) Return instance + diagnostics for unmatched edges
        # --------------------------------------------------
        # --------------------------------------------------
        # 9b) Apply user-specified weights (optional)
        # --------------------------------------------------
        # user_weights_df columns: src_sector_from, root_sector_to, weight
        # Rows matching (src_sector_from, root_sector_to) get weight_mode='user'
        # and the supplied raw weight. _impute_weights() normalises all weights
        # per src_edge group, treating uniform rows as weight=1 before normalising.
        if user_weights_df is not None and not user_weights_df.empty:
            tab = tab.merge(
                user_weights_df[['root_sector_from', 'root_sector_to', 'weight']]
                .rename(columns={'weight': '_uw'}),
                on=['root_sector_from', 'root_sector_to'],
                how='left',
            )
            mask = tab['_uw'].notna()
            tab.loc[mask, 'weight_value'] = tab.loc[mask, '_uw']
            tab.loc[mask, 'weight_mode']  = 'user'
            tab.drop(columns=['_uw'], inplace=True)

        # tab was built from a correct edge-join — skip redundant re-validation
        instance = cls(tab=tab, acc_struct=acc_struct, id=id, _skip_validation=True)
        instance.unmatched_raw_rows = unmatched
        return instance

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, **kwargs):
        os.makedirs(self.odir, exist_ok=True)
        path = kwargs.get('path', os.path.join(self.odir, f"{self.id}.pkl"))
        with open(path, "wb") as f:
            pickle.dump({
                "id":         self.id,
                "acc_id":     self.acc_used,
                "tab":        self.tab,
                "src_edges":  self.src_edges,
            }, f)

    @classmethod
    def load(cls, id: str, acc_struct, **kwargs) -> "ConcordanceTable":
        odir = kwargs.get('odir', "data/proc/conc")
        path = kwargs.get('path', os.path.join(odir, f"{id}.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls.__new__(cls)
        # Bypass __init__; restore state directly
        obj.id          = data['id']
        obj.odir        = odir
        obj.acc_struct  = acc_struct
        obj.acc_used    = data['acc_id']
        obj.tab         = data['tab']
        obj.src_edges   = data['src_edges']
        obj.src_edge_to_id = {tuple(row): i for i, row in data['src_edges'].iterrows()}
        obj.n_src_edges  = len(obj.src_edges)
        obj.n_root_edges = len(acc_struct.edge_index)
        return obj


# =============================================================================
# WEIGHT MATRIX
# =============================================================================

class WeightMatrix(DataFeed):
    """
    Sparse W[src_edge, root_edge] matrix built from a ConcordanceTable.

    Receives a fully constructed ConcordanceTable; does not rebuild it.
    """

    def __init__(self,
                 concordance: ConcordanceTable,
                 prop_shares: Optional[pd.DataFrame] = None,
                 id: Optional[str] = None,
                 odir: str = "data/proc/conc"):

        # Inherit acc_struct and odir from the concordance table
        super().__init__(
            id=id or concordance.id,
            acc_struct=concordance.acc_struct,
            odir=odir,
        )

        self.concordance  = concordance
        self.prop_shares  = prop_shares

        self._build_weight_matrix()

    # ------------------------------------------------------------------ #
    # Root-edge lookup (internal helper)                                  #
    # ------------------------------------------------------------------ #

    # _get_root_edge_id removed — superseded by vectorised merge in _build_weight_matrix

    # ------------------------------------------------------------------ #
    # Matrix construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_weight_matrix(self):
        ct = self.concordance
        node_to_id = self.acc_struct.node_to_id

        # ── Step 1: map each concordance row's root nodes → from_id / to_id ──
        # Using a map Series is ~100× faster than iterrows for large tab.
        tab = ct.tab.copy()

        # Build (sector, entity, region) tuple columns for from/to, then map to int id.
        tab['_from_id'] = tab.apply(
            lambda r: node_to_id[(r['root_sector_from'], r['root_entity_from'], r['root_region_from'])],
            axis=1)
        tab['_to_id'] = tab.apply(
            lambda r: node_to_id[(r['root_sector_to'], r['root_entity_to'], r['root_region_to'])],
            axis=1)

        # ── Step 2: resolve root edge_id via a single vectorised merge ──────
        edge_lookup = self.acc_struct.edge_index[
            ['from_id','to_id','time','valuation','edge_id']
        ].copy()
        edge_lookup['time'] = edge_lookup['time'].astype(str)

        tab = tab.merge(
            edge_lookup.rename(columns={
                'from_id': '_from_id', 'to_id': '_to_id',
                'time': 'root_time', 'valuation': 'root_valuation',
                'edge_id': '_root_edge_id',
            }),
            on=['_from_id', '_to_id', 'root_time', 'root_valuation'],
            how='left',
        )

        if tab['_root_edge_id'].isna().any():
            n_bad = tab['_root_edge_id'].isna().sum()
            raise ValueError(f"Could not resolve root edge_id for {n_bad} concordance rows")

        # ── Step 3: resolve src_id via dict lookup (already O(1) per row) ──
        tab['_src_id'] = tab.apply(
            lambda r: ct.src_edge_to_id[tuple(r[c] for c in ct.SRC_COLS)], axis=1)

        # ── Step 4: build COO sparse matrix ─────────────────────────────────
        src_ids  = tab['_src_id'].to_numpy(dtype=np.int64)
        root_ids = tab['_root_edge_id'].to_numpy(dtype=np.int64)
        weights  = tab['weight_value'].to_numpy(dtype=float)

        self.weight_matrix = sp.coo_matrix(
            (weights, (src_ids, root_ids)),
            shape=(ct.n_src_edges, ct.n_root_edges),
        ).tocsr()
            # ------------------------------------------------------------------ #
    # Apply                                                                #
    # ------------------------------------------------------------------ #

    def apply(self, F_src: np.ndarray) -> np.ndarray:
        """Map a vector over src_edges → root_edges via W.T @ F_src."""
        return self.weight_matrix.T @ F_src

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, **kwargs):
        os.makedirs(self.odir, exist_ok=True)
        path = kwargs.get('path', os.path.join(self.odir, f"{self.id}.pkl"))
        with open(path, "wb") as f:
            pickle.dump({
                "id":            self.id,
                "acc_id":        self.acc_used,
                "weight_matrix": self.weight_matrix,
                # Embed the concordance so a round-trip load is self-contained
                "concordance_tab":        self.concordance.tab,
                "concordance_src_edges":  self.concordance.src_edges,
                "concordance_id":         self.concordance.id,
            }, f)

    @classmethod
    def load(cls, id: str, acc_struct, **kwargs) -> "WeightMatrix":
        odir = kwargs.get('odir', "data/proc/conc")
        path = kwargs.get('path', os.path.join(odir, f"{id}.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Reconstruct concordance shell (no re-validation needed)
        ct = ConcordanceTable.__new__(ConcordanceTable)
        ct.id            = data['concordance_id']
        ct.odir          = odir
        ct.acc_struct    = acc_struct
        ct.acc_used      = data['acc_id']
        ct.tab           = data['concordance_tab']
        ct.src_edges     = data['concordance_src_edges']
        ct.src_edge_to_id = {
            tuple(row): i for i, row in data['concordance_src_edges'].iterrows()
        }
        ct.n_src_edges  = len(ct.src_edges)
        ct.n_root_edges = len(acc_struct.edge_index)

        obj = cls.__new__(cls)
        obj.id            = data['id']
        obj.odir          = odir
        obj.acc_struct    = acc_struct
        obj.acc_used      = data['acc_id']
        obj.concordance   = ct
        obj.prop_shares   = None
        obj.weight_matrix = data['weight_matrix']
        return obj