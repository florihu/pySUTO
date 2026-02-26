'''
Build the G matrix for region-sector mass constraints

'''

import logging
import os
import re
from datetime import datetime
from itertools import product
from typing import Iterable, Tuple, List, Optional
import sys



#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sparse as sp
from tqdm import tqdm
from pathlib import Path

from src.core.datafeed.util import get_latest_index


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



index_path = get_latest_index('data/proc/index')

acc_folder = Path("data/proc/datafeed/icsg")

account_paths =     {
    'E': 'E_prod_icsg_20251020_101950.csv',
    'O': 'O_prod_icsg_20251020_101950.csv',
    'P': 'P_prod_icsg_20251020_101950.csv',
    'S': 'S_prod_icsg_20251020_101950.csv',
    'U': 'U_prod_icsg_20251020_101950.csv',
    'Y': 'Y_prod_icsg_20251020_101950.csv'
}


def consistency_check_index(data, index, logger):
    """
    Check that all origin/destination values in `data` are present in the corresponding
    origin/destination columns of `index` for Sectors, Regions, and Entities.

    Parameters
    ----------
    data : pandas.DataFrame
        The main dataset containing origin/destination columns (e.g., 'Sector_origin', 'Sector_destination').
    index : pandas.DataFrame
        The reference index containing valid origin/destination pairs.
    
    Logs
    ----
    Issues warnings (via `logger.warning`) if `data` contains origin/destination entries
    not present in `index`.
    """
    def check_consistency(prefix):
        """Helper to check origin/destination consistency for a given prefix."""
        cols = [f"{prefix}_origin", f"{prefix}_destination"]
        if not all(c in index.columns for c in cols):
            return  # skip if index lacks required columns
        
        valid = set(pd.unique(pd.concat([index[cols[0]], index[cols[1]]])))
        for col in cols:
            if col in data.columns:
                missing = set(data[col].unique()) - valid
                if missing:
                    logger.warning(f"Missing {prefix.lower()}s in index for column {col}: {missing}")

    for prefix in ["Sector", "Region", "Entity"]:
        check_consistency(prefix)


def build_regtot_const(index_path, s_path, sup_use, sanity_check=True):
    """
    Efficient builder for region totals constraints (supply or use).
    Returns dict with: 'G' (COO), 'c' (np.ndarray), 'meta' (dict)
    """

    # --- Step 1: Load
    keys_df = pd.read_parquet(index_path).reset_index(drop=True)
    s_df = pd.read_csv(s_path)
    consistency_check_index(s_df, keys_df, logger)

    type_ = f"{sup_use}_region_totals"
    n_tots, n_vars = len(s_df), len(keys_df)

    # --- Step 2: Build composite join key
    s_df_r = s_df.reset_index().rename(columns={'index': 'row_index'})
    keys_idx = keys_df.reset_index().rename(columns={'index': 'var_index'})

    if sup_use == 'supply':
        s_df_r['merge_key'] = (
            s_df_r['Region_origin'].astype(str) + '|' +
            s_df_r['Sector_origin'].astype(str) + '|' +
            s_df_r['Sector_destination'].astype(str)
        )
        keys_idx['merge_key'] = (
            keys_idx['Region_origin'].astype(str) + '|' +
            keys_idx['Sector_origin'].astype(str) + '|' +
            keys_idx['Sector_destination'].astype(str)
        )
    elif sup_use == 'use':
        s_df_r['merge_key'] = (
            s_df_r['Region_destination'].astype(str) + '|' +
            s_df_r['Sector_origin'].astype(str) + '|' +
            s_df_r['Sector_destination'].astype(str)
        )
        keys_idx['merge_key'] = (
            keys_idx['Region_destination'].astype(str) + '|' +
            keys_idx['Sector_origin'].astype(str) + '|' +
            keys_idx['Sector_destination'].astype(str)
        )
    else:
        raise ValueError("sup_use must be 'supply' or 'use'")

    # --- Step 3: Merge (vectorized join)
    merged = s_df_r.merge(
        keys_idx[['merge_key', 'var_index']],
        on='merge_key',
        how='left',
        indicator=True
    )

    # Warn about missing matches
    missing = merged[merged['_merge'] != 'both']
    if not missing.empty:
        logger.warning(
            f"{len(missing)} constraints had no matches in keys_df "
            f"and will produce zero rows in G and zeros in c."
        )
        # print some examples
        logger.warning(f"Examples of missing matches:\n{missing[['Region_origin', 'Region_destination', 'Sector_origin', 'Sector_destination', 'Value']].head()}")

    logger.info('No issues detected during merge.')


    matched = merged[merged['_merge'] == 'both']

    # --- Step 4: Build COO correctly (coords first, data second)
    if len(matched) == 0:
        coords = np.zeros((2, 0), dtype=int)
        data = np.zeros(0, dtype=np.int8)
    else:
        row = matched['row_index'].to_numpy(dtype=int)
        col = matched['var_index'].to_numpy(dtype=int)
        coords = np.vstack((row, col))
        data = np.ones(len(row), dtype=np.int8)

    G = sp.COO(coords, data, shape=(n_tots, n_vars))

    # Build full-length c vector
    c = np.zeros(n_tots, dtype=float)
    if not matched.empty:
        c[matched['row_index'].to_numpy(dtype=int)] = matched['Value'].to_numpy(dtype=float)


    c_sigma = c *.02  # 2% tolerance

        # Select columns based on mode
    if sup_use == 'supply':
        region_col, sector_col = 'Region_origin', 'Sector_origin'
    elif sup_use == 'use':
        region_col, sector_col = 'Region_destination', 'Sector_destination'
    else:
        raise ValueError(f"Invalid sup_use: {sup_use}")

    # Build c_index DataFrame
    c_index = s_df_r[[region_col, sector_col]].copy()
    c_index['type'] = type_

    # Build identifier directly as a Series
    c_idx = c_index['type'] + '_' + c_index[region_col] + '_' + c_index[sector_col]
    # to numpy
    c_idx = c_idx.to_numpy()


    # --- Step 5: Sanity checks
    if sanity_check:
        assert G.shape == (n_tots, n_vars), "Matrix shape mismatch"
        assert len(c) == n_tots, "Constraint vector length mismatch"

    # --- Step 6: Logging & save
    density = (G.nnz / (n_tots * n_vars)) * 100 if n_tots * n_vars else 0
    mem_MB = (G.data.nbytes + G.coords.nbytes) / (1024**2) if G.nnz > 0 else 0
    nnz = G.nnz

    logger.info(f"G built: shape={G.shape}, density={density:.6e}%, mem={mem_MB:.2f} MB, nnz={nnz}")

    out_folder = os.path.join("data", "proc", "const")
    os.makedirs(out_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path_G = os.path.join(out_folder, f"G_{type_}_{timestamp}.npz")
    out_path_c = os.path.join(out_folder, f"c_{type_}_{timestamp}.npy")
    out_path_csigma = os.path.join(out_folder, f"c_sigma_{type_}_{timestamp}.npy")
    out_c_idx = os.path.join(out_folder, f"c_idx_{type_}_{timestamp}.npy")    


    sp.save_npz(out_path_G, G)
    np.save(out_path_c, c)
    np.save(out_path_csigma, c_sigma)
    np.save(out_c_idx, c_idx)


    logger.info(f"G saved to {out_path_G}, c saved to {out_path_c}, c_sigma saved to {out_path_csigma}, c_idx saved to {out_c_idx}")


    return None



def build_implicit_mass_balance(index_p, acc_p):
    """
    Build the implicit mass balance constraint matrix (G) with two rows per s:
    - flow perspective
    - process perspective (tech + boundary)

    Goal for every process and flow value in s must be an associated mass balance constraint in G, with the same value in c. 
    This allows us to solve for the flow values in s while ensuring mass balance is implicitly satisfied. 
    The first n_s rows of G correspond to the flow perspective, and the next n_s rows correspond to the process perspective. 
    Each row in G will have a 1 for the corresponding variable in index that matches the sector/region pairs from s, and 0 otherwise. The c vector will simply duplicate the values from s for both perspectives.
    """

    # --- Read input ---
    index = pd.read_parquet(index_p, engine= 'fastparquet').reset_index(drop=True)
    
    # read and concatenate all accounts into a single DataFrame
    account_dfs = []
    for acc, path in acc_p.items():
        path = os.path.join(acc_folder, path)
        df = pd.read_csv(path)
        df['Account'] = acc  # add account column to identify source
        account_dfs.append(df)

    df = pd.concat(account_dfs, ignore_index=True)

    # n = len(df)
    
    # # --- Build COO data ---
    # c_vals = []
    # row_coords = []
    # col_coords = []
    # data_vals = []



    # # --- Build sparse matrix ---
    # shape = (2 * n, len(index))  # <-- two rows per s
    # G = sp.COO(coords=[row_coords, col_coords], data=data_vals, shape=shape)

    # # RHS vector: duplicate s['Value'] for flow and process rows
    # c_vals = np.concatenate([s['Value'].to_numpy(), s['Value'].to_numpy()])

    return None #G, c_vals 




if __name__ == "__main__":
    

    build_implicit_mass_balance(index_path, account_paths)

    