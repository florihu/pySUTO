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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from model.core.datafeed.util import get_latest_index


def build_regtot_const(index_path, s_path, u_path):

    '''
    Steps
    1. merge s on index.
    2. build G from merged., build c, c_idx, c_sigma
        2.1 G: rows constraint and cols variables (index)
    3. sanity check
    '''

    # --- Step 1: Merge s on index
    keys_df = pd.read_parquet(index_path).reset_index(drop=True)
    n_vars = keys_df.shape[0]
    s_df = pd.read_csv(s_path)

    G_data = [] # contains 1 for affected variables 0 for none
    G_rows = [] # rows for G = constraints
    G_cols = [] # variables affected 
    

    for i in s_df.iterrows():
        row = i[1]
        # find identify indexes in keys_df
        # mask all ids affected by this row and append a 1 in G_data
        mask = (keys_df.Region_origin == row['Region']) & (keys_df.Sector_origin == row['Sector']) & (keys_df.Entity_origin == 'Flow')
        idxs = keys_df.index[mask].tolist()
        const

        
        
    


    
    return None

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# optional: pydata.sparse (preferred if you use pydata everywhere)
try:
    import sparse as pydata_sparse
    _HAS_PYDATA = True
except Exception:
    from scipy import sparse as scipy_sparse
    _HAS_PYDATA = False

def build_regtot_const_from_s_rows(
    index_path,
    s_path,
    u_path=None,
    s_to_index_map=None,
    sign_col=None,            # name of column in s_df containing +1/-1 sign for constraint entries (optional)
    value_col="Value",        # column in s_df that gives the RHS (c) for each constraint
    drop_empty=True,          # drop constraints that select zero variables
    sigma_default_rel=1e-6,
    return_pydata=True,       # return a pydata.sparse.COO if available, else scipy sparse
):
    """
    Build G, c, c_idx, c_sigma by looping through s_df rows.
    Columns (variables) preserve keys_df order.

    Parameters
    ----------
    index_path : str
        parquet path to keys_df (must contain columns like Region_origin, Sector_origin, Entity_origin).
    s_path : str
        CSV path for supply-use rows. Each row defines one constraint by matching some fields against keys_df.
    u_path : str or None
        optional path for sigma (per-constraint uncertainties). If None, a default relative sigma is used.
    s_to_index_map : dict or None
        mapping from columns in s_df to columns in keys_df to match on.
        Default: {'Region': 'Region_origin', 'Sector': 'Sector_origin', 'Entity': 'Entity_origin'}
    sign_col : str or None
        if present in s_df, its value (+1 or -1) will be used for entries; otherwise entries are +1.
    value_col : str
        column in s_df that contains the target sum for that constraint.
    drop_empty : bool
        if True, constraints that match zero variables are skipped.
    return_pydata : bool
        try to return pydata.sparse.COO when available; otherwise returns scipy.sparse.coo_matrix.
    """
    # defaults
    if s_to_index_map is None:
        s_to_index_map = {'Region': 'Region_origin', 'Sector': 'Sector_origin', 'Entity': 'Entity_origin'}

    # --- load index and s
    keys_df = pd.read_parquet(index_path).reset_index(drop=True)
    n_vars = keys_df.shape[0]

    s_df = pd.read_csv(s_path)
    if value_col not in s_df.columns:
        raise KeyError(f"'{value_col}' column not found in s_df")

    # Prepare containers for COO data
    G_data = []
    G_rows = []
    G_cols = []
    c_list = []
    c_idx = []  # human readable identifier for each constraint row

    current_row = 0
    # iterate preserving s_df order
    for s_row in tqdm(s_df.itertuples(index=False), desc="Building regtot constraints"):
        # build boolean mask (start all True, apply AND for each mapping)
        mask = np.ones(n_vars, dtype=bool)
        # for readability, create dict-like access to s_row fields
        s_row_dict = s_row._asdict() if hasattr(s_row, "_asdict") else s_row.__dict__

        for s_col, idx_col in s_to_index_map.items():
            if s_col in s_row_dict:
                # prefer exact match; cast to string for robust comparisons if mixed dtypes exist
                mask &= (keys_df[idx_col].astype(str).values == str(s_row_dict[s_col]))
            else:
                # if s_df does not have this column skip it
                continue

        var_idxs = np.flatnonzero(mask)
        if var_idxs.size == 0 and drop_empty:
            # skip empty constraint
            continue

        # decide sign for entries in this row
        sign = 1
        if (sign_col is not None) and (sign_col in s_row_dict):
            try:
                sign = int(s_row_dict[sign_col])
            except Exception:
                sign = 1

        # append row entries
        if var_idxs.size > 0:
            G_data.extend([sign] * int(var_idxs.size))
            G_rows.extend([current_row] * int(var_idxs.size))
            G_cols.extend(var_idxs.tolist())

        # RHS c
        c_val = float(s_row_dict.get(value_col, 0.0))
        c_list.append(c_val)

        # create a human-friendly identifier for this row (you can change format)
        # use s_df original columns that were used for matching to describe the constraint
        id_parts = []
        for s_col, idx_col in s_to_index_map.items():
            if s_col in s_row_dict:
                id_parts.append(f"{s_col}={s_row_dict[s_col]}")
        # if no id parts, fall back to row index in s_df
        if not id_parts:
            id_str = f"s_row_{current_row}"
        else:
            id_str = "|".join(id_parts)
        c_idx.append(id_str)

        current_row += 1

    # finalize
    n_constraints = current_row
    if n_constraints == 0:
        # Return empty structures
        if _HAS_PYDATA and return_pydata:
            G = pydata_sparse.COO(np.empty(0), coords=(np.empty((2,0), dtype=int)), shape=(0, n_vars))
        else:
            from scipy.sparse import coo_matrix
            G = coo_matrix(([], ([], [])), shape=(0, n_vars))
        c = np.zeros((0,), dtype=float)
        c_sigma = np.zeros((0,), dtype=float)
        return G, c, c_idx, c_sigma

    G_data = np.asarray(G_data, dtype=float)
    G_row = np.asarray(G_rows, dtype=int)
    G_col = np.asarray(G_cols, dtype=int)
    c = np.asarray(c_list, dtype=float)

    # build sparse matrix (prefer pydata if available)
    if _HAS_PYDATA and return_pydata:
        # pydata.sparse.COO expects coords as shape (ndim, nnz)
        coords = np.vstack([G_row, G_col])
        G = pydata_sparse.COO(coords, G_data, shape=(n_constraints, n_vars))
    else:
        from scipy.sparse import coo_matrix
        G = coo_matrix((G_data, (G_row, G_col)), shape=(n_constraints, n_vars))

    # c_sigma: try to read from u_path if provided (simple mapping by s_df row order),
    # otherwise use default relative sigma
    if (u_path is not None) and os.path.exists(u_path):
        try:
            u_df = pd.read_csv(u_path)
            # If u_df directly aligns row-by-row with s_df, and we dropped empty rows,
            # we select only the matching ones; otherwise fall back to default.
            if u_df.shape[0] == len(c_list):
                c_sigma = u_df.iloc[:len(c_list)].iloc[:, 0].astype(float).to_numpy()
            else:
                # best-effort: if u_df has 'Value' or 'sigma' column try to map by same matching keys
                if 'sigma' in u_df.columns:
                    sigma_col = 'sigma'
                elif 'Value' in u_df.columns:
                    sigma_col = 'Value'
                else:
                    sigma_col = None

                if sigma_col is not None:
                    # naive: take first column values (you can supply a mapping function if needed)
                    c_sigma = u_df[sigma_col].astype(float).to_numpy()
                    # if lengths mismatch, fall back to default for missing
                    if c_sigma.shape[0] < len(c_list):
                        c_sigma = np.concatenate([c_sigma, np.full(len(c_list) - c_sigma.shape[0], np.nan)])
                    c_sigma = c_sigma[:len(c_list)]
                    missing = ~np.isfinite(c_sigma)
                    c_sigma[missing] = sigma_default_rel * np.maximum(1.0, np.abs(c[missing]))
                else:
                    c_sigma = sigma_default_rel * np.maximum(1.0, np.abs(c))
        except Exception:
            c_sigma = sigma_default_rel * np.maximum(1.0, np.abs(c))
    else:
        c_sigma = sigma_default_rel * np.maximum(1.0, np.abs(c))

    return G, c, c_idx, c_sigma

if __name__ == "__main__":
    index_path = get_latest_index(r'data\proc\index')
    s_path  = r'data\proc\datafeed\icsg\region_totals\S_20250929_134742.csv'
    u_path  = r'data\proc\datafeed\icsg\region_totals\U_20250929_134330.csv'
    G, c, c_idx, c_sigma = build_regtot_const_from_s_rows(index_path, s_path, u_path)

    