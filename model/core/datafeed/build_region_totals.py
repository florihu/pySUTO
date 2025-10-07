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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_regtot_const(index_path, s_path, type_):

    '''
    Steps
    1. merge s on index.
    2. build G from merged., build c, c_idx, c_sigma
        2.1 G: rows constraint and cols variables (index)
    3. sanity check
    '''

    # --- Step 1: Merge s on index
    keys_df = pd.read_parquet(index_path).reset_index(drop=True)

    s_df = pd.read_csv(s_path)
    n_tots = s_df.shape[0]
    n_vars = keys_df.shape[0]

    # --- Step 2: Build G, c, c_idx, c_sigma

    
    G_data, G_row, G_col = [], [], []

    # for each row in s_df, find matching rows in keys_df
    c_list, c_idx_list = [], []

    for _, s_row in tqdm(s_df.iterrows(), total=s_df.shape[0], desc="Building constraints"):
        region = s_row['Region_origin']
        sector_origin = s_row['Sector_origin']
        sector_dest = s_row['Sector_destination']
        value = s_row['Value']

        # find matching rows in keys_df
        mask = (
            (keys_df['Region_origin'] == region) &
            (keys_df['Sector_origin'] == sector_origin) &
            (keys_df['Sector_destination'] == sector_dest)
        )
        matched_indices = keys_df.index[mask].tolist()
        if not matched_indices:
            logger.warning(f"No matches found for constraint ({type_}, {region}, {sector_origin}, {sector_dest})")
            continue
        # Add to G
        row_idx = len(c_list)

        for col_idx in matched_indices:
            G_data.append(1.0)
            G_row.append(row_idx)
            G_col.append(col_idx)

        c_list.append(value)
        c_idx_list.append((type_, region, sector_origin, sector_dest))

    # Convert to numpy arrays
    G_data = np.array(G_data, dtype=np.int64)
    G_row = np.array(G_row, dtype=np.int64)
    G_col = np.array(G_col, dtype=np.int64)

    # Build sparse COO matrix
    G = sparse.COO((G_data, (G_row, G_col)), shape=(n_tots, n_vars))
    c = np.zeros(n_tots, dtype=np.int64)

    density = G.nnz / (n_tots * n_vars) * 100
    logger.info(f"Sparse matrix G built with shape {G.shape} and density {density:.6e}%")
    G_mem_MB = (G.data.nbytes + G.coords.nbytes) / (1024**2)
    logger.info(f"Sparse matrix G memory size: {G_mem_MB:.2f} MB")


    if sanity_check: 
         # G must have the same index order as Keys_df
        assert G.shape[1] == keys_df.shape[0], "Mismatch in G columns and index length"
    
       
    return None


if __name__ == "__main__":
    index_path = get_latest_index(r'data\proc\index')
    s_path  = r'data\proc\datafeed\icsg\region_totals\S_20250929_134742.csv'
    u_path  = r'data\proc\datafeed\icsg\region_totals\U_20250929_134330.csv'
    build_regtot_const(index_path, s_path, type_='supply_region_totals')

    