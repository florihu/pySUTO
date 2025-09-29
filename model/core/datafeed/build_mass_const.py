
'''
First build fundamental supply-use constraints
'''

import logging
import os
import re
from datetime import datetime
from itertools import product
from typing import Iterable, Tuple, List, Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm



def build_mass_balance_constraints_from_index(
    index_path,
    mb_rel_ent=("Flow", "Process"),
    out_folder="data/proc/const",
    out_name_G="G_mass_balance",
    out_name_c="c_mass_balance",
    out_name_c_sigma="c_sigma_mass_balance",
    out_name_c_idx="c_mass_balance_idx",
    const_type="mass_balance",
    sanity_check=True,
    ):
    """
    Build sparse mass balance constraint matrix G and vector c from an index.

    - Order of rows in keys_df defines the variable ordering (columns of G).
    - Rows of G = (region, sector) constraints.
    - Columns of G = variables from keys_df (must align 1:1 with optimization model).

    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Step 1: Load index and enforce continuous integer index
    keys_df = pd.read_parquet(index_path).reset_index(drop=True)
    n_vars = keys_df.shape[0]

    # --- Step 2: Relevant sectors & regions
    origin = keys_df[keys_df.Entity_origin.isin(mb_rel_ent)].Sector_origin.unique()
    dest = keys_df[keys_df.Entity_destination.isin(mb_rel_ent)].Sector_destination.unique()
    sectors = np.union1d(origin, dest)
    regions = keys_df["Region_origin"].unique()

    n_sectors = len(sectors)
    n_regions = len(regions)
    n_constraints = n_sectors * n_regions

    logger.info(f"Number of variables: {n_vars}")
    logger.info(f"Constraints to build: {n_constraints}")

    # --- Step 3: Build sparse matrix G and vector c
    reg_sec_comb = list(product(regions, sectors))
    constraint_idx_map = {(const_type, reg, sec): i for i, (reg, sec) in enumerate(reg_sec_comb)}

    # Preallocate lists
    G_data, G_row, G_col = [], [], []

    for constr_idx, (reg, sec) in enumerate(tqdm(reg_sec_comb, desc="Building constraints")):
        # Origin = supply (-1)
        supply_mask = (keys_df["Region_origin"] == reg) & (keys_df["Sector_origin"] == sec)
        supply_vars = np.flatnonzero(supply_mask.values)
        if supply_vars.size > 0:
            G_data.extend([-1] * len(supply_vars))
            G_row.extend([constr_idx] * len(supply_vars))
            G_col.extend(supply_vars)

        # Destination = use (+1)
        use_mask = (keys_df["Region_destination"] == reg) & (keys_df["Sector_destination"] == sec)
        use_vars = np.flatnonzero(use_mask.values)
        if use_vars.size > 0:
            G_data.extend([1] * len(use_vars))
            G_row.extend([constr_idx] * len(use_vars))
            G_col.extend(use_vars)

    # Convert to numpy arrays once
    G_data = np.array(G_data, dtype=np.int8)
    G_row = np.array(G_row, dtype=np.int32)
    G_col = np.array(G_col, dtype=np.int32)

    # Build sparse matrix
    G = sparse.coo_matrix((G_data, (G_row, G_col)), shape=(n_constraints, n_vars)).tocsr()
    c = np.zeros(n_constraints, dtype=np.int8)

    density = G.nnz / (n_constraints * n_vars) * 100
    logger.info(f"Sparse matrix G built with shape {G.shape} and density {density:.6e}%")
    # Memory size of G in MB
    G_mem_MB = (G.data.nbytes + G.indptr.nbytes + G.indices.nbytes) / (1024**2)
    logger.info(f"Sparse matrix G memory size: {G_mem_MB:.2f} MB")

    if sanity_check:
        # Sanity check: each variable appears in exactly one supply and one use constraint
        row_sums = np.abs(G).sum(axis=0).A1
        if not np.all(row_sums <= 2):
            raise ValueError("Sanity check failed: Some variables appear in more than two constraints.")
        logger.info("Sanity check passed: Each variable appears in at most two constraints.")

        # print unique values of row_sums
        unique, counts = np.unique(row_sums, return_counts=True)
        logger.info(f"Variable participation in constraints: {dict(zip(unique, counts))}")

        # Check that all constraints have at least one variable
        col_sums = np.abs(G).sum(axis=1).A1
        if not np.all(col_sums > 0):
            raise ValueError("Sanity check failed: Some constraints have no variables.")
        logger.info("Sanity check passed: All constraints have at least one variable.")
    
        # print unique values of col_sums
        unique_c, counts_c = np.unique(col_sums, return_counts=True)
        

        # get an example id of a constraint for every unique value in unique_c
        example_constraints = {}
        for val in unique_c:
            example_idx = np.where(col_sums == val)[0][0]
            example_constraints[val] = (example_idx, reg_sec_comb[example_idx])
        
        # make a dataframe out of unique_c, counts_c, example_constraints
        example_df = pd.DataFrame({
            'Num_Variables': unique_c,
            'Count': counts_c,
            'Example_Constraint': [example_constraints[val] for val in unique_c]
        })
        logger.info(f"Constraint variable counts:\n{example_df}")

    # lets assume a tonne deviation of the constraint is acceptable
    c_sigma = np.ones(n_constraints, dtype=np.float32) * 10**-3  # 1 t tolerance

    # get time stamp
    time_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") 
    
    # --- Step 4: Save

    out_folder = out_folder +'/_mass_balance' + f'_{time_stamp}'
    os.makedirs(out_folder, exist_ok=True)
    sparse.save_npz(os.path.join(out_folder, out_name_G), G)
    np.save(os.path.join(out_folder, out_name_c), c)
    np.save(os.path.join(out_folder, out_name_c_idx), constraint_idx_map)
    np.save(os.path.join(out_folder, out_name_c_sigma), c_sigma)
    logger.info(f"Mass balance constraints saved to '{out_folder}'")

    return None




def _parse_timestamp_from_name(fname: str) -> Optional[datetime]:
    m = _TS_RE.search(fname)
    if not m:
        return None
    ts_text = m.group(1)
    try:
        return datetime.strptime(ts_text, "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def get_latest_index(folder: str) -> str:
    """
    Return the path to the latest parquet file in `folder`.
    Priority for deciding "latest":
      1) timestamp parsed from filename using pattern YYYYMMDD_HHMMSS
      2) file modification time (fallback)
    Raises FileNotFoundError when no .parquet files exist.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    candidates = []
    for f in files:
        full = os.path.join(folder, f)
        ts = _parse_timestamp_from_name(f)
        if ts is None:
            # fallback to file modification time
            ts = datetime.fromtimestamp(os.path.getmtime(full))
        candidates.append((ts, full))

    # pick the newest timestamp
    latest_path = max(candidates, key=lambda x: x[0])[1]
    return latest_path


def plot_G():
    root_folder = r"data\processed\data_feed\constraints"
    G_path = root_folder + r"\G_mass_balance.npz"
    G = sparse.load_npz(G_path)

    # Get matrix shape
    n_rows, n_cols = G.shape
    aspect_ratio = n_cols / n_rows

    density = G.nnz / (n_rows * n_cols) *100

    # Adjust figure size based on matrix shape
    plt.figure(figsize=(30,5))
    plt.spy(G[:,:10000], markersize=0.5)  # slightly larger marker for visibility
    plt.title(f"Sparsity pattern of constraint matrix G (mass balance)\nShape: {n_rows} x {n_cols}, Density: {density:.4f} %")
    plt.xlabel("Variables")
    plt.ylabel("Constraints")
    plt.tight_layout()
    plt.savefig(r'figs\explo\G_sparse.png', dpi=300)
    plt.show()


if __name__ == "__main__":

    _TS_RE = re.compile(r'(\d{8}_\d{6})')  # matches YYYYMMDD_HHMMSS
    path = get_latest_index(r"data\proc\index")
    build_mass_balance_constraints_from_index(path)