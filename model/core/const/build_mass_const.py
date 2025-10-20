
'''
First build fundamental supply-use constraints
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

def build_mass_balance_constraints_from_index(
    index_path,
    mb_rel_ent=("Flow", "Process"),
    out_folder="data/proc/const/mb",
    out_name_G="G_mass_balance",
    out_name_c="c_mass_balance",
    out_name_c_sigma="c_sigma_mass_balance",
    out_name_c_idx="c_mass_balance_idx",
    const_type="mass_balance",
    sanity_check=True,
):
    """
    Build sparse mass balance constraint matrix G and vector c from an index using pydata/sparse.

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

    # Convert to numpy arrays
    G_data = np.array(G_data, dtype=np.int32)
    G_row = np.array(G_row, dtype=np.int32)
    G_col = np.array(G_col, dtype=np.int32)

    # Build sparse COO matrix
    G = sparse.COO((G_data, (G_row, G_col)), shape=(n_constraints, n_vars))
    c = np.zeros(n_constraints, dtype=np.int32)

    density = G.nnz / (n_constraints * n_vars) * 100
    logger.info(f"Sparse matrix G built with shape {G.shape} and density {density:.6e}%")
    G_mem_MB = (G.data.nbytes + G.coords.nbytes) / (1024**2)
    logger.info(f"Sparse matrix G memory size: {G_mem_MB:.2f} MB")

    # --- Step 4: Sanity check
    if sanity_check:
        row_sums = np.abs(G).sum(axis=0).todense()
        if not np.all(row_sums <= 2):
            raise ValueError("Sanity check failed: Some variables appear in more than two constraints.")
        logger.info("Sanity check passed: Each variable appears in at most two constraints.")

        unique, counts = np.unique(row_sums, return_counts=True)
        logger.info(f"Variable participation in constraints: {dict(zip(unique, counts))}")

        col_sums = np.abs(G).sum(axis=1).todense()
        if not np.all(col_sums > 0):
            raise ValueError("Sanity check failed: Some constraints have no variables.")
        logger.info("Sanity check passed: All constraints have at least one variable.")

        # Example constraint mapping
        example_constraints = {val: (np.where(col_sums == val)[0][0], reg_sec_comb[np.where(col_sums == val)[0][0]]) for val in np.unique(col_sums)}
        example_df = pd.DataFrame({
            'Num_Variables': np.unique(col_sums),
            'Count': [np.sum(col_sums == val) for val in np.unique(col_sums)],
            'Example_Constraint': [example_constraints[val] for val in np.unique(col_sums)]
        })
        logger.info(f"Constraint variable counts:\n{example_df}")

    # --- Step 5: Constraint tolerance
    c_sigma = np.ones(n_constraints, dtype=np.float32) * 1e-3  # 1 t tolerance

    # --- Step 6: Save
    os.makedirs(out_folder, exist_ok=True)
    time_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    sparse.save_npz(os.path.join(out_folder, out_name_G + f'_{time_stamp}.npz'), G)
    np.save(os.path.join(out_folder, out_name_c + f'_{time_stamp}.npy'), c)
    np.save(os.path.join(out_folder, out_name_c_sigma + f'_{time_stamp}.npy'), c_sigma)
    np.save(os.path.join(out_folder, out_name_c_idx + f'_{time_stamp}.npy'), constraint_idx_map)

    logger.info(f"Mass balance constraints saved to '{out_folder}'")


def plot_G(
    root_folder=r"data\proc\const\mb",
    G_name=r"G_mass_balance_20250930_101428.npz",
    out_path = r"figs\explo\G_sparse"
):
    # --- Load matrix ---
    path = os.path.join(root_folder, G_name)
    G = sparse.load_npz(path)

    # Shape and density
    n_rows, n_cols = G.shape
    density = G.nnz / (n_rows * n_cols) * 100

    # --- Plot sparsity pattern ---
    fig, ax = plt.subplots(figsize=(15, 10))

    # markersize scales inversely with matrix size
    markersize = max(0.1, 2000 / max(n_rows, n_cols))

    ax.spy(G, markersize=markersize, rasterized=True)
    ax.set_title(
        f"Sparsity pattern of G (mass balance)\n"
        f"Shape: {n_rows} x {n_cols}, Density: {density:.4f}%"
    )
    ax.set_xlabel("Variables")
    ax.set_ylabel("Constraints")
    
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

    plt.tight_layout()
    plt.savefig(out_path + f'_{time_stamp}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved sparsity plot to {out_path}")

if __name__ == "__main__":

    # _TS_RE = re.compile(r'(\d{8}_\d{6})')  # matches YYYYMMDD_HHMMSS
    path = get_latest_index(r"data\proc\index")
    # build_mass_balance_constraints_from_index(path)

    build_mass_balance_constraints_from_index(index_path=path)