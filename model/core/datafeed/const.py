
'''
First build fundamental supply-use constraints
'''
import numpy as np
import pandas as pd
from scipy import sparse
from itertools import product
from typing import Iterable, Tuple, List
import os
import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns




def build_origin_dest_constraints(
    ie_parquet_path: str = r"data\processed\data_feed\ie\ie_matrix.parquet",
    flow_categories: Iterable[str] = ("Flow", "SEM", "Nature", "FD"),
    normalize_case: bool = True,
) -> Tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    """
    Build binary constraint mask G and nominal RHS c = G @ x (no pos/neg distinction).

    Returns:
      G: scipy.sparse.csr_matrix, shape (n_constraints, n_vars), values in {-1, 0, +1}
      c : numpy.ndarray, constraint values (here always zero)
      constraint_keys : DataFrame with constraint_id, Region, Sector, kind ('origin'/'destination') and type ('mass_balance')
    """
    # --- Load IE table
    df = pd.read_parquet(ie_parquet_path)
    df.reset_index(inplace=True)

    # Collapse FD/SEM/Nature → Flow
    ent_rename = {"FD": "Flow", "Nature": "Flow", "SEM": "Flow"}
    df["Entity_origin"] = df["Entity_origin"].replace(ent_rename)
    df["Entity_destination"] = df["Entity_destination"].replace(ent_rename)

    n_vars = len(df)

    regions = pd.Index(pd.concat([df["Region_origin"], df["Region_destination"]]).unique())
    sectors = pd.Index(pd.concat([df["Sector_origin"], df["Sector_destination"]]).unique())

    G_rows = []
    c_keys = []


    for r, s in tqdm.tqdm (product(regions, sectors), total=len(regions) * len(sectors),  desc="Building constraints"):
        # the input (origin) gets plus -1 the output (destination) gets minus 1
        or_mask = (df["Region_origin"] == r) & (df["Sector_origin"] == s)
        de_mask = (df["Region_destination"] == r) & (df["Sector_destination"] == s)
        
        G_row_or = sparse.csr_matrix((-np.ones(or_mask.sum()), (np.zeros(or_mask.sum()), np.where(or_mask)[0])), shape=(1, n_vars))
        G_row_de = sparse.csr_matrix((np.ones(de_mask.sum()), (np.zeros(de_mask.sum()), np.where(de_mask)[0])), shape=(1, n_vars))
        G_row = G_row_or + G_row_de
        G_rows.append(G_row)
        # constraint key
        c_keys.append(('Mass_bal', r, s))

        # sum the rows together

    # Build sparse matrix in one go
    G= sparse.vstack(G_rows).tocsr()
    c = np.zeros(G.shape[0], dtype=float)
    keys = pd.DataFrame(c_keys, columns=["Type", "Region", "Sector"])

    # safe stuff
    root_folder = r"data\processed\data_feed\constraints"
    G_path = root_folder + r"\G_mass_balance.npz"
    c_path = root_folder + r"\c_mass_balance.npy"
    keys_path = root_folder + r"\c_keys_mass_balance.parquet"

    #save
    sparse.save_npz(G_path, G)
    np.save(c_path, c)
    keys.to_parquet(keys_path, index=False)


    return None


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
    plot_G()