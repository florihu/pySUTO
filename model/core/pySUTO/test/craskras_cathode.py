import numpy as np
import sparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds pySUTO to path
from opt.ProblemData import ProblemData
from opt.KRASCRASOptimizer import KRASCRASOptimizer
from opt.Diagnostics import Diagnostics



# --- Define paths (Linux-compatible) ---
G_path = "data/proc/const/G_20251028_145401.npz"
c_path = "data/proc/const/c_20251028_145401.npy"
c_sigma_path = "data/proc/const/c_sigma_20251028_145401.npy"
c_idx_path = "data/proc/const/c_idx_20251028_145401.npy"
a0_path = "data/proc/ie/ie_20251028_113607.npy"
var_p = "data/proc/index/index_20251020_154514.parquet"

# --- Load data ---
G = sparse.load_npz(Path(G_path))                      # load sparse matrix
c = np.load(Path(c_path))                              # load constraint targets
c_sigma = np.load(Path(c_sigma_path))                  # load constraint tolerances
a0 = np.load(Path(a0_path))                            # load initial guess
c_idx = np.load(Path(c_idx_path), allow_pickle=True)   # load constraint index map
vars = pd.read_parquet(Path(var_p))


# generate c_idx_map
c_idx_map = {i:idx for i, idx in enumerate(c_idx)}
# generate c_idx_map
vars_map = {i: (row.Region_origin, row.Sector_origin, row.Entity_origin, row.Region_destination, row.Sector_destination, row.Entity_destination) for i, row in vars.iterrows()}


problem = ProblemData(
    G=G,
    c=c,
    sigma=c_sigma,
    a0=a0,
    c_idx_map=c_idx_map,
    vars_map=vars_map,
    c_to_var=None
)


optimizer = KRASCRASOptimizer(problem, max_iter=10000, tol=1e-4, verbose=True, n_jobs=1, alpha=1.0, n_verb = 10, alpha_decline = False)


result = optimizer.solve()
result.dump('craskras_cathode_test_run')







