# optimization/data.py
import numpy as np
import pandas as pd
import sparse
import logging

class ProblemData:
    """
    Encapsulates all data structures required for optimization algorithms.
    """
    def __init__(self, G, c, sigma, a0, c_idx_map, c_to_var, vars_map = None, ):
        self.G = G
        self.c = c
        self.sigma = sigma
        self.a0 = a0
        self.c_idx_map = c_idx_map
        self.vars_map = vars_map
        self.c_to_var = c_to_var

    @classmethod
    def from_files(cls, paths):
        """Load a problem instance from file paths (e.g., from init_problem)."""
        G = sparse.load_npz(paths["G"])
        c = np.load(paths["c"])
        sigma = np.load(paths["sigma"])
        a0 = np.load(paths["a0"])
        c_idx = np.load(paths["c_idx"], allow_pickle=True)
        vars_df = pd.read_parquet(paths["vars"])

        c_idx_map = {i: idx for i, idx in enumerate(c_idx)}
        vars_map = {
            i: (row.Region_origin, row.Sector_origin, row.Entity_origin,
                row.Region_destination, row.Sector_destination, row.Entity_destination)
            for i, row in vars_df.iterrows()
        }

        c_to_var = cls._build_constraint_to_var_map(c_idx, G)
        return cls(G, c, sigma, a0, c_idx_map, vars_map, c_to_var)

    @staticmethod
    def _build_constraint_to_var_map(c_idx, G):
        """Map constraints to variables that appear in them."""
        Gcoo = G.tocoo() if hasattr(G, "tocoo") else G
        mapping = {c: set() for c in c_idx}
        for k in range(Gcoo.data.shape[0]):
            i, j = int(Gcoo.coords[0][k]), int(Gcoo.coords[1][k])
            mapping[c_idx[i]].add(j)
        return mapping

    def summary(self):
        logging.info(f"G shape: {self.G.shape}, "
                     f"Constraints: {len(self.c)}, Variables: {self.a0.size}")