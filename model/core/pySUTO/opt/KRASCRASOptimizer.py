import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from joblib import Parallel, delayed
import numpy as np
import logging
from tqdm import tqdm
import sparse


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds pySUTO to path
from opt.BaseOptimizer import BaseOptimizer
from opt.ProblemData import ProblemData
from opt.Diagnostics import Diagnostics
from opt.util import check_g_degenerate


class KRASCRASOptimizer(BaseOptimizer):
    """KRAS-CRAS optimization with sparse parallel updates and pre-initialized parameters."""
    
    DEFAULTS = {
        "alpha": 0.8,
        "tol": 1e-6,
        "delta": 1e-3,
        "max_iter": 10000,
        "structural_zero_mask": None,
        "tiny": 1e-12,
        "big_number": 1e12,
        "seed_": 42,
        "r_min_global": 1e-6,
        "r_max_global": 1e6,
        "use_residual_ordering": True,
        "n_verb": 10,
        "n_jobs": 1,
        "out_name": "results",
        "K_top_residuals": 200,
        "l": None,
        "u": None,
        "alpha_decline": True,
        "alpha_max": 1,
        "alpha_min": 1e-3,
        "c_lower": None,
        "c_upper": None,

    }
    
    def __init__(self, problem_data, **params):
        super().__init__(problem_data)
        
        merged = {**self.DEFAULTS, **params}

        # Dynamically set as attributes
        for k, v in merged.items():
            setattr(self, k, v)

        # Keep full parameter record
        self.params = merged
        
    #


    @BaseOptimizer.timed
    def solve(self):
        problem = self.problem
        rng = np.random.default_rng(self.seed_)

        # --- Initial data arrays ---
        a = np.asarray(problem.a0, dtype=float).ravel().copy()
        c = np.asarray(problem.c, dtype=float).ravel().copy()
        sigma = np.asarray(problem.sigma, dtype=float).ravel().copy()
        N = a.size
        NC, GN = problem.G.shape
        if GN != N:
            raise ValueError("G must have number of columns equal to a0.size")

        # --- Apply bounds to a ---
        l = np.zeros(N) if self.l is None else np.asarray(self.l, dtype=float).ravel().copy()
        u = np.full(N, 1e8) if self.u is None else np.asarray(self.u, dtype=float).ravel().copy()
        a = np.clip(a, l, u)

        # --- Apply structural zero mask ---
        sz_mask = np.zeros(N, dtype=bool) if self.structural_zero_mask is None else np.asarray(self.structural_zero_mask, dtype=bool)
        a[sz_mask] = 0.0

        # --- Set bounds for c ---
        c_lower = self.c_lower if getattr(self, "c_lower", None) is not None else 0.0
        c_upper = self.c_upper if getattr(self, "c_upper", None) is not None else self.big_number
        c = np.clip(c, c_lower, c_upper)

        # --- Sparse COO/CSR preparation ---
        Gcoo = problem.G.tocoo() if hasattr(problem.G, "tocoo") else problem.G
        row_idx, col_idx, data_vals = Gcoo.coords[0], Gcoo.coords[1], Gcoo.data

        rows_nonzero = [[] for _ in range(NC)]
        for k in range(data_vals.shape[0]):
            i, j, v = int(row_idx[k]), int(col_idx[k]), data_vals[k]
            rows_nonzero[i].append((j, v))

        # --- Graph coloring to build independent row blocks ---
        var_to_rows = [[] for _ in range(N)]
        for i, nz in enumerate(rows_nonzero):
            for j, _ in nz:
                var_to_rows[j].append(i)

        row_colors = -np.ones(NC, dtype=int)
        max_color = -1
        for i in range(NC):
            neighbor_colors = set()
            for j, _ in rows_nonzero[i]:
                for r in var_to_rows[j]:
                    if row_colors[r] >= 0:
                        neighbor_colors.add(row_colors[r])
            color = 0
            while color in neighbor_colors:
                color += 1
            row_colors[i] = color
            max_color = max(max_color, color)
        blocks = [np.where(row_colors == color)[0] for color in range(max_color + 1)]

        if len(blocks) == 1 or all(len(b) < 2 for b in blocks):
            self.logger.warning("No independent row blocks detected — running single-threaded for efficiency.")
            self.n_jobs = 1

        # --- History tracking ---
        history = {"max_rel_resid": [], "residuals": [], "rel_resid": [], "mean_rel_resid": [], "alpha": [], "iteration": []}
        if problem.c_idx_map is not None:
            history["id_c_rel_max"] = []

        prev_max_rel = np.inf
        converged = False

        if self.verbose:
            self.logger.info("Starting KRAS-CRAS optimization")
            self.logger.info('Number of blocks: {}'.format(len(blocks)))

        # --- Optimized update_block function ---
        def update_block(block, a):
            if len(block) == 0:
                return a

            # Collect indices and values for the block
            js_all, vs_all, row_idx_all = [], [], []
            for i in block:
                if not rows_nonzero[i]:
                    continue
                js, vs = zip(*rows_nonzero[i])
                js_all.extend(js)
                vs_all.extend(vs)
                row_idx_all.extend([i]*len(js))

            if not js_all:
                return a

            js_all = np.array(js_all, dtype=int)
            vs_all = np.array(vs_all, dtype=float)
            row_idx_all = np.array(row_idx_all, dtype=int)

            contrib = vs_all * a[js_all]

            # Accumulate positive and negative contributions per row
            S_pos = np.zeros(len(block), dtype=float)
            S_neg = np.zeros(len(block), dtype=float)
            block_idx_map = {b: k for k, b in enumerate(block)}
            indices = np.array([block_idx_map[i] for i in row_idx_all])

            pos_mask = contrib > self.tiny
            neg_mask = contrib < -self.tiny
            np.add.at(S_pos, indices[pos_mask], contrib[pos_mask])
            np.add.at(S_neg, indices[neg_mask], -contrib[neg_mask])

            # Compute r_cands
            targets = c[block]
            r_cands = np.ones_like(targets)
            mask = S_pos > self.tiny
            disc = np.zeros_like(targets)
            disc[mask] = targets[mask]**2 + 4*S_pos[mask]*S_neg[mask]
            r_cands[mask] = (targets[mask] + np.sqrt(np.maximum(disc[mask], 0.0))) / (2.0*np.maximum(S_pos[mask], self.tiny))

            mask_invalid = (~np.isfinite(r_cands)) | (r_cands <= 0)
            r_cands[mask_invalid] = np.where(np.abs(S_pos[mask_invalid]-S_neg[mask_invalid])>self.tiny,
                                            targets[mask_invalid]/(S_pos[mask_invalid]-S_neg[mask_invalid]),
                                            1.0)
            r_cands = np.clip(r_cands, self.r_min_global, self.r_max_global)

            # Update a for the block (vectorized)
            for i_row, r in zip(block, r_cands):
                nz = rows_nonzero[i_row]
                if not nz:
                    continue
                js, vs = zip(*nz)
                js = np.array(js, dtype=int)
                vs = np.array(vs, dtype=float)
                contrib = vs * a[js]
                pos_mask = contrib > self.tiny
                neg_mask = contrib < -self.tiny
                if np.any(pos_mask):
                    a[js[pos_mask]] *= r
                    np.clip(a[js[pos_mask]], l[js[pos_mask]], u[js[pos_mask]], out=a[js[pos_mask]])
                if np.any(neg_mask):
                    a[js[neg_mask]] /= r
                    np.clip(a[js[neg_mask]], l[js[neg_mask]], u[js[neg_mask]], out=a[js[neg_mask]])

            a[sz_mask] = 0.0
            return a

        # --- Main loop ---
        for it in tqdm(range(1, self.max_iter + 1), desc="KRAS-CRAS iterations"):
            Ga = problem.G.dot(a)
            residuals = Ga - c

            # Residual ordering
            if self.use_residual_ordering:
                row_order = np.argsort(-np.abs(residuals))
                topK = row_order[:self.K_top_residuals]
                rng.shuffle(topK)
                row_order = np.concatenate([topK, row_order[self.K_top_residuals:]])
            else:
                row_order = rng.permutation(NC)

            # --- Update all blocks (single-threaded or block-level parallel) ---
            for block in blocks:
                a = update_block(block, a)

            # --- Convergence check ---
            Ga = problem.G.dot(a)
            residuals = Ga - c
            rel_resid = np.abs(residuals)/(np.abs(c)+self.tiny)
            max_rel = float(np.max(rel_resid))
            mean_rel = float(np.mean(rel_resid))

            if problem.c_idx_map is not None:
                id_rel_max = np.argmax(rel_resid)
                max_rel_id_c = problem.c_idx_map.get(id_rel_max, None)
                history["id_c_rel_max"].append(max_rel_id_c)

            history["max_rel_resid"].append(max_rel)
            history["residuals"].append(residuals.copy())
            history["rel_resid"].append(rel_resid.copy())
            history["mean_rel_resid"].append(mean_rel)
            history["alpha"].append(self.alpha)
            history["iteration"].append(it)

            if self.verbose and (it % self.n_verb == 0):
                self.verbose_report(rel_resid, it, max_rel, mean_rel, self.alpha,
                                    id_c=problem.c_idx_map, n_verb=self.n_verb)

            if max_rel <= self.tol:
                converged = True
                self.logger.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
                break

            # --- CRAS adjustment with c bounds ---
            improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
            improvement_ratio = improvement / prev_max_rel if prev_max_rel != np.inf and prev_max_rel > self.tiny else 0.0

            if improvement < self.delta:
                diff = Ga - c
                adj = np.sign(diff) * np.minimum(self.alpha*sigma, np.abs(diff))
                c += adj
                c = np.clip(c, c_lower, c_upper)

            prev_max_rel = max_rel

        if self.verbose and not converged:
            self.logger.info(f"Reached max iterations ({self.max_iter}) without convergence. Final max_rel={max_rel:.2e}")

        a = np.clip(a, l, u)

        diagnostics = {
            "converged": converged,
            "params": self.params,
            "a_init": problem.a0,
            "c_init": problem.c,
            "a_final": a,
            "c_final": c,
            "iterations": it,
            "history": history,
            "final_residuals": residuals,
            "final_rel_resid": rel_resid,
            "idx_map_c": problem.c_idx_map,
        }

        self.res = Diagnostics(**diagnostics, logger=self.logger)
        return self.res

if __name__ == "__main__":
    
    G = sparse.random(shape=(100, 500), density=0.01, format='coo', random_state=42)
    c = np.random.rand(100)
    sigma = 0.1 * np.ones(100)
    a0 = np.random.rand(500)
    c_idx_map = {i: i for i in range(100)}
    vars_map = {i: ("Region1", "SectorA", "EntityX", "Region2", "SectorB", "EntityY") for i in range(500)}
    problem = ProblemData(G=G, c=c, sigma=sigma, a0=a0, c_idx_map=c_idx_map, vars_map=vars_map, c_to_var=None)
    optimizer = KRASCRASOptimizer(problem, max_iter=1000, tol=1e-3, verbose=True, n_jobs=1, alpha=1)
    result = optimizer.solve()
    print("Converged:", result.converged)

