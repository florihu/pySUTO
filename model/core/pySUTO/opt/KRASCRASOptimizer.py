from joblib import Parallel, delayed
import numpy as np
import logging
from tqdm import tqdm
from .BaseOptimizer import BaseOptimizer
from .Diagnostics import Diagnostics
from .util import check_g_degenerate, interval_for_pos, interval_for_neg



class KRASCRASOptimizer(BaseOptimizer):
    """KRAS-CRAS optimization with sparse parallel updates and pre-initialized parameters."""
    
    DEFAULTS = {
        "alpha": 0.8,
        "tol": 1e-6,
        "delta": 1e-3,
        "max_iter": 10000,
        "structural_zero_mask": None,
        "tiny": 1e-12,
        "seed_": 42,
        "r_min_global": 1e-6,
        "r_max_global": 1e6,
        "use_residual_ordering": True,
        "n_verb": 10,
        "n_jobs": -1,
        "out_name": "results",
        "K_top_residuals": 200,
        "l": None,
        "u": None,
    }
    
    def __init__(self, problem_data, **params):
        super().__init__(problem_data)
        
        merged = {**self.DEFAULTS, **params}

        # Dynamically set as attributes
        for k, v in merged.items():
            setattr(self, k, v)

        # Keep full parameter record
        self.params = merged
        
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

        # --- Apply bounds ---
        l = np.zeros(N) if self.l is None else np.asarray(self.l, dtype=float).ravel().copy()
        u = np.full(N, 1e8) if self.u is None else np.asarray(self.u, dtype=float).ravel().copy()
        a = np.clip(a, l, u)

        # --- Apply structural zero mask ---
        sz_mask = np.zeros(N, dtype=bool) if self.structural_zero_mask is None else np.asarray(self.structural_zero_mask, dtype=bool)
        a[sz_mask] = 0.0

        # --- Sparse COO preparation ---
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

        # --- History tracking ---
        history = {"max_rel_resid": [], "residuals": [], "rel_resid": [], "mean_rel_resid": [], "alpha": [], "iteration": []}
        if problem.c_idx_map is not None:
            history["id_c_rel_max"] = []

        prev_max_rel = np.inf
        converged = False

        if self.verbose:
            self.logger.info("Starting KRAS-CRAS optimization")

        # --- Main loop ---
        for it in tqdm(range(1, self.max_iter + 1), desc="KRAS-CRAS iterations"):

            # Compute residuals
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

            # --- Parallel block updates ---
            def update_block(block):
                if len(block) == 0:
                    return np.zeros_like(a)
                a_block = a.copy()
                js_all, vs_all, rows = [], [], []
                for i in block:
                    if not rows_nonzero[i]:
                        continue
                    js, vs = zip(*rows_nonzero[i])
                    js_all.extend(js)
                    vs_all.extend(vs)
                    rows.extend([i]*len(js))
                js_all = np.array(js_all, dtype=int)
                vs_all = np.array(vs_all, dtype=float)
                rows = np.array(rows, dtype=int)
                a_sub = a_block[js_all]
                contrib = vs_all * a_sub
                S_pos = np.zeros_like(block, dtype=float)
                S_neg = np.zeros_like(block, dtype=float)
                block_idx_map = {b: k for k, b in enumerate(block)}
                for i_row, val in zip(rows, contrib):
                    idx = block_idx_map[i_row]
                    if val > self.tiny:
                        S_pos[idx] += val
                    elif val < -self.tiny:
                        S_neg[idx] -= val
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
                a_new = a_block.copy()
                for i_row, r in zip(block, r_cands):
                    nz = rows_nonzero[i_row]
                    if not nz:
                        continue
                    js, vs = zip(*nz)
                    js = np.array(js, dtype=int)
                    vs = np.array(vs, dtype=float)
                    contrib = vs * a_new[js]
                    pos_mask = contrib > self.tiny
                    neg_mask = contrib < -self.tiny
                    if np.any(pos_mask):
                        a_new[js[pos_mask]] = np.clip(a_new[js[pos_mask]]*r, l[js[pos_mask]], u[js[pos_mask]])
                    if np.any(neg_mask):
                        a_new[js[neg_mask]] = np.clip(a_new[js[neg_mask]]/r, l[js[neg_mask]], u[js[neg_mask]])
                a_new[sz_mask] = 0.0
                return a_new

            for block in blocks:
                updated = Parallel(n_jobs=self.n_jobs)(delayed(update_block)([i]) for i in block)
                for a_block in updated:
                    a = np.where(a_block != 0, a_block, a)

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
            

            if self.verbose and (it % self.n_verb == 0):
                self.verbose_report(rel_resid, it, max_rel, mean_rel, self.alpha, 
                                    id_c=problem.c_idx_map, n_verb=self.n_verb)

            if max_rel <= self.tol:
                converged = True
                self.logger.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
                break

            # --- CRAS adjustment ---
            improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
            if improvement < self.delta:
                diff = Ga - c
                adj = np.sign(diff) * np.minimum(self.alpha*sigma, np.abs(diff))
                c += adj
                self.alpha *= 0.9
            else:
                self.alpha = min(self.alpha, self.alpha*1.05)
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
