import numpy as np
import math
from sparse import COO
from numba import njit, prange
from tqdm import tqdm
from opt.BaseOptimizer import BaseOptimizer
from opt.Diagnostics import Diagnostics

import numpy as np
from numba import njit, prange
from tqdm import tqdm



import numpy as np


try:
    from numba import njit
except Exception:
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func



@njit(cache=True, parallel=True)
def _numba_apply_updates(data, indices, indptr, rows, a_in, l, u, sz_mask, r_cands, tiny):
    a_new = a_in.copy()
    nrows = rows.shape[0]
    for ii in prange(nrows):
        row = rows[ii]
        start = indptr[row]
        end = indptr[row + 1]
        if start >= end:
            continue
        r = r_cands[ii]
        for p in range(start, end):
            j = indices[p]
            v = data[p]
            contrib = v * a_new[j]
            if contrib > tiny:
                a_new[j] = a_new[j] * r
                if a_new[j] > u[j]:
                    a_new[j] = u[j]
                elif a_new[j] < l[j]:
                    a_new[j] = l[j]
            elif contrib < -tiny:
                if r != 0.0:
                    a_new[j] = a_new[j] / r
                    if a_new[j] > u[j]:
                        a_new[j] = u[j]
                    elif a_new[j] < l[j]:
                        a_new[j] = l[j]
    for jj in range(a_new.shape[0]):
        if sz_mask[jj]:
            a_new[jj] = 0.0
    return a_new

class KRASCRASOptimizer(BaseOptimizer):
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
        "alpha_decline": True
    }

    def __init__(self, problem_data, **params):
        super().__init__(problem_data)
        merged = {**self.DEFAULTS, **params}
        for k, v in merged.items():
            setattr(self, k, v)
        self.params = merged

    @BaseOptimizer.timed
    def solve(self):
        problem = self.problem
        rng = np.random.default_rng(self.seed_)

        a = np.asarray(problem.a0, dtype=np.float64).ravel().copy()
        c = np.asarray(problem.c, dtype=np.float64).ravel().copy()
        sigma = np.asarray(problem.sigma, dtype=np.float64).ravel().copy()
        N = a.size
        NC, GN = problem.G.shape
        if GN != N:
            raise ValueError("G must have number of columns equal to a0.size")

        l = np.zeros(N, dtype=np.float64) if self.l is None else np.asarray(self.l, dtype=np.float64).ravel().copy()
        u = np.full(N, 1e8, dtype=np.float64) if self.u is None else np.asarray(self.u, dtype=np.float64).ravel().copy()
        a = np.clip(a, l, u)

        sz_mask = np.zeros(N, dtype=np.bool_) if self.structural_zero_mask is None else np.asarray(self.structural_zero_mask, dtype=np.bool_)
        a[sz_mask] = 0.0

        # Convert problem.G to sparse.COO from the sparse library
        if not isinstance(problem.G, COO):
            Gcoo = COO.from_scipy_sparse(problem.G)  # if input is scipy sparse
        else:
            Gcoo = problem.G

        data = Gcoo.data
        row = Gcoo.coords[0]
        col = Gcoo.coords[1]
        shape = Gcoo.shape

        # Build CSR-like arrays for numba inner loop
        indptr = np.zeros(shape[0] + 1, dtype=np.int64)
        indices = col.copy()
        sorted_idx = np.argsort(row)
        row_sorted = row[sorted_idx]
        indices = indices[sorted_idx]
        data = data[sorted_idx]
        count = np.bincount(row_sorted, minlength=shape[0])
        indptr[1:] = np.cumsum(count)

        # Graph coloring remains unchanged
        rows_nonzero = [[] for _ in range(NC)]
        for i in range(NC):
            start = indptr[i]
            end = indptr[i+1]
            for p in range(start, end):
                rows_nonzero[i].append((indices[p], data[p]))

        var_to_rows = [[] for _ in range(N)]
        for i, nz in enumerate(rows_nonzero):
            for j, _ in nz:
                var_to_rows[j].append(i)

        row_colors = -np.ones(NC, dtype=np.int64)
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
            if color > max_color:
                max_color = color

        blocks = [np.where(row_colors == color)[0] for color in range(max_color + 1)]

        if len(blocks) == 1 or all(len(b) < 2 for b in blocks):
            self.logger.warning("No independent row blocks detected — single-threaded mode.")
            self.n_jobs = 1

        history = {"max_rel_resid": [], "residuals": [], "rel_resid": [], "mean_rel_resid": [], "alpha": [], "iteration": []}
        if problem.c_idx_map is not None:
            history["id_c_rel_max"] = []

        prev_max_rel = np.inf
        converged = False
        imp_count = 0

        if self.verbose:
            self.logger.info("Starting KRAS-CRAS with sparse library and Numba inner loop")
            self.logger.info(f'Number of blocks: {len(blocks)}')

        for it in tqdm(range(1, self.max_iter + 1), desc="KRAS-CRAS iterations"):
            Ga = np.zeros(NC, dtype=np.float64)
            for p in range(data.shape[0]):
                Ga[row[p]] += data[p] * a[col[p]]
            residuals = Ga - c

            if self.use_residual_ordering:
                K = min(self.K_top_residuals, residuals.shape[0])
                if K > 0:
                    topK_idx = np.argpartition(-np.abs(residuals), K-1)[:K]
                    rng.shuffle(topK_idx)
                    row_order = np.concatenate([topK_idx, np.setdiff1d(np.arange(NC), topK_idx)])
                else:
                    row_order = np.arange(NC)
            else:
                row_order = rng.permutation(NC)

            for block in blocks:
                if len(block) == 0:
                    continue
                rows_arr = np.array(block, dtype=np.int64)
                sub_contrib = np.zeros(len(block), dtype=np.float64)
                for i, r in enumerate(block):
                    start = indptr[r]
                    end = indptr[r+1]
                    sub_contrib[i] = np.sum(data[start:end] * a[indices[start:end]])
                S_pos = np.where(sub_contrib > self.tiny, sub_contrib, 0.0)
                S_neg = np.where(sub_contrib < -self.tiny, -sub_contrib, 0.0)
                targets = c[rows_arr]
                r_cands = np.ones_like(targets, dtype=np.float64)
                mask = S_pos > self.tiny
                if np.any(mask):
                    disc = targets[mask]**2 + 4*S_pos[mask]*S_neg[mask]
                    r_cands[mask] = (targets[mask] + np.sqrt(np.maximum(disc,0.0))) / (2.0*np.maximum(S_pos[mask],self.tiny))
                mask_invalid = (~np.isfinite(r_cands)) | (r_cands <= 0.0)
                if np.any(mask_invalid):
                    denom = S_pos[mask_invalid]-S_neg[mask_invalid]
                    r_cands[mask_invalid] = np.where(np.abs(denom)>self.tiny, targets[mask_invalid]/denom, 1.0)
                r_cands = np.clip(r_cands, self.r_min_global, self.r_max_global)

                a_block_updated = _numba_apply_updates(data, indices, indptr, rows_arr, a, l, u, sz_mask, r_cands, self.tiny)
                changed = a_block_updated != 0.0
                a = np.where(changed, a_block_updated, a)

            Ga = np.zeros(NC, dtype=np.float64)
            for p in range(data.shape[0]):
                Ga[row[p]] += data[p] * a[col[p]]
            residuals = Ga - c
            rel_resid = np.abs(residuals)/(np.abs(c)+self.tiny)
            max_rel = float(np.max(rel_resid))
            mean_rel = float(np.mean(rel_resid))

            if problem.c_idx_map is not None:
                id_rel_max = np.argmax(rel_resid)
                history["id_c_rel_max"].append(problem.c_idx_map.get(id_rel_max,None))

            history["max_rel_resid"].append(max_rel)
            history["residuals"].append(residuals.copy())
            history["rel_resid"].append(rel_resid.copy())
            history["mean_rel_resid"].append(mean_rel)
            history["alpha"].append(self.alpha)

            if self.verbose and it%self.n_verb==0:
                self.verbose_report(rel_resid,it,max_rel,mean_rel,self.alpha,id_c=problem.c_idx_map,n_verb=self.n_verb)

            if max_rel <= self.tol:
                converged = True
                self.logger.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
                break

            improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
            if improvement < self.delta:
                imp_count += 1
                diff = Ga - c
                adj = np.sign(diff)*np.minimum(self.alpha*sigma, np.abs(diff))
                c += adj
                if imp_count>=5 and self.alpha_decline:
                    self.alpha *= 0.9
                    imp_count = 0
            else:
                imp_count = 0
                if self.alpha_decline:
                    self.alpha = min(self.alpha, self.alpha*1.05)
            prev_max_rel = max_rel

        a = np.clip(a,l,u)

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

