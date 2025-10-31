from joblib import Parallel, delayed
import numpy as np
import logging
from tqdm import tqdm
from .BaseOptimizer import BaseOptimizer
from .Diagnostics import Diagnostics
from .util import check_g_degenerate, interval_for_pos, interval_for_neg


class KRASCRASOptimizer(BaseOptimizer):
    """Implementation of the KRAS-CRAS optimization algorithm with sparse parallel updates."""
    def __init__(self, problem_data, **params):
        super().__init__(problem_data)
        self.params = params

    def solve(self):

        a0 = self.problem.a0
        G = self.problem.G
        c0 = self.problem.c
        sigma =self.problem.sigma
        l=self.params.get("l", None)
        u=self.params.get("u", None)
        id_c=self.problem.c_idx_map
        vars_map=self.problem.vars_map
        c_to_var=self.problem.c_to_var
        alpha=self.params.get("alpha", 0.8)
        tol=self.params.get("tol", 1e-6)
        delta=self.params.get("delta", 1e-3)
        max_iter=self.params.get("max_iter", 10000)
        structural_zero_mask=self.params.get("structural_zero_mask", None)
        tiny=self.params.get("tiny", 1e-12)
        verbose=self.params.get("verbose", False)
        seed_=self.params.get("seed_", 42)
        r_min_global=self.params.get("r_min_global", 1e-6)
        r_max_global=self.params.get("r_max_global", 1e6)
        use_residual_ordering=self.params.get("use_residual_ordering", True)
        n_verb=self.params.get("n_verb", 10)
        n_jobs=self.params.get("n_jobs", -1)
        out_name = self.params.get("out_name", 'results')

        c = c0.copy()

        if verbose:
            self.logger.info("Starting KRAS-CRAS optimization")
            

        a = np.asarray(a0, dtype=float).ravel().copy()
        c = np.asarray(c, dtype=float).ravel().copy()
        sigma = np.asarray(sigma, dtype=float).ravel().copy()
        N = a.size
        NC, GN = G.shape
        if GN != N:
            raise ValueError("G must have number of columns equal to a0.size")

        if l is None:
            l = np.zeros(N, dtype=float)
        else:
            l = np.asarray(l, dtype=float).ravel().copy()
        if u is None:
            u = np.full(N, 1e8, dtype=float)
        else:
            u = np.asarray(u, dtype=float).ravel().copy()
        a = np.clip(a, l, u)

        sz_mask = np.zeros(N, dtype=bool) if structural_zero_mask is None else np.asarray(structural_zero_mask, dtype=bool)
        a[sz_mask] = 0.0

        # --- Prepare COO sparse structure
        Gcoo = G.tocoo() if hasattr(G, "tocoo") else G
        row_idx = Gcoo.coords[0]
        col_idx = Gcoo.coords[1]
        data_vals = Gcoo.data

        # Build row-wise non-zero indices
        rows_nonzero = [[] for _ in range(NC)]
        for k in range(data_vals.shape[0]):
            i, j, v = int(row_idx[k]), int(col_idx[k]), data_vals[k]
            rows_nonzero[i].append((j, v))

        # --- Build independent row blocks (graph coloring)
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

        history = {"max_rel_resid": [], "residuals": [], "rel_resid": [], "mean_rel_resid": []}
        if id_c is not None:
            history["id_c_rel_max"] = []

        prev_max_rel = np.inf
        converged = False
        rng = np.random.default_rng(seed_)

        # --- Main loop ---
        for it in tqdm(range(1, max_iter + 1), desc="KRAS-CRAS iterations"):
            Ga = G.dot(a)
            residuals = Ga - c

            if use_residual_ordering:
                row_order = np.argsort(-np.abs(residuals))
                K = min(200, len(row_order))
                topK = row_order[:K]
                rng.shuffle(topK)
                row_order = np.concatenate([topK, row_order[K:]])
            else:
                row_order = rng.permutation(NC)

            # --- Vectorized update per block ---
            def update_block_vectorized(block):
                if len(block) == 0:
                    return np.zeros_like(a)
                a_block = a.copy()
                # concatenate all indices and values in the block
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
                pos_mask = contrib > tiny
                neg_mask = contrib < -tiny

                S_pos = np.zeros_like(block, dtype=float)
                S_neg = np.zeros_like(block, dtype=float)
                block_idx_map = {b: k for k, b in enumerate(block)}
                for i_row, p, n, t in zip(rows, pos_mask, neg_mask, contrib):
                    idx = block_idx_map[i_row]
                    if p:
                        S_pos[idx] += t
                    elif n:
                        S_neg[idx] -= t

                # Compute r candidates for all rows in block
                targets = c[block]
                r_cands = np.ones_like(targets)
                mask = S_pos > tiny
                disc = np.zeros_like(targets)
                disc[mask] = targets[mask]**2 + 4*S_pos[mask]*S_neg[mask]
                r_cands[mask] = (targets[mask] + np.sqrt(np.maximum(disc[mask],0.0))) / (2.0*np.maximum(S_pos[mask], tiny))
                mask_invalid = (~np.isfinite(r_cands)) | (r_cands <= 0)
                r_cands[mask_invalid] = np.where(np.abs(S_pos[mask_invalid]-S_neg[mask_invalid])>tiny,
                                                targets[mask_invalid]/(S_pos[mask_invalid]-S_neg[mask_invalid]),
                                                1.0)
                r_cands = np.clip(r_cands, r_min_global, r_max_global)

                # Apply multiplicative updates
                a_new = a_block.copy()
                for i_row, r in zip(block, r_cands):
                    nz = rows_nonzero[i_row]
                    if not nz:
                        continue
                    js, vs = zip(*nz)
                    js = np.array(js, dtype=int)
                    vs = np.array(vs, dtype=float)
                    contrib = vs * a_new[js]
                    pos_mask = contrib > tiny
                    neg_mask = contrib < -tiny
                    if np.any(pos_mask):
                        js_pos = js[pos_mask]
                        a_new[js_pos] = np.clip(a_new[js_pos]*r, l[js_pos], u[js_pos])
                    if np.any(neg_mask):
                        js_neg = js[neg_mask]
                        a_new[js_neg] = np.clip(a_new[js_neg]/r, l[js_neg], u[js_neg])
                if sz_mask is not None:
                    a_new[sz_mask] = 0.0
                return a_new

            # --- Parallel update of blocks ---
            for block in blocks:
                updated = Parallel(n_jobs=n_jobs)(delayed(update_block_vectorized)([i]) for i in block)
                for a_block in updated:
                    a = np.where(a_block != 0, a_block, a)

            # Recompute residuals
            Ga = G.dot(a)
            residuals = Ga - c
            rel_resid = np.abs(residuals)/(np.abs(c)+tiny)
            max_rel = float(np.max(rel_resid))
            mean_rel = float(np.mean(rel_resid))

            if id_c is not None:
                id_rel_max = np.argmax(rel_resid)
                max_rel_id_c = id_c.get(id_rel_max, None)
                history["id_c_rel_max"].append(max_rel_id_c)

            history["max_rel_resid"].append(max_rel)
            history["residuals"].append(residuals.copy())
            history["rel_resid"].append(rel_resid.copy())
            history["mean_rel_resid"].append(mean_rel)

            if verbose and (it % 10 == 0):
                # i want to inherit the verbose report function from base optimizer
                self.verbose_report(rel_resid, it, max_rel, mean_rel, alpha, self.logger, id_c=id_c, n_verb=n_verb)

            if max_rel <= tol:
                converged = True
                self.logger.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
                break

            # CRAS adjustment
            improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
            if improvement < delta:
                diff = Ga - c
                adj = np.sign(diff) * np.minimum(alpha*sigma, np.abs(diff))
                c += adj
                alpha *= 0.9
            else:
                alpha = min(alpha, alpha*1.05)
            prev_max_rel = max_rel

        if verbose and not converged:
            self.logger.info(f"Reached max iterations ({max_iter}) without convergence. Final max_rel={max_rel:.2e}")

        a = np.clip(a, l, u)
        diagnostics = {
            "converged": converged,
            "parameters": self.params,
            "a_init": a0,
            "c_init": c0,
            "a_final": a,
            "c_final": c,
            "iterations": it,
            "history": history,
            "final_residuals": residuals,
            "final_rel_resid": rel_resid,
        }

        res = Diagnostics(**diagnostics, logger = self.logger, out_name=out_name)
        res.to_json(f'kras_cras_diagnostics.json')

        self.result = res

        return res
