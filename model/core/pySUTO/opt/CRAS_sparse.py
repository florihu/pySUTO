import numpy as np
import sparse  # PyData sparse
from sparse import COO
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import logging
import os
import json


# def kras_cras_sparse_pdata(
#     a0,
#     G,                  # a sparse.PyData sparse array, shape (NC, N)
#     c,
#     sigma,
#     id_c =None,         # dict mapping constraint indices to original IDs (for logging)
#     alpha=0.8,          # max fraction of sigma to adjust c by
#     tol=1e-6,           # tolerance for convergence
#     delta=1e-3,         # min improvement to trigger CRAS adjustment
#     max_iter=10000,
#     structural_zero_mask=None,
#     tiny=1e-12,         # to avoid divide-by-zero
#     verbose=False,
#     seed_ = 42,         # random seed for reproducibility
# ):
#     """
#     Robust KRAS + CRAS for PyData sparse arrays (sparse.COO, GCXS, etc.).

#     todo : we need at some point bounds on a (min, max) per element

#     """
#     # --- Setup logging
#     if verbose:
#         logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
#         logger = logging.getLogger("KRAS-CRAS")
#     # --- Prepare arrays
#     a = np.asarray(a0, dtype=float).ravel().copy()
#     c = np.asarray(c, dtype=float).ravel().copy()
#     sigma = np.asarray(sigma, dtype=float).ravel().copy()
#     N = a.size
#     NC, GN = G.shape
#     if GN != N:
#         raise ValueError("G must have number of columns equal to a0.size")

#     if structural_zero_mask is None:
#         sz_mask = np.zeros((N,), dtype=bool)
#     else:
#         sz_mask = np.asarray(structural_zero_mask, dtype=bool)
#         if sz_mask.shape != (N,):
#             raise ValueError("structural_zero_mask must have same shape as a0")
#         a[sz_mask] = 0.0

#     # --- Prepare sparse COO structure
#     Gcoo = G.tocoo() if hasattr(G, "tocoo") else G
#     row_idx = Gcoo.coords[0]
#     col_idx = Gcoo.coords[1]
#     data_vals = Gcoo.data

#     # Build row-wise non-zero indices
#     rows_nonzero = [[] for _ in range(NC)]
#     for k in range(data_vals.shape[0]):
#         i = int(row_idx[k])
#         j = int(col_idx[k])
#         v = data_vals[k]
#         rows_nonzero[i].append((j, v))

#     history = {
#         "max_rel_resid": [],
#         "residuals": [],
#         "rel_resid": [],
#     }


#     if id_c is not None:
#         history["id_c_rel_max"] = []

#     prev_max_rel = np.inf
#     converged = False

#     for it in tqdm(range(1, max_iter + 1), desc="KRAS-CRAS iterations"):
#         # random row order for better convergence
#         row_order = np.random.Generator.permutation(np.random.default_rng(seed_), NC)
#         # --- KRAS update over all constraints
#         for i in row_order:
#             nz = rows_nonzero[i]
#             if not nz:
#                 continue

#             js, vs = zip(*nz)
#             js = np.array(js, dtype=int)
#             vs = np.array(vs, dtype=float)
#             a_sub = a[js]
#             contrib = vs * a_sub

#             pos_mask = contrib > 0
#             neg_mask = contrib < 0

#             S_pos = contrib[pos_mask].sum() if np.any(pos_mask) else 0.0
#             S_neg = -contrib[neg_mask].sum() if np.any(neg_mask) else 0.0
#             target = float(c[i])

#             # Skip tiny contributions
#             if S_pos <= tiny and S_neg <= tiny:
#                 continue

#             # Compute scaling factor r safely
#             if S_pos > tiny:
#                 disc = target**2 + 4.0 * S_pos * S_neg
#                 disc = max(disc, 0.0)
#                 denom = 2.0 * max(S_pos, tiny)
#                 r = (target + np.sqrt(disc)) / denom
#                 if not np.isfinite(r) or r <= 0.0:
#                     gcur = S_pos - S_neg
#                     r = target / (gcur + tiny) if abs(gcur) > tiny else 1.0
#             else:
#                 gcur = S_pos - S_neg
#                 r = target / (gcur + tiny) if abs(gcur) > tiny else 1.0

#             r = np.clip(r, 1e-6, 1e6)

#             # Apply multiplicative update
#             if np.any(pos_mask):
#                 js_pos = js[pos_mask]
#                 a[js_pos] = np.clip(a[js_pos] * r, 1e-12, 1e12)

#             if np.any(neg_mask):
#                 js_neg = js[neg_mask]
#                 a[js_neg] = np.clip(a[js_neg] / r, 1e-12, 1e12)

#             if np.any(sz_mask):
#                 a[sz_mask] = 0.0

#         # Recompute residuals
#         Ga = G.dot(a)
#         residuals = Ga - c
#         rel_resid = np.abs(residuals) / (np.abs(c) + tiny)
#         max_rel = float(np.max(rel_resid))

#         if id_c is not None:
#             id_rel_max = np.argmax(rel_resid)
#             # find value in id_c
#             max_rel_id_c = id_c.get(id_rel_max, None)
#             history["id_c_rel_max"].append(max_rel_id_c)

#         history["max_rel_resid"].append(max_rel)
#         history["residuals"].append(residuals.copy())
#         history["rel_resid"].append(rel_resid.copy())

#         if verbose and (it % 10 == 0):
#             logger.info(f"[KRAS-sparse] it {it:4d} \n max_rel={max_rel:.2e} \n"
#                         + (f", max_rel_id_c={max_rel_id_c}" if id_c is not None else "")
#                         )
            
#         # Check convergence
#         if max_rel <= tol:
#             converged = True
#             break

#         # CRAS adjustment if insufficient improvement
#         improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
#         if improvement < delta:
#             diff = Ga - c
#             adj = np.sign(diff) * np.minimum(alpha * sigma, np.abs(diff))
#             c = c + adj
            
#         prev_max_rel = max_rel

#     diagnostics = {
#         "converged": converged,      
#         "iterations": it,
#         "history": history,
#         "final_residuals": residuals,
#         "final_rel_resid": rel_resid,
#     }

#     return a, c, diagnostics

# helper functions to compute per-element interval for pos or neg case

    



def kras_cras_sparse_pdata_bounds(
    a0,
    G,                  # a sparse.PyData sparse array, shape (NC, N)
    c,
    sigma,
    l=None,             # lower bounds per element (None -> very wide)
    u=None,             # upper bounds per element (None -> very wide)
    id_c =None,         # dict mapping constraint indices to original IDs (for logging)
    vars_map = None,  # dict mapping variable indices to original variable IDs (for logging)
    c_to_var = None, # dict mapping constraint indices to sets of variable indices (for logging)
    alpha=0.8,          # max fraction of sigma to adjust c by
    tol=1e-6,           # tolerance for convergence
    delta=1e-3,         # min improvement to trigger CRAS adjustment
    max_iter=10000,
    structural_zero_mask=None,
    tiny=1e-12,         # to avoid divide-by-zero
    verbose=False,
    seed_ = 42, # random seed for reproducibility
    r_min_global = 1e-6, # global min r allowed
    r_max_global = 1e6, # global max r allowed
    use_residual_ordering = True, 
    n_verb = 10  # whether to order constraints by residual magnitude each iteration
            
):
    """
    Robust KRAS + CRAS for PyData sparse arrays (sparse.COO, GCXS, etc.)
    Now supports per-element lower bounds `l` and upper bounds `u`.

    Notes on bounds:
    - `l` and `u` must be broadcastable to shape (N,). If omitted, very wide bounds are used.
    - The update chooses a candidate r as before, then intersects the feasible r-intervals
      induced by element-wise bounds for the affected indices and clips r into that interval.
    - If an element is tiny/zero and zero is outside [l,u], the element is nudged so multiplicative
      updates can move it toward the feasible region (multiplicative updates cannot change exact zeros).
    """
    # --- Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger("KRAS-CRAS")

    # --- Prepare arrays
    a = np.asarray(a0, dtype=float).ravel().copy()
    c = np.asarray(c, dtype=float).ravel().copy()
    sigma = np.asarray(sigma, dtype=float).ravel().copy()
    N = a.size
    NC, GN = G.shape
    if GN != N:
        raise ValueError("G must have number of columns equal to a0.size")

    # --- bounds handling: set defaults if None and validate
    if l is None:
        l = np.full((N,), 0, dtype=float)
    else:
        l = np.asarray(l, dtype=float).ravel().copy()
    if u is None:
        u = np.full((N,), 1e8, dtype=float)
    else:
        u = np.asarray(u, dtype=float).ravel().copy()

    if l.shape != (N,) or u.shape != (N,):
        raise ValueError("l and u must have shape equal to a0.size")
    if np.any(l > u):
        raise ValueError("elementwise l must be <= u")

    # apply structural zero mask (if provided)
    if structural_zero_mask is None:
        sz_mask = np.zeros((N,), dtype=bool)
    else:
        sz_mask = np.asarray(structural_zero_mask, dtype=bool)
        if sz_mask.shape != (N,):
            raise ValueError("structural_zero_mask must have same shape as a0")
        a[sz_mask] = 0.0

    # basic check: ensure initial a respects bounds (clip)
    a = np.minimum(np.maximum(a, l), u)

    # --- Prepare sparse COO structure
    Gcoo = G.tocoo() if hasattr(G, "tocoo") else G
    row_idx = Gcoo.coords[0]
    col_idx = Gcoo.coords[1]
    data_vals = Gcoo.data

    # Build row-wise non-zero indices
    rows_nonzero = [[] for _ in range(NC)]
    for k in range(data_vals.shape[0]):
        i = int(row_idx[k])
        j = int(col_idx[k])
        v = data_vals[k]
        rows_nonzero[i].append((j, v))

    history = {
        "max_rel_resid": [],
        "residuals": [],
        "rel_resid": [],
        "mean_rel_resid": [],
    }

    if id_c is not None:
        history["id_c_rel_max"] = []

    prev_max_rel = np.inf
    converged = False

  
    for it in tqdm(range(1, max_iter + 1), desc="KRAS-CRAS iterations"):
        # random row order for better convergence

        Ga = G.dot(a)
        residuals = Ga - c

        rng = np.random.default_rng(seed_)

        if use_residual_ordering:
            # Order by descending absolute residual (largest first)
            row_order = np.argsort(-np.abs(residuals))  # largest absolute residual first
            #Optionally, you can shuffle within top K to avoid cycles:
            K = min(200, len(row_order))
            topK = row_order[:K]
            rng.shuffle(topK)
            row_order = np.concatenate([topK, row_order[K:]])
        else:
            
            row_order = rng.permutation(NC)
        

        # --- KRAS update over all constraints
        for i in row_order:
            nz = rows_nonzero[i]
            if not nz:
                continue

            js, vs = zip(*nz)
            js = np.array(js, dtype=int)
            vs = np.array(vs, dtype=float)
            a_sub = a[js]
            contrib = vs * a_sub

            pos_mask = contrib > 0
            neg_mask = contrib < 0

            S_pos = contrib[pos_mask].sum() if np.any(pos_mask) else 0.0
            S_neg = -contrib[neg_mask].sum() if np.any(neg_mask) else 0.0
            target = float(c[i])

            # Skip tiny contributions
            if S_pos <= tiny and S_neg <= tiny:
                continue

            # Compute scaling factor r candidate safely (as before, but robust)
            if S_pos > tiny:
                disc = target**2 + 4.0 * S_pos * S_neg
                disc = max(disc, 0.0)
                denom = 2.0 * max(S_pos, tiny)
                r_cand = (target + np.sqrt(disc)) / denom
                if not np.isfinite(r_cand) or r_cand <= 0.0:
                    gcur = S_pos - S_neg
                    r_cand = target / (gcur + tiny) if abs(gcur) > tiny else 1.0
            else:
                gcur = S_pos - S_neg
                r_cand = target / (gcur + tiny) if abs(gcur) > tiny else 1.0

            # Clip candidate into a globally allowable range immediately
            r_cand = float(np.clip(r_cand, r_min_global, r_max_global))

            # --- compute feasible r-interval induced by bounds for affected indices
            # start with very wide feasible interval, then intersect
            r_lower = r_min_global
            r_upper = r_max_global

            # intersect intervals for pos indices
            if np.any(pos_mask):
                js_pos = js[pos_mask]
                for j_idx in js_pos:
                    rlo_j, rhi_j = interval_for_pos(a[j_idx], l[j_idx], u[j_idx], tiny, r_min_global, r_max_global)
                    # only positive r matters; intersect
                    r_lower = max(r_lower, rlo_j)
                    r_upper = min(r_upper, rhi_j)
                    if r_lower > r_upper:
                        break

            # intersect intervals for neg indices
            if r_lower <= r_upper and np.any(neg_mask):
                js_neg = js[neg_mask]
                for j_idx in js_neg:
                    rlo_j, rhi_j = interval_for_neg(a[j_idx], l[j_idx], u[j_idx], tiny, r_min_global, r_max_global)
                    r_lower = max(r_lower, rlo_j)
                    r_upper = min(r_upper, rhi_j)
                    if r_lower > r_upper:
                        break

            # Ensure interval is positive and not degenerate
            r_lower = max(r_lower, r_min_global)
            r_upper = min(r_upper, r_max_global)

            # If feasible interval is empty, we cannot find an r that respects *all* element-wise bounds simultaneously.
            # Strategy: clip candidate r to the nearest endpoint (this will keep as many values feasible as possible),
            # apply it and then clip the a entries themselves to [l,u] as a safety net.
            if r_lower > r_upper:
                # infeasible: choose r that is as close as possible to candidate but positive
                if r_cand < r_lower:
                    r_safe = r_lower
                elif r_cand > r_upper:
                    r_safe = r_upper
                else:
                    r_safe = r_cand
            else:
                # normal case: clip candidate into feasible range
                r_safe = float(np.clip(r_cand, r_lower, r_upper))

            # final safety bounds on r
            r_safe = float(np.clip(r_safe, r_min_global, r_max_global))

            # --- Apply multiplicative update with r_safe
            if np.any(pos_mask):
                js_pos = js[pos_mask]
                # multiply and clip to bounds per-element
                new_vals = a[js_pos] * r_safe
                # final per-element clip: clamp to [l,u]
                new_vals = np.minimum(np.maximum(new_vals, l[js_pos]), u[js_pos])
                a[js_pos] = new_vals

            if np.any(neg_mask):
                js_neg = js[neg_mask]
                # divide and clip
                new_vals = a[js_neg] / r_safe
                new_vals = np.minimum(np.maximum(new_vals, l[js_neg]), u[js_neg])
                a[js_neg] = new_vals

            if np.any(sz_mask):
                a[sz_mask] = 0.0

        # Recompute residuals
        Ga = G.dot(a)
        residuals = Ga - c
        rel_resid = np.abs(residuals) / (np.abs(c) + tiny)
        max_rel = float(np.max(rel_resid))
        mean_rel = float(np.mean(rel_resid))

        if id_c is not None:
            id_rel_max = np.argmax(rel_resid)
            # find value in id_c
            max_rel_id_c = id_c.get(id_rel_max, None)
            history["id_c_rel_max"].append(max_rel_id_c)

        history["max_rel_resid"].append(max_rel)
        history["residuals"].append(residuals.copy())
        history["rel_resid"].append(rel_resid.copy())
        history["mean_rel_resid"].append(mean_rel)

        if verbose and (it % 10 == 0):

            verbose_report(rel_resid, it, max_rel, mean_rel, alpha, c_to_vars=c_to_var, id_c=id_c, n_verb =n_verb)


        # Check convergence
        if max_rel <= tol:
            converged = True
            logging.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
            break

        no_improve = 0
        alpha0 = alpha
        alpha_decay = 0.9

        # CRAS adjustment if insufficient improvement
        improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
        if improvement < delta:
            no_improve += 1
            diff = Ga - c
            adj = np.sign(diff) * np.minimum(alpha * sigma, np.abs(diff))
            c = c + adj

            # decay
            if no_improve >= 5:
                alpha = alpha * alpha_decay

        else:
            no_improve = 0
            alpha = min(alpha0, alpha *1.05)

        prev_max_rel = max_rel



    if verbose and not converged:
        logging.info(f"Reached max iterations ({max_iter}) without convergence. Final max_rel={max_rel:.2e}")

    # Final safety: ensure a in [l,u]
    a = np.minimum(np.maximum(a, l), u)

    diagnostics = {
        "converged": converged,
        "iterations": it,
        "history": history,
        "final_residuals": residuals,
        "final_rel_resid": rel_resid,
    }

    store_stuff_as_json(a, c, diagnostics)

    return a, c, diagnostics





    # # based on the constraint ids in report data get the variables involved
    # if c_to_vars is not None:
    #     var_lists = []
    #     for idx in report_data["Constraint ID"]:

    #         var_set = c_to_vars.get(idx, set())
    #         var_lists.append(", ".join(map(str, sorted(var_set))))
    #     report_data["Involved Variables"] = var_lists



    report_df = pd.DataFrame(report_data)
    logging.info(f"Iteration {it}, max_rel={max_rel:.2e}, mean_rel={mean_rel:.2e}, alpha={alpha} \n")
    logging.info("Top constraints by relative residual:\n" + report_df.to_string(index=False))







def init_problem():
    
    G_path = r"data\proc\const\G_20251028_145401.npz"
    c_path = r"data\proc\const\c_20251028_145401.npy"
    c_sigma_path = r"data\proc\const\c_sigma_20251028_145401.npy"
    c_idx_path = r"data\proc\const\c_idx_20251028_145401.npy"
    
    a0_path = r"data/proc/ie/ie_20251028_113607.npy"
    
    var_p = r"data/proc/index/index_20251020_154514.parquet"
    

    # --- Load data ---
    G = sparse.load_npz(G_path)  # load sparse matrix
    c = np.load(c_path)     # load constraint targets
    c_sigma = np.load(c_sigma_path)  # load constraint tolerances
    a0 = np.load(a0_path)     # load initial guessc
    c_idx = np.load(c_idx_path, allow_pickle=True)  # load constraint index map
    vars = pd.read_parquet(var_p)

    # generate c_idx_map
    c_idx_map = {i:idx for i, idx in enumerate(c_idx)}
    #vars map
    vars_map = {i: (row.Region_origin, row.Sector_origin, row.Entity_origin, row.Region_destination, row.Sector_destination, row.Entity_destination) for i, row in vars.iterrows()}

    c_to_var = cs_to_vars(c_idx, G)

    return G, c, c_sigma, a0, c_idx_map, vars_map, c_to_var



def cs_to_vars(c_idx, G):
    """
    Build a dictionary mapping each constraint (identified by c_idx)
    to the set of variable indices that are active (nonzero) in G.

    Parameters
    ----------
    c_idx : array-like of str
        Identifiers for each constraint (length = number of rows in G)
    G : sparse.COO or similar
        Sparse constraint matrix (rows = constraints, cols = variables)

    Returns
    -------
    dict
        {constraint_id: set(variable_indices)}
    """
    Gcoo = G.tocoo() if hasattr(G, "tocoo") else G

    row_idx = Gcoo.coords[0]
    col_idx = Gcoo.coords[1]

    c_to_vars = {c: set() for c in c_idx}

    for k in range(Gcoo.data.shape[0]):
        i = int(row_idx[k])  # constraint index
        j = int(col_idx[k])  # variable index
        c_to_vars[c_idx[i]].add(j)

    return c_to_vars


def kras_cras_sparse_pdata_bounds_parallel_vec(
    a0,
    G,
    c,
    sigma,
    l=None,
    u=None,
    id_c=None,
    vars_map=None,
    c_to_var=None,
    alpha=0.8,
    tol=1e-6,
    delta=1e-3,
    max_iter=10000,
    structural_zero_mask=None,
    tiny=1e-12,
    verbose=False,
    seed_=42,
    r_min_global=1e-6,
    r_max_global=1e6,
    use_residual_ordering=True,
    n_verb=10,
    n_jobs=-1
):
    """
    Fully vectorized, conflict-free parallel KRAS + CRAS for sparse arrays.
    Updates are done block-wise in parallel using Joblib.
    """
    # --- Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger("KRAS-CRAS")

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
            updated = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(update_block_vectorized)([i]) for i in block)
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
            verbose_report(rel_resid, it, max_rel, mean_rel, alpha, c_to_vars=c_to_var, id_c=id_c, n_verb=n_verb)
        
        elif verbose and (it % 50 == 0):
            plot_max_rel_resid(history, tol=tol)
        


        # Check convergence
        if max_rel <= tol:
            converged = True
            logging.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
            break

        # CRAS adjustment
        improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf

        improve_count = 0
        alpha0 = alpha
        if improvement < delta:
            diff = Ga - c
            adj = np.sign(diff) * np.minimum(alpha*sigma, np.abs(diff))
            c += adj

            if improve_count >= 5:
                alpha *= 0.9 # decay
        else:
            alpha = min(alpha0, alpha*1.05)
            improve_count = 0
        prev_max_rel = max_rel

    if verbose and not converged:
        logging.info(f"Reached max iterations ({max_iter}) without convergence. Final max_rel={max_rel:.2e}")

    a = np.clip(a, l, u)
    diagnostics = {
        "converged": converged,
        "iterations": it,
        "history": history,
        "final_residuals": residuals,
        "final_rel_resid": rel_resid,
    }

    return a, c, diagnostics



def store_stuff_as_json(a, c, diagnostics):
    folder_path = r"data/proc/recon"
    os.makedirs(folder_path, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"recon_{timestamp}.json")

    data_to_store = {
        "a": a.tolist() if hasattr(a, "tolist") else a,
        "c": c.tolist() if hasattr(c, "tolist") else c,
        "diagnostics": diagnostics
    }

    with open(file_path, 'w') as f:
        json.dump(data_to_store, f, indent=4)

    print(f"✅ Data saved to {file_path}")



def plot_max_rel_resid(history, tol=None, display_time=5):
    """
    Plot max_rel_resid and automatically close the figure after display_time seconds.
    
    Parameters
    ---------- 
    tol : float, optional
        Convergence tolerance for reference line.
    display_time : float
        Time in seconds to display the figure before closing.
    """
    max_rel = history['max_rel_resid']
    change_max_rel = np.diff(max_rel)
    iterations = range(1, len(max_rel) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(iterations, max_rel, marker='o', linestyle='-', alpha=0.7, label='max_rel_resid')
    plt.plot(iterations[1:], change_max_rel, marker='x', linestyle='--', alpha=0.7, label='change in max_rel_resid')
    if tol is not None:
        plt.axhline(tol, color='red', linestyle='--', label=f'Tolerance = {tol}')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Max relative residual')
    plt.title('KRAS-CRAS Convergence')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.show(block=False)  # non-blocking
    plt.pause(display_time)  # display for 'display_time' seconds
    plt.close()  # close automatically


if __name__ == "__main__":
    G, c, sigma, a0, c_idx, v_map, c_to_var = init_problem()
    a_star, c_adj, diag = kras_cras_sparse_pdata_bounds_parallel_vec(
        a0,
        G,
        c,
        sigma,
        l=None,
        u=None,
        id_c=c_idx,
        vars_map=v_map,
        c_to_var=c_to_var,
        alpha=1,
        tol=1e-2,
        delta=1e-2,
        max_iter=10000,
        structural_zero_mask=None,
        tiny=1e-12,
        verbose=True,
        seed_=42,
        r_min_global=1e-6,
        r_max_global=1e6,
        use_residual_ordering=True,
        n_verb=10,
        n_jobs=-1
    )
