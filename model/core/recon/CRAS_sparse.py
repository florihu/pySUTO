import numpy as np
import sparse  # PyData sparse
from sparse import COO
from tqdm import tqdm
import pandas as pd

import logging

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
def interval_for_pos(a_j, l_j, u_j, tiny, r_min_global, r_max_global):
    # requires l_j <= a_j * r <= u_j  => r in [l_j/a_j, u_j/a_j]
    if abs(a_j) <= tiny:
        # multiplicative update cannot move exact zero; if zero is already feasible any r is fine
        if (l_j <= 0.0 <= u_j):
            return (r_min_global, r_max_global)
        # otherwise nudge a_j to small nonzero in direction of midpoint so we can compute interval
        mid = (l_j + u_j) / 2.0
        sign = np.sign(mid) if mid != 0.0 else 1.0
        a0 = np.clip(sign * max(tiny, abs(l_j) if l_j != 0.0 else tiny), l_j, u_j)
        vals = np.array([l_j / a0, u_j / a0])
        return (float(np.min(vals)), float(np.max(vals)))
    else:
        vals = np.array([l_j / a_j, u_j / a_j])
        return (float(np.min(vals)), float(np.max(vals)))

def interval_for_neg(a_j, l_j, u_j, tiny, r_min_global, r_max_global):
    # requires l_j <= a_j / r <= u_j  => r in [a_j/u_j, a_j/l_j]
    # watch division by zero in l_j or u_j
    if abs(a_j) <= tiny:
        if (l_j <= 0.0 <= u_j):
            return (r_min_global, r_max_global)
        mid = (l_j + u_j) / 2.0
        sign = np.sign(mid) if mid != 0.0 else 1.0
        a0 = np.clip(sign * max(tiny, abs(l_j) if l_j != 0.0 else tiny), l_j, u_j)
        vals = []
        if abs(u_j) > tiny:
            vals.append(a0 / u_j)
        else:
            vals.append(np.sign(a0) * 1e12)
        if abs(l_j) > tiny:
            vals.append(a0 / l_j)
        else:
            vals.append(np.sign(a0) * 1e12)
        vals = np.array(vals)
        return (float(np.min(vals)), float(np.max(vals)))
    else:
        vals = []
        if abs(u_j) > tiny:
            vals.append(a_j / u_j)
        else:
            vals.append(np.sign(a_j) * 1e12)
        if abs(l_j) > tiny:
            vals.append(a_j / l_j)
        else:
            vals.append(np.sign(a_j) * 1e12)
        vals = np.array(vals)
        return (float(np.min(vals)), float(np.max(vals)))

def kras_cras_sparse_pdata_bounds(
    a0,
    G,                  # a sparse.PyData sparse array, shape (NC, N)
    c,
    sigma,
    l=None,             # lower bounds per element (None -> very wide)
    u=None,             # upper bounds per element (None -> very wide)
    id_c =None,         # dict mapping constraint indices to original IDs (for logging)
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

            verbose_report(rel_resid, it, max_rel, mean_rel, id_c=id_c, n_verb =n_verb)


        # Check convergence
        if max_rel <= tol:
            converged = True
            logging.info(f"Converged at iteration {it} with max_rel={max_rel:.2e}")
            break

        # CRAS adjustment if insufficient improvement
        improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
        if improvement < delta:
            diff = Ga - c
            adj = np.sign(diff) * np.minimum(alpha * sigma, np.abs(diff))
            c = c + adj

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

    return a, c, diagnostics


def verbose_report(rel_resid, it, max_rel, mean_rel, id_c=None, n_verb = 20):
    '''
    Print a df of the n idc with highest relative residuals. plus the corresponing id_c
    '''
    idx_sorted = np.argsort(-rel_resid)  # descending order
    topn_idx = idx_sorted[:n_verb]
    report_data = {
        "Constraint Index": topn_idx,
        "Relative Residual": rel_resid[topn_idx],
    }
    if id_c is not None:
        report_data["Constraint ID"] = [id_c.get(idx, None) for idx in topn_idx]

    report_df = pd.DataFrame(report_data)
    logging.info(f"Iteration {it}, max_rel={max_rel:.2e}, mean_rel={mean_rel:.2e} \n")
    logging.info("Top constraints by relative residual:\n" + report_df.to_string(index=False))



def check_g_degenerate(G):
    """
    Check if any rows of G are all zeros (degenerate constraints).
    """
    if not isinstance(G, COO):
        G = G.tocoo() if hasattr(G, "tocoo") else G

    NC, N = G.shape
    row_sums = np.zeros((NC,), dtype=float)
    for k in range(G.data.shape[0]):
        i = int(G.coords[0][k])
        v = float(G.data[k])
        row_sums[i] += abs(v)

    degenerate_rows = np.where(row_sums == 0.0)[0]
    n_degenerate = degenerate_rows.size

    if n_degenerate > 0:
        logging.warning(f"Found {n_degenerate} degenerate (all-zero) constraints in G.")
        return True, degenerate_rows.tolist()
    else:
        logging.info("No degenerate constraints found in G.")
        return False, []

def init_problem():
    
    G_path = r"data/proc/const/G_20251027_091720.npz"
    c_path = r"data/proc/const/c_20251027_091720.npy"
    c_sigma_path = r"data/proc/const/c_sigma_20251027_091720.npy"
    c_idx_path = r"data/proc/const/c_idx_20251027_091720.npy"
    a0_path = r"data/proc/ie/ie_20251021_092053.npy"
    
    

    # --- Load data ---
    G = sparse.load_npz(G_path)  # load sparse matrix
    c = np.load(c_path)     # load constraint targets
    c_sigma = np.load(c_sigma_path)  # load constraint tolerances
    a0 = np.load(a0_path)     # load initial guessc
    c_idx = np.load(c_idx_path, allow_pickle=True)  # load constraint index map

    c_sigma[2701] = 1
    

    # generate c_idx_map
    c_idx_map = {i:idx for i, idx in enumerate(c_idx)}


    return G, c, c_sigma, a0, c_idx_map



if __name__ == "__main__":
    G, c, sigma, a0, c_idx = init_problem()
    a_star, c_adj, diag = kras_cras_sparse_pdata_bounds(
        a0, G, c, sigma, id_c=c_idx,
        alpha=1,
        tol=1e-6,
        delta=1e-2,
        max_iter=10000,
        structural_zero_mask=None,
        tiny=1e-12,
        verbose=True
    )

