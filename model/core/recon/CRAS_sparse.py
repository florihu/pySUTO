import numpy as np
import sparse  # PyData sparse
from sparse import COO

def kras_cras_sparse_pdata(
    a0,
    G,                # a sparse.PyData sparse array, shape (NC, m*n)
    c,
    sigma,
    alpha=0.5,
    tol=1e-6,
    delta=1e-8,
    max_iter=10000,
    structural_zero_mask=None,
    tiny=1e-12,
    verbose=False,
):
    """
    KRAS + CRAS, adapted to use PyData sparse arrays (sparse.COO, GCXS etc.)
    for very sparse G (density << 1%).
    """

    m, n = a0.shape
    N = m * n

    # Convert G to a sparse.COO (or an efficient sparse format) if not already
    # COO is good for iterating non-zero entries; GCXS maybe faster for matmul etc.
    if not isinstance(G, sparse.SparseArray):
        G = sparse.asarray(G)
    # Optionally convert to a format good for row-slicing / fast indexing
    # e.g. GCXS with row-major compressed layout
    # (depending on version of sparse, may use .asformat or set_layout)
    # For now let’s assume COO or any format where we can get coords + data

    c = np.asarray(c, dtype=float).ravel().copy()
    sigma = np.asarray(sigma, dtype=float).ravel().copy()
    NC, GN = G.shape
    if GN != N:
        raise ValueError("G must have number of columns equal to m*n (vectorized A0).")

    if structural_zero_mask is None:
        sz_mask = np.zeros((m, n), dtype=bool)
    else:
        sz_mask = np.asarray(structural_zero_mask, dtype=bool)
        if sz_mask.shape != (m, n):
            raise ValueError("structural_zero_mask must have same shape as A0")

    # vectorize a
    a = a0.astype(float).ravel()
    if np.any(sz_mask):
        a[sz_mask.ravel()] = 0.0

    eps = tiny

    # Compute initial Ga = G @ a  (sparse mat-vec)
    # sparse library supports .dot or @ for matmul
    Ga = G.dot(a)  # or G @ a

    history = {
        "max_rel_resid": [],
        "amad": [],
        "residuals": [],
        "rel_resid": [],
    }

    prev_max_rel = np.inf
    converged = False

    # For iterating rows non-zero structure, get G’s non-zero coords
    # If G is COO, coords are (2, nnz), with first dimension row indices
    # Let’s extract once
    Gcoo = G.tocoo() if hasattr(G, "tocoo") else G  # ensure a COO-like view
    row_idx = Gcoo.coords[0]
    col_idx = Gcoo.coords[1]
    data_vals = Gcoo.data

    # Build for each row a list of non-zero (col, value) pairs
    # This could be memory-heavy, but for very sparse G ok
    rows_nonzero = [[] for _ in range(NC)]
    for k in range(data_vals.shape[0]):
        i = int(row_idx[k])
        j = int(col_idx[k])
        v = data_vals[k]
        rows_nonzero[i].append( (j, v) )

    for it in range(1, max_iter + 1):
        # Sweep over constraints
        for i in range(NC):
            nz = rows_nonzero[i]
            if not nz:
                continue

            # gather j, v, a[j]
            js, vs = zip(*nz)  # list of column indices & corresponding G entries
            js = np.array(js, dtype=int)
            vs = np.array(vs, dtype=float)
            a_sub = a[js]
            contrib = vs * a_sub

            pos_mask = contrib > 0.0
            neg_mask = contrib < 0.0

            if np.any(pos_mask):
                S_pos = contrib[pos_mask].sum()
            else:
                S_pos = 0.0

            if np.any(neg_mask):
                S_neg = -contrib[neg_mask].sum()
            else:
                S_neg = 0.0

            target = float(c[i])

            # Skip if essentially no dependency
            if S_pos <= tiny and S_neg <= tiny:
                continue

            # Compute scaling r
            if S_pos > tiny:
                disc = target * target + 4.0 * S_pos * S_neg
                if disc < 0.0:
                    disc = 0.0
                r = (target + np.sqrt(disc)) / (2.0 * S_pos)
                if r <= 0.0 or not np.isfinite(r):
                    gcur = S_pos - S_neg
                    r = (target / (gcur + eps)) if abs(gcur) > tiny else 1.0
            else:
                gcur = S_pos - S_neg
                r = (target / (gcur + eps)) if abs(gcur) > tiny else 1.0

            r = max(1e-6, min(1e6, r))

            # Apply multiplicative update on a[js] only
            # And update Ga accordingly (so we don't recompute G @ a from scratch)
            # Keep old_sub for neg-mask and pos-mask
            if np.any(pos_mask):
                # For pos contrib: a_js *= r
                js_pos = js[pos_mask]
                old_vals_pos = a[js_pos].copy()
                a[js_pos] = old_vals_pos * r

                # Ga (for *all* constraints) changes by G[:, js_pos] * (a_new - a_old)
                # But we only know G at rows that include those js_pos
                # For simplicity, update only Ga[i] here (row i) since this update only acts to satisfy constraint i
                # To fully update other constraints you'd need the row structure for all rows involving those js_pos
                # For now, we recompute Ga = G.dot(a) after the full sweep (see below)
            if np.any(neg_mask):
                js_neg = js[neg_mask]
                old_vals_neg = a[js_neg].copy()
                a[js_neg] = old_vals_neg / r

            if np.any(sz_mask):
                a[sz_mask.ravel()] = 0.0

        # After full sweep, recompute Ga
        Ga = G.dot(a)

        residuals = Ga - c
        rel_resid = np.abs(residuals) / (np.abs(c) + tiny)
        max_rel = float(np.max(rel_resid))
        amad = float(np.mean(np.abs(residuals) / (sigma + tiny)))

        history["max_rel_resid"].append(max_rel)
        history["amad"].append(amad)
        history["residuals"].append(residuals.copy())
        history["rel_resid"].append(rel_resid.copy())

        if verbose and (it % 10 == 0):
            print(f"[KRAS-sparse-PData] it {it:4d} max_rel={max_rel:.2e}, amad={amad:.2e}")

        if max_rel <= tol:
            converged = True
            break

        improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
        if improvement < delta:
            # CRAS adjustment
            diff = Ga - c
            adj = np.sign(diff) * np.minimum(alpha * sigma, np.abs(diff))
            c = c + adj
        prev_max_rel = max_rel

    A_final = a.reshape((m, n))
    diagnostics = {
        "converged": converged,
        "iterations": it,
        "history": history,
        "final_residuals": residuals,
        "final_rel_resid": rel_resid,
    }
    return A_final, c, diagnostics
