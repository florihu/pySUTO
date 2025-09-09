import numpy as np

def _vec(A):
    return A.T.reshape(-1)   # column-stack to 1d

def _mat(a_vec, m, n):
    return a_vec.reshape((n, m)).T



def build_rowcol_G(m, n):
    G_row = np.zeros((m, m*n))
    for i in range(m):
        idx = [i + j*m for j in range(n)]
        G_row[i, idx] = 1.0
    G_col = np.zeros((n, m*n))
    for j in range(n):
        idx = [i + j*m for i in range(m)]
        G_col[j, idx] = 1.0
    return np.vstack([G_row, G_col])



def kras_cras(
    A0,
    G,
    c,
    sigma,
    alpha=0.5,
    tol=1e-6,
    delta=1e-8,
    max_iter=1000,
    structural_zero_mask=None,
    tiny=1e-12,
    verbose=False
):
    """
    KRAS + CRAS implementation following Lenzen et al. (2009).

    Parameters
    ----------
    A0 : array_like, shape (m, n)
        Initial matrix (can have mixed sign).
    G : array_like, shape (NC, m*n)
        Constraint coefficient matrix acting on vectorized A (column-stacked).
    c : array_like, shape (NC,)
        Constraint targets.
    sigma : array_like, shape (NC,)
        Standard errors for constraints (reliability).
    alpha : float in [0,1]
        CRAS step size (how strongly to move c toward realized G a).
    tol : float
        Convergence tolerance for max relative residual: max_i |Ga_i - c_i| / (|c_i|+tiny)
    delta : float
        Minimal improvement threshold used to detect stalling/oscillation.
    max_iter : int
    structural_zero_mask : bool array (m,n) or None
        True where entries are structural zeros (kept exactly zero).
    tiny : float
        Numerical safeguard.
    verbose : bool

    Returns
    -------
    A_final : (m, n) balanced matrix (column-stack inverse)
    c_adj : adjusted constraints (NC,)
    diagnostics : dict with history, residuals, converged flag
    """
    A0 = np.asarray(A0, dtype=float)
    m, n = A0.shape
    N = m * n
    G = np.asarray(G, dtype=float)
    c = np.asarray(c, dtype=float).ravel().copy()
    sigma = np.asarray(sigma, dtype=float).ravel().copy()
    NC = G.shape[0]

    if G.shape[1] != N:
        raise ValueError("G must have number of columns equal to m*n (vectorized A0).")

    if structural_zero_mask is None:
        sz_mask = np.zeros((m, n), dtype=bool)
    else:
        sz_mask = np.asarray(structural_zero_mask, dtype=bool)
        if sz_mask.shape != (m, n):
            raise ValueError("structural_zero_mask must be same shape as A0")

    # vectorize
    a = _vec(A0).astype(float)   # can contain negative values
    # enforce structural zeros in vectorized form
    if np.any(sz_mask):
        a[_vec(sz_mask).astype(bool)] = 0.0

    # Ensure no exact zero denominators in intermediate calculations
    # (we allow zeros in a, but for multiplicative updates we ensure tiny positive for computations)
    eps = tiny

    history = {
        "max_rel_resid": [],
        "residuals": [],
        "c_history": [c.copy()],
        "a_history_sample": []
    }

    prev_max_rel = np.inf
    converged = False

    for it in range(1, max_iter + 1):
        a_before_sweep = a.copy()

        # Sweep constraints sequentially
        for i in range(NC):
            gi = G[i, :]

            # compute contributions g_{ij} * a_j
            contrib = gi * a

            # classify indices
            pos_idx = contrib > 0.0
            neg_idx = contrib < 0.0

            S_pos = float(np.sum(contrib[pos_idx]))    # positive sum (>=0)
            # S_neg is positive magnitude of negative contributions
            S_neg = float(-np.sum(contrib[neg_idx]))   # >=0

            target = float(c[i])

            # If both sums are ~0 and target ~0, no scaling necessary
            if (S_pos <= tiny and S_neg <= tiny):
                # nothing to do for this constraint (no dependent variables)
                continue

            # Compute scalar r using Lenzen's quadratic formula where possible
            if S_pos > tiny:
                # r_i = (c_i + sqrt(c_i^2 + 4 * S_pos * S_neg)) / (2 * S_pos)
                disc = target * target + 4.0 * S_pos * S_neg
                # numeric safety
                disc = max(disc, 0.0)
                r = (target + np.sqrt(disc)) / (2.0 * S_pos)
                # r should be positive; add protective clamp
                if r <= 0.0 or not np.isfinite(r):
                    # fallback: ratio-based small step
                    gcur = S_pos - S_neg
                    r = max(1e-6, min(1e6, (target / (gcur + eps)) if abs(gcur) > tiny else 1.0))
            else:
                # no positive contributors -> cannot compute quadratic formula
                # fallback: try small corrective step aimed at closing gap
                gcur = S_pos - S_neg  # negative or zero
                r = 1.0
                if abs(gcur) > tiny:
                    r = max(1e-6, min(1e6, target / (gcur + eps)))

            # clamp r to avoid explosion
            r = max(1e-6, min(1e6, r))

            # Multiplicative update: a_j <- a_j * r^{ sign(contrib_j) }
            # sign(contrib_j) = +1 if contrib>0 ; -1 if contrib<0 ; 0 if contrib==0
            # So we multiply positives by r and divide negatives by r.
            # For numerical robust, update in-place using masks
            if np.any(pos_idx):
                a[pos_idx] = a[pos_idx] * (r ** 1.0)
            if np.any(neg_idx):
                a[neg_idx] = a[neg_idx] / (r ** 1.0)

            # restore structural zeros if present
            if np.any(sz_mask):
                a[_vec(sz_mask).astype(bool)] = 0.0

        # after full sweep compute residuals
        Ga = G @ a
        residuals = Ga - c
        rel_resid = np.abs(residuals) / (np.abs(c) + tiny)
        max_rel = float(np.max(rel_resid))

        history["max_rel_resid"].append(max_rel)
        history["residuals"].append(residuals.copy())
        history["c_history"].append(c.copy())
        if it % 10 == 0:
            history["a_history_sample"].append(a.copy())

        if verbose:
            print(f"[KRAS] it {it:4d}  max_rel_resid={max_rel:.3e}")

        # Check convergence
        if max_rel <= tol:
            converged = True
            if verbose:
                print(f"Converged at iteration {it}.")
            break

        # Check improvement vs previous sweep
        improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf

        if improvement >= delta:
            # good improvement -> continue
            prev_max_rel = max_rel
            continue
        else:
            # little/no improvement -> we are stalling/oscillating.
            # Apply CRAS constraint adjustment using reliability sigma and step alpha (Eq.13)
            if verbose:
                print(f"[CRAS] small improvement {improvement:.3e} < delta ({delta:.3e}) -> adjusting constraints (alpha={alpha}).")

            # compute Ga again (realization)
            Ga = G @ a
            diff = Ga - c    # positive -> realization > target -> increase c toward Ga

            # Adjust each constraint by at most alpha * sigma_i, bounded by |diff_i|
            adj = np.sign(diff) * np.minimum(alpha * sigma, np.abs(diff))
            # update c towards realization
            c = c + adj

            # record and continue (do not reset prev_max_rel to inf; keep for next loop)
            history["c_history"].append(c.copy())
            prev_max_rel = max_rel

    A_final = _mat(a, m, n)
    diagnostics = {
        "converged": converged,
        "iterations": it,
        "history": history,
        "final_residuals": residuals,
        "final_rel_resid": rel_resid
    }

    return A_final, c, diagnostics




if __name__ == "__main__":
    m, n = 3, 3
    A0 = np.array([[ 2.0, -0.1, 0.3],
                [ 0.0,  1.5, 0.2],
                [ 0.1,  0.2, 1.0]])
    G = build_rowcol_G(m, n)
    # Suppose row sums (supply) and column sums (use) we want:
    row_sum = np.array([2.5, 1.9, 1.5])
    col_sum = np.array([1.9, 2.1, 1.9])
    c = np.hstack([row_sum, col_sum])
    # give sigma as 2% of c (example)
    sigma = 0.02 * np.maximum(np.abs(c), 1.0)

    A_bal, c_adj, diag = kras_cras(A0, G, c, sigma, alpha=0.5, tol=1e-8, verbose=True)