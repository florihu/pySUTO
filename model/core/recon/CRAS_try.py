import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def kras_cras(
    a0,
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
    a0 : array_like, shape (m*n,)
        Initial vectorized matrix (column-stacked).
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
    sigma_a : estimated standard errors for a (m,n) 
    sigma_c: estimated standard errors for c (NC,)


    Todo
    ----
    - Add error propagation for sigma_a from sigma_c 
    """
    N = a0.shape[0]
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
    a = a0.astype(float)   # can contain negative values
    # enforce structural zeros in vectorized form
    if np.any(sz_mask):
        a[_vec(sz_mask).astype(bool)] = 0.0

    # Ensure no exact zero denominators in intermediate calculations
    # (we allow zeros in a, but for multiplicative updates we ensure tiny positive for computations)
    eps = tiny

    history = {
        "max_rel_resid": [],
        "residuals": [],
        "rel_resid": [],
        "c_history": [c.copy()],
        "a_history": [a.copy()], 
        'sigma_a_history': [],
        'sigma_c_history': []

    }

    prev_max_rel = np.inf
    converged = False
    # Main iteration loop
    for it in range(1, max_iter + 1):
        a_before_sweep = a.copy()

        # Sweep constraints sequentially
        for i in range(NC):
            # current constraint row
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
                    # prevent runaway if gcur ~ 0 this means there are no positive contributors
                    # (target / (gcur + eps) dirty slop for relative change + eps for numeric safety)
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
                a[pos_idx] = a[pos_idx] * r
            if np.any(neg_idx):
                a[neg_idx] = a[neg_idx] / r

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
        history["rel_resid"].append(rel_resid.copy())

        history["a_history"].append(a.copy())

        # lets calculate the sigma_a empirically from a_history
        a_stack = np.vstack(history["a_history"])
        sigma_a = np.std(a_stack, axis=0)
        history['sigma_a_history'].append(sigma_a.copy())

        # estimate sigma_c from residuals and sigma
        c_stack = np.vstack(history["c_history"])
        sigma_c = np.std(c_stack, axis=0)
        history['sigma_c_history'].append(sigma_c.copy())


        if verbose:
            print(f"[KRAS] it {it:4d}  max_rel_resid={max_rel:.3e}, max_resid={np.max(np.abs(residuals)):.3e}")

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
            

    a_final = a.reshape((m, n))

    diagnostics = {
        "converged": converged,
        "iterations": it,
        "history": history,
        "final_residuals": residuals,
        "final_rel_resid": rel_resid,
        'final_sigma_a': sigma_a,
        'final_sigma_c': sigma_c
    }

    return a_final, c, diagnostics


if __name__ == "__main__":
    

    m, n = 2, 2

    a = 1
    b = 2
    c = 2

    A0 = np.array([[ a, b],
                [ c,  1]])
    #col stacked
    a = A0.reshape(-1)

    # build c
    row_sum = np.array([1, 3])
    col_sum = np.array([1, 3])
    c = np.hstack([row_sum, col_sum])
    # set the constraint for a2,2 to 1
    c = np.hstack([c, 1])


    sigma = 0.02 * np.maximum(np.abs(c), 1.0)

    # c2 hihg uncertainty
    sigma[1] = 0.5 * np.maximum(np.abs(c[1]), 1.0)


    # build G matrix constrain row and column sums plus field a2,2 to 1
    G= [[1, 1, 0, 0],  # row 1 sum
        [0, 0, 1, 1],  # row 2 sum
        [1, 0, 1, 0],  # col 1 sum
        [0, 1, 0, 1],  # col 2 sum
        [0, 0, 0, 1]]  # a2,2 = 1


    

    def plot_max_rel_resid_per_alpha(alphas= [0.01, 0.1, 0.5, 1.0]):
        
        store = {}

        for alpha in alphas:
            a_star, c_star, diag = kras_cras(
            a,
            G,
            c,
            sigma.copy(),
            alpha=alpha,
            tol=1e-6,
            delta=1e-8,
            max_iter=100000,
            structural_zero_mask=None,
            tiny=1e-12,
            verbose=True
            )
            if diag is not None:
                diag_df = pd.DataFrame(diag['history']['max_rel_resid'], columns=['max_rel_resid'])
                store[alpha] = diag_df
        
        # plot rel res per alpha
        plt.figure(figsize=(10, 6))
        for alpha, df in store.items():
            plt.plot(df.index + 1, df['max_rel_resid'], marker='o', label=f'alpha={alpha}')
        plt.xlabel('Iteration')
        plt.ylabel('Max Relative Residual')
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Convergence of KRAS + CRAS Algorithm for Different Alpha')
        plt.legend()
        plt.savefig(r'figs\explo\kras_cras_convergence_per_alpha.png')
        plt.show()
        

        return None

    
    plot_max_rel_resid_per_alpha()

    # diag_df = pd.DataFrame(diag['history']['max_rel_resid'], columns=['max_rel_resid'])

    # print("Final balanced matrix A:")
    # print(a.reshape((m, n)))

    # # plot iterations vs. max_rel_resid
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=diag_df, marker='o')
    # plt.xlabel('Iteration')
    # plt.ylabel('Max Relative Residual')
    # plt.title('Convergence of KRAS + CRAS Algorithm')
    # plt.show()
    # plt.savefig(r'figs\explo\kras_cras_convergence.png')
    

    # # plot for the constraints the history of c values
    # c_history = np.array(diag['history']['c_history'])
    # c_df = pd.DataFrame(c_history, columns=[f'c_{i+1}' for i in range(c_history.shape[1])])
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=c_df)
    # plt.xlabel('Iteration')
    # plt.ylabel('Constraint Values')
    # plt.title('History of Constraint Values')
    # plt.savefig(r'figs\explo\kras_cras_c_history.png')
    # plt.show()


    
