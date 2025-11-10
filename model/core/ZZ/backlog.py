# def kras_cras_sparse_pdata_parallel(
#     a0,
#     G,                
#     c,
#     sigma,
#     alpha=0.8,
#     tol=1e-6,
#     delta=1e-5,
#     max_iter=10000,
#     structural_zero_mask=None,
#     tiny=1e-12,
#     n_jobs=-1,         # number of parallel jobs (-1 = all cores)
#     verbose=False,
# ):
#     """
#     KRAS + CRAS with PyData sparse arrays, parallelized and randomized row updates.
#     """

#     # --- Setup logging
#     logger = logging.getLogger("KRAS-CRAS")
#     logger.setLevel(logging.INFO if verbose else logging.WARNING)
#     if not logger.hasHandlers():
#         handler = logging.StreamHandler()
#         handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
#         logger.addHandler(handler)

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
#         "amad": [],
#         "residuals": [],
#         "rel_resid": [],
#     }

#     prev_max_rel = np.inf
#     converged = False

#     def update_constraint(i, a_local):
#         """Update a subset of variables for one constraint row."""
#         nz = rows_nonzero[i]
#         if not nz:
#             return np.zeros_like(a_local)  # no change

#         js, vs = zip(*nz)
#         js = np.array(js, dtype=int)
#         vs = np.array(vs, dtype=float)
#         a_sub = a_local[js]
#         contrib = vs * a_sub

#         pos_mask = contrib > 0
#         neg_mask = contrib < 0

#         S_pos = contrib[pos_mask].sum() if np.any(pos_mask) else 0.0
#         S_neg = -contrib[neg_mask].sum() if np.any(neg_mask) else 0.0
#         target = float(c[i])

#         if S_pos <= tiny and S_neg <= tiny:
#             return np.zeros_like(a_local)

#         # Compute scaling factor r
#         if S_pos > tiny:
#             disc = target**2 + 4.0 * S_pos * S_neg
#             disc = max(disc, 0.0)
#             denom = 2.0 * max(S_pos, tiny)
#             r = (target + np.sqrt(disc)) / denom
#             if not np.isfinite(r) or r <= 0.0:
#                 gcur = S_pos - S_neg
#                 r = target / (gcur + tiny) if abs(gcur) > tiny else 1.0
#         else:
#             gcur = S_pos - S_neg
#             r = target / (gcur + tiny) if abs(gcur) > tiny else 1.0

#         r = np.clip(r, 1e-6, 1e6)

#         delta_a = np.zeros_like(a_local)
#         if np.any(pos_mask):
#             js_pos = js[pos_mask]
#             delta_a[js_pos] = a_local[js_pos] * (r - 1.0)
#         if np.any(neg_mask):
#             js_neg = js[neg_mask]
#             delta_a[js_neg] += a_local[js_neg] * (1.0/r - 1.0)

#         return delta_a

#     for it in range(1, max_iter + 1):

#         # Randomize row order for better convergence
#         row_order = np.random.permutation(NC)

#         # --- Parallel update over constraints
#         deltas = Parallel(n_jobs=n_jobs, backend="loky")(
#             delayed(update_constraint)(i, a) for i in row_order
#         )

#         # Apply updates
#         for delta_a in deltas:
#             a += delta_a
#         if np.any(sz_mask):
#             a[sz_mask] = 0.0
#         a = np.clip(a, 1e-12, 1e12)

#         # Recompute residuals
#         Ga = G.dot(a)
#         residuals = Ga - c
#         rel_resid = np.abs(residuals) / (np.abs(c) + tiny)
#         max_rel = float(np.max(rel_resid))
#         amad = float(np.mean(np.abs(residuals) / (sigma + tiny)))

#         history["max_rel_resid"].append(max_rel)
#         history["amad"].append(amad)
#         history["residuals"].append(residuals.copy())
#         history["rel_resid"].append(rel_resid.copy())

#         if verbose and (it % 10 == 0):
#             logger.info(f"[KRAS-sparse-PData] it {it:4d} max_rel={max_rel:.2e}, amad={amad:.2e}")

#         if max_rel <= tol:
#             converged = True
#             break

#         # CRAS adjustment if insufficient improvement
#         improvement = prev_max_rel - max_rel if prev_max_rel != np.inf else np.inf
#         if improvement < delta:
#             diff = Ga - c
#             adj = np.sign(diff) * np.minimum(alpha * sigma, np.abs(diff))
#             c += adj
#         prev_max_rel = max_rel

#     diagnostics = {
#         "converged": converged,
#         "iterations": it,
#         "history": history,
#         "final_residuals": residuals,
#         "final_rel_resid": rel_resid,
#     }

#     return a, c, diagnostics