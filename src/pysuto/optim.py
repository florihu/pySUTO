# @dataclass
# class OptimResult:
#     t:             int
#     F_opt:         np.ndarray      # (E,)
#     sigma_opt:     np.ndarray      # (E,) — filled by OptUncertainty
#     residuals:     np.ndarray      # G @ F_opt − c
#     objective:     float
#     solver_status: str
#     G:             np.ndarray
#     c:             np.ndarray

# class LinearOptimiser:
#     """
#     Minimise  (f − f0)ᵀ Σ⁻¹ (f − f0)  subject to  G f = c.

#     Σ = diag(σ_agg²)  (inverse-variance weighting, consistent with
#     the aggregation step).  Warm-start: f0 for t>0 is F_opt[t-1].

#     Solved via scipy.optimize.lsq_linear or a direct KKT solve when
#     G has full row rank.
#     """
#     def __init__(self, acc_struct)
#     def solve(self, G: np.ndarray, c: np.ndarray,
#               F_agg: np.ndarray, sigma_agg: np.ndarray,
#               warm_start: np.ndarray | None = None) -> OptimResult