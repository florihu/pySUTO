# class OptUncertainty:
#     """
#     Propagate uncertainty from c into F_opt using the KKT sensitivity.

#     For the equality-constrained WLS:
#         F_opt = f0 + Σ Gᵀ (G Σ Gᵀ)⁻¹ (c − G f0)

#     Jacobian ∂F_opt/∂c = Σ Gᵀ (G Σ Gᵀ)⁻¹

#     So:  Σ_opt = J · Σ_c · Jᵀ
#     and  σ_opt = sqrt(diag(Σ_opt))

#     Σ_c = diag(σ_agg²) — uncertainty in the constraint RHS
#     propagated from the aggregation step.
#     """
#     def compute(self, G: np.ndarray, Sigma_diag: np.ndarray,
#                 f0: np.ndarray) -> np.ndarray   # returns σ_opt (E,)