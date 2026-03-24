# # class ConstraintBuilder:
#     """
#     Assembles the G matrix and c vector for one timestep t.

#     Two constraint types, both expressed as rows in G @ f = c:

#     1. Supply = Use balance (from AccountingStructure):
#          for each node n:  Σ(inflows) − Σ(outflows) = 0

#     2. Parametric efficiency:
#          f_b − α · f_a = 0
#          (same relations as ParametricImputer, just expressed as constraints)
#     """
#     def __init__(self, acc_struct, parametric_relations: dict)
#     def build(self, t: int,
#               F_agg: np.ndarray,
#               sigma_agg: np.ndarray) -> tuple[np.ndarray, np.ndarray]
#               # returns G (M_constraints × E), c (M_constraints,)
#     def add_custom_row(self, coefficients: dict[int, float], rhs: float)