# class OptimPipeline:
#     """
#     Outer loop over T timesteps.  For each t:
#       1. ConstraintBuilder.build(t)  →  G_t, c_t
#       2. LinearOptimiser.solve(...)  →  OptimResult (no σ_opt yet)
#       3. OptUncertainty.compute(...) →  fills OptimResult.sigma_opt
#       4. Appends to OptimRun

#     Warm-start: passes OptimResult[t-1].F_opt as f0 to t.
#     Persists the full OptimRun to the registry on completion.
#     """
#     def __init__(self, acc_struct, registry: DataFeedRegistry,
#                  constraint_builder: ConstraintBuilder,
#                  optimiser: LinearOptimiser,
#                  unc: OptUncertainty)
#     def run(self, agg_result: AggregationResult,
#             timesteps: list[int]) -> "OptimRun"

# @dataclass
# class OptimRun:
#     run_id:           str
#     results:          list[OptimResult]   # len == T
#     agg_result_id:    str
#     registry_ids:     list[str]           # feed ids that went in
#     created_at:       str
#     solver_meta:      dict                # scipy version, tolerance, etc.