
# import numpy as np
# from abc import ABC, abstractmethod
# from typing import Literal
# from .collector import AggregationResult


# class ImputationEngine:
#     """Runs registered imputers in order; logs every fill."""
#     def __init__(self, acc_struct)
#     def register(self, imputer: "BaseImputer") -> None
#     def run(self, agg_result: AggregationResult) -> AggregationResult

# class BaseImputer(ABC):
#     @abstractmethod
#     def impute(self, F: np.ndarray, sigma: np.ndarray,
#                mask_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]

# class ParametricImputer(BaseImputer):
#     """
#     Two edges linked by a parameter α: f_b = α · f_a.
#     If f_a known and f_b missing → f_b = α · f_a, σ_b = α · σ_a.
#     Parametric relations come from acc_struct or a user-supplied dict.
#     """
#     def __init__(self, relations: dict[tuple, tuple, float])
#     # relations: {(edge_b_id): (edge_a_id, alpha)}

# class TimeSeriesImputer(BaseImputer):
#     """
#     Fill a missing edge at time t by linear interpolation or
#     last-observation-carried-forward from adjacent timesteps.
#     """
#     def __init__(self, method: Literal["linear", "locf", "nocb"])