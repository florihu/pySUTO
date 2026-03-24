# import numpy as np
# from dataclasses import dataclass
# from pysuto.registry import DataFeedRegistry


# class FeedCollector:
    
#     Loads a selection of feeds from the registry and assembles
#     their F_root / σ_root vectors into aligned arrays.
    
#     def __init__(self, registry: DataFeedRegistry, acc_struct)
#     def select(self, feed_ids: list[str]) -> "FeedCollector"
#     def load_results(self) -> dict[str, dict]   # {id: {F_root, σ_root}}
#     def aggregate(self) -> "AggregationResult"  # delegates to agg.py

# @dataclass
# class AggregationResult:
#     F_agg:           np.ndarray          # shape (E,)
#     sigma_agg:       np.ndarray          # shape (E,)
#     source_weights:  np.ndarray          # shape (N_feeds, E)
#     feed_ids:        list[str]
#     imputation_log:  list[dict]          # filled in by ImputationEngine
#     agg_id:          str