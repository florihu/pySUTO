# optimization/base_optimizer.py
from abc import ABC, abstractmethod
import numpy as np
import logging


class BaseOptimizer(ABC):
    """Abstract base class for all optimization algorithms."""
    def __init__(self, problem_data):
        self.problem = problem_data
        self.result = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def solve(self, **kwargs):
        pass

    def diagnostics(self):
        return getattr(self, "result", None)
    
    def verbose_report(self, rel_resid, it, max_rel, mean_rel, alpha, logger, id_c=None, n_verb = 20):
        '''
        Print a df of the n idc with highest relative residuals. plus the corresponing id_c
        '''
        idx_sorted = np.argsort(-rel_resid)  # descending order
        topn_idx = idx_sorted[:n_verb]
        # the variables involved in each constraint


        report_data = {
            "Constraint Index": topn_idx,
            "Relative Residual": rel_resid[topn_idx],
        }


        if id_c is not None:
            report_data["Constraint ID"] = [id_c.get(idx, None) for idx in topn_idx]

        # to df
        import pandas as pd
        report_df = pd.DataFrame(report_data)

        # give logger info
        logger.info(f"Iteration {it}: max_rel={max_rel:.4e}, mean_rel={mean_rel:.4e}, alpha={alpha:.4e}")
        logger.info("Top constraints by relative residual:\n" + report_df.to_string(index=False))