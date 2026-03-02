# optimization/base_optimizer.py
from abc import ABC, abstractmethod
import numpy as np
import logging
import sys
import time
import functools

class BaseOptimizer(ABC):
    """Abstract base class for all optimization algorithms."""
    def __init__(self, problem_data, verbose=True):
        self.problem = problem_data
        self.res = None
        self.verbose = verbose
        # Create a class-specific logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)  # or DEBUG if needed
        
        if verbose and not self.logger.handlers:
            # Only add handler if none exist to avoid duplicates
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False  # prevent double logging in root

    @abstractmethod
    def solve(self, **kwargs):
        pass

    def diagnostics(self):
        return getattr(self, "result", None)
    
    # --- Compact human-readable runtime formatter ---
    @staticmethod
    def _format_runtime(sec: float) -> str:
        if sec >= 3600:
            return f"{sec / 3600:.2f} h"
        if sec >= 100:
            return f"{sec / 60:.2f} min"
        return f"{sec:.2f} s"

    # --- Elegant decorator for timing any solver ---
    @staticmethod
    def timed(method):
        """Decorator that measures runtime and stores it in the solver diagnostics."""
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = method(self, *args, **kwargs)
            runtime = time.perf_counter() - t0

            # Format intelligently (s, min, h)
            if runtime < 100:
                runtime_str = f"{runtime:.2f} s"
            elif runtime < 6000:
                runtime_str = f"{runtime / 60:.2f} min"
            else:
                runtime_str = f"{runtime / 3600:.2f} h"

            # Update Diagnostics directly
            if getattr(self, "res", None) is not None:
                self.res.update(runtime_sec=runtime, runtime_str=runtime_str)

            # Optional log
            if hasattr(self, "logger") and self.logger:
                self.logger.info(f"Runtime: {runtime_str}")

            return result
        return wrapper
    
    def verbose_report(self, rel_resid, it, max_rel, mean_rel, alpha, id_c=None, n_verb = 20):
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
        self.logger.info(f"Iteration {it}: max_rel={max_rel:.4e}, mean_rel={mean_rel:.4e}, alpha={alpha:.4e}")
        self.logger.info("Top constraints by relative residual:\n" + report_df.to_string(index=False))