"""
base.py
=======
``BaseImputer`` + the two helpers shared across every imputer:

* :func:`_apply_cv_cap` — σ-cap to cv_cap × |f| (4 callers across 3 imputers).
* :func:`_log`           — one dict per filled edge (used by every imputer).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_cv_cap(sigma_val: float, f_val: float, cv_cap: float) -> tuple[float, bool]:
    """
    Cap σ at cv_cap × |f|. Returns (sigma_capped, was_capped).

    cv_cap <= 0 or cv_cap == inf disables the cap. f == 0 disables the cap.
    """
    if (cv_cap <= 0.0 or not np.isfinite(cv_cap)
            or f_val == 0.0 or not np.isfinite(sigma_val) or not np.isfinite(f_val)):
        return sigma_val, False
    cap = cv_cap * abs(f_val)
    if sigma_val > cap:
        return cap, True
    return sigma_val, False


def _log(
    edge_id: int, step: int, method: str,
    source_edges: list[int], coefficients: list[float],
    f_after: float, sigma_after: float,
    description: str, param_id: Optional[str],
) -> dict:
    return {
        "edge_id":      edge_id,
        "step":         step,
        "method":       method,
        "source_edges": list(source_edges),
        "coefficients": list(coefficients),
        "f_before":     np.nan,
        "f_after":      float(f_after),
        "sigma_before": np.nan,
        "sigma_after":  float(sigma_after),
        "description":  description,
        "param_id":     param_id,
    }


# ---------------------------------------------------------------------------
# BaseImputer
# ---------------------------------------------------------------------------

class BaseImputer(ABC):
    """
    Abstract base for all imputers.

    impute() receives the current f and sigma arrays (modified in-place),
    the missing mask at the START of this step (bool, True where NaN),
    the pipeline step index for logging, and the accounting structure.
    Returns a list of log-entry dicts (one per filled edge).
    """

    @abstractmethod
    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        ...

    @property
    def source_paths(self) -> list[str]:
        return []

    def config(self) -> dict:
        """
        Return a JSON-serialisable snapshot of this imputer's configuration.

        Default impl: every public attribute (no leading underscore) whose
        value is JSON-able. Subclasses can override for richer detail.
        """
        import json as _json
        out: dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                _json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)
        return out
