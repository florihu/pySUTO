"""
lp.py
=====
Pure-math Lp / power-mean σ combiner.

Single source of truth for ( Σ |x_i|^p )^(1/p) used across Stage 1 / 2 / 3.
Imported by every imputer and by ConstraintEngine's σ-regularisation.
"""

from __future__ import annotations

import numpy as np


def lp_combine(values, p: float) -> float:
    """
    Power-mean (Lp) combiner: ( Σ |x_i|^p )^(1/p).

    With p=2 this matches Gaussian quadrature. With p<2 large terms are
    weighted relatively less, suppressing σ inflation when source σ values
    have correlated structure. The pySUTO range is p ∈ [1.1, 1.5];
    default p = 1.3.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    s = float(np.sum(np.abs(arr) ** p))
    if s <= 0.0:
        return 0.0
    return s ** (1.0 / p)
