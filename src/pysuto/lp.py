"""
lp.py
=====
Pure-math Lp / power-mean σ combiner.

Single source of truth for ( Σ |x_i|^p )^(1/p) used across Stage 1 / 2 / 3.
Imported by the Lp-combining imputers (fill, time, network), the Stage-2
aggregation and the Stage-3 σ builders. ``P_DEFAULT`` is the one definition of the
library's default exponent; a case's ``P_PROP`` knob overrides it (copper's:
``scripts/copper.py``). Stage 1's propagation (``UncertaintyMixin``, ``feedpipe``)
keeps its sparse-matrix form of the sum, which is not an axis sum of one array.
"""

from __future__ import annotations

import numpy as np

P_DEFAULT: float = 1.3

def lp_combine(values, p: float) -> float:
    """
    Power-mean (Lp) combiner: ( Σ |x_i|^p )^(1/p).

    With p=2 this matches Gaussian quadrature. With p<2 large terms are
    weighted relatively less, suppressing σ inflation when source σ values
    have correlated structure. The pySUTO range is p ∈ [1.1, 1.5]; the shipped
    value is ``P_DEFAULT`` (every caller passes ``p``).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    s = float(np.sum(np.abs(arr) ** p))
    if s <= 0.0:
        return 0.0
    return s ** (1.0 / p)


def lp_combine_axis(array, p: float, axis=None):
    """
    Power-mean (Lp) combiner along ``axis``: ( Σ |x_i|^p )^(1/p).

    The vectorised twin of :func:`lp_combine` (same ``np.sum(np.abs(x) ** p)``
    followed by ``** (1/p)``, no zero short-circuit: an all-zero axis gives 0.0). Bit-identical to the inline sums it replaces
    for the pySUTO range p ∈ [1.1, 1.5]; at p = 2 ``** 0.5`` takes numpy's sqrt path,
    which can differ from ``np.power`` by one ulp.
    ``axis=None`` combines the whole array and returns a numpy scalar.
    """
    arr = np.asarray(array, dtype=float)
    return np.sum(np.abs(arr) ** p, axis=axis) ** (1.0 / p)
