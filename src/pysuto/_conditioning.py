"""
_conditioning.py
================
Shared numerical conditioning helpers for the Stage-3 prior (σ_a, a_0)
and Stage-3 constraint-row uncertainty (σ_c).

Extracted from :class:`pysuto.solver.ClarabelSolver` so that downstream
consumers — currently :class:`pysuto.uncprop.KalmanUncProp` — apply the
*same* floors and ceilings the solver applied when it produced ``a*``.
Keeping these rules in one module rules out silent drift between the
QP that was solved and the posterior that is then propagated against it.

Two helpers:

* :func:`condition_prior` — floor / ceiling / NaN handling for σ_a;
  NaN handling for a_0. Identical to the historical
  ``ClarabelSolver._condition_prior`` body.
* :func:`condition_sigma_c` — soft-row σ_c flooring, with an optional
  ``hard_mask`` argument that zeroes out hard rows (used by uncprop,
  not by the solver — see ``stage5_uncprop_integration_plan.md``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .logging import get_logger

logger = get_logger("pysuto.conditioning")


def condition_prior(
    a_0:            np.ndarray,
    sigma_a:        np.ndarray,
    *,
    floor_abs:      float = 1e-8,
    floor_rel:      float = 1e-5,
    ceiling_factor: float = 1e5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply NaN, floor and ceiling rules to the Stage-3 prior.

    NaN in ``a_0`` is replaced with 0 (flat prior). NaN in ``sigma_a``
    is replaced with the ceiling (no-information prior). Then::

        floor   = max(floor_abs, floor_rel · |a_0|)
        ceiling = ceiling_factor · max(finite, positive σ_a)
        σ_a    ← clip(σ_a, floor, ceiling)

    Returns the conditioned ``(a_0, sigma_a)`` as fresh arrays; inputs
    are not mutated.
    """
    a = np.array(a_0,     dtype=float, copy=True)
    s = np.array(sigma_a, dtype=float, copy=True)

    n_nan_a = int(np.isnan(a).sum())
    if n_nan_a:
        a[np.isnan(a)] = 0.0
        logger.warning(
            "condition_prior: %d NaN entries in a_0 substituted with 0 "
            "(prefer registering FlatPriorImputer in Stage 2 to avoid this).",
            n_nan_a,
        )

    finite = s[np.isfinite(s) & (s < 1e30) & (s > 0)]
    ceiling = ceiling_factor * float(finite.max()) if finite.size else 1e12

    n_nan_s = int(np.isnan(s).sum())
    if n_nan_s:
        s[np.isnan(s)] = ceiling
        logger.warning(
            "condition_prior: %d NaN entries in sigma_a substituted with "
            "ceiling=%.3e (no-information prior).",
            n_nan_s, ceiling,
        )

    floor = np.maximum(floor_abs, floor_rel * np.abs(a))
    s = np.maximum(s, floor)
    s = np.minimum(s, ceiling)
    return a, s


def condition_sigma_c(
    sigma_c:   np.ndarray,
    *,
    floor:     float = 1e-9,
    hard_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply NaN→floor and a numerical floor to ``sigma_c``.

    If ``hard_mask`` is provided, hard rows are forced to exact zero
    *after* flooring — preserves the ``Σ_c^full = blkdiag(0_h, Σ_c_s)``
    block structure that uncprop's posterior derivation relies on.

    The solver omits ``hard_mask`` (its hard rows pick up the floor like
    everything else, which is fine for QP solving). uncprop passes
    ``hard_mask`` so that hard rows behave as exact equality constraints
    when contributing to variance reduction.
    """
    s = np.array(sigma_c, dtype=float, copy=True)
    s = np.where(np.isnan(s), floor, s)
    s = np.maximum(s, floor)
    if hard_mask is not None:
        s = s.copy()
        s[np.asarray(hard_mask, dtype=bool)] = 0.0
    return s
