"""
_conditioning.py
================
Shared numerical conditioning helpers for the Stage-3 prior (σ_a, a_0)
and Stage-3 constraint-row uncertainty (σ_c).

Extracted from :class:`pysuto.solver.Solver` so that downstream
consumers — currently :class:`pysuto.uncprop.KalmanUncProp` — apply the
*same* floors and ceilings the solver applied when it produced ``a*``.
Keeping these rules in one module rules out silent drift between the
QP that was solved and the posterior that is then propagated against it.

Two helpers:

* :func:`condition_prior` — floor / ceiling rules for σ_a; a NaN in a_0 or
  σ_a is refused, never filled.
* :func:`condition_sigma_c` — σ_c flooring, with an optional ``hard_mask``
  argument that zeroes out hard rows (used by uncprop, not by the solver,
  which conditions only its soft rows); a NaN σ_c is refused.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .logging import get_logger

logger = get_logger("pysuto.conditioning")


def _refuse_nan(x: np.ndarray, what: str) -> None:
    """Raise ``ValueError`` naming the first NaN positions of ``x``.

    A Stage-3 run never stores a NaN in σ_a, a_0 or σ_c (the engine refuses NaN
    ``f_agg`` / ``σ_agg``); a NaN here means the run is corrupt, and a fill would
    turn "not covered" into a number (CLAUDE.md: validation never fills).
    """
    bad = np.flatnonzero(np.isnan(x))
    if bad.size:
        raise ValueError(
            f"condition: {bad.size} NaN entries in {what} (first positions "
            f"{bad[:5].tolist()}); a NaN is 'not covered' and is never filled — fix "
            f"the input and rebuild the constraint run."
        )


# The only definition of the default floors and ceiling. The solver (Stage 4)
# records the values it used in the opt sidecar's ``stage_config``; uncprop
# (Stage 5) reads them back from there and falls back to these.
SIGMA_A_FLOOR_ABS      = 1e-8
SIGMA_A_FLOOR_REL      = 1e-5
SIGMA_A_CEILING_FACTOR = 1e5
SIGMA_C_FLOOR          = 1e-9

# σ_a at or above _SIGMA_HUGE counts as "no information" and does not set the
# ceiling; with no usable σ_a at all the ceiling is _CEILING_EMPTY.
_SIGMA_HUGE    = 1e30
_CEILING_EMPTY = 1e12

DEFAULT_CONDITIONING = {
    "sigma_a_floor_abs":      SIGMA_A_FLOOR_ABS,
    "sigma_a_floor_rel":      SIGMA_A_FLOOR_REL,
    "sigma_a_ceiling_factor": SIGMA_A_CEILING_FACTOR,
    "sigma_c_floor":          SIGMA_C_FLOOR,
}


def recorded_conditioning(stage_config: dict, run_id: str = "") -> dict:
    """The σ conditioning floors an opt run recorded in its ``stage_config``.

    A run that predates the record (any key missing) gets the library defaults
    in :data:`DEFAULT_CONDITIONING` and a logged warning; the defaults never
    changed, so this is exact for every such run.

    Parameters
    ----------
    stage_config : dict
        The opt sidecar's ``stage_config``.
    run_id : str
        The opt run id, named in the warning.

    Returns
    -------
    dict
        The four :data:`DEFAULT_CONDITIONING` keys, as floats.
    """
    cfg = stage_config or {}
    missing = [k for k in DEFAULT_CONDITIONING if k not in cfg]
    if missing:
        logger.warning(
            "opt run %r records no %s in its stage_config; using the library "
            "defaults %s.", run_id, ", ".join(missing), DEFAULT_CONDITIONING,
        )
    return {k: float(cfg.get(k, v)) for k, v in DEFAULT_CONDITIONING.items()}


def condition_prior(
    a_0:            np.ndarray,
    sigma_a:        np.ndarray,
    *,
    floor_abs:      float = SIGMA_A_FLOOR_ABS,
    floor_rel:      float = SIGMA_A_FLOOR_REL,
    ceiling_factor: float = SIGMA_A_CEILING_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the floor and ceiling rules to the Stage-3 prior.

    A NaN in ``a_0`` or ``sigma_a`` raises ``ValueError``. Then::

        floor   = max(floor_abs, floor_rel · |a_0|)
        ceiling = ceiling_factor · max(finite, positive σ_a)
        σ_a    ← clip(σ_a, floor, ceiling)

    Returns the conditioned ``(a_0, sigma_a)`` as fresh arrays; inputs
    are not mutated.
    """
    a = np.array(a_0,     dtype=float, copy=True)
    s = np.array(sigma_a, dtype=float, copy=True)

    _refuse_nan(a, "a_0")
    _refuse_nan(s, "sigma_a")

    finite = s[np.isfinite(s) & (s < _SIGMA_HUGE) & (s > 0)]
    ceiling = ceiling_factor * float(finite.max()) if finite.size else _CEILING_EMPTY

    floor = np.maximum(floor_abs, floor_rel * np.abs(a))
    s = np.maximum(s, floor)
    s = np.minimum(s, ceiling)
    return a, s


def condition_sigma_c(
    sigma_c:   np.ndarray,
    *,
    floor:     float = SIGMA_C_FLOOR,
    hard_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a numerical floor to ``sigma_c``; a NaN raises ``ValueError``.

    If ``hard_mask`` is provided, hard rows are forced to exact zero
    *after* flooring — preserves the ``Σ_c^full = blkdiag(0_h, Σ_c_s)``
    block structure that uncprop's posterior derivation relies on.

    The solver omits ``hard_mask`` and passes only its soft rows (a hard row
    has no σ_c in the QP). uncprop passes every row and ``hard_mask``, so that
    hard rows behave as exact equality constraints when contributing to
    variance reduction.
    """
    s = np.array(sigma_c, dtype=float, copy=True)
    _refuse_nan(s, "sigma_c")
    s = np.maximum(s, floor)
    if hard_mask is not None:
        s = s.copy()
        s[np.asarray(hard_mask, dtype=bool)] = 0.0
    return s
