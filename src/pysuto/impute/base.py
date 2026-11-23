"""
base.py
=======
``BaseImputer`` + the two helpers shared across every imputer:

* :func:`_apply_cv_cap` — σ-cap to cv_cap × |f| (callers across the fill and the
  network imputers).
* :func:`_reg_sensitivity` — bounded σ sensitivity of an inverted coefficient (the fill's
  backward branch, the network's recipe inversion).
* :func:`_log`           — one dict per filled edge (built by the ledger).
"""

from __future__ import annotations

import json
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


def _reg_sensitivity(coeff: float, q: float, alpha_min: float) -> float:
    """
    Bounded effective sensitivity used in the σ propagation of an inverted coefficient.

        s(α) = |α| / (|α|^q + α_min^q)^(1/q)

    Asymptotes to 1 for |α| >> α_min (caps coefficient amplification of σ),
    and to |α|/α_min for |α| << α_min (linearly damps very small coefficients).
    q controls the knee sharpness (q=3 is a moderate default).
    """
    a = abs(float(coeff))
    am = abs(float(alpha_min))
    if a == 0.0 and am == 0.0:
        return 0.0
    return a / (a ** q + am ** q) ** (1.0 / q)


def _log(
    edge_id: int, step: int, method: str,
    source_edges: list[int], coefficients: list[float],
    f_after: float, sigma_after: float,
    description: str, param_id: Optional[str],
    relaxed: bool = False, assumed_absent: Optional[list[int]] = None,
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
        "relaxed":      bool(relaxed),
        "assumed_absent": [int(i) for i in (assumed_absent or [])],
    }


# ---------------------------------------------------------------------------
# BaseImputer
# ---------------------------------------------------------------------------

class BaseImputer(ABC):
    """
    Abstract base for all imputers.

    The engine drives each imputer through four explicit hooks: :meth:`prepare`
    (once, at registration), :meth:`attach` (before each step),
    :meth:`writer_scope` (before each step) and :meth:`impute`. ``impute`` reads
    ``ledger.f`` / ``ledger.sigma`` and writes only through the
    :class:`~pysuto.impute.ledger.PriorLedger`, which enforces no-overwrite and the
    scope :meth:`writer_scope` declares.

    ``id`` is a provenance label stamped onto every log entry and sidecar
    step record by :class:`ImputationEngine`. Each concrete ``__init__``
    accepts an optional ``id`` and defaults it to the class name; pass an
    explicit id (e.g. ``"conc"``) to disambiguate same-class imputers.
    """

    #: Provenance id; set by each concrete __init__ (default = class name).
    id: str = ""

    def prepare(self, acc) -> None:
        """Resolve everything that depends on the accounting structure; once."""

    def attach(self, agg_result) -> None:
        """Receive the (read-only) aggregation result before each step."""

    @abstractmethod
    def writer_scope(self, acc) -> np.ndarray:
        """Bool ``(n_edges,)``: the edges this imputer is permitted to write."""

    @abstractmethod
    def impute(self, ledger, step: int, acc_struct) -> None:
        """Fill edges through ``ledger`` (its ``f`` and ``sigma`` are the prior)."""

    @property
    def source_paths(self) -> list[str]:
        return []

    def config(self) -> dict:
        """
        Return a JSON-serialisable snapshot of this imputer's configuration.

        Default impl: every public attribute (no leading underscore) whose
        value is JSON-able. Subclasses can override for richer detail.
        """
        out: dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)
        return out
