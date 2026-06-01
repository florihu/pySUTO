"""
base.py
=======
Shared building blocks for the constraint subpackage:

* :class:`ConstraintBlock`     — sparse block of constraint rows from one builder.
* :func:`_empty_block`         — zero-row block (used by every builder + engine).
* :class:`BaseConstraintBuilder` — abstract base for every builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


# =============================================================================
# ConstraintBlock  — output of a single builder for one phase (static/dynamic)
# =============================================================================

@dataclass
class ConstraintBlock:
    """
    Sparse block of constraint rows from one builder.

    G_rows    : (M_k × E) sparse CSR matrix
    c         : (M_k,) RHS values
    sigma_c   : (M_k,) per-row uncertainty (0 for hard constraints)
    row_meta  : DataFrame with one row per G row; columns:
                  constraint_id (str), type (str), is_hard (bool), cone_type (str)
    """
    G_rows:   sp.csr_matrix
    c:        np.ndarray
    sigma_c:  np.ndarray
    row_meta: pd.DataFrame


def _empty_block(n_edges: int) -> ConstraintBlock:
    return ConstraintBlock(
        G_rows   = sp.csr_matrix((0, n_edges)),
        c        = np.empty(0),
        sigma_c  = np.empty(0),
        row_meta = pd.DataFrame(columns=["constraint_id", "type", "is_hard", "cone_type"]),
    )



# =============================================================================
# BaseConstraintBuilder
# =============================================================================

class BaseConstraintBuilder(ABC):
    """
    Abstract base for all constraint builders.

    build_static()  — rows whose G coefficients never change across timesteps.
                      Called once per run.
    build_dynamic() — rows whose c or sigma_c depend on a_agg / sigma_agg at time t.
                      Default: returns empty block (override only when needed).
    """

    @abstractmethod
    def build_static(self, acc_struct) -> ConstraintBlock:
        ...

    def build_dynamic(
        self,
        acc_struct,
        a_agg:     np.ndarray,
        sigma_agg: np.ndarray,
        t:         str,
    ) -> ConstraintBlock:
        return _empty_block(acc_struct.n_edges)

    @property
    def source_paths(self) -> list[str]:
        return []


# =============================================================================
# MassBalanceBuilder
# =============================================================================

