"""
base.py
=======
Shared building blocks for the constraint subpackage:

* :class:`ConstraintBlock`     — sparse block of constraint rows from one builder.
* :func:`_empty_block`         — zero-row block (used by every builder + engine).
* :class:`BaseConstraintBuilder` — abstract base for every builder.

Under the structural refactor every builder emits one row per structural
constraint identity (no time replication). Per-period activation lives on
``ConstraintBlock.active`` (boolean, shape ``(M_k,)``); time-varying
coefficients live on the per-period ``G_rows.data`` returned by
``build_dynamic(t)`` and are stitched into :class:`StructG`'s data tape by
the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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

    G_rows    : (M_k × n_edges) sparse CSR matrix
    c         : (M_k,) RHS values (per-period when emitted by build_dynamic)
    sigma_c   : (M_k,) per-row uncertainty (0 for hard rows)
    row_meta  : DataFrame with one row per G row; columns:
                  constraint_id (str), type (str), is_hard (bool), cone_type (str)
                Under the structural model rows have no ``time`` column;
                per-period activation is tracked in ``active`` instead.
    active    : (M_k,) bool — per-row activation flag for this builder's
                rows at the period this block represents. Defaults to all
                True if not provided. Replaces SIGMA_C_INACTIVE /
                BOUND_C_INACTIVE sentinels.
    """
    G_rows:   sp.csr_matrix
    c:        np.ndarray
    sigma_c:  np.ndarray
    row_meta: pd.DataFrame
    active:   Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.active is None:
            self.active = np.ones(self.G_rows.shape[0], dtype=bool)


def _empty_block(n_edges: int) -> ConstraintBlock:
    return ConstraintBlock(
        G_rows   = sp.csr_matrix((0, n_edges)),
        c        = np.empty(0),
        sigma_c  = np.empty(0),
        row_meta = pd.DataFrame(columns=["constraint_id", "type", "is_hard", "cone_type"]),
        active   = np.empty(0, dtype=bool),
    )



# =============================================================================
# BaseConstraintBuilder
# =============================================================================

class BaseConstraintBuilder(ABC):
    """
    Abstract base for all constraint builders.

    build_static()  — emits the builder's structural rows: one row per
                      constraint identity, with the sparsity pattern + the
                      default coefficient values. The pattern is the invariant
                      contract: ``build_dynamic`` MUST preserve it.
    build_dynamic() — per-period overlay. Returns a block whose ``G_rows``
                      shares the same sparsity as ``build_static``'s result
                      (same ``indptr``/``indices``) but may carry different
                      ``data`` (e.g. ratio coefficient ``r(t)``), and per-period
                      ``c``, ``sigma_c``, ``active`` arrays. Default: returns
                      empty block (subclasses override).

    a_agg / sigma_agg passed to ``build_dynamic`` are the period slice
    ``(n_edges,)`` for the requested ``t_idx`` (not the full 2-D agg
    matrix), so builders don't need to know which period they're in.
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

