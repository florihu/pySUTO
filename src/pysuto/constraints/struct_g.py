"""
struct_g.py
===========
``StructG`` — first-class operator that exposes the constraint matrix as a
structurally-invariant sparsity pattern with a time-varying coefficient
tape.

Under the structural model the constraint graph is identical across every
period; only the numeric coefficients in a small subset of entries vary.
``StructG`` captures that fact by storing the CSR ``indptr`` / ``indices``
once and a single contiguous ``data`` tape of shape ``(nnz, n_timesteps)``.
Period materialization is a view::

    G_t = StructG.at(t_idx) → scipy.sparse.csr_matrix

The returned CSR shares ``indptr`` and ``indices`` with the operator
(``copy=False``); only the per-period data column is a new array.

Invariants
----------
* ``data.shape == (len(indices), n_timesteps)``
* ``len(indptr) == shape[0] + 1``
* ``indices.max() < shape[1]`` (column references in bounds)
* ``row_meta`` has ``shape[0]`` rows; carries no ``time`` column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class StructG:
    indptr:   np.ndarray
    indices:  np.ndarray
    data:     np.ndarray
    shape:    tuple
    row_meta: pd.DataFrame

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"StructG.data must be 2-D (nnz, n_timesteps); got shape {self.data.shape}"
            )
        if self.data.shape[0] != self.indices.shape[0]:
            raise ValueError(
                f"StructG.data axis-0 ({self.data.shape[0]}) must match "
                f"len(indices) ({self.indices.shape[0]})"
            )
        if self.indptr.shape[0] != self.shape[0] + 1:
            raise ValueError(
                f"StructG.indptr length ({self.indptr.shape[0]}) must equal "
                f"shape[0]+1 ({self.shape[0] + 1})"
            )
        if len(self.row_meta) != self.shape[0]:
            raise ValueError(
                f"StructG.row_meta length ({len(self.row_meta)}) must equal "
                f"shape[0] ({self.shape[0]})"
            )

    @property
    def n_constraints(self) -> int:
        return int(self.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_timesteps(self) -> int:
        return int(self.data.shape[1])

    def at(self, t_idx: int) -> sp.csr_matrix:
        """Materialize period ``t_idx`` as a ``csr_matrix`` view."""
        return sp.csr_matrix(
            (self.data[:, t_idx], self.indices, self.indptr),
            shape=self.shape,
            copy=False,
        )

    @classmethod
    def from_static(
        cls,
        G_static: sp.csr_matrix,
        row_meta: pd.DataFrame,
        n_timesteps: int,
    ) -> "StructG":
        """
        Build a StructG whose data tape is the static G_rows.data broadcast
        across every period.

        Use when no builder overrides per-period coefficients (the topological
        builders); subsequent in-place writes to ``data[:, t_idx]`` from
        ``build_dynamic`` can still vary per-period entries.
        """
        G = G_static.tocsr() if not sp.isspmatrix_csr(G_static) else G_static
        data = np.broadcast_to(G.data[:, None], (G.nnz, n_timesteps)).copy()
        return cls(
            indptr   = G.indptr,
            indices  = G.indices,
            data     = data,
            shape    = tuple(G.shape),
            row_meta = row_meta.reset_index(drop=True),
        )
