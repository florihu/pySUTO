"""
feed.py
=======
DataFeed  –  base class of the concordance / weight objects.
UncertaintyMixin  –  pedigree-based error propagation using scipy.sparse CSR.

Matrix convention
-----------------
weight_matrix : scipy.sparse.csr_matrix, shape (n_src, n_root)
    W[i, j]  is the weight that source edge i contributes to root edge j.

Propagation (CV-invariant power-mean, ``UncertaintyMixin.apply_with_uncertainty``):
    F_root     = W.T  @  F_src                       dense (n_root,)
    ω_ij       = |W_ij · f_i| / Σ_k |W_kj · f_k|     fan-in shares, Σ_i ω_ij = 1
    cv_root[j] = cv · (Σ_i ω_ij^p)^(1/p)
    σ_root[j]  = cv_root[j] · |F_root[j]|

where cv = PedigreeScores.total_cv() is the dataset-level coefficient of
variation derived from the pedigree matrix (Laner et al. 2016) and p the Lp
exponent (README "Uncertainty propagation").
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from pysuto.unc import UncertaintyMixin, PedigreeScores


# =============================================================================
# BASE CLASS
# =============================================================================

class DataFeed(UncertaintyMixin):
    """
    Base of the concordance/weight objects; ``WeightMatrix`` is the persisted one
    and defines ``save`` / ``load``.

    Inherits UncertaintyMixin, which adds:
        set_pedigree(pedigree)
        set_F_src(F_src)
        apply_with_uncertainty(F_src, pedigree) → PropagationResult

    All matrix operations inside UncertaintyMixin use scipy.sparse CSR,
    so subclasses must store their weight matrix as a csr_matrix.
    """

    def __init__(self, id: str, acc_struct, odir: str = "data/proc/conc"):
        self.id         = id
        self.odir       = odir
        self.acc_struct = acc_struct
        self.acc_used   = acc_struct.id

        # Optional slots — populated lazily
        self.pedigree:       Optional[PedigreeScores] = None
        self.F_src:          Optional[np.ndarray]     = None
        self.weight_matrix:  Optional[sp.csr_matrix]  = None
