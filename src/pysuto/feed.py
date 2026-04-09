"""
feed.py
=======
DataFeed  –  abstract base class for all concordance / weight objects.
UncertaintyMixin  –  Gaussian error propagation using scipy.sparse CSR.

Matrix convention
-----------------
weight_matrix : scipy.sparse.csr_matrix, shape (n_src, n_root)
    W[i, j]  is the weight that source edge i contributes to root edge j.

Propagation (first-order Gaussian):
    F_root   = W.T  @  F_src                         dense (n_root,)
    var_root = cv²  *  (W ⊙ W).T  @  F_src²          element-wise square
    σ_root   = sqrt(var_root)
    cv_root  = σ_root / |F_root|

where cv = PedigreeScores.total_cv() is the dataset-level coefficient of
variation derived from the pedigree matrix (Laner et al. 2016).
"""

from __future__ import annotations

import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.unc import UncertaintyMixin, PedigreeScores, PropagationResult


# =============================================================================
# BASE CLASS
# =============================================================================

class DataFeed(UncertaintyMixin, ABC):
    """
    Abstract base for all concordance/weight objects.

    Inherits UncertaintyMixin, which adds:
        set_pedigree(pedigree)
        set_pedigree_from_dict(d)
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

    @abstractmethod
    def save(self, **kwargs): ...

    @classmethod
    @abstractmethod
    def load(cls, id: str, acc_struct, **kwargs): ...
