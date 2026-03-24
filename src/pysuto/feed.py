from __future__ import annotations

import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import sparse


from pysuto.unc import UncertaintyMixin, PedigreeScores, PropagationResult


# =============================================================================
# BASE CLASS
# =============================================================================

class DataFeed(UncertaintyMixin, ABC):
    """
    Abstract base for all concordance/weight objects.
 
    Now inherits UncertaintyMixin, which adds:
      - set_pedigree(pedigree)
      - set_pedigree_from_dict(d)
      - set_F_src(F_src)
      - apply_with_uncertainty(F_src, pedigree) -> PropagationResult
 
    Every concrete subclass (WeightMatrix etc.) automatically gains these
    capabilities without any changes to their own code.
    """
 
    def __init__(self, id: str, acc_struct, odir: str = "data/proc/conc"):
        self.id = id
        self.odir = odir
        self.acc_struct = acc_struct
        self.acc_used = acc_struct.id
 
        # Optional slots — populated lazily
        self.pedigree: Optional[PedigreeScores] = None
        self.F_src:    Optional[np.ndarray]     = None
 
    @abstractmethod
    def save(self, **kwargs): ...
 
    @classmethod
    @abstractmethod
    def load(cls, id: str, acc_struct, **kwargs): ...


