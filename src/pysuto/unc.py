from __future__ import annotations
 
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict
 
import numpy as np
 
 
# =============================================================================
# PEDIGREE SCORES  (Laner et al. 2016, Table 1)
# =============================================================================
 
VALID_SCORES = frozenset({1, 2, 3, 4})
 
SENSITIVITY_PARAMS: Dict[str, float] = {
    # a-parameter for each sensitivity level (b is universal = 1.105)
    "not_sensitive":    0.375,
    "medium_sensitive": 0.750,
    "highly_sensitive": 1.500,
}
 
B_UNIVERSAL = 1.105          # shared exponential base parameter
 
 
@dataclass
class PedigreeScores:
    """
    Holds the five data-quality indicator scores (1–4) and the
    sensitivity level for each non-reliability indicator.
 
    Parameters
    ----------
    reliability : int
        Score for data reliability (1 = best, 4 = worst).
        Uses its own exponential function with a non-zero CV at score 1.
    completeness : int
    temporal_correlation : int
    geographical_correlation : int
    other_correlation : int
        Scores for the remaining four indicators.
    completeness_sensitivity : str
    temporal_sensitivity : str
    geographical_sensitivity : str
    other_sensitivity : str
        One of {"not_sensitive", "medium_sensitive", "highly_sensitive"}.
        Defaults to "medium_sensitive" for all.
    """
    reliability:               int = 1
    completeness:              int = 1
    temporal_correlation:      int = 1
    geographical_correlation:  int = 1
    other_correlation:         int = 1
 
    completeness_sensitivity:     str = "medium_sensitive"
    temporal_sensitivity:         str = "medium_sensitive"
    geographical_sensitivity:     str = "medium_sensitive"
    other_sensitivity:            str = "medium_sensitive"
 
    def __post_init__(self):
        for attr in ("reliability", "completeness", "temporal_correlation",
                     "geographical_correlation", "other_correlation"):
            v = getattr(self, attr)
            if v not in VALID_SCORES:
                raise ValueError(
                    f"Score '{attr}' must be in {{1,2,3,4}}, got {v}."
                )
        for attr in ("completeness_sensitivity", "temporal_sensitivity",
                     "geographical_sensitivity", "other_sensitivity"):
            v = getattr(self, attr)
            if v not in SENSITIVITY_PARAMS:
                raise ValueError(
                    f"Sensitivity '{attr}' must be one of "
                    f"{set(SENSITIVITY_PARAMS)}, got '{v}'."
                )
 
    # ------------------------------------------------------------------
    # Individual CVs  (Laner et al. 2016, Equations 1 & 2, Table 2)
    # ------------------------------------------------------------------
 
    @staticmethod
    def _cv_reliability(score: int) -> float:
        """
        Equation (2) from Laner et al.:
            CV = a * exp(b * score)
        with a=0.75, b=1.105.
        Produces a non-zero CV even at score=1 (best data always has
        some generation imperfection).
        """
        a = 0.750
        return a * np.exp(B_UNIVERSAL * score) / 100.0   # return as fraction
 
    @staticmethod
    def _cv_other(score: int, sensitivity: str) -> float:
        """
        Equation (1) from Laner et al.:
            CV = 0           if score == 1
            CV = a * exp(b*(score-1))   otherwise
        Returns CV as a fraction (not percent).
        """
        if score == 1:
            return 0.0
        a = SENSITIVITY_PARAMS[sensitivity]
        return a * np.exp(B_UNIVERSAL * (score - 1)) / 100.0
 
    # ------------------------------------------------------------------
    # Aggregated total CV  (Equation 3)
    # ------------------------------------------------------------------
 
    def total_cv(self) -> float:
        r"""
        Aggregate individual CVs via sum-in-quadrature:
 
            CV_tot = sqrt(CV_rel^2 + CV_comp^2 + CV_temp^2
                          + CV_geo^2  + CV_other^2)
 
        Returns
        -------
        float
            Total coefficient of variation as a *fraction* (e.g. 0.15
            means ±15 % at 1 σ).
        """
        cv_rel  = self._cv_reliability(self.reliability)
        cv_comp = self._cv_other(self.completeness,
                                 self.completeness_sensitivity)
        cv_temp = self._cv_other(self.temporal_correlation,
                                 self.temporal_sensitivity)
        cv_geo  = self._cv_other(self.geographical_correlation,
                                 self.geographical_sensitivity)
        cv_oth  = self._cv_other(self.other_correlation,
                                 self.other_sensitivity)
 
        return float(
            np.sqrt(cv_rel**2 + cv_comp**2 + cv_temp**2
                    + cv_geo**2 + cv_oth**2)
        )
 
    def summary(self) -> dict:
        """Return a dict with all individual CVs and the total."""
        return {
            "cv_reliability":              self._cv_reliability(self.reliability),
            "cv_completeness":             self._cv_other(self.completeness,
                                               self.completeness_sensitivity),
            "cv_temporal_correlation":     self._cv_other(self.temporal_correlation,
                                               self.temporal_sensitivity),
            "cv_geographical_correlation": self._cv_other(self.geographical_correlation,
                                               self.geographical_sensitivity),
            "cv_other_correlation":        self._cv_other(self.other_correlation,
                                               self.other_sensitivity),
            "cv_total":                    self.total_cv(),
        }
 
 
# =============================================================================
# PROPAGATION RESULT
# =============================================================================
 
@dataclass
class PropagationResult:
    """
    Container returned by DataFeed.apply_with_uncertainty().
 
    Attributes
    ----------
    F_root : np.ndarray, shape (N_root,)
        Point-estimate root-edge flow vector  W^T @ F_src.
    sigma_root : np.ndarray, shape (N_root,)
        Per-edge standard deviation after error propagation.
    cv_root : np.ndarray, shape (N_root,)
        Per-edge coefficient of variation (sigma / F_root).
        NaN where F_root == 0.
    cv_src : float
        The dataset-level total CV used as input.
    """
    F_root:    np.ndarray
    sigma_root: np.ndarray
    cv_root:   np.ndarray
    cv_src:    float
 
    def __repr__(self) -> str:
        return (
            f"PropagationResult("
            f"F_root.shape={self.F_root.shape}, "
            f"cv_src={self.cv_src:.4f}, "
            f"mean_cv_root={np.nanmean(self.cv_root):.4f})"
        )
 
 
# =============================================================================
# EXTENSION MIXIN  –  add to DataFeed (or use directly on WeightMatrix)
# =============================================================================
 
class UncertaintyMixin:
    """
    Mixin that adds pedigree-based uncertainty and error propagation
    to any object that owns:
      - self.weight_matrix  : sparse.COO  (N_src × N_root)
      - self.F_src          : np.ndarray  (N_src,)   [optional default]
      - self.pedigree       : PedigreeScores
 
    The mixin is designed to be mixed into DataFeed subclasses, but can
    also be used standalone for testing.
    """
 
    # ------------------------------------------------------------------
    # Attach pedigree
    # ------------------------------------------------------------------
 
    def set_pedigree(self, pedigree: PedigreeScores) -> None:
        """Attach a PedigreeScores object to this feed."""
        self.pedigree: PedigreeScores = pedigree
 
    def set_pedigree_from_dict(self, d: dict) -> None:
        """
        Convenience: build and attach a PedigreeScores from a flat dict.
 
        Expected keys (all optional; defaults to score=1, medium sensitivity):
            reliability, completeness, temporal_correlation,
            geographical_correlation, other_correlation,
            completeness_sensitivity, temporal_sensitivity,
            geographical_sensitivity, other_sensitivity
        """
        self.pedigree = PedigreeScores(**{
            k: v for k, v in d.items()
            if k in PedigreeScores.__dataclass_fields__
        })
 
    # ------------------------------------------------------------------
    # Core propagation
    # ------------------------------------------------------------------
 
    def apply_with_uncertainty(
        self,
        F_src: Optional[np.ndarray] = None,
        pedigree: Optional[PedigreeScores] = None,
    ) -> PropagationResult:
        r"""
        Map F_src through the weight matrix and propagate uncertainty.
 
        Point estimate
        --------------
            F_root = W^T @ F_src
 
        Error propagation (Gaussian, linear map)
        ----------------------------------------
        The dataset-level pedigree gives a single CV_tot that applies
        uniformly to all source edges:
 
            Σ_src = CV_tot² · diag(F_src²)
 
        The covariance of the root vector is:
 
            Σ_root = W^T · Σ_src · W
                   = CV_tot² · W^T · diag(F_src²) · W
 
        The variance of root edge j is the j-th diagonal:
 
            Var(F_j^root) = CV_tot² · Σ_i  W_ij² · F_i²
 
        and the standard deviation vector is:
 
            σ_root = CV_tot · sqrt( W^T · (F_src ⊙ F_src) )
 
        where ⊙ is elementwise multiplication and sqrt is elementwise.
 
        Parameters
        ----------
        F_src : array-like, shape (N_src,), optional
            Source flow vector.  Falls back to self.F_src if not given.
        pedigree : PedigreeScores, optional
            Override the instance-level pedigree for this call.
 
        Returns
        -------
        PropagationResult
        """
        # --- resolve inputs -------------------------------------------
        if F_src is None:
            if not hasattr(self, "F_src") or self.F_src is None:
                raise ValueError(
                    "No F_src supplied and none stored on the instance."
                )
            F_src = self.F_src
        F_src = np.asarray(F_src, dtype=float)
 
        pg = pedigree if pedigree is not None else getattr(self, "pedigree", None)
        if pg is None:
            raise ValueError(
                "No PedigreeScores attached.  Call set_pedigree() first "
                "or pass pedigree= directly."
            )
 
        cv_tot = pg.total_cv()
 
        # --- point estimate -------------------------------------------
        #   W is (N_src × N_root); W^T @ F_src → shape (N_root,)
        W = self.weight_matrix   # sparse COO
        F_root = (W.T @ F_src).todense() if hasattr(W.T @ F_src, "todense") \
                 else np.asarray(W.T @ F_src)
 
        # --- variance propagation  ------------------------------------
        #   σ²_j = CV² · Σ_i  W_ij² · F_i²
        #
        #   Implemented as:  W^T @ (F_src² element-wise)
        #   then multiply by CV² and take sqrt.
        #
        #   For sparse W we exploit:
        #       (W ⊙ W)^T @ F_src² = W^T @ F_src²   when using W²
        F_src_sq = F_src ** 2
 
        # W² (elementwise) then transpose-multiply
        import sparse as sp_mod
        W_dense_sq = (W * W)           # elementwise square, stays sparse
        var_root = cv_tot**2 * np.asarray((W_dense_sq.T @ F_src_sq))
        sigma_root = np.sqrt(np.maximum(var_root, 0.0))   # guard fp noise
 
        # --- CV per root edge -----------------------------------------
        with np.errstate(invalid="ignore", divide="ignore"):
            cv_root = np.where(F_root != 0.0, sigma_root / np.abs(F_root), np.nan)
 
        return PropagationResult(
            F_root=F_root,
            sigma_root=sigma_root,
            cv_root=cv_root,
            cv_src=cv_tot,
        )
 
    # ------------------------------------------------------------------
    # Convenience: store F_src on instance
    # ------------------------------------------------------------------
 
    def set_F_src(self, F_src: np.ndarray) -> None:
        """Store a source-flow vector on the instance for repeated use."""
        self.F_src = np.asarray(F_src, dtype=float)