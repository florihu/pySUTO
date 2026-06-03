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
    row_penalized : np.ndarray, shape (N_root,), optional
        Boolean provenance mask flagging the edges whose σ was inflated by a
        per-edge ``cv_root_scale`` (e.g. the ROW pedigree penalty). ``None``
        when no penalty was applied. Set by the caller, not by propagation.
    """
    F_root:    np.ndarray
    sigma_root: np.ndarray
    cv_root:   np.ndarray
    cv_src:    float
    row_penalized: Optional[np.ndarray] = None
 
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
        p: float = 1.3,
        cv_root_scale: Optional[np.ndarray] = None,
    ) -> PropagationResult:
        r"""
        Map F_src through the weight matrix and propagate uncertainty.

        Point estimate
        --------------
            F_root = W^T @ F_src

        Error propagation (CV-invariant power-mean)
        -------------------------------------------
        For every root edge j, define normalized fan-in shares from the
        absolute source contributions:

            ω_ij = |W_ij · f_i| / Σ_k |W_kj · f_k|       (Σ_i ω_ij = 1)

        Then propagate uncertainty as:

            CV_root[j] = CV_tot · ( Σ_i ω_ij^p )^(1/p)
            σ_root[j]  = CV_root[j] · |F_root[j]|

        Properties:
            - CV_root[j] ≤ CV_tot for every j (diversification holds for
              p ≥ 1: power-mean of normalized weights is ≤ 1).
            - σ_root scales linearly with |F_root[j]|, so sign cancellation
              in F_root does not produce an inflated absolute σ.
            - For W ≥ 0 and F ≥ 0 with rows summing to 1, this matches the
              previous Lp formula σ = CV · (Σ |W·f|^p)^(1/p) exactly.

        With p = 2 this is the inverse-variance / diversification form.
        The pySUTO range is p ∈ [1.1, 1.5]; default p = 1.3.

        Parameters
        ----------
        F_src : array-like, shape (N_src,), optional
            Source flow vector. Falls back to self.F_src if not given.
        pedigree : PedigreeScores, optional
            Override the instance-level pedigree for this call.
        p : float, default 1.3
            Power-mean exponent. p ≥ 1 required for the diversification
            invariant to hold.

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
        W = self.weight_matrix   # scipy.sparse CSR (N_src × N_root)
        F_root = np.asarray(W.T @ F_src).ravel()

        # --- CV-invariant σ propagation ------------------------------
        # A_ij = |W_ij · f_i|  (entry-wise absolute-value contributions)
        # S[j] = Σ_i A_ij      (column sum: total absolute fan-in at j)
        # ω_ij = A_ij / S[j]   (normalized shares; column j sums to 1)
        # CV_root[j] = CV_tot · ( Σ_i ω_ij^p )^(1/p)
        # σ_root[j]  = CV_root[j] · |F_root[j]|
        if hasattr(W, "data") and getattr(W.data, "ndim", None) == 1:
            # scipy.sparse CSR/CSC path — operate on .data (no densification)
            A = W.copy()
            row_idx = A.tocoo(copy=False).row
            A.data = np.abs(A.data * F_src[row_idx])
            S = np.asarray(A.sum(axis=0)).ravel()       # length N_root
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_S = np.where(S > 0.0, 1.0 / S, 0.0)
            # ω_ij = A_ij / S[j]; multiply each column by inv_S[col]
            Omega = A.copy()
            col_idx = Omega.tocoo(copy=False).col
            Omega.data = Omega.data * inv_S[col_idx]
            # Σ ω_ij^p along axis=0
            Omega_p = Omega.copy()
            Omega_p.data = Omega_p.data ** p
            sum_omega_p = np.asarray(Omega_p.sum(axis=0)).ravel()
        else:
            # Dense ndarray fallback (used by tests with numpy W)
            W_dense = np.asarray(W, dtype=float)
            A = np.abs(W_dense * F_src[:, None])         # (N_src, N_root)
            S = A.sum(axis=0)                            # (N_root,)
            with np.errstate(divide="ignore", invalid="ignore"):
                Omega = np.where(S > 0.0, A / S[None, :], 0.0)
            sum_omega_p = (Omega ** p).sum(axis=0)

        cv_root_arr = cv_tot * np.power(np.maximum(sum_omega_p, 0.0), 1.0 / p)
        cv_root_arr = np.where(np.asarray(sum_omega_p) > 0.0, cv_root_arr, 0.0)
        # Optional per-root-edge CV multiplier (ROW penalty supplies this; the
        # mixin stays agnostic about why an edge is scaled).
        if cv_root_scale is not None:
            cv_root_arr = cv_root_arr * np.asarray(cv_root_scale, dtype=float)
        sigma_root = cv_root_arr * np.abs(F_root)

        # --- reported per-edge CV (sigma / |F|; NaN where F=0) -------
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