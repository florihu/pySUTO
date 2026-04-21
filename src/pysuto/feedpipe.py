"""
feedpipe.py
===========
Canonical four-step DataFeed lifecycle:

    A  Read metadata (Excel) → DataFeedMetadata
    B  Build ConcordanceTable
    C  Build WeightMatrix
    D  Apply + propagate uncertainty  →  PropagationResult
       Register + persist everything  →  DataFeedRegistry

All matrix operations use scipy.sparse (CSR format).

Run this file directly for a self-contained smoke test (no real
AccountingStructure or Excel file required).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.meta import DataFeedMetadata
from pysuto.conc import ConcordanceTable, WeightMatrix
from pysuto.unc import UncertaintyMixin, PropagationResult
from pysuto.registry import DataFeedRegistry
from pysuto.logging import get_logger

_log = get_logger("pysuto.feedpipe")


# ===========================================================================
# STEP A — Read metadata
# ===========================================================================

def step_A_read_metadata(excel_path: str) -> DataFeedMetadata:
    """
    Parse the Excel sidecar that describes this dataset.

    Returns a DataFeedMetadata with all identification, pedigree,
    and provenance fields populated.
    """
    _log.info("A  reading metadata: %s", excel_path)
    meta = DataFeedMetadata.from_excel(excel_path)
    _log.info("   feed_name=%s  iedc_type=%s  version=%s",
              meta.feed_name, meta.iedc_type, meta.version)
    return meta


# ===========================================================================
# STEP B — Build ConcordanceTable
# ===========================================================================

def step_B_build_concordance(
    raw_df:           pd.DataFrame,
    acc_struct,
    sector_map_df:    pd.DataFrame,
    region_map_df:    pd.DataFrame,
    time_map_df:      pd.DataFrame,
    valuation_map_df: pd.DataFrame,
    var_map_df:       pd.DataFrame,
    feed_id:          str,
    aux_col:          Optional[str] = None,
    allow_unmatched_edges: bool = True,
    user_weights_df:  Optional[pd.DataFrame] = None,
) -> ConcordanceTable:
    """
    Build the src→root edge mapping (ConcordanceTable) from raw data.

    Parameters
    ----------
    allow_unmatched_edges : passed through to ConcordanceTable.from_raw so
        that rows that cannot be matched to a root edge are collected in
        ct.unmatched_raw_rows rather than raising an exception.
    user_weights_df : optional DataFrame with columns [root_sector_from,
        root_sector_to, weight].  Rows matching (root_sector_from,
        root_sector_to) override the default equal-split weight.  Other rows
        keep the uniform default (weight=1, normalised per src_edge group).

    Returns a validated ConcordanceTable ready for weight-matrix assembly.
    """
    _log.info("B  building concordance  feed='%s'  raw_rows=%d", feed_id, len(raw_df))
    if user_weights_df is not None and not user_weights_df.empty:
        _log.info("   user_weights: %d entries", len(user_weights_df))
    ct = ConcordanceTable.from_raw(
        raw_df                = raw_df,
        acc_struct            = acc_struct,
        sector_map_df         = sector_map_df,
        region_map_df         = region_map_df,
        time_map_df           = time_map_df,
        valuation_map_df      = valuation_map_df,
        var_map_df            = var_map_df,
        aux_col               = aux_col,
        allow_unmatched_edges = allow_unmatched_edges,
        id                    = feed_id,
        user_weights_df       = user_weights_df,
    )
    unmatched = getattr(ct, "unmatched_raw_rows", [])
    _log.info("   src_edges=%d  root_edges=%d  conc_rows=%d  unmatched=%d",
              ct.n_src_edges, ct.n_root_edges, len(ct.tab), len(unmatched))
    if unmatched:
        _log.warning("   %d raw rows could not be matched to a root edge", len(unmatched))
    return ct


# ===========================================================================
# STEP C — Build WeightMatrix
# ===========================================================================

def step_C_build_weight_matrix(ct: ConcordanceTable) -> WeightMatrix:
    """
    Assemble the sparse W[src_edge, root_edge] matrix from the concordance.

    The weight matrix is stored internally as a scipy.sparse.csr_matrix.
    Returns a WeightMatrix whose .apply() and .apply_with_uncertainty()
    methods are ready.
    """
    _log.info("C  building weight matrix")
    wm = WeightMatrix(concordance=ct, id=ct.id)
    _log.info("   shape=%s  nnz=%d", wm.weight_matrix.shape, wm.weight_matrix.nnz)
    return wm


# ===========================================================================
# STEP D — Apply + propagate
# ===========================================================================

def step_D_apply_and_propagate(
    wm:       WeightMatrix,
    F_src:    np.ndarray,
    metadata: DataFeedMetadata,
) -> PropagationResult:
    """
    1.  Derive PedigreeScores from metadata and attach to the weight matrix.
    2.  Store F_src on the weight matrix.
    3.  Run the combined apply + Gaussian error propagation (scipy.sparse).

    Propagation formula (Gaussian, first-order):
        F_root  = W.T @ F_src
        var_root = cv² * (W² .T @ F_src²)      element-wise W²
        σ_root  = sqrt(var_root)
        cv_root = σ_root / |F_root|

    Returns
    -------
    PropagationResult with F_root, sigma_root, cv_root, cv_src.
    """
    _log.info("D  propagating uncertainty  F_src_nnz=%d  F_src_sum=%.1f kt",
              int(np.count_nonzero(F_src)), float(F_src.sum()))
    pg = metadata.to_pedigree_scores()
    wm.set_pedigree(pg)
    wm.set_F_src(F_src)
    result = wm.apply_with_uncertainty()
    _log.info("   cv_src=%.2f%%  F_root_sum=%.1f kt  mean_cv_root=%.2f%%",
              result.cv_src * 100,
              float(result.F_root.sum()),
              float(np.nanmean(result.cv_root)) * 100)
    return result


# ===========================================================================
# REGISTRATION — store everything
# ===========================================================================

def register_feed(
    wm:        WeightMatrix,
    metadata:  DataFeedMetadata,
    result:    PropagationResult,
    registry:  DataFeedRegistry,
    overwrite: bool = False,
) -> None:
    """
    Persist the weight matrix, metadata sidecar, and result arrays,
    then flush the registry index to disk.

    File layout created under registry.odir:
        registry.json
        <feed_id>/
            metadata.json          ← human-readable, filled by fill_computed()
            <feed_id>.pkl          ← weight matrix + concordance (pickle)
            result.npz             ← F_root, sigma_root, cv_root, cv_src
    """
    _log.info("E  registering feed '%s'", metadata.feed_name)
    registry.register(wm, metadata, result=result, overwrite=overwrite)
    registry.save()
    _log.info("   registry saved → %s", registry.odir)


# ===========================================================================
# FULL PIPELINE (convenience wrapper)
# ===========================================================================

def run_feed_pipeline(
    excel_path:       str,
    raw_df:           pd.DataFrame,
    acc_struct,
    sector_map_df:    pd.DataFrame,
    region_map_df:    pd.DataFrame,
    time_map_df:      pd.DataFrame,
    valuation_map_df: pd.DataFrame,
    var_map_df:       pd.DataFrame,
    F_src:            np.ndarray,
    registry:         DataFeedRegistry,
    feed_id:          Optional[str] = None,
    aux_col:          Optional[str] = None,
    allow_unmatched_edges: bool = True,
    overwrite:        bool = False,
) -> tuple[WeightMatrix, DataFeedMetadata, PropagationResult]:
    """
    Execute steps A → D and register, returning (wm, metadata, result).

    Parameters
    ----------
    excel_path            : path to the per-dataset Excel metadata file
    raw_df                : raw source data (flows)
    acc_struct            : AccountingStructure instance
    *_map_df              : concordance mapping tables
    F_src                 : source flow vector aligned to concordance src edges
    registry              : DataFeedRegistry to register into
    feed_id               : override; defaults to metadata.feed_name
    aux_col               : optional supply/use discriminator column name
    allow_unmatched_edges : collect unmatched rows instead of raising
    overwrite             : allow replacing an existing registry entry
    """
    meta   = step_A_read_metadata(excel_path)
    fid    = feed_id or meta.feed_name or "unnamed_feed"
    ct     = step_B_build_concordance(
                 raw_df, acc_struct,
                 sector_map_df, region_map_df,
                 time_map_df, valuation_map_df, var_map_df,
                 feed_id=fid, aux_col=aux_col,
                 allow_unmatched_edges=allow_unmatched_edges)
    wm     = step_C_build_weight_matrix(ct)
    result = step_D_apply_and_propagate(wm, F_src, meta)
    register_feed(wm, meta, result, registry, overwrite=overwrite)
    return wm, meta, result


# ===========================================================================
# SMOKE TEST (no real project dependencies)
# ===========================================================================

if __name__ == "__main__":
    import sys
    import json
    import dataclasses

    sys.path.insert(0, os.path.dirname(__file__))

    # ------------------------------------------------------------------ #
    # Minimal stand-in for WeightMatrix using scipy.sparse throughout     #
    # ------------------------------------------------------------------ #
    class _MockWeightMatrix(UncertaintyMixin):
        """Stand-in for WeightMatrix; uses scipy.sparse CSR for all ops."""

        def __init__(self, W_dense: np.ndarray, feed_id: str, acc_id: str):
            self.id           = feed_id
            self.acc_used     = acc_id
            self.odir         = ""
            # Store as CSR — the same format WeightMatrix uses in production
            self.weight_matrix = sp.csr_matrix(W_dense)

        def save(self, **kwargs):
            import pickle
            path = kwargs.get("path", f"{self.id}.pkl")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({"id": self.id, "W": self.weight_matrix}, fh)

        def apply_with_uncertainty(
            self,
            F_src: Optional[np.ndarray] = None,
            pedigree=None,
        ) -> PropagationResult:
            """
            Gaussian error propagation with scipy.sparse:

                F_root   = W.T  @  F_src          (dense result via CSR)
                var_root = cv²  *  W².T  @  F_src²
                σ_root   = sqrt(var_root)
                cv_root  = σ_root / |F_root|
            """
            F   = np.asarray(
                F_src if F_src is not None else self.F_src,
                dtype=float,
            )
            pg  = pedigree or self.pedigree
            cv  = pg.total_cv()

            W: sp.csr_matrix = self.weight_matrix          # (n_src, n_root)

            # F_root = W.T @ F_src  →  dense (n_root,)
            F_root: np.ndarray = np.asarray(W.T @ F).ravel()

            # Element-wise W² (efficient: just square the stored data array)
            W2 = W.copy()
            W2.data **= 2

            var_root: np.ndarray = cv**2 * np.asarray(W2.T @ F**2).ravel()
            sigma: np.ndarray    = np.sqrt(np.maximum(var_root, 0.0))

            with np.errstate(invalid="ignore", divide="ignore"):
                cv_root = np.where(
                    F_root != 0.0,
                    sigma / np.abs(F_root),
                    np.nan,
                )

            return PropagationResult(
                F_root=F_root,
                sigma_root=sigma,
                cv_root=cv_root,
                cv_src=float(cv),
            )

    # ------------------------------------------------------------------ #
    # A  Metadata (constructed directly, simulating an Excel read)        #
    # ------------------------------------------------------------------ #
    meta = DataFeedMetadata(
        feed_name                = "icsg",
        iedc_type                = "1_F_",
        reliability              = 1,
        completeness             = 1,
        temporal_correlation     = 1,
        geographical_correlation = 1,
        other_correlation        = 1,
        version                  = 0,
        date_of_assessment       = "2025-06-11",
        person                   = "Florian_Huber",
        dataset_link             = "https://www.icsg.org",
    )
    print("A  metadata loaded:", meta.feed_name, meta.iedc_type)

    # ------------------------------------------------------------------ #
    # B  (skipped — ConcordanceTable needs a real AccountingStructure)    #
    # ------------------------------------------------------------------ #
    print("B  [skipped in smoke test — ConcordanceTable needs AccountingStructure]")

    # ------------------------------------------------------------------ #
    # C  Synthetic scipy.sparse weight matrix                             #
    # ------------------------------------------------------------------ #
    W_dense = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.3, 0.7],
    ])  # 4 src edges → 3 root edges
    wm = _MockWeightMatrix(W_dense, feed_id="icsg", acc_id="acc_test")
    print("C  weight matrix shape:", wm.weight_matrix.shape,
          "  nnz:", wm.weight_matrix.nnz,
          "  type:", type(wm.weight_matrix).__name__)

    # ------------------------------------------------------------------ #
    # D  Apply + propagate                                                #
    # ------------------------------------------------------------------ #
    F_src  = np.array([100.0, 200.0, 150.0, 80.0])
    result = step_D_apply_and_propagate(wm, F_src, meta)

    print("\nD  Propagation result")
    print(f"   CV_src       : {result.cv_src * 100:.4f} %")
    print(f"   F_root       : {result.F_root}")
    print(f"   sigma_root   : {result.sigma_root}")
    print(f"   CV_root (%)  : {result.cv_root * 100}")

    # ------------------------------------------------------------------ #
    # Register                                                            #
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = DataFeedRegistry(odir=tmpdir)
        entry    = registry.register(wm, meta, result=result)
        registry.save()

        print("\nRegistry entry:")
        print(json.dumps(entry.to_dict(), indent=2))

        # Round-trip: reload and verify
        registry2 = DataFeedRegistry(odir=tmpdir)
        registry2.load_index()
        print("\nRegistry summary:\n", registry2.summary_df().T.to_string())

        reloaded_meta   = registry2.load_metadata("icsg")
        reloaded_result = registry2.load_result("icsg")
        print(f"\nReloaded cv_total   : {reloaded_meta.cv_total * 100:.4f} %")
        print(f"Reloaded F_root     : {reloaded_result['F_root']}")
        print(f"Reloaded sigma_root : {reloaded_result['sigma_root']}")

    print("\n✓ Smoke test passed.")
