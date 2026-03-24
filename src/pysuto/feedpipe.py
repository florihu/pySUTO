"""
feedpipe.py
===========
Canonical four-step DataFeed lifecycle:

    A  Read metadata (Excel) → DataFeedMetadata
    B  Build ConcordanceTable
    C  Build WeightMatrix
    D  Apply + propagate uncertainty  →  PropagationResult
       Register + persist everything  →  DataFeedRegistry

Run this file directly for a self-contained smoke test (no real
AccountingStructure or Excel file required).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stdlib / third-party
# ---------------------------------------------------------------------------
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pysuto.meta import DataFeedMetadata
from pysuto.conc import ConcordanceTable, WeightMatrix
from pysuto.unc import UncertaintyMixin, PropagationResult
from pysuto.registry import DataFeedRegistry

# ---------------------------------------------------------------------------
# Project imports  (adjust paths as needed)
# ---------------------------------------------------------------------------
# from datafeed_uncertainty import PedigreeScores, PropagationResult
# from datafeed_registry    import DataFeedMetadata, DataFeedRegistry
# from concordance          import ConcordanceTable, WeightMatrix
#
# For the smoke test below we define minimal stand-ins so the file runs
# without the full project tree.


# ===========================================================================
# STEP A ── Read metadata
# ===========================================================================

def step_A_read_metadata(excel_path: str) -> "DataFeedMetadata":
    """
    Parse the Excel sidecar that describes this dataset.

    Returns a DataFeedMetadata with all identification, pedigree,
    and provenance fields populated.
    """
    meta = DataFeedMetadata.from_excel(excel_path)
    return meta


# ===========================================================================
# STEP B ── Build ConcordanceTable
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
    aux_col:          str | None = None,
) -> "ConcordanceTable":
    """
    Build the src→root edge mapping (ConcordanceTable) from raw data.

    Returns a validated ConcordanceTable ready for weight-matrix assembly.
    """
    ct = ConcordanceTable.from_raw(
        raw_df           = raw_df,
        acc_struct       = acc_struct,
        sector_map_df    = sector_map_df,
        region_map_df    = region_map_df,
        time_map_df      = time_map_df,
        valuation_map_df = valuation_map_df,
        var_map_df       = var_map_df,
        aux_col          = aux_col,
        id               = feed_id,
    )
    return ct


# ===========================================================================
# STEP C ── Build WeightMatrix
# ===========================================================================

def step_C_build_weight_matrix(ct: "ConcordanceTable") -> "WeightMatrix":
    """
    Assemble the sparse W[src_edge, root_edge] matrix from the concordance.

    Returns a WeightMatrix whose .apply() and .apply_with_uncertainty()
    methods are now ready.
    """
    from pysuto.conc import WeightMatrix              # your existing module
    wm = WeightMatrix(concordance=ct, id=ct.id)
    return wm


# ===========================================================================
# STEP D ── Apply + propagate
# ===========================================================================

def step_D_apply_and_propagate(
    wm:       "WeightMatrix",
    F_src:    np.ndarray,
    metadata: "DataFeedMetadata",
) -> "PropagationResult":
    """
    1.  Attach the pedigree from metadata to the weight matrix.
    2.  Store F_src on the weight matrix.
    3.  Run the combined apply + Gaussian error propagation.

    Returns
    -------
    PropagationResult with F_root, sigma_root, cv_root, cv_src.
    """
    pg = metadata.to_pedigree_scores()
    wm.set_pedigree(pg)
    wm.set_F_src(F_src)
    result = wm.apply_with_uncertainty()
    return result


# ===========================================================================
# REGISTRATION  ── store everything
# ===========================================================================

def register_feed(
    wm:       "WeightMatrix",
    metadata: "DataFeedMetadata",
    result:   "PropagationResult",
    registry: "DataFeedRegistry",
    overwrite: bool = False,
) -> None:
    """
    Persist the weight matrix, metadata sidecar, and result arrays,
    then update the registry index.

    File layout created:
        <registry.odir>/
            registry.json
            <feed_id>/
                metadata.json          ← human-readable
                <feed_id>.pkl          ← weight matrix + concordance
                result.npz             ← F_root, sigma_root, cv_root
    """
    registry.register(wm, metadata, result=result, overwrite=overwrite)
    registry.save()


# ===========================================================================
# FULL PIPELINE  (convenience wrapper)
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
    registry:         "DataFeedRegistry",
    feed_id:          str | None = None,
    aux_col:          str | None = None,
    overwrite:        bool = False,
) -> tuple:
    """
    Execute steps A → D and register, returning (wm, metadata, result).

    Parameters
    ----------
    excel_path   : path to the per-dataset Excel metadata file
    raw_df       : raw source data (flows)
    acc_struct   : AccountingStructure instance
    *_map_df     : concordance mapping tables
    F_src        : source flow vector aligned to concordance src edges
    registry     : DataFeedRegistry to register into
    feed_id      : override; defaults to metadata.feed_name
    aux_col      : optional supply/use discriminator column name
    overwrite    : allow replacing an existing registry entry
    """

    meta    = step_A_read_metadata(excel_path)
    fid     = feed_id or meta.feed_name or "unnamed_feed"
    ct      = step_B_build_concordance(
                  raw_df, acc_struct,
                  sector_map_df, region_map_df,
                  time_map_df, valuation_map_df, var_map_df,
                  feed_id=fid, aux_col=aux_col)
    wm      = step_C_build_weight_matrix(ct)
    result  = step_D_apply_and_propagate(wm, F_src, meta)
    register_feed(wm, meta, result, registry, overwrite=overwrite)
    return wm, meta, result


# ===========================================================================
# SMOKE TEST  (no real project dependencies)
# ===========================================================================

if __name__ == "__main__":
    import sys, json, dataclasses
    import tempfile, os

    # ------------------------------------------------------------------ #
    # Minimal stand-ins so this file runs standalone                      #
    # ------------------------------------------------------------------ #
    sys.path.insert(0, os.path.dirname(__file__))

    # Synthetic sparse-free WeightMatrix stand-in
    class _MockWeightMatrix(UncertaintyMixin):
        """Stand-in for WeightMatrix without sparse / AccountingStructure."""

        def __init__(self, W_dense: np.ndarray, feed_id: str, acc_id: str):
            self.id          = feed_id
            self.acc_used    = acc_id
            self.odir        = ""
            # Store as a plain numpy array; UncertaintyMixin expects *something*
            # that supports elementwise * and .T @ vec.  ndarray does both.
            self.weight_matrix = W_dense

        def save(self, **kwargs):
            import pickle
            path = kwargs.get("path", f"{self.id}.pkl")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"id": self.id, "W": self.weight_matrix}, f)

        # Override apply_with_uncertainty to skip sparse-specific calls
        def apply_with_uncertainty(self, F_src=None, pedigree=None):
            F_src = np.asarray(F_src if F_src is not None else self.F_src, dtype=float)
            pg    = pedigree or self.pedigree
            cv    = pg.total_cv()
            W     = self.weight_matrix
            F_root    = W.T @ F_src
            var_root  = cv**2 * ((W**2).T @ F_src**2)
            sigma     = np.sqrt(np.maximum(var_root, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                cv_root = np.where(F_root != 0.0, sigma / np.abs(F_root), np.nan)
            return PropagationResult(F_root=F_root, sigma_root=sigma,
                                     cv_root=cv_root, cv_src=cv)

    # ------------------------------------------------------------------ #
    # A  Metadata (from dict, simulating Excel read)                      #
    # ------------------------------------------------------------------ #
    meta = DataFeedMetadata(
        feed_name  = "icsg",
        iedc_type  = "1_F_",
        reliability              = 1,
        completeness             = 1,
        temporal_correlation     = 1,
        geographical_correlation = 1,
        other_correlation        = 1,
        version            = 0,
        date_of_assessment = "2025-06-11",
        person             = "Florian_Huber",
        dataset_link       = "https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37",
    )
    print("A  metadata loaded:", meta.feed_name, meta.iedc_type)

    # ------------------------------------------------------------------ #
    # B  (skipped – concordance needs real AccountingStructure)           #
    # ------------------------------------------------------------------ #
    print("B  [skipped in smoke test – ConcordanceTable needs AccountingStructure]")

    # ------------------------------------------------------------------ #
    # C  Synthetic weight matrix                                          #
    # ------------------------------------------------------------------ #
    W = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.3, 0.7],
    ])                              # 4 src edges → 3 root edges
    wm = _MockWeightMatrix(W, feed_id="icsg", acc_id="acc_test")
    print("C  weight matrix shape:", W.shape)

    # ------------------------------------------------------------------ #
    # D  Apply + propagate                                                #
    # ------------------------------------------------------------------ #
    F_src  = np.array([100.0, 200.0, 150.0, 80.0])
    pg     = meta.to_pedigree_scores()
    wm.set_pedigree(pg)
    wm.set_F_src(F_src)
    result = wm.apply_with_uncertainty()

    print("\nD  Propagation result")
    print(f"   CV_src       : {result.cv_src*100:.4f} %")
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

        print("\nRegistry entry")
        print(json.dumps(entry.to_dict(), indent=2))

        # Round-trip: reload and print summary
        registry2 = DataFeedRegistry(odir=tmpdir)
        registry2.load_index()
        print("\nRegistry summary:\n", registry2.summary_df().T.to_string())

        reloaded_meta   = registry2.load_metadata("icsg")
        reloaded_result = registry2.load_result("icsg")
        print(f"\nReloaded cv_total  : {reloaded_meta.cv_total*100:.4f} %")
        print(f"Reloaded F_root    : {reloaded_result['F_root']}")
        print(f"Reloaded sigma_root: {reloaded_result['sigma_root']}")

    print("\n✓ Smoke test passed.")