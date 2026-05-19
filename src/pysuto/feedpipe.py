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
    p:        float = 1.3,
) -> PropagationResult:
    """
    1.  Derive PedigreeScores from metadata and attach to the weight matrix.
    2.  Store F_src on the weight matrix.
    3.  Run the combined apply + Lp uncertainty propagation (scipy.sparse).

    Propagation formula (power-mean / Lp norm):
        F_root      = W.T @ F_src
        σ_root[j]   = CV · ( Σ_i (|W_ij| · |f_i|)^p )^(1/p)
        cv_root     = σ_root / |F_root|

    Returns
    -------
    PropagationResult with F_root, sigma_root, cv_root, cv_src.
    """
    _log.info("D  propagating uncertainty  F_src_nnz=%d  F_src_sum=%.1f kt",
              int(np.count_nonzero(F_src)), float(F_src.sum()))
    pg = metadata.to_pedigree_scores()
    wm.set_pedigree(pg)
    wm.set_F_src(F_src)
    result = wm.apply_with_uncertainty(p=p)
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
    *,
    parents:   dict,
    F_src:     Optional[np.ndarray] = None,
    label:     Optional[str] = None,
    stage_config:  Optional[dict] = None,
    stage_summary: Optional[dict] = None,
):
    """
    Persist the weight matrix, metadata sidecar, result arrays and the
    source flow vector under a freshly minted ``run_id``, then flush
    the registry index to disk.

    ``parents`` follows the harmonized sidecar schema:
    ``{"inputs": {"acc": {"id", "version"},
                  "feed": {"id", "version"}}, "upstream": {}}``.

    Returns the registry entry (with ``.run_dir`` and ``.run_id``).
    """
    _log.info("E  registering feed '%s'", metadata.feed_name)
    entry = registry.register(
        wm, metadata,
        parents       = parents,
        result        = result,
        F_src         = F_src,
        label         = label,
        stage_config  = stage_config,
        stage_summary = stage_summary,
    )
    registry.save()
    _log.info("   run_id        → %s", entry.run_id)
    _log.info("   run dir       → %s", entry.run_dir)
    return entry


# ===========================================================================
# FULL PIPELINE (convenience wrapper)
# ===========================================================================

def _apply_edge_scope_mask(ct, acc_struct, mask_name: str) -> None:
    """Filter ``ct.tab`` in-place to drop rows whose resolved root edge is not
    in ``acc_struct.masks[mask_name]``. Missing mask is a logged warning.
    """
    if not mask_name:
        return
    masks = getattr(acc_struct, "masks", None) or {}
    if mask_name not in masks:
        _log.warning(
            "edge_scope_mask=%r is not present in acc_struct.masks "
            "(available: %s); skipping scope filter.",
            mask_name, sorted(masks),
        )
        return

    scope = masks[mask_name]
    n_before = len(ct.tab)

    # Resolve each tab row to its edge_id via (from_id, to_id, time, valuation).
    tab = ct.tab
    from_id = tab.apply(
        lambda r: acc_struct.node_to_id[
            (r["root_sector_from"], r["root_entity_from"], r["root_region_from"])
        ], axis=1,
    )
    to_id = tab.apply(
        lambda r: acc_struct.node_to_id[
            (r["root_sector_to"], r["root_entity_to"], r["root_region_to"])
        ], axis=1,
    )
    keys = pd.DataFrame({
        "from_id":   from_id.values,
        "to_id":     to_id.values,
        "time":      tab["root_time"].astype(str).values,
        "valuation": tab["root_valuation"].astype(str).values,
    })

    ei = acc_struct.edge_index[["from_id", "to_id", "time", "valuation", "edge_id"]].copy()
    ei["time"]      = ei["time"].astype(str)
    ei["valuation"] = ei["valuation"].astype(str)
    joined = keys.merge(ei, on=["from_id", "to_id", "time", "valuation"], how="left")
    edge_ids = joined["edge_id"].to_numpy()

    in_scope = np.zeros(n_before, dtype=bool)
    valid = ~pd.isna(edge_ids)
    in_scope[valid] = scope[edge_ids[valid].astype(int)]

    ct.tab = ct.tab.loc[in_scope].reset_index(drop=True)
    n_after = len(ct.tab)
    if n_after != n_before:
        _log.info(
            "edge_scope_mask=%r: kept %d / %d concordance rows (%d dropped)",
            mask_name, n_after, n_before, n_before - n_after,
        )


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
    *,
    parents:          dict,
    feed_id:          Optional[str] = None,
    aux_col:          Optional[str] = None,
    allow_unmatched_edges: bool = True,
    diag_plot:        bool = True,
    conc_file:        Optional[str] = None,
    p:                float = 1.3,
    label:            Optional[str] = None,
) -> tuple[WeightMatrix, DataFeedMetadata, PropagationResult, "object"]:
    """
    Execute steps A → D, register the run and (optionally) render the
    feed diagnostic figure into the per-run ``diag/`` directory.

    ``parents`` carries the harmonized parent block referencing the
    ``acc`` + ``feed`` entries from the input registries.

    Returns ``(wm, metadata, result, entry)`` where ``entry.run_id``
    and ``entry.run_dir`` identify the freshly registered run.
    """
    meta   = step_A_read_metadata(excel_path)
    fid    = feed_id or meta.feed_name or "unnamed_feed"
    ct     = step_B_build_concordance(
                 raw_df, acc_struct,
                 sector_map_df, region_map_df,
                 time_map_df, valuation_map_df, var_map_df,
                 feed_id=fid, aux_col=aux_col,
                 allow_unmatched_edges=allow_unmatched_edges)
    _apply_edge_scope_mask(ct, acc_struct, meta.edge_scope_mask)
    wm     = step_C_build_weight_matrix(ct)
    result = step_D_apply_and_propagate(wm, F_src, meta, p=p)
    entry  = register_feed(
        wm, meta, result, registry,
        parents = parents,
        F_src   = F_src,
        label   = label,
        stage_config = {"p": p, "allow_unmatched_edges": allow_unmatched_edges},
    )

    if diag_plot:
        from pysuto.diag import FeedDiagnostics
        acc_id_lab = (parents.get("inputs", {}).get("acc", {}).get("id") or "")
        diag = FeedDiagnostics(
            feed_run_id = entry.run_id,
            feed_id     = fid,
            acc_id      = acc_id_lab,
            raw_df      = raw_df,
            ct          = ct,
            F_src       = F_src,
            result      = result,
            conc_file   = conc_file,
        )
        diag.plot(odir=os.path.join(entry.run_dir, "diag"))

    return wm, meta, result, entry


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
            p: float = 1.3,
        ) -> PropagationResult:
            """Lp uncertainty propagation with scipy.sparse."""
            F   = np.asarray(
                F_src if F_src is not None else self.F_src,
                dtype=float,
            )
            pg  = pedigree or self.pedigree
            cv  = pg.total_cv()

            W: sp.csr_matrix = self.weight_matrix          # (n_src, n_root)

            # F_root = W.T @ F_src  →  dense (n_root,)
            F_root: np.ndarray = np.asarray(W.T @ F).ravel()

            Wp = W.copy()
            Wp.data = np.abs(Wp.data) ** p
            sum_p: np.ndarray = np.asarray(Wp.T @ (np.abs(F) ** p)).ravel()
            sigma: np.ndarray = cv * np.power(np.maximum(sum_p, 0.0), 1.0 / p)

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
