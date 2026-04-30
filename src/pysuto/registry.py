"""
registry.py
===========
DataFeedMetadata  –  structured container for all per-feed metadata
                     (identification, pedigree scores, provenance,
                      and computed outputs).

DataFeedRegistry  –  persistent catalogue of every DataFeed ever registered.

_RegistryEntry    –  index record for one feed (stored in registry.json).

_NpEncoder        –  JSON encoder that handles numpy scalars and ndarrays.

File layout on disk
-------------------
<odir>/
    registry.json              ← flat index, one entry per feed
    <feed_id>/
        metadata.json          ← DataFeedMetadata (human-readable)
        <feed_id>.pkl          ← WeightMatrix / ConcordanceTable pickle
        result.npz             ← F_root, sigma_root, cv_root, cv_src arrays

Import contract
---------------
This module only imports from the standard library, numpy, and pandas.
It does NOT import from pysuto.unc, pysuto.conc, or pysuto.feed so that
it can be imported in isolation for testing or tooling without pulling in
the full package dependency tree.  The only pysuto type that crosses the
boundary is the duck-typed ``feed`` argument to ``DataFeedRegistry.register``
(must have ``.id``, ``.acc_used``, ``.save(path=...)``) and the duck-typed
``result`` argument (must have ``.F_root``, ``.sigma_root``, ``.cv_root``,
``.cv_src``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


from pysuto.meta import DataFeedMetadata

# =============================================================================
# JSON helpers
# =============================================================================

class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, arrays, and date objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def _json_dump(obj: Any, path: str) -> None:
    """Write *obj* to *path* as indented JSON, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def _json_load(path: str) -> Any:
    """Load JSON from *path*."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class _RegistryEntry:
    """One row in ``registry.json``."""

    feed_id:       str
    feed_class:    str   # e.g. "WeightMatrix"
    acc_id:        str   # AccountingStructure id used to build this feed
    odir:          str   # directory that holds this feed's files
    metadata_path: str   # absolute path to metadata.json
    created_at:    str   # UTC ISO-8601 timestamp

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "_RegistryEntry":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# DataFeedRegistry
# =============================================================================

class DataFeedRegistry:
    """
    Persistent catalogue of all registered DataFeed instances.

    The registry owns one index file (``registry.json``) and a
    sub-directory per feed containing::

        <feed_id>/
            metadata.json    – DataFeedMetadata (human-readable JSON)
            <feed_id>.pkl    – feed object (pickle, written by feed.save())
            result.npz       – F_root, sigma_root, cv_root, cv_src arrays

    Typical workflow::

        registry = DataFeedRegistry(odir="data/proc/feeds")
        registry.load_index()          # load existing index (no-op if absent)

        entry = registry.register(wm, metadata, result=prop_result)
        registry.save()                # flush index to disk

        # later session:
        registry.load_index()
        meta   = registry.load_metadata("icsg")
        arrays = registry.load_result("icsg")

    Parameters
    ----------
    odir : str
        Root directory for all registry files.
    """

    REGISTRY_FILENAME = "registry.json"

    def __init__(self, odir: str = "data/proc/feeds") -> None:
        self.odir = odir
        self._index: Dict[str, _RegistryEntry] = {}

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """
        Load the registry index from ``<odir>/registry.json``.

        No-op if the file does not exist (fresh registry).
        """
        path = os.path.join(self.odir, self.REGISTRY_FILENAME)
        if not os.path.exists(path):
            return
        raw = _json_load(path)
        self._index = {k: _RegistryEntry.from_dict(v) for k, v in raw.items()}

    def save(self) -> None:
        """Flush the in-memory index to ``<odir>/registry.json``."""
        Path(self.odir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(self.odir, self.REGISTRY_FILENAME)
        _json_dump({k: v.to_dict() for k, v in self._index.items()}, path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        feed: Any,
        metadata: DataFeedMetadata,
        result: Any = None,
        overwrite: bool = False,
    ) -> _RegistryEntry:
        """
        Persist a feed and create its registry entry.

        Steps (in order)
        ----------------
        1. Raise ``ValueError`` if ``feed.id`` is already registered
           and ``overwrite=False``.
        2. Call ``metadata.fill_computed(result)`` if result is not None.
        3. Write ``metadata.json`` sidecar.
        4. Call ``feed.save(path=<pkl_path>)`` to write the pickle.
        5. Write ``result.npz`` if result is not None.
        6. Create and store the ``_RegistryEntry`` in the in-memory index.

        Note: ``save()`` is **not** called automatically — flush the index
        explicitly with ``registry.save()`` after registering.

        Parameters
        ----------
        feed      : object with ``.id``, ``.acc_used``, ``.save(path=...)``
        metadata  : DataFeedMetadata instance
        result    : optional PropagationResult (provides F_root, sigma_root,
                    cv_root, cv_src)
        overwrite : if True, silently replace an existing entry

        Returns
        -------
        _RegistryEntry
        """
        fid = feed.id
        if fid in self._index and not overwrite:
            raise ValueError(
                f"Feed '{fid}' is already registered. "
                "Pass overwrite=True to replace it."
            )

        feed_odir = os.path.join(self.odir, fid)
        Path(feed_odir).mkdir(parents=True, exist_ok=True)

        # 2. fill computed fields
        if result is not None:
            metadata.fill_computed(result)

        # 3. metadata sidecar
        meta_path = os.path.join(feed_odir, "metadata.json")
        _json_dump(metadata.to_dict(), meta_path)

        # 4. feed pickle
        pkl_path = os.path.join(feed_odir, f"{fid}.pkl")
        feed.save(path=pkl_path)

        # 5. result arrays
        if result is not None:
            np.savez(
                os.path.join(feed_odir, "result.npz"),
                F_root     = result.F_root,
                sigma_root = result.sigma_root,
                cv_root    = result.cv_root,
                cv_src     = np.array([result.cv_src]),
            )

        # 6. index entry
        entry = _RegistryEntry(
            feed_id       = fid,
            feed_class    = type(feed).__name__,
            acc_id        = feed.acc_used,
            odir          = feed_odir,
            metadata_path = meta_path,
            created_at    = datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
        self._index[fid] = entry
        return entry

    # ------------------------------------------------------------------
    # Lookup / container protocol
    # ------------------------------------------------------------------

    def __getitem__(self, feed_id: str) -> _RegistryEntry:
        return self._index[feed_id]

    def __contains__(self, feed_id: str) -> bool:
        return feed_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def ids(self) -> List[str]:
        """Return a list of all registered feed IDs."""
        return list(self._index.keys())

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def load_metadata(self, feed_id: str) -> DataFeedMetadata:
        """Load and return the DataFeedMetadata for *feed_id*."""
        entry = self[feed_id]
        return DataFeedMetadata.from_dict(_json_load(entry.metadata_path))

    def load_result(self, feed_id: str) -> Dict[str, np.ndarray]:
        """
        Load the result arrays for *feed_id*.

        Returns
        -------
        dict with keys ``F_root``, ``sigma_root``, ``cv_root``, ``cv_src``
        """
        npz_path = os.path.join(self._index[feed_id].odir, "result.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"No result.npz found for feed '{feed_id}'. "
                "Was a PropagationResult passed to register()?"
            )
        z = np.load(npz_path)
        return {k: z[k] for k in z.files}

    def load_feed(self, feed_id: str, feed_cls: Any, acc_struct: Any) -> Any:
        """
        Reload a feed object from its pickle.

        Parameters
        ----------
        feed_id    : registered feed ID
        feed_cls   : class with a ``.load(id, acc_struct, path=...)`` classmethod
        acc_struct : AccountingStructure instance to pass to feed_cls.load
        """
        entry = self[feed_id]
        pkl   = os.path.join(entry.odir, f"{feed_id}.pkl")
        return feed_cls.load(feed_id, acc_struct, path=pkl)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per registered feed.

        Columns come from the registry index plus selected metadata fields.
        Feeds whose metadata cannot be loaded produce a row with NaN
        metadata columns and emit a warning.
        """

        if not self._index:
            return pd.DataFrame(columns=["feed_id"]).set_index("feed_id")
        rows = []
        for fid, entry in self._index.items():
            row = entry.to_dict()
            try:
                meta = self.load_metadata(fid)
                row.update({
                    "feed_name":       meta.feed_name,
                    "iedc_type":       meta.iedc_type,
                    "cv_total_%":      round(meta.cv_total * 100, 3)
                                       if meta.cv_total is not None else None,
                    "F_root_sum":      meta.F_root_sum,
                    "sigma_root_mean": meta.sigma_root_mean,
                    "person":          meta.person,
                    "dataset_link":    meta.dataset_link,
                })
            except Exception as exc:
                warnings.warn(f"Could not load metadata for '{fid}': {exc}")
            rows.append(row)
        return pd.DataFrame(rows).set_index("feed_id")


# =============================================================================
# ImputationRunRegistry
# =============================================================================

class ImputationRunRegistry:
    """
    Persistent registry for Stage 2 imputation runs.

    Mirrors ConstraintRunRegistry: thin JSON index + per-run subdirectory
    holding arrays, log, and a metadata sidecar.

    Directory layout::

        {storage_dir}/
        ├── registry.json
        └── {run_id}/
            ├── metadata.json          ← run + step + feed + imputer-config detail
            ├── agg_result.npz         ← arrays (f_agg, sigma_agg, Omega, F, Σ, coverage, feed_ids)
            └── imputation_log.parquet ← per-edge fill log

    Run-id format: ``imp_<label>_<YYYYMMDD_HHMMSS>_<shorthash>``.
    """

    REGISTRY_FILENAME = "registry.json"

    def __init__(self, storage_dir: str = "data/proc/imp") -> None:
        self.storage_dir = storage_dir
        self._index: Dict[str, dict] = {}
        os.makedirs(self.storage_dir, exist_ok=True)
        self._index_path = os.path.join(self.storage_dir, self.REGISTRY_FILENAME)
        self._load_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        if os.path.exists(self._index_path):
            self._index = _json_load(self._index_path)

    def _save_index(self) -> None:
        _json_dump(self._index, self._index_path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        agg_result,
        *,
        label: str,
        feeds: List[dict],
        imputers: List[Any],
        acc_id: str,
    ) -> str:
        """
        Persist one imputation run and return its run_id.

        Parameters
        ----------
        agg_result : AggregationResult after ImputationEngine.run().
                     Must have .imputation_log set (raises otherwise).
        label      : human-readable label (folded into the run_id).
        feeds      : list of dicts {"feed_id", "version", "registry_path"}
                     as produced by FeedCollector.select().
        imputers   : list of BaseImputer instances (engine.imputers).
        acc_id     : AccountingStructure id.
        """
        import uuid

        if agg_result.imputation_log is None:
            raise ValueError(
                "AggregationResult.imputation_log is None — register only "
                "after ImputationEngine.run(). Stage 2a-only runs are not "
                "supported by this registry."
            )

        created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        ts_compact = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short      = uuid.uuid4().hex[:8]
        safe_label = label.replace(" ", "_") if label else "run"
        run_id     = f"imp_{safe_label}_{ts_compact}_{short}"

        run_dir = os.path.join(self.storage_dir, run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        # 1. arrays -----------------------------------------------------
        np.savez(
            os.path.join(run_dir, "agg_result.npz"),
            f_agg     = agg_result.f_agg,
            sigma_agg = agg_result.sigma_agg,
            Omega     = agg_result.Omega,
            F         = agg_result.F,
            Sigma     = agg_result.Sigma,
            coverage  = agg_result.coverage,
            feed_ids  = np.array(agg_result.feed_ids, dtype=object),
        )

        # 2. imputation log (parquet) -----------------------------------
        log_df = agg_result.imputation_log.as_dataframe()
        log_df.to_parquet(
            os.path.join(run_dir, "imputation_log.parquet"),
            index=False,
        )

        # 3. metadata sidecar ------------------------------------------
        n_total    = int(agg_result.f_agg.shape[0])
        n_imputed  = int(len(log_df))
        n_after    = int(np.isfinite(agg_result.f_agg).sum())
        coverage   = agg_result.coverage
        n_before   = int(coverage.sum())

        meta_records: list[dict] = []
        if agg_result.imputation_metadata is not None:
            meta_records = agg_result.imputation_metadata.as_dataframe().to_dict("records")
        # Splice imputer config snapshots into matching step records.
        for rec in meta_records:
            step = int(rec.get("step", -1))
            if 0 <= step < len(imputers):
                imp = imputers[step]
                rec["config"] = imp.config() if hasattr(imp, "config") else {}

        metadata_obj = {
            "run_id":     run_id,
            "label":      label,
            "acc_id":     acc_id,
            "created_at": created_at,
            "feeds":      feeds,
            "imputers":   meta_records,
            "totals": {
                "n_total":          n_total,
                "n_covered_before": n_before,
                "n_covered_after":  n_after,
                "n_imputed":        n_imputed,
            },
        }
        _json_dump(metadata_obj, os.path.join(run_dir, "metadata.json"))

        # 4. update index ----------------------------------------------
        self._index[run_id] = {
            "run_id":     run_id,
            "label":      label,
            "acc_id":     acc_id,
            "created_at": created_at,
            "n_edges":    n_total,
            "n_filled":   n_imputed,
            "n_steps":    len(meta_records),
            "feed_ids":   [f["feed_id"] for f in feeds],
        }
        self._save_index()

        # 5. stamp run_id back onto the result for downstream provenance
        agg_result._imp_run_id = run_id

        return run_id

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, run_id: str):
        """
        Reconstruct an AggregationResult (with imputation_log attached) from
        disk. Returns the dataclass; metadata.json contents available via
        load_metadata().
        """
        from pysuto.agg import AggregationResult
        from pysuto.impute import ImputationLog

        if run_id not in self._index:
            raise KeyError(f"Imputation run '{run_id}' not in registry")

        run_dir = os.path.join(self.storage_dir, run_id)
        data    = np.load(os.path.join(run_dir, "agg_result.npz"), allow_pickle=True)

        log_path = os.path.join(run_dir, "imputation_log.parquet")
        log_obj  = ImputationLog()
        if os.path.exists(log_path):
            df = pd.read_parquet(log_path)
            log_obj.extend(df.to_dict("records"))

        result = AggregationResult(
            f_agg     = data["f_agg"],
            sigma_agg = data["sigma_agg"],
            Omega     = data["Omega"],
            F         = data["F"],
            Sigma     = data["Sigma"],
            feed_ids  = list(data["feed_ids"]),
            coverage  = data["coverage"].astype(bool),
        )
        result.imputation_log = log_obj
        result._imp_run_id    = run_id
        return result

    def load_metadata(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"Imputation run '{run_id}' not in registry")
        return _json_load(os.path.join(self.storage_dir, run_id, "metadata.json"))

    # ------------------------------------------------------------------
    # Container protocol & summary
    # ------------------------------------------------------------------

    def __getitem__(self, run_id: str) -> dict:
        return self._index[run_id]

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def ids(self) -> List[str]:
        return list(self._index.keys())

    def summary_df(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self._index, orient="index").reset_index(drop=True)
