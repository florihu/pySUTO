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


# =============================================================================
# DataFeedMetadata
# =============================================================================

@dataclass
class DataFeedMetadata:
    """
    All metadata for one DataFeed.

    Fields mirror the per-feed Excel sidecar layout::

        Grouping            | Meta_name                 | Value
        ─────────────────────────────────────────────────────
        Identification      | Feed_name                 | icsg
                            | IEDC_type                 | 1_F_
        Unc (pedigree)      | Reliability               | 1
                            | Completeness              | 1
                            | Temporal_correlation      | 1
                            | Geographical_correlation  | 1
                            | Other_correlation         | 1
        Submission/review   | Version                   | 0
                            | Date_of_assessment        | 2025-06-11
                            | Person                    | Florian_Huber
                            | Dataset_link              | https://...
        Computed (derived)  | cv_total                  | (filled by registry)
                            | F_root_sum                | (filled by registry)
                            | sigma_root_mean           | (filled by registry)
                            | computation_timestamp     | (filled by registry)
    """

    # --- Identification ---------------------------------------------------
    feed_name: str = ""
    iedc_type: str = ""

    # --- Pedigree (Laner et al. 2016) -------------------------------------
    reliability:              int = 1
    completeness:             int = 1
    temporal_correlation:     int = 1
    geographical_correlation: int = 1
    other_correlation:        int = 1

    completeness_sensitivity:   str = "medium_sensitive"
    temporal_sensitivity:       str = "medium_sensitive"
    geographical_sensitivity:   str = "medium_sensitive"
    other_sensitivity:          str = "medium_sensitive"

    # --- Submission / review ----------------------------------------------
    version:            Any = 0
    date_of_assessment: str = ""
    person:             str = ""
    dataset_link:       str = ""

    # --- Computed (filled by DataFeedRegistry.register) -------------------
    cv_total:              Optional[float] = None
    F_root_sum:            Optional[float] = None
    sigma_root_mean:       Optional[float] = None
    computation_timestamp: str             = ""

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet_name: Any = 0,
        name_col:  str = "Meta_name",
        value_col: str = "Value",
    ) -> "DataFeedMetadata":
        """
        Parse an Excel metadata table (name→value layout).

        The lookup is case-insensitive on ``Meta_name`` values, so
        ``Reliability`` and ``reliability`` both resolve correctly.

        Parameters
        ----------
        path       : path to the Excel file
        sheet_name : sheet index or name (default: first sheet)
        name_col   : column containing the field name
        value_col  : column containing the field value
        """
        df = pd.read_excel(path, sheet_name=sheet_name)
        df.columns = [c.strip() for c in df.columns]
        lookup: Dict[str, Any] = {
            str(row[name_col]).strip(): row[value_col]
            for _, row in df.iterrows()
            if pd.notna(row[name_col])
        }

        def get(key: str, default: Any = None) -> Any:
            for k, v in lookup.items():
                if k.lower() == key.lower():
                    return v
            return default

        def get_int(key: str, default: int = 1) -> int:
            v = get(key, default)
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        return cls(
            feed_name                = str(get("Feed_name", "")),
            iedc_type                = str(get("IEDC_type",  "")),
            reliability              = get_int("Reliability",              1),
            completeness             = get_int("Completeness",             1),
            temporal_correlation     = get_int("Temporal_correlation",     1),
            geographical_correlation = get_int("Geographical_correlation", 1),
            other_correlation        = get_int("Other_correlation",        1),
            version            = get("Version",            0),
            date_of_assessment = str(get("Date_of_assessment", "")),
            person             = str(get("Person",             "")),
            dataset_link       = str(get("Dataset_link",       "")),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DataFeedMetadata":
        """Construct from a plain dict, ignoring unknown keys."""
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        """Serialise to a plain dict (all fields, including computed ones)."""
        return dataclasses.asdict(self)

    # ------------------------------------------------------------------
    # Computed-field population
    # ------------------------------------------------------------------

    def fill_computed(self, result: Any) -> None:
        """
        Populate the four derived fields from a ``PropagationResult``.

        Parameters
        ----------
        result : object with attributes F_root, sigma_root, cv_src
            Typically ``pysuto.unc.PropagationResult``.
        """
        self.cv_total              = float(result.cv_src)
        self.F_root_sum            = float(np.nansum(result.F_root))
        self.sigma_root_mean       = float(np.nanmean(result.sigma_root))
        self.computation_timestamp = (
            datetime.utcnow().isoformat(timespec="seconds") + "Z"
        )


# =============================================================================
# _RegistryEntry  (internal index record)
# =============================================================================

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
