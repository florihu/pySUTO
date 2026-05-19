"""
registry.py
===========
Harmonized output registries for Stage 1 (feed) and Stage 2 (imp).

All stage output registries share one run-id pattern and one sidecar
schema, defined in :class:`BaseOutputRegistry`:

* run-id literal format ``{stage}_{YYYYMMDD_HHMMSS}_{8-char-uuid4-hex}``
  — no label / feed-id / acc-id / parent run-id embedded.
* per-run sidecar ``metadata.json`` with ``parents.inputs`` (refs into
  the input registries) and ``parents.upstream`` (refs into upstream
  output registries), plus free-form ``stage_config`` / ``stage_summary``.
* top-level ``registry.json`` index trimmed to four fields per entry::

      {"run_id": "...", "stage": "feed", "label": "...", "created_at": "..."}

Everything richer (parents, config, summary, stage-specific metadata)
lives in the per-run sidecar.

File layout on disk (Stage 1 example)::

    data/proc/feed/registry/
    ├── registry.json                       ← trimmed index
    └── feed_20260518_143022_a4f1c3d2/
        ├── metadata.json                   ← sidecar
        ├── feed.pkl                        ← Stage-1-specific
        ├── result.npz                      ← Stage-1-specific
        ├── F_src.npy                       ← Stage-1-specific
        └── diag/<run_id>_feed.png          ← Stage-1-specific

Stage 2 has the same shape, with ``agg_result.npz`` and
``imputation_log.parquet`` instead of the Stage-1 artefacts.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def _json_load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _now_iso() -> str:
    """Current UTC time as ISO-8601 with trailing ``Z`` (seconds resolution)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _short_code() -> str:
    return uuid.uuid4().hex[:8]


def make_run_id(stage: str) -> str:
    """Mint a harmonized run-id ``{stage}_{ts}_{code}``."""
    return f"{stage}_{_ts_compact()}_{_short_code()}"


# =============================================================================
# Trimmed index entry
# =============================================================================

@dataclass
class _RegistryEntry:
    """One row in the top-level ``registry.json`` (4 fields on disk).

    On load, :attr:`run_dir` is filled in from ``{storage_dir}/{run_id}``
    as a convenience for callers — it is *not* persisted.
    """

    run_id:     str
    stage:      str
    label:      str
    created_at: str
    run_dir:    str = ""

    INDEX_FIELDS = ("run_id", "stage", "label", "created_at")

    def to_index_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.INDEX_FIELDS}

    @classmethod
    def from_index_dict(cls, d: dict, *, storage_dir: str = "") -> "_RegistryEntry":
        return cls(
            run_id     = d["run_id"],
            stage      = d["stage"],
            label      = d.get("label", ""),
            created_at = d["created_at"],
            run_dir    = os.path.join(storage_dir, d["run_id"]) if storage_dir else "",
        )


# =============================================================================
# Sidecar schema helpers
# =============================================================================

def build_sidecar(
    *,
    run_id:        str,
    stage:         str,
    label:         str,
    created_at:    str,
    parents:       dict,
    stage_config:  Optional[dict] = None,
    stage_summary: Optional[dict] = None,
    extra:         Optional[dict] = None,
) -> dict:
    """Assemble the per-run sidecar payload.

    ``parents`` must follow the schema
    ``{"inputs": {...}, "upstream": {...}}``; both sub-dicts may be empty.
    ``extra`` lets stage-specific code stash additional fields (e.g.
    ``feed_metadata`` for Stage 1, ``feeds`` array for Stage 2)."""
    payload = {
        "run_id":        run_id,
        "stage":         stage,
        "label":         label,
        "created_at":    created_at,
        "parents":       {
            "inputs":   parents.get("inputs",   {}),
            "upstream": parents.get("upstream", {}),
        },
        "stage_config":  stage_config  or {},
        "stage_summary": stage_summary or {},
    }
    if extra:
        payload.update(extra)
    return payload


# =============================================================================
# BaseOutputRegistry
# =============================================================================

class BaseOutputRegistry:
    """Shared scaffolding for the four output registries.

    Owns:
      * the harmonized run-id format (``{stage}_{ts}_{code}``),
      * the trimmed top-level ``registry.json`` (4 fields per entry),
      * the per-run ``metadata.json`` sidecar IO,
      * the container protocol (``[]``, ``in``, ``len``, iter).

    Concrete subclasses set :attr:`STAGE` and provide stage-specific
    ``register()`` / ``load_*`` logic that calls
    :meth:`_persist_run_dir` / :meth:`_write_sidecar` underneath.
    """

    STAGE: str = ""
    REGISTRY_FILENAME = "registry.json"
    SIDECAR_FILENAME  = "metadata.json"

    def __init__(self, storage_dir: str) -> None:
        if not self.STAGE:
            raise NotImplementedError(f"{type(self).__name__}.STAGE not set")
        self.storage_dir = storage_dir
        self._index: Dict[str, _RegistryEntry] = {}
        self._index_path = os.path.join(storage_dir, self.REGISTRY_FILENAME)

    # ── lifecycle ─────────────────────────────────────────────────────

    def load_index(self) -> None:
        """Load the trimmed index from ``{storage_dir}/registry.json``.

        Missing file is a no-op (fresh registry).
        """
        if not os.path.exists(self._index_path):
            self._index = {}
            return
        raw = _json_load(self._index_path)
        self._index = {
            k: _RegistryEntry.from_index_dict(v, storage_dir=self.storage_dir)
            for k, v in raw.items()
        }

    def save(self) -> None:
        """Flush the trimmed index to ``{storage_dir}/registry.json``."""
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        _json_dump(
            {k: v.to_index_dict() for k, v in self._index.items()},
            self._index_path,
        )

    # ── registration scaffolding ──────────────────────────────────────

    def _new_run_id(self) -> str:
        return make_run_id(self.STAGE)

    def _make_run_dir(self, run_id: str) -> str:
        run_dir = os.path.join(self.storage_dir, run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_sidecar(self, run_dir: str, payload: dict) -> str:
        path = os.path.join(run_dir, self.SIDECAR_FILENAME)
        _json_dump(payload, path)
        return path

    def _read_sidecar(self, run_id: str) -> dict:
        run_dir = os.path.join(self.storage_dir, run_id)
        return _json_load(os.path.join(run_dir, self.SIDECAR_FILENAME))

    def _add_entry(
        self,
        *,
        run_id:     str,
        label:      str,
        created_at: str,
    ) -> _RegistryEntry:
        entry = _RegistryEntry(
            run_id     = run_id,
            stage      = self.STAGE,
            label      = label,
            created_at = created_at,
            run_dir    = os.path.join(self.storage_dir, run_id),
        )
        self._index[run_id] = entry
        return entry

    # ── container protocol ────────────────────────────────────────────

    def __getitem__(self, run_id: str) -> _RegistryEntry:
        return self._index[run_id]

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def run_ids(self) -> List[str]:
        return list(self._index.keys())

    def labels(self) -> List[str]:
        return sorted({e.label for e in self._index.values() if e.label})

    def latest_by_label(self, label: str) -> _RegistryEntry:
        runs = [e for e in self._index.values() if e.label == label]
        if not runs:
            raise KeyError(f"No registered run for label '{label}'")
        runs.sort(key=lambda e: e.created_at)
        return runs[-1]


# =============================================================================
# DataFeedRegistry — Stage 1
# =============================================================================

class DataFeedRegistry(BaseOutputRegistry):
    """Stage-1 registry.

    Per-run directory layout::

        data/proc/feed/registry/
        ├── registry.json
        └── feed_<ts>_<code>/
            ├── metadata.json         ← harmonized sidecar (+ ``feed_metadata`` block)
            ├── feed.pkl              ← WeightMatrix + concordance pickle
            ├── result.npz            ← F_root, sigma_root, cv_root, cv_src
            ├── F_src.npy             ← (optional) source flow vector
            └── diag/<run_id>_feed.png

    The sidecar's ``parents.inputs.feed.id`` carries the upstream
    input-feed-id (e.g. ``"baci"``); :meth:`latest_by_input_feed_id`
    follows that link to support feed-id lookups even though the
    run-id no longer embeds the feed-id.
    """

    STAGE = "feed"

    def __init__(self, odir: str = "data/proc/feed/registry") -> None:
        super().__init__(storage_dir=odir)
        self.odir = odir          # alias — keeps backwards-compatible kwarg name
        self._input_feed_idx: dict[str, list[str]] = {}  # input_feed_id -> [run_ids]

    # ── overridden lifecycle: rebuild reverse index ───────────────────

    def load_index(self) -> None:
        super().load_index()
        self._rebuild_input_feed_index()

    def _rebuild_input_feed_index(self) -> None:
        self._input_feed_idx.clear()
        for run_id, entry in self._index.items():
            try:
                sidecar = self._read_sidecar(run_id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            feed_ref = (sidecar.get("parents", {})
                              .get("inputs", {})
                              .get("feed"))
            if not feed_ref:
                continue
            fid = feed_ref.get("id")
            if not fid:
                continue
            self._input_feed_idx.setdefault(fid, []).append(run_id)
        # Sort each bucket by created_at ascending so [-1] = latest.
        for fid, run_ids in self._input_feed_idx.items():
            run_ids.sort(key=lambda r: self._index[r].created_at)

    # ── register ──────────────────────────────────────────────────────

    def register(
        self,
        feed: Any,
        metadata: DataFeedMetadata,
        *,
        parents:       dict,
        result:        Any = None,
        F_src:         Optional[np.ndarray] = None,
        label:         Optional[str] = None,
        stage_config:  Optional[dict] = None,
        stage_summary: Optional[dict] = None,
    ) -> _RegistryEntry:
        """Persist a Stage-1 run and append it to the index.

        Parameters
        ----------
        feed
            Duck-typed object with ``.save(path=...)``.
        metadata
            ``DataFeedMetadata``; ``fill_computed(result)`` is invoked
            if *result* is provided.
        parents
            Required.  Must look like
            ``{"inputs": {"acc": {"id": ..., "version": ...},
                          "feed": {"id": ..., "version": ...}},
               "upstream": {}}``.
        label
            If omitted, falls back to ``metadata.feed_name``.
        result, F_src, stage_config, stage_summary
            All optional sidecar / artefact extras.

        ``save()`` is not called automatically — flush after batch
        registrations with :meth:`save`.
        """
        if "inputs" not in parents:
            raise ValueError("parents must contain an 'inputs' block")

        run_id     = self._new_run_id()
        created_at = _now_iso()
        if label is None:
            label = getattr(metadata, "feed_name", "") or ""

        run_dir = self._make_run_dir(run_id)
        Path(os.path.join(run_dir, "diag")).mkdir(parents=True, exist_ok=True)

        if result is not None:
            metadata.fill_computed(result)

        feed.save(path=os.path.join(run_dir, "feed.pkl"))

        if result is not None:
            np.savez(
                os.path.join(run_dir, "result.npz"),
                F_root     = result.F_root,
                sigma_root = result.sigma_root,
                cv_root    = result.cv_root,
                cv_src     = np.array([result.cv_src]),
            )

        if F_src is not None:
            np.save(os.path.join(run_dir, "F_src.npy"),
                    np.asarray(F_src, dtype=float))

        sidecar = build_sidecar(
            run_id        = run_id,
            stage         = self.STAGE,
            label         = label,
            created_at    = created_at,
            parents       = parents,
            stage_config  = stage_config,
            stage_summary = stage_summary,
            extra         = {"feed_metadata": metadata.to_dict()},
        )
        self._write_sidecar(run_dir, sidecar)

        entry = self._add_entry(run_id=run_id, label=label, created_at=created_at)

        feed_ref = parents.get("inputs", {}).get("feed")
        if feed_ref and feed_ref.get("id"):
            self._input_feed_idx.setdefault(feed_ref["id"], []).append(run_id)

        return entry

    # ── lookup ────────────────────────────────────────────────────────

    def latest_by_input_feed_id(self, input_feed_id: str) -> _RegistryEntry:
        """Return the most recently registered run whose
        ``parents.inputs.feed.id`` equals *input_feed_id*."""
        run_ids = self._input_feed_idx.get(input_feed_id, [])
        if not run_ids:
            raise KeyError(f"No registered run for input feed_id '{input_feed_id}'")
        return self._index[run_ids[-1]]

    def input_feed_ids(self) -> List[str]:
        """Distinct input-feed-ids referenced by registered runs."""
        return sorted(self._input_feed_idx.keys())

    def __contains__(self, key: str) -> bool:
        if key in self._index:
            return True
        return key in self._input_feed_idx

    def resolve(self, key: str) -> _RegistryEntry:
        """Resolve *key* (run-id, or input-feed-id → latest run)."""
        if key in self._index:
            return self._index[key]
        return self.latest_by_input_feed_id(key)

    # ── lazy loaders ──────────────────────────────────────────────────

    def load_metadata(self, key: str) -> DataFeedMetadata:
        entry = self.resolve(key)
        sidecar = self._read_sidecar(entry.run_id)
        feed_meta_dict = sidecar.get("feed_metadata")
        if feed_meta_dict is None:
            raise KeyError(
                f"sidecar for {entry.run_id} has no 'feed_metadata' block"
            )
        return DataFeedMetadata.from_dict(feed_meta_dict)

    def load_sidecar(self, key: str) -> dict:
        entry = self.resolve(key)
        return self._read_sidecar(entry.run_id)

    def load_result(self, key: str) -> Dict[str, np.ndarray]:
        entry = self.resolve(key)
        npz_path = os.path.join(entry.run_dir, "result.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"No result.npz found for feed run '{entry.run_id}'. "
                "Was a PropagationResult passed to register()?"
            )
        z = np.load(npz_path)
        return {k: z[k] for k in z.files}

    def load_feed(self, key: str, feed_cls: Any, acc_struct: Any) -> Any:
        entry = self.resolve(key)
        pkl = os.path.join(entry.run_dir, "feed.pkl")
        sidecar = self._read_sidecar(entry.run_id)
        feed_id = (sidecar.get("parents", {})
                          .get("inputs", {})
                          .get("feed", {})
                          .get("id"))
        return feed_cls.load(feed_id, acc_struct, path=pkl)

    # ── summary ───────────────────────────────────────────────────────

    def summary_df(self) -> pd.DataFrame:
        """One row per registered run, joining index + sidecar fields."""
        if not self._index:
            return pd.DataFrame(columns=["run_id"]).set_index("run_id")

        rows = []
        for run_id, entry in self._index.items():
            row: dict[str, Any] = entry.to_index_dict()
            try:
                sidecar = self._read_sidecar(run_id)
                meta = DataFeedMetadata.from_dict(sidecar.get("feed_metadata", {}))
                row.update({
                    "input_feed_id":   (sidecar.get("parents", {})
                                                .get("inputs", {})
                                                .get("feed", {})
                                                .get("id")),
                    "input_acc_id":    (sidecar.get("parents", {})
                                                .get("inputs", {})
                                                .get("acc", {})
                                                .get("id")),
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
                warnings.warn(f"Could not load sidecar for '{run_id}': {exc}")
            rows.append(row)
        return pd.DataFrame(rows).set_index("run_id")


# =============================================================================
# ImputationRunRegistry — Stage 2
# =============================================================================

class ImputationRunRegistry(BaseOutputRegistry):
    """Stage-2 registry.

    Per-run directory layout::

        data/proc/imp/
        ├── registry.json
        └── imp_<ts>_<code>/
            ├── metadata.json          ← harmonized sidecar
            ├── agg_result.npz         ← f_agg, sigma_agg, Omega, F, Σ, coverage, feed_ids
            └── imputation_log.parquet
    """

    STAGE = "imp"

    def __init__(self, storage_dir: str = "data/proc/imp") -> None:
        super().__init__(storage_dir=storage_dir)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.load_index()

    # ── register ──────────────────────────────────────────────────────

    def register(
        self,
        agg_result,
        *,
        label:         str,
        parents:       dict,
        imputers:      Optional[List[Any]] = None,
        stage_config:  Optional[dict] = None,
        stage_summary: Optional[dict] = None,
    ) -> str:
        """Persist a Stage-2 imputation run and return its run-id.

        Parameters
        ----------
        agg_result
            AggregationResult after ``ImputationEngine.run()`` — must
            carry an ``imputation_log``.
        parents
            Harmonized parents block.  Expects
            ``parents.inputs = {"acc": {...}, "params": [...]}`` and
            ``parents.upstream = {"feed_run_ids": [...]}``.
        imputers
            Optional list of imputer instances; their ``.config()`` and
            the per-step metadata are folded into the sidecar.

        Returns
        -------
        str
            ``run_id``.  Also stamped onto ``agg_result._imp_run_id`` so
            downstream constraint builds can auto-resolve the parent.
        """
        if agg_result.imputation_log is None:
            raise ValueError(
                "AggregationResult.imputation_log is None — register only "
                "after ImputationEngine.run()."
            )
        if "inputs" not in parents:
            raise ValueError("parents must contain an 'inputs' block")

        run_id     = self._new_run_id()
        created_at = _now_iso()
        run_dir    = self._make_run_dir(run_id)

        # 1. arrays
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

        # 2. imputation log
        log_df = agg_result.imputation_log.as_dataframe()
        log_df.to_parquet(
            os.path.join(run_dir, "imputation_log.parquet"),
            index=False,
        )

        # 3. assemble per-imputer step records (matches old metadata.json shape)
        meta_records: list[dict] = []
        if agg_result.imputation_metadata is not None:
            meta_records = agg_result.imputation_metadata.as_dataframe().to_dict("records")
        if imputers:
            for rec in meta_records:
                step = int(rec.get("step", -1))
                if 0 <= step < len(imputers):
                    imp = imputers[step]
                    rec["config"] = imp.config() if hasattr(imp, "config") else {}

        # 4. summary numbers
        n_total   = int(agg_result.f_agg.shape[0])
        n_imputed = int(len(log_df))
        n_after   = int(np.isfinite(agg_result.f_agg).sum())
        n_before  = int(agg_result.coverage.sum())

        summary = (stage_summary or {}) | {
            "n_total":          n_total,
            "n_covered_before": n_before,
            "n_covered_after":  n_after,
            "n_imputed":        n_imputed,
            "n_steps":          len(meta_records),
        }

        # 5. sidecar
        sidecar = build_sidecar(
            run_id        = run_id,
            stage         = self.STAGE,
            label         = label,
            created_at    = created_at,
            parents       = parents,
            stage_config  = stage_config,
            stage_summary = summary,
            extra         = {"imputers": meta_records},
        )
        self._write_sidecar(run_dir, sidecar)

        # 6. index + save (imp registry auto-saves on register, like before)
        self._add_entry(run_id=run_id, label=label, created_at=created_at)
        self.save()

        # 7. stamp run_id back for downstream provenance
        agg_result._imp_run_id = run_id

        return run_id

    # ── loaders ───────────────────────────────────────────────────────

    def load(self, run_id: str):
        """Reconstruct an ``AggregationResult`` with attached log."""
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

    def load_sidecar(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"Imputation run '{run_id}' not in registry")
        return self._read_sidecar(run_id)

    def ids(self) -> List[str]:
        return list(self._index.keys())

    def summary_df(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame()
        rows = []
        for run_id, entry in self._index.items():
            row = entry.to_index_dict()
            try:
                sidecar = self._read_sidecar(run_id)
                summary = sidecar.get("stage_summary", {})
                row.update({
                    "n_total":          summary.get("n_total"),
                    "n_imputed":        summary.get("n_imputed"),
                    "n_covered_after":  summary.get("n_covered_after"),
                    "feed_run_ids":     (sidecar.get("parents", {})
                                                 .get("upstream", {})
                                                 .get("feed_run_ids", [])),
                })
            except Exception as exc:
                warnings.warn(f"Could not load sidecar for '{run_id}': {exc}")
            rows.append(row)
        return pd.DataFrame(rows)
