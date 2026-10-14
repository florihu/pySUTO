"""
feed.py
=======
``DataFeedRegistry`` — Stage-1 output registry.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pysuto.meta import DataFeedMetadata

from ._sidecar import (
    _RegistryEntry,
    _now_iso,
    build_sidecar,
    make_run_id,
)
from .base import BaseOutputRegistry


class DataFeedRegistry(BaseOutputRegistry):
    """Stage-1 registry.

    Per-run directory layout::

        data/proc/feed/registry/
        ├── registry.json
        └── feed_<id>_<ts>_<code>/
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

        _fid = (parents.get("inputs", {}).get("feed", {}).get("id") or
                getattr(metadata, "feed_name", "") or "")
        run_id = make_run_id(f"feed_{_fid}") if _fid else self._new_run_id()
        created_at = _now_iso()
        if label is None:
            label = getattr(metadata, "feed_name", "") or ""

        run_dir = self._make_run_dir(run_id)
        Path(os.path.join(run_dir, "diag")).mkdir(parents=True, exist_ok=True)

        if result is not None:
            metadata.fill_computed(result)

        feed.save(path=os.path.join(run_dir, "feed.pkl"))

        if result is not None:
            arrays = dict(
                F_root     = result.F_root,
                sigma_root = result.sigma_root,
                cv_root    = result.cv_root,
                cv_src     = np.array([result.cv_src]),
            )
            row_pen = getattr(result, "row_penalized", None)
            if row_pen is not None:
                arrays["row_penalized"] = np.asarray(row_pen, dtype=bool)
            obs_zero = getattr(result, "observed_zero", None)
            if obs_zero is not None:
                arrays["observed_zero"] = np.asarray(obs_zero, dtype=bool)
            np.savez(os.path.join(run_dir, "result.npz"), **arrays)

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
