"""
test_registry.py
================
Tests for harmonized output registries in ``pysuto.registry``:

* run-id format helpers (``make_run_id``, ``_ts_compact``, ``_short_code``)
* ``_RegistryEntry`` (trimmed 4-field on-disk, enriched on load with ``run_dir``)
* ``build_sidecar`` schema
* ``DataFeedRegistry`` (Stage 1) — register, sidecar shape, container protocol,
  reverse-index by input-feed-id, lazy loaders, summary_df
* ``ImputationRunRegistry`` (Stage 2) — register, sidecar parents.upstream,
  load round-trip

Stubs replace heavy objects (WeightMatrix, PropagationResult,
AggregationResult).
"""

from __future__ import annotations

import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto.registry import (
    _NpEncoder,
    _RegistryEntry,
    _json_dump,
    _json_load,
    build_sidecar,
    make_run_id,
    DataFeedMetadata,
    DataFeedRegistry,
    ImputationRunRegistry,
    ConstraintRunRegistry,
    OptimRunRegistry,
)


_RUN_ID_RE = re.compile(r"^(feed|imp|const|opt)_\d{8}_\d{6}_[a-f0-9]{8}$")


# =============================================================================
# Stubs
# =============================================================================


class _Feed:
    """Minimal duck-typed WeightMatrix stub."""

    def __init__(self, feed_id: str):
        self.id = feed_id
        self.save_calls: list[str] = []

    def save(self, **kwargs) -> None:
        path = kwargs["path"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"id": self.id}, f)
        self.save_calls.append(path)


class _Result:
    def __init__(
        self,
        F_root: np.ndarray | None = None,
        sigma_root: np.ndarray | None = None,
        cv_root: np.ndarray | None = None,
        cv_src: float = 0.022644,
    ):
        self.F_root     = F_root     if F_root     is not None else np.array([100.0, 200.0, 50.0])
        self.sigma_root = sigma_root if sigma_root is not None else np.array([2.0, 4.0, 1.0])
        self.cv_root    = cv_root    if cv_root    is not None else np.array([0.02, 0.02, 0.02])
        self.cv_src     = cv_src


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parents_feed():
    """Harmonized parents block for a Stage-1 run."""
    return {
        "inputs": {
            "acc":  {"id": "copper", "version": 1},
            "feed": {"id": "icsg",   "version": 1},
        },
        "upstream": {},
    }


@pytest.fixture
def parents_feed_bgs():
    return {
        "inputs": {
            "acc":  {"id": "copper", "version": 1},
            "feed": {"id": "bgs",    "version": 1},
        },
        "upstream": {},
    }


@pytest.fixture
def feed():
    return _Feed("icsg")


@pytest.fixture
def feed2():
    return _Feed("bgs")


@pytest.fixture
def metadata():
    return DataFeedMetadata(
        feed_name="icsg",
        iedc_type="1_F_",
        reliability=1,
        completeness=1,
        temporal_correlation=1,
        geographical_correlation=1,
        other_correlation=1,
        person="Florian_Huber",
        dataset_link="https://example.com/icsg",
    )


@pytest.fixture
def metadata2():
    return DataFeedMetadata(
        feed_name="bgs",
        iedc_type="2_F_",
        person="Jane_Doe",
    )


@pytest.fixture
def result():
    return _Result()


@pytest.fixture
def registry(tmp_path):
    return DataFeedRegistry(odir=str(tmp_path))


@pytest.fixture
def populated_registry(tmp_path, feed, metadata, result, parents_feed):
    reg = DataFeedRegistry(odir=str(tmp_path))
    reg.register(feed, metadata, parents=parents_feed, result=result)
    return reg


# =============================================================================
# JSON helpers (carried forward unchanged)
# =============================================================================


class TestNpEncoder:

    def test_numpy_int(self):
        assert json.loads(json.dumps(np.int64(42), cls=_NpEncoder)) == 42

    def test_numpy_float(self):
        out = json.loads(json.dumps(np.float64(3.14), cls=_NpEncoder))
        assert out == pytest.approx(3.14)

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0])
        assert json.loads(json.dumps(arr, cls=_NpEncoder)) == [1.0, 2.0]

    def test_datetime(self):
        dt = datetime(2025, 6, 11, 12, 0, 0)
        assert "2025-06-11" in json.dumps(dt, cls=_NpEncoder)


class TestJsonHelpers:

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "x.json")
        payload = {"a": 1, "b": [1.0, 2.0], "c": np.array([3, 4])}
        _json_dump(payload, path)
        loaded = _json_load(path)
        assert loaded["a"] == 1
        assert loaded["b"] == [1.0, 2.0]
        assert loaded["c"] == [3, 4]

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "x.json")
        _json_dump({"x": 1}, path)
        assert os.path.isfile(path)


# =============================================================================
# Run-id helpers
# =============================================================================


class TestRunIdFormat:

    @pytest.mark.parametrize("stage", ["feed", "imp", "const", "opt"])
    def test_run_id_matches_pattern(self, stage):
        run_id = make_run_id(stage)
        assert _RUN_ID_RE.match(run_id), f"unexpected format: {run_id}"
        assert run_id.startswith(f"{stage}_")

    def test_two_calls_disambiguate(self):
        # Same second → uuid suffix must differentiate.
        ids = {make_run_id("feed") for _ in range(50)}
        assert len(ids) == 50


# =============================================================================
# _RegistryEntry
# =============================================================================


class TestRegistryEntry:

    def test_index_dict_has_four_fields(self):
        entry = _RegistryEntry(
            run_id="feed_20260518_143022_a4f1c3d2",
            stage="feed",
            label="icsg",
            created_at="2026-05-18T14:30:22Z",
            run_dir="/some/dir",
        )
        d = entry.to_index_dict()
        assert set(d.keys()) == {"run_id", "stage", "label", "created_at"}
        assert "run_dir" not in d        # run_dir is in-memory only

    def test_from_index_dict_fills_run_dir(self):
        d = {
            "run_id":     "feed_20260518_143022_a4f1c3d2",
            "stage":      "feed",
            "label":      "icsg",
            "created_at": "2026-05-18T14:30:22Z",
        }
        entry = _RegistryEntry.from_index_dict(d, storage_dir="/data/proc/feed/registry")
        assert entry.run_dir.endswith("feed_20260518_143022_a4f1c3d2")


# =============================================================================
# build_sidecar
# =============================================================================


class TestSidecarBuilder:

    def test_schema_shape(self):
        sc = build_sidecar(
            run_id     = "imp_20260518_143055_b7e2f3a1",
            stage      = "imp",
            label      = "copper baseline",
            created_at = "2026-05-18T14:30:55Z",
            parents    = {
                "inputs":   {"acc": {"id": "copper", "version": 2}},
                "upstream": {"feed_run_ids": ["feed_..."]},
            },
            stage_config  = {"p": 1.3},
            stage_summary = {"n_total": 100},
        )
        assert sc["run_id"] == "imp_20260518_143055_b7e2f3a1"
        assert sc["stage"]  == "imp"
        assert sc["label"]  == "copper baseline"
        assert sc["parents"]["inputs"]["acc"]["id"] == "copper"
        assert sc["parents"]["upstream"]["feed_run_ids"] == ["feed_..."]
        assert sc["stage_config"] == {"p": 1.3}
        assert sc["stage_summary"] == {"n_total": 100}

    def test_empty_parents_blocks_default_to_dict(self):
        sc = build_sidecar(
            run_id="opt_x", stage="opt", label="", created_at="t",
            parents={},
        )
        assert sc["parents"]["inputs"]   == {}
        assert sc["parents"]["upstream"] == {}

    def test_extra_block_appended(self):
        sc = build_sidecar(
            run_id="feed_x", stage="feed", label="", created_at="t",
            parents={}, extra={"feed_metadata": {"feed_name": "icsg"}},
        )
        assert sc["feed_metadata"]["feed_name"] == "icsg"


# =============================================================================
# DataFeedRegistry — construction
# =============================================================================


class TestRegistryConstruction:

    def test_odir_stored(self, tmp_path):
        reg = DataFeedRegistry(odir=str(tmp_path))
        assert reg.odir == str(tmp_path)
        assert reg.storage_dir == str(tmp_path)

    def test_empty_on_init(self, registry):
        assert len(registry) == 0

    def test_run_ids_empty_on_init(self, registry):
        assert registry.run_ids() == []

    def test_registry_filename_constant(self):
        assert DataFeedRegistry.REGISTRY_FILENAME == "registry.json"


# =============================================================================
# DataFeedRegistry — register
# =============================================================================


class TestRegistryRegister:

    def test_returns_registry_entry(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert isinstance(entry, _RegistryEntry)

    def test_run_id_format(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        # DataFeedRegistry embeds the feed_id: feed_<id>_<ts>_<code>
        _feed_run_id_re = re.compile(r"^feed_\w+_\d{8}_\d{6}_[a-f0-9]{8}$")
        assert _feed_run_id_re.match(entry.run_id), f"unexpected format: {entry.run_id}"
        assert entry.stage == "feed"

    def test_label_defaults_to_feed_name(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert entry.label == "icsg"

    def test_explicit_label_overrides_meta(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(
            feed, metadata, parents=parents_feed, result=result, label="explicit-label",
        )
        assert entry.label == "explicit-label"

    def test_created_at_ends_with_z(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert entry.created_at.endswith("Z")

    def test_run_dir_is_under_storage(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert entry.run_dir == os.path.join(registry.odir, entry.run_id)
        assert os.path.isdir(entry.run_dir)

    def test_sidecar_written(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert os.path.isfile(os.path.join(entry.run_dir, "metadata.json"))

    def test_sidecar_contents(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        sc = _json_load(os.path.join(entry.run_dir, "metadata.json"))
        assert sc["run_id"] == entry.run_id
        assert sc["stage"]  == "feed"
        assert sc["label"]  == "icsg"
        assert sc["parents"]["inputs"]["feed"]["id"] == "icsg"
        assert sc["parents"]["inputs"]["acc"]["id"]  == "copper"
        assert sc["feed_metadata"]["feed_name"] == "icsg"
        assert sc["feed_metadata"]["cv_total"] == pytest.approx(result.cv_src)

    def test_pickle_written(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert os.path.isfile(os.path.join(entry.run_dir, "feed.pkl"))

    def test_feed_save_called_with_correct_path(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        expected = os.path.join(entry.run_dir, "feed.pkl")
        assert expected in feed.save_calls

    def test_result_npz_written(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert os.path.isfile(os.path.join(entry.run_dir, "result.npz"))

    def test_no_result_no_npz(self, registry, feed, metadata, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=None)
        assert not os.path.isfile(os.path.join(entry.run_dir, "result.npz"))

    def test_no_result_no_fill_computed(self, registry, feed, metadata, parents_feed):
        registry.register(feed, metadata, parents=parents_feed, result=None)
        assert metadata.cv_total is None

    def test_added_to_index_by_run_id(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert entry.run_id in registry

    def test_input_feed_id_resolves(self, registry, feed, metadata, result, parents_feed):
        registry.register(feed, metadata, parents=parents_feed, result=result)
        assert "icsg" in registry            # via reverse index
        latest = registry.latest_by_input_feed_id("icsg")
        assert latest.stage == "feed"
        assert latest.label == "icsg"

    def test_repeat_register_creates_new_run(self, registry, feed, metadata, result, parents_feed):
        e1 = registry.register(feed, metadata, parents=parents_feed, result=result)
        e2 = registry.register(feed, metadata, parents=parents_feed, result=result)
        assert e1.run_id != e2.run_id
        assert len(registry) == 2

    def test_latest_by_input_feed_id_returns_newest(self, registry, feed, metadata, result, parents_feed):
        e1 = registry.register(feed, metadata, parents=parents_feed, result=result)
        e2 = registry.register(feed, metadata, parents=parents_feed, result=result)
        latest = registry.latest_by_input_feed_id("icsg")
        assert latest.run_id in (e1.run_id, e2.run_id)

    def test_parents_required(self, registry, feed, metadata, result):
        with pytest.raises(ValueError, match="parents"):
            registry.register(feed, metadata, parents={}, result=result)

    def test_created_at_is_iso_parseable(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        datetime.fromisoformat(entry.created_at.rstrip("Z"))

    def test_result_arrays_values(self, registry, feed, metadata, result, parents_feed):
        entry = registry.register(feed, metadata, parents=parents_feed, result=result)
        z = np.load(os.path.join(entry.run_dir, "result.npz"))
        np.testing.assert_allclose(z["F_root"],     result.F_root)
        np.testing.assert_allclose(z["sigma_root"], result.sigma_root)
        assert float(z["cv_src"][0]) == pytest.approx(result.cv_src)


# =============================================================================
# DataFeedRegistry — save / load_index round-trip
# =============================================================================


class TestRegistrySaveLoad:

    def test_save_creates_registry_json(self, populated_registry):
        populated_registry.save()
        path = os.path.join(populated_registry.odir, "registry.json")
        assert os.path.isfile(path)

    def test_registry_json_has_only_four_fields(self, populated_registry):
        populated_registry.save()
        path = os.path.join(populated_registry.odir, "registry.json")
        with open(path) as f:
            payload = json.load(f)
        for run_id, entry in payload.items():
            assert set(entry.keys()) == {"run_id", "stage", "label", "created_at"}

    def test_save_creates_odir_if_absent(self, tmp_path):
        odir = str(tmp_path / "new" / "dir")
        reg = DataFeedRegistry(odir=odir)
        reg.save()
        assert os.path.isdir(odir)

    def test_load_index_no_op_on_missing_file(self, tmp_path):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.load_index()
        assert len(reg) == 0

    def test_round_trip_len(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        assert len(reg2) == 1

    def test_round_trip_input_feed_id_lookup(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        # Reverse index must be rebuilt by reading sidecars.
        assert "icsg" in reg2
        e2 = reg2.latest_by_input_feed_id("icsg")
        e1 = populated_registry.latest_by_input_feed_id("icsg")
        assert e2.run_id == e1.run_id

    def test_round_trip_entry_fields(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        run_id = populated_registry.run_ids()[0]
        e1 = populated_registry[run_id]
        e2 = reg2[run_id]
        assert e2.run_id     == e1.run_id
        assert e2.label      == e1.label
        assert e2.stage      == e1.stage
        assert e2.created_at == e1.created_at

    def test_multiple_feeds_round_trip(
        self, tmp_path, feed, feed2, metadata, metadata2, result,
        parents_feed, parents_feed_bgs,
    ):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed,  metadata,  parents=parents_feed,     result=result)
        reg.register(feed2, metadata2, parents=parents_feed_bgs, result=result)
        reg.save()

        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        assert len(reg2) == 2
        assert set(reg2.input_feed_ids()) == {"icsg", "bgs"}


# =============================================================================
# DataFeedRegistry — lazy loaders
# =============================================================================


class TestRegistryLoaders:

    def test_load_sidecar_returns_dict(self, populated_registry):
        sc = populated_registry.load_sidecar("icsg")
        assert sc["stage"] == "feed"
        assert sc["parents"]["inputs"]["feed"]["id"] == "icsg"

    def test_load_metadata_returns_data_feed_metadata(self, populated_registry):
        m = populated_registry.load_metadata("icsg")
        assert isinstance(m, DataFeedMetadata)
        assert m.feed_name == "icsg"

    def test_load_metadata_cv_total(self, populated_registry, result):
        m = populated_registry.load_metadata("icsg")
        assert m.cv_total == pytest.approx(result.cv_src)

    def test_load_metadata_missing_id_raises(self, populated_registry):
        with pytest.raises(KeyError):
            populated_registry.load_metadata("nonexistent")

    def test_load_result_keys(self, populated_registry):
        d = populated_registry.load_result("icsg")
        assert set(d.keys()) == {"F_root", "sigma_root", "cv_root", "cv_src"}

    def test_load_result_values(self, populated_registry, result):
        d = populated_registry.load_result("icsg")
        np.testing.assert_allclose(d["F_root"],     result.F_root)
        np.testing.assert_allclose(d["sigma_root"], result.sigma_root)

    def test_load_result_missing_npz_raises_file_not_found(
        self, tmp_path, feed, metadata, parents_feed,
    ):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed, metadata, parents=parents_feed, result=None)
        with pytest.raises(FileNotFoundError):
            reg.load_result("icsg")


# =============================================================================
# DataFeedRegistry — container protocol
# =============================================================================


class TestRegistryContainerProtocol:

    def test_contains_run_id(self, populated_registry):
        run_id = populated_registry.run_ids()[0]
        assert run_id in populated_registry

    def test_contains_input_feed_id(self, populated_registry):
        assert "icsg" in populated_registry

    def test_not_contains_unregistered(self, populated_registry):
        assert "nope" not in populated_registry

    def test_len(self, populated_registry):
        assert len(populated_registry) == 1

    def test_getitem_returns_entry(self, populated_registry):
        run_id = populated_registry.run_ids()[0]
        assert isinstance(populated_registry[run_id], _RegistryEntry)

    def test_getitem_missing_raises_key_error(self, populated_registry):
        with pytest.raises(KeyError):
            _ = populated_registry["nonexistent_run"]

    def test_iter_yields_run_ids(
        self, tmp_path, feed, feed2, metadata, metadata2, result,
        parents_feed, parents_feed_bgs,
    ):
        reg = DataFeedRegistry(odir=str(tmp_path))
        e1 = reg.register(feed,  metadata,  parents=parents_feed,     result=result)
        e2 = reg.register(feed2, metadata2, parents=parents_feed_bgs, result=result)
        assert set(reg) == {e1.run_id, e2.run_id}


# =============================================================================
# DataFeedRegistry — summary_df
# =============================================================================


class TestRegistrySummaryDf:

    def test_returns_dataframe(self, populated_registry):
        df = populated_registry.summary_df()
        assert isinstance(df, pd.DataFrame)

    def test_index_is_run_id(self, populated_registry):
        df = populated_registry.summary_df()
        assert df.index.name == "run_id"

    def test_has_one_row(self, populated_registry):
        df = populated_registry.summary_df()
        assert len(df) == 1

    def test_input_feed_id_column(self, populated_registry):
        df = populated_registry.summary_df()
        assert (df["input_feed_id"] == "icsg").any()

    def test_cv_total_percent_value(self, populated_registry, result):
        df = populated_registry.summary_df()
        expected = round(result.cv_src * 100, 3)
        row = df[df["input_feed_id"] == "icsg"].iloc[0]
        assert row["cv_total_%"] == pytest.approx(expected)

    def test_empty_registry_returns_empty_df(self, registry):
        df = registry.summary_df()
        assert len(df) == 0


# =============================================================================
# ImputationRunRegistry
# =============================================================================


class _ImpLog:
    def __init__(self, n=0):
        self._n = n

    def as_dataframe(self):
        return pd.DataFrame({"edge_id": range(self._n), "step": [0] * self._n})


class _AggResult:
    """Minimal duck-typed AggregationResult stub."""

    def __init__(self, n_edges=4, n_feeds=2, n_imputed=2):
        self.a_agg     = np.array([1.0, 2.0, np.nan, 4.0][:n_edges])
        self.sigma_agg = np.array([0.1, 0.2, np.nan, 0.4][:n_edges])
        self.Omega     = np.ones((n_feeds, n_edges))
        self.F         = np.ones((n_feeds, n_edges))
        self.Sigma     = np.ones((n_feeds, n_edges))
        self.coverage  = np.array([True, True, False, True][:n_edges])
        self.feed_ids  = [f"feed_{i}" for i in range(n_feeds)]
        self.imputation_log      = _ImpLog(n=n_imputed)
        self.imputation_metadata = None
        self._imp_run_id         = None


class TestImputationRunRegistry:

    @pytest.fixture
    def imp_registry(self, tmp_path):
        return ImputationRunRegistry(storage_dir=str(tmp_path))

    @pytest.fixture
    def imp_parents(self):
        return {
            "inputs": {
                "acc":    {"id": "copper", "version": 1},
                "params": [
                    {"id": "conc", "version": 1},
                    {"id": "eff",  "version": 1},
                ],
            },
            "upstream": {
                "feed_run_ids": [
                    "feed_20260518_141055_aaaaaaaa",
                    "feed_20260518_141056_bbbbbbbb",
                ],
            },
        }

    def test_register_returns_run_id(self, imp_registry, imp_parents):
        agg = _AggResult()
        run_id = imp_registry.register(agg, label="copper baseline", parents=imp_parents)
        assert _RUN_ID_RE.match(run_id)
        assert run_id.startswith("imp_")

    def test_sidecar_written(self, imp_registry, imp_parents):
        agg = _AggResult()
        run_id = imp_registry.register(agg, label="copper", parents=imp_parents)
        run_dir = os.path.join(imp_registry.storage_dir, run_id)
        assert os.path.isfile(os.path.join(run_dir, "metadata.json"))

    def test_sidecar_parents_upstream(self, imp_registry, imp_parents):
        agg = _AggResult()
        run_id = imp_registry.register(agg, label="copper", parents=imp_parents)
        sc = imp_registry.load_sidecar(run_id)
        assert sc["parents"]["upstream"]["feed_run_ids"] \
            == imp_parents["upstream"]["feed_run_ids"]
        assert sc["parents"]["inputs"]["acc"]["id"] == "copper"
        assert {p["id"] for p in sc["parents"]["inputs"]["params"]} == {"conc", "eff"}

    def test_sidecar_stage_summary(self, imp_registry, imp_parents):
        agg = _AggResult(n_edges=4, n_imputed=2)
        run_id = imp_registry.register(agg, label="copper", parents=imp_parents)
        sc = imp_registry.load_sidecar(run_id)
        assert sc["stage_summary"]["n_total"]    == 4
        assert sc["stage_summary"]["n_imputed"]  == 2

    def test_run_id_stamped_on_agg_result(self, imp_registry, imp_parents):
        agg = _AggResult()
        run_id = imp_registry.register(agg, label="copper", parents=imp_parents)
        assert agg._imp_run_id == run_id

    def test_register_requires_imputation_log(self, imp_registry, imp_parents):
        agg = _AggResult()
        agg.imputation_log = None
        with pytest.raises(ValueError, match="imputation_log"):
            imp_registry.register(agg, label="x", parents=imp_parents)

    def test_register_requires_parents_inputs(self, imp_registry):
        agg = _AggResult()
        with pytest.raises(ValueError, match="parents"):
            imp_registry.register(agg, label="x", parents={})

    def test_index_has_only_four_fields(self, imp_registry, imp_parents):
        agg = _AggResult()
        imp_registry.register(agg, label="copper", parents=imp_parents)
        with open(os.path.join(imp_registry.storage_dir, "registry.json")) as f:
            payload = json.load(f)
        for run_id, entry in payload.items():
            assert set(entry.keys()) == {"run_id", "stage", "label", "created_at"}

    def test_load_round_trip(self, imp_registry, imp_parents):
        agg = _AggResult(n_edges=4, n_feeds=2)
        run_id = imp_registry.register(agg, label="copper", parents=imp_parents)

        # Fresh registry instance — must reload from disk.
        reg2 = ImputationRunRegistry(storage_dir=imp_registry.storage_dir)
        loaded = reg2.load(run_id)
        np.testing.assert_allclose(
            np.nan_to_num(loaded.a_agg), np.nan_to_num(agg.a_agg)
        )
        assert loaded._imp_run_id == run_id

    def test_load_missing_raises(self, imp_registry):
        with pytest.raises(KeyError):
            imp_registry.load("imp_nonexistent")


# =============================================================================
# Stage-3: ConstraintRunRegistry  (direct API; no ConstraintEngine fixtures)
# =============================================================================


def _toy_constraint_run(run_id: str = "const_20260519_000000_aaaa1111"):
    """Hand-rolled ConstraintRun with the smallest structural payload:
    2 constraints, 2 edges, 1 timestep."""
    from pysuto.constraints import ConstraintRun, StructG
    # Sparsity pattern: row 0 touches edges {0, 1}; row 1 touches edge 1.
    indptr  = np.array([0, 2, 3], dtype=np.int32)
    indices = np.array([0, 1, 1], dtype=np.int32)
    data    = np.array([[1.0], [-1.0], [1.0]])   # (nnz=3, n_timesteps=1)
    row_meta = pd.DataFrame({
        "constraint_id": ["c0", "c1"],
        "type":          ["mass_balance", "ratio"],
        "is_hard":       [True, False],
        "cone_type":     ["zero", "free"],
    })
    G = StructG(
        indptr   = indptr,
        indices  = indices,
        data     = data,
        shape    = (2, 2),
        row_meta = row_meta,
    )
    return ConstraintRun(
        G          = G,
        row_meta   = row_meta,
        a0         = np.array([10.0, 20.0]),
        sigma_a    = np.array([[1.0], [2.0]]),    # (n_edges, n_timesteps)
        u          = np.array([1e9, 1e9]),
        l          = np.array([0.0, 0.0]),
        c          = np.array([[0.0], [5.0]]),     # (n_constraints, n_timesteps)
        sigma_c    = np.array([[1.0], [1.0]]),
        active     = np.array([[True], [True]]),
        timesteps  = ["2020"],
        run_id     = run_id,
        acc_id     = "toy",
        agg_id     = "imp_20260519_000000_bbbb2222",
        created_at = "2026-05-19T00:00:00Z",
        label      = "toy-const",
        _p         = 1.3,
    )


class TestConstraintRunRegistry:

    @pytest.fixture
    def const_registry(self, tmp_path):
        return ConstraintRunRegistry(storage_dir=str(tmp_path / "const"))

    def test_run_id_prefix(self, const_registry):
        assert const_registry.STAGE == "const"

    def test_round_trip(self, const_registry):
        run = _toy_constraint_run()
        const_registry.register(run)
        reg2 = ConstraintRunRegistry(storage_dir=const_registry.storage_dir)
        loaded = reg2.load(run.run_id)

        # 1-D / 2-D round-trip equality.
        np.testing.assert_allclose(loaded.a0,      run.a0)
        np.testing.assert_allclose(loaded.sigma_a, run.sigma_a)
        np.testing.assert_allclose(loaded.u,       run.u)
        np.testing.assert_allclose(loaded.l,       run.l)
        np.testing.assert_allclose(loaded.c,       run.c)
        np.testing.assert_allclose(loaded.sigma_c, run.sigma_c)
        np.testing.assert_array_equal(loaded.active, run.active)

        # StructG round-trips with the same sparsity + data tape.
        np.testing.assert_array_equal(loaded.G.indptr,  run.G.indptr)
        np.testing.assert_array_equal(loaded.G.indices, run.G.indices)
        np.testing.assert_allclose(loaded.G.data,       run.G.data)
        assert loaded.G.shape == run.G.shape

        assert loaded.timesteps == run.timesteps
        assert loaded.acc_id == "toy"
        assert loaded.agg_id == run.agg_id
        assert loaded._p     == pytest.approx(1.3)

    def test_index_has_only_four_fields(self, const_registry):
        const_registry.register(_toy_constraint_run())
        with open(os.path.join(const_registry.storage_dir, "registry.json")) as f:
            payload = json.load(f)
        for entry in payload.values():
            assert set(entry.keys()) == {"run_id", "stage", "label", "created_at"}

    def test_sidecar_parents_default(self, const_registry):
        run = _toy_constraint_run()
        const_registry.register(run)
        sc = const_registry.load_sidecar(run.run_id)
        assert sc["parents"]["inputs"]["acc"]["id"] == "toy"
        assert sc["parents"]["upstream"]["imp_run_id"] == run.agg_id

    def test_load_missing_raises(self, const_registry):
        with pytest.raises(KeyError):
            const_registry.load("const_nonexistent")


# =============================================================================
# Stage-4: OptimRunRegistry
# =============================================================================


def _toy_optim_results():
    """Build a 2-timestep dict[str, OptimResult] with no iter_history."""
    from pysuto.solver import OptimResult
    out = {}
    for t, obj in [("2020", 1.23), ("2021", 4.56)]:
        out[t] = OptimResult(
            timestep      = t,
            F_opt         = np.array([1.0, 2.0]),
            eps_opt       = np.array([0.0]),
            c_star        = np.array([0.0, 5.0]),
            residuals     = np.array([0.0, 0.0]),
            objective     = obj,
            iterations    = 5,
            solve_time_s  = 0.01,
            solver_status = "Solved",
            convergence   = {"primal_residual": 1e-9},
        )
    return out


class TestOptimRunRegistry:

    @pytest.fixture
    def opt_registry(self, tmp_path):
        return OptimRunRegistry(storage_dir=str(tmp_path / "opt"))

    def test_stage_constant(self, opt_registry):
        assert opt_registry.STAGE == "opt"

    def test_register_returns_opt_prefixed_id(self, opt_registry):
        run_id = opt_registry.register(
            _toy_optim_results(),
            parents={"inputs": {}, "upstream": {"const_run_id": "const_x"}},
        )
        assert run_id.startswith("opt_")
        assert _RUN_ID_RE.match(run_id)

    def test_index_has_only_four_fields(self, opt_registry):
        opt_registry.register(
            _toy_optim_results(),
            parents={"inputs": {}, "upstream": {"const_run_id": "const_x"}},
            label="copper-2020-21",
        )
        with open(os.path.join(opt_registry.storage_dir, "registry.json")) as f:
            payload = json.load(f)
        for entry in payload.values():
            assert set(entry.keys()) == {"run_id", "stage", "label", "created_at"}
            assert entry["label"] == "copper-2020-21"

    def test_sidecar_records_parents_and_summary(self, opt_registry):
        run_id = opt_registry.register(
            _toy_optim_results(),
            parents={"inputs": {}, "upstream": {"const_run_id": "const_x"}},
        )
        sc = opt_registry.load_sidecar(run_id)
        assert sc["parents"]["upstream"]["const_run_id"] == "const_x"
        assert sc["stage_summary"]["timesteps"] == ["2020", "2021"]
        assert sc["stage_summary"]["per_timestep"]["2020"]["objective"] == pytest.approx(1.23)
        assert sc["stage_summary"]["per_timestep"]["2021"]["solver_status"] == "Solved"

    def test_load_round_trip(self, opt_registry):
        original = _toy_optim_results()
        run_id = opt_registry.register(
            original,
            parents={"inputs": {}, "upstream": {"const_run_id": "const_x"}},
        )
        reg2 = OptimRunRegistry(storage_dir=opt_registry.storage_dir)
        loaded = reg2.load(run_id)
        assert set(loaded.keys()) == {"2020", "2021"}
        np.testing.assert_allclose(loaded["2020"].F_opt, original["2020"].F_opt)
        assert loaded["2020"].objective    == pytest.approx(1.23)
        assert loaded["2021"].solver_status == "Solved"
        assert loaded["2021"].iterations    == 5

    def test_load_missing_raises(self, opt_registry):
        with pytest.raises(KeyError):
            opt_registry.load("opt_nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
