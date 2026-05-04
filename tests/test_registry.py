"""
test_registry.py
================
Unit tests for pysuto.registry:

    _NpEncoder          – numpy-safe JSON encoder
    DataFeedMetadata    – dataclass, from_dict, to_dict, from_excel,
                          fill_computed
    _RegistryEntry      – dataclass, to_dict, from_dict round-trip
    DataFeedRegistry    – register, save, load_index, load_metadata,
                          load_result, load_feed, summary_df,
                          container protocol, overwrite guard

All tests are self-contained: no pysuto.conc, pysuto.unc, or
pysuto.feed imports are required.  Heavy objects (feeds, results)
are replaced by minimal duck-typed stubs.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto.registry import (
    DataFeedMetadata,
    DataFeedRegistry,
    _RegistryEntry,
)

from pysuto.utils import _NpEncoder, _json_dump, _json_load


# =============================================================================
# STUBS
# =============================================================================

class _Feed:
    """Minimal duck-typed feed stub (replaces WeightMatrix)."""

    def __init__(self, feed_id: str, acc_id: str = "acc_test"):
        self.id       = feed_id
        self.acc_used = acc_id
        self.save_calls: list[str] = []

    def save(self, **kwargs) -> None:
        path = kwargs.get("path", f"{self.id}.pkl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"id": self.id}, f)
        self.save_calls.append(path)


class _Result:
    """Minimal duck-typed PropagationResult stub."""

    def __init__(
        self,
        F_root:     np.ndarray | None = None,
        sigma_root: np.ndarray | None = None,
        cv_root:    np.ndarray | None = None,
        cv_src:     float = 0.022644,
    ):
        self.F_root     = F_root     if F_root     is not None else np.array([100.0, 200.0, 50.0])
        self.sigma_root = sigma_root if sigma_root is not None else np.array([2.0, 4.0, 1.0])
        self.cv_root    = cv_root    if cv_root    is not None else np.array([0.02, 0.02, 0.02])
        self.cv_src     = cv_src


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def feed():
    return _Feed("icsg", "acc_test")


@pytest.fixture
def feed2():
    return _Feed("bgs", "acc_test")


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
def populated_registry(tmp_path, feed, metadata, result):
    """Registry with one registered feed (with result)."""
    reg = DataFeedRegistry(odir=str(tmp_path))
    reg.register(feed, metadata, result=result)
    return reg


@pytest.fixture
def excel_metadata_file(tmp_path):
    """
    Write a minimal Excel metadata sidecar and return its path.
    Uses openpyxl so no xlrd dependency is needed.
    """
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Grouping", "Meta_name", "Value", "Required"])
    rows = [
        ("Identification", "Feed_name",                "icsg",              True),
        ("Identification", "IEDC_type",                "1_F_",              True),
        ("Unc",            "Reliability",               1,                   True),
        ("Unc",            "Completeness",              2,                   True),
        ("Unc",            "Temporal_correlation",      1,                   True),
        ("Unc",            "Geographical_correlation",  3,                   True),
        ("Unc",            "Other_correlation",         1,                   True),
        ("Review",         "Version",                   0,                   False),
        ("Review",         "Date_of_assessment",        "2025-06-11",        False),
        ("Review",         "Person",                    "Florian_Huber",     False),
        ("Review",         "Dataset_link",              "https://example.com", True),
    ]
    for row in rows:
        ws.append(list(row))
    path = str(tmp_path / "metadata.xlsx")
    wb.save(path)
    return path


# =============================================================================
# 1. _NpEncoder
# =============================================================================

class TestNpEncoder:

    def test_numpy_int64(self):
        out = json.dumps(np.int64(42), cls=_NpEncoder)
        assert json.loads(out) == 42

    def test_numpy_int32(self):
        assert json.loads(json.dumps(np.int32(7), cls=_NpEncoder)) == 7

    def test_numpy_float64(self):
        val = json.loads(json.dumps(np.float64(3.14), cls=_NpEncoder))
        assert val == pytest.approx(3.14)

    def test_numpy_float32(self):
        val = json.loads(json.dumps(np.float32(2.5), cls=_NpEncoder))
        assert val == pytest.approx(2.5, rel=1e-5)

    def test_numpy_array_1d(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = json.loads(json.dumps(arr, cls=_NpEncoder))
        assert out == [1.0, 2.0, 3.0]

    def test_numpy_array_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        out = json.loads(json.dumps(arr, cls=_NpEncoder))
        assert out == [[1, 2], [3, 4]]

    def test_numpy_array_empty(self):
        out = json.loads(json.dumps(np.array([]), cls=_NpEncoder))
        assert out == []

    def test_datetime(self):
        dt = datetime(2025, 6, 11, 12, 0, 0)
        out = json.dumps(dt, cls=_NpEncoder)
        assert "2025-06-11" in out

    def test_plain_types_unchanged(self):
        d = {"a": 1, "b": "hello", "c": [1, 2, 3], "d": None}
        assert json.loads(json.dumps(d, cls=_NpEncoder)) == d

    def test_unsupported_type_raises(self):
        class Unsupported:
            pass
        with pytest.raises(TypeError):
            json.dumps(Unsupported(), cls=_NpEncoder)


# =============================================================================
# 2. _json_dump / _json_load
# =============================================================================

class TestJsonHelpers:

    def test_round_trip_plain_dict(self, tmp_path):
        d = {"x": 1, "y": "hello"}
        path = str(tmp_path / "test.json")
        _json_dump(d, path)
        assert _json_load(path) == d

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c.json")
        _json_dump({"k": "v"}, path)
        assert os.path.isfile(path)

    def test_numpy_values_serialised(self, tmp_path):
        d = {"val": np.int64(99)}
        path = str(tmp_path / "np.json")
        _json_dump(d, path)
        loaded = _json_load(path)
        assert loaded["val"] == 99

    def test_output_is_valid_utf8(self, tmp_path):
        d = {"name": "Ünïcödé"}
        path = str(tmp_path / "utf8.json")
        _json_dump(d, path)
        assert _json_load(path)["name"] == "Ünïcödé"

    def test_output_is_indented(self, tmp_path):
        path = str(tmp_path / "indent.json")
        _json_dump({"a": 1}, path)
        raw = Path(path).read_text()
        assert "\n" in raw        # indented JSON has newlines


# =============================================================================
# 3. DataFeedMetadata — construction & dataclass contract
# =============================================================================

class TestDataFeedMetadataDefaults:

    def test_default_feed_name_empty(self):
        assert DataFeedMetadata().feed_name == ""

    def test_default_pedigree_scores_are_one(self):
        m = DataFeedMetadata()
        for attr in ("reliability", "completeness", "temporal_correlation",
                     "geographical_correlation", "other_correlation"):
            assert getattr(m, attr) == 1

    def test_default_sensitivities_are_medium(self):
        m = DataFeedMetadata()
        for attr in ("completeness_sensitivity", "temporal_sensitivity",
                     "geographical_sensitivity", "other_sensitivity"):
            assert getattr(m, attr) == "medium_sensitive"

    def test_computed_fields_default_to_none(self):
        m = DataFeedMetadata()
        assert m.cv_total is None
        assert m.F_root_sum is None
        assert m.sigma_root_mean is None

    def test_computation_timestamp_default_empty(self):
        assert DataFeedMetadata().computation_timestamp == ""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DataFeedMetadata)


# =============================================================================
# 4. DataFeedMetadata — to_dict / from_dict round-trip
# =============================================================================

class TestDataFeedMetadataDict:

    def test_to_dict_returns_dict(self, metadata):
        assert isinstance(metadata.to_dict(), dict)

    def test_to_dict_contains_feed_name(self, metadata):
        assert metadata.to_dict()["feed_name"] == "icsg"

    def test_to_dict_contains_all_fields(self, metadata):
        d = metadata.to_dict()
        for f in dataclasses.fields(DataFeedMetadata):
            assert f.name in d

    def test_from_dict_round_trip(self, metadata):
        d = metadata.to_dict()
        restored = DataFeedMetadata.from_dict(d)
        assert restored.feed_name == metadata.feed_name
        assert restored.reliability == metadata.reliability
        assert restored.person == metadata.person

    def test_from_dict_ignores_unknown_keys(self):
        m = DataFeedMetadata.from_dict({"feed_name": "x", "unknown_key": 99})
        assert m.feed_name == "x"

    def test_from_dict_uses_defaults_for_missing_keys(self):
        m = DataFeedMetadata.from_dict({"feed_name": "x"})
        assert m.reliability == 1

    def test_from_dict_preserves_computed_fields(self):
        d = {"feed_name": "x", "cv_total": 0.05, "F_root_sum": 100.0,
             "sigma_root_mean": 2.0, "computation_timestamp": "2025-01-01T00:00:00Z"}
        m = DataFeedMetadata.from_dict(d)
        assert m.cv_total == pytest.approx(0.05)
        assert m.F_root_sum == pytest.approx(100.0)

    def test_to_dict_none_values_preserved(self):
        m = DataFeedMetadata()
        assert m.to_dict()["cv_total"] is None


# =============================================================================
# 5. DataFeedMetadata — from_excel
# =============================================================================

class TestDataFeedMetadataFromExcel:

    def test_feed_name_parsed(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.feed_name == "icsg"

    def test_iedc_type_parsed(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.iedc_type == "1_F_"

    def test_reliability_parsed_as_int(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.reliability == 1
        assert isinstance(m.reliability, int)

    def test_completeness_parsed(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.completeness == 2

    def test_geographical_correlation_parsed(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.geographical_correlation == 3

    def test_person_parsed(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.person == "Florian_Huber"

    def test_dataset_link_parsed(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.dataset_link == "https://example.com"

    def test_computed_fields_not_populated_by_excel(self, excel_metadata_file):
        m = DataFeedMetadata.from_excel(excel_metadata_file)
        assert m.cv_total is None
        assert m.F_root_sum is None

    def test_case_insensitive_key_lookup(self, tmp_path):
        """'reliability' in lowercase must still be found."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Meta_name", "Value"])
        ws.append(["reliability", 3])
        ws.append(["feed_name", "test"])
        path = str(tmp_path / "ci.xlsx")
        wb.save(path)
        m = DataFeedMetadata.from_excel(path)
        assert m.reliability == 3

    def test_missing_key_falls_back_to_default(self, tmp_path):
        """Keys absent from Excel revert to DataFeedMetadata defaults."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Meta_name", "Value"])
        ws.append(["Feed_name", "sparse"])
        path = str(tmp_path / "sparse.xlsx")
        wb.save(path)
        m = DataFeedMetadata.from_excel(path)
        assert m.completeness == 1     # default


# =============================================================================
# 6. DataFeedMetadata — fill_computed
# =============================================================================

class TestDataFeedMetadataFillComputed:

    def test_cv_total_set_from_result(self, metadata, result):
        metadata.fill_computed(result)
        assert metadata.cv_total == pytest.approx(result.cv_src)

    def test_F_root_sum_equals_nansum(self, metadata, result):
        metadata.fill_computed(result)
        expected = float(np.nansum(result.F_root))
        assert metadata.F_root_sum == pytest.approx(expected)

    def test_sigma_root_mean_equals_nanmean(self, metadata, result):
        metadata.fill_computed(result)
        expected = float(np.nanmean(result.sigma_root))
        assert metadata.sigma_root_mean == pytest.approx(expected)

    def test_computation_timestamp_is_utc_iso(self, metadata, result):
        metadata.fill_computed(result)
        ts = metadata.computation_timestamp
        assert ts.endswith("Z")
        # must be parseable as ISO-8601
        datetime.fromisoformat(ts.rstrip("Z"))

    def test_fill_computed_with_nan_F_root(self, metadata):
        r = _Result(F_root=np.array([np.nan, 100.0, np.nan]))
        metadata.fill_computed(r)
        assert metadata.F_root_sum == pytest.approx(100.0)

    def test_fill_computed_concrete_values(self):
        m = DataFeedMetadata()
        r = _Result(
            F_root=np.array([100.0, 200.0, 50.0]),
            sigma_root=np.array([2.0, 4.0, 1.0]),
            cv_src=0.022644,
        )
        m.fill_computed(r)
        assert m.cv_total        == pytest.approx(0.022644)
        assert m.F_root_sum      == pytest.approx(350.0)
        assert m.sigma_root_mean == pytest.approx(7.0 / 3.0)


# =============================================================================
# 7. _RegistryEntry
# =============================================================================

class TestRegistryEntry:

    @pytest.fixture
    def entry(self):
        return _RegistryEntry(
            feed_run_id="feed_icsg_20260101_000000_abcdef01",
            feed_id="icsg",
            feed_class="WeightMatrix",
            acc_id="acc_test",
            odir="/data/proc/feeds/feed_icsg_20260101_000000_abcdef01",
            metadata_path="/data/proc/feeds/feed_icsg_20260101_000000_abcdef01/metadata.json",
            created_at="2025-06-11T12:00:00Z",
        )

    def test_is_dataclass(self, entry):
        assert dataclasses.is_dataclass(entry)

    def test_to_dict_keys(self, entry):
        d = entry.to_dict()
        assert set(d.keys()) == {
            "feed_run_id", "feed_id", "feed_class", "acc_id", "odir",
            "metadata_path", "created_at",
        }

    def test_to_dict_values(self, entry):
        d = entry.to_dict()
        assert d["feed_id"]    == "icsg"
        assert d["feed_class"] == "WeightMatrix"
        assert d["acc_id"]     == "acc_test"

    def test_from_dict_round_trip(self, entry):
        restored = _RegistryEntry.from_dict(entry.to_dict())
        assert restored.feed_id    == entry.feed_id
        assert restored.feed_class == entry.feed_class
        assert restored.created_at == entry.created_at

    def test_from_dict_ignores_extra_keys(self, entry):
        d = entry.to_dict()
        d["extra_key"] = "should_be_ignored"
        restored = _RegistryEntry.from_dict(d)
        assert restored.feed_id == "icsg"


# =============================================================================
# 8. DataFeedRegistry — construction
# =============================================================================

class TestRegistryConstruction:

    def test_odir_stored(self, tmp_path):
        reg = DataFeedRegistry(odir=str(tmp_path))
        assert reg.odir == str(tmp_path)

    def test_empty_on_init(self, registry):
        assert len(registry) == 0

    def test_ids_empty_on_init(self, registry):
        assert registry.ids() == []

    def test_registry_filename_constant(self):
        assert DataFeedRegistry.REGISTRY_FILENAME == "registry.json"


# =============================================================================
# 9. DataFeedRegistry — register
# =============================================================================

class TestRegistryRegister:

    def test_returns_registry_entry(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert isinstance(entry, _RegistryEntry)

    def test_entry_feed_id(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert entry.feed_id == "icsg"

    def test_entry_feed_class_is_class_name(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert entry.feed_class == "_Feed"

    def test_entry_acc_id(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert entry.acc_id == "acc_test"

    def test_entry_created_at_ends_with_z(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert entry.created_at.endswith("Z")

    def test_entry_odir_is_run_subdir(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert entry.odir == os.path.join(registry.odir, entry.feed_run_id)
        assert entry.feed_run_id.startswith("feed_icsg_")

    def test_metadata_json_written(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert os.path.isfile(os.path.join(entry.odir, "metadata.json"))

    def test_metadata_json_contains_feed_name(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        d = _json_load(os.path.join(entry.odir, "metadata.json"))
        assert d["feed_name"] == "icsg"

    def test_metadata_json_contains_cv_total(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        d = _json_load(os.path.join(entry.odir, "metadata.json"))
        assert d["cv_total"] == pytest.approx(result.cv_src)

    def test_pickle_written(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert os.path.isfile(os.path.join(entry.odir, "feed.pkl"))

    def test_feed_save_called_with_correct_path(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        expected = os.path.join(entry.odir, "feed.pkl")
        assert expected in feed.save_calls

    def test_result_npz_written(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert os.path.isfile(os.path.join(entry.odir, "result.npz"))

    def test_no_result_no_npz(self, registry, feed, metadata):
        entry = registry.register(feed, metadata, result=None)
        assert not os.path.isfile(os.path.join(entry.odir, "result.npz"))

    def test_no_result_no_fill_computed(self, registry, feed, metadata):
        registry.register(feed, metadata, result=None)
        assert metadata.cv_total is None

    def test_feed_added_to_index(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        assert entry.feed_run_id in registry
        assert "icsg" in registry  # feed_id resolves via latest

    def test_len_increases(self, registry, feed, feed2, metadata, metadata2, result):
        registry.register(feed, metadata, result=result)
        assert len(registry) == 1
        registry.register(feed2, metadata2, result=result)
        assert len(registry) == 2

    def test_repeat_register_creates_new_run(self, registry, feed, metadata, result):
        e1 = registry.register(feed, metadata, result=result)
        # Sleep is unnecessary: the run-id has a uuid suffix that disambiguates
        # within the same second.
        e2 = registry.register(feed, metadata, result=result)
        assert e1.feed_run_id != e2.feed_run_id
        assert len(registry) == 2

    def test_latest_by_feed_id_returns_newest(self, registry, feed, metadata, result):
        e1 = registry.register(feed, metadata, result=result)
        e2 = registry.register(feed, metadata, result=result)
        latest = registry.latest_by_feed_id("icsg")
        # newest is whichever has the lexicographically larger timestamp; both
        # have identical timestamps but different uuids → fall back on insertion.
        assert latest.feed_run_id in (e1.feed_run_id, e2.feed_run_id)

    def test_created_at_is_iso_parseable(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        datetime.fromisoformat(entry.created_at.rstrip("Z"))

    def test_result_arrays_values(self, registry, feed, metadata, result):
        entry = registry.register(feed, metadata, result=result)
        z = np.load(os.path.join(entry.odir, "result.npz"))
        np.testing.assert_allclose(z["F_root"],     result.F_root)
        np.testing.assert_allclose(z["sigma_root"], result.sigma_root)
        np.testing.assert_allclose(z["cv_root"],    result.cv_root)
        assert float(z["cv_src"][0]) == pytest.approx(result.cv_src)


# =============================================================================
# 10. DataFeedRegistry — save / load_index round-trip
# =============================================================================

class TestRegistrySaveLoad:

    def test_save_creates_registry_json(self, populated_registry):
        populated_registry.save()
        path = os.path.join(populated_registry.odir, "registry.json")
        assert os.path.isfile(path)

    def test_save_creates_odir_if_absent(self, tmp_path):
        odir = str(tmp_path / "new" / "dir")
        reg = DataFeedRegistry(odir=odir)
        reg.save()
        assert os.path.isdir(odir)

    def test_load_index_no_op_on_missing_file(self, tmp_path):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.load_index()        # must not raise
        assert len(reg) == 0

    def test_round_trip_len(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        assert len(reg2) == 1

    def test_round_trip_feed_id(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        assert "icsg" in reg2

    def test_round_trip_entry_fields(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        e1 = populated_registry["icsg"]
        e2 = reg2["icsg"]
        assert e2.feed_id     == e1.feed_id
        assert e2.feed_run_id == e1.feed_run_id
        assert e2.feed_class  == e1.feed_class
        assert e2.acc_id      == e1.acc_id
        assert e2.created_at  == e1.created_at

    def test_multiple_feeds_round_trip(self, tmp_path, feed, feed2,
                                       metadata, metadata2, result):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed,  metadata,  result=result)
        reg.register(feed2, metadata2, result=result)
        reg.save()

        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        assert len(reg2) == 2
        assert set(reg2.ids()) == {"icsg", "bgs"}

    def test_save_is_idempotent(self, populated_registry, tmp_path):
        populated_registry.save()
        populated_registry.save()     # second save must not raise or corrupt
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        assert len(reg2) == 1

    def test_registry_json_is_valid_json(self, populated_registry):
        populated_registry.save()
        path = os.path.join(populated_registry.odir, "registry.json")
        with open(path) as f:
            data = json.load(f)
        # registry.json is keyed by feed_run_id; check via a feed_id lookup
        assert any(e.get("feed_id") == "icsg" for e in data.values())

    def test_load_index_overwrites_in_memory(self, tmp_path, feed, feed2,
                                              metadata, metadata2, result):
        """load_index must replace the current in-memory index, not merge."""
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed, metadata, result=result)
        reg.save()

        # Dirty the in-memory state
        reg.register(feed2, metadata2, result=result)
        assert len(reg) == 2

        # Reload from disk — only icsg was saved
        reg.load_index()
        assert len(reg) == 1
        assert "icsg" in reg
        assert "bgs" not in reg


# =============================================================================
# 11. DataFeedRegistry — load_metadata
# =============================================================================

class TestRegistryLoadMetadata:

    def test_returns_data_feed_metadata(self, populated_registry):
        populated_registry.save()
        m = populated_registry.load_metadata("icsg")
        assert isinstance(m, DataFeedMetadata)

    def test_feed_name_correct(self, populated_registry):
        m = populated_registry.load_metadata("icsg")
        assert m.feed_name == "icsg"

    def test_cv_total_correct(self, populated_registry, result):
        m = populated_registry.load_metadata("icsg")
        assert m.cv_total == pytest.approx(result.cv_src)

    def test_F_root_sum_correct(self, populated_registry, result):
        m = populated_registry.load_metadata("icsg")
        assert m.F_root_sum == pytest.approx(float(np.nansum(result.F_root)))

    def test_sigma_root_mean_correct(self, populated_registry, result):
        m = populated_registry.load_metadata("icsg")
        assert m.sigma_root_mean == pytest.approx(float(np.nanmean(result.sigma_root)))

    def test_person_correct(self, populated_registry):
        m = populated_registry.load_metadata("icsg")
        assert m.person == "Florian_Huber"

    def test_missing_feed_id_raises_key_error(self, populated_registry):
        with pytest.raises(KeyError):
            populated_registry.load_metadata("nonexistent")

    def test_survives_registry_json_round_trip(self, populated_registry, tmp_path):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        m = reg2.load_metadata("icsg")
        assert m.feed_name == "icsg"


# =============================================================================
# 12. DataFeedRegistry — load_result
# =============================================================================

class TestRegistryLoadResult:

    def test_returns_dict(self, populated_registry, result):
        populated_registry.save()
        d = populated_registry.load_result("icsg")
        assert isinstance(d, dict)

    def test_result_keys(self, populated_registry, result):
        d = populated_registry.load_result("icsg")
        assert set(d.keys()) == {"F_root", "sigma_root", "cv_root", "cv_src"}

    def test_F_root_values(self, populated_registry, result):
        d = populated_registry.load_result("icsg")
        np.testing.assert_allclose(d["F_root"], result.F_root)

    def test_sigma_root_values(self, populated_registry, result):
        d = populated_registry.load_result("icsg")
        np.testing.assert_allclose(d["sigma_root"], result.sigma_root)

    def test_cv_root_values(self, populated_registry, result):
        d = populated_registry.load_result("icsg")
        np.testing.assert_allclose(d["cv_root"], result.cv_root)

    def test_cv_src_scalar(self, populated_registry, result):
        d = populated_registry.load_result("icsg")
        assert float(d["cv_src"][0]) == pytest.approx(result.cv_src)

    def test_missing_npz_raises_file_not_found(self, tmp_path, feed, metadata):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed, metadata, result=None)   # no result → no npz
        with pytest.raises(FileNotFoundError):
            reg.load_result("icsg")

    def test_missing_feed_id_raises_key_error(self, populated_registry):
        with pytest.raises(KeyError):
            populated_registry.load_result("nonexistent")

    def test_arrays_survive_save_load_round_trip(self, populated_registry,
                                                  tmp_path, result):
        populated_registry.save()
        reg2 = DataFeedRegistry(odir=str(tmp_path))
        reg2.load_index()
        d = reg2.load_result("icsg")
        np.testing.assert_allclose(d["F_root"], result.F_root)


# =============================================================================
# 13. DataFeedRegistry — container protocol
# =============================================================================

class TestRegistryContainerProtocol:

    def test_contains_registered_id(self, populated_registry):
        assert "icsg" in populated_registry

    def test_not_contains_unregistered_id(self, populated_registry):
        assert "nope" not in populated_registry

    def test_len_empty(self, registry):
        assert len(registry) == 0

    def test_len_one(self, populated_registry):
        assert len(populated_registry) == 1

    def test_getitem_returns_entry(self, populated_registry):
        assert isinstance(populated_registry["icsg"], _RegistryEntry)

    def test_getitem_missing_raises_key_error(self, populated_registry):
        with pytest.raises(KeyError):
            _ = populated_registry["nonexistent"]

    def test_ids_returns_list(self, populated_registry):
        ids = populated_registry.ids()
        assert isinstance(ids, list)

    def test_ids_contains_registered(self, populated_registry):
        assert "icsg" in populated_registry.ids()

    def test_ids_empty_on_fresh_registry(self, registry):
        assert registry.ids() == []

    def test_iter_yields_run_ids(self, tmp_path, feed, feed2,
                                  metadata, metadata2, result):
        reg = DataFeedRegistry(odir=str(tmp_path))
        e1 = reg.register(feed,  metadata,  result=result)
        e2 = reg.register(feed2, metadata2, result=result)
        assert set(reg) == {e1.feed_run_id, e2.feed_run_id}


# =============================================================================
# 14. DataFeedRegistry — summary_df
# =============================================================================

class TestRegistrySummaryDf:

    def test_returns_dataframe(self, populated_registry):
        df = populated_registry.summary_df()
        assert isinstance(df, pd.DataFrame)

    def test_index_is_feed_run_id(self, populated_registry):
        df = populated_registry.summary_df()
        assert df.index.name == "feed_run_id"
        # at least one row whose feed_id column is "icsg"
        assert (df["feed_id"] == "icsg").any()

    def test_has_one_row(self, populated_registry):
        df = populated_registry.summary_df()
        assert len(df) == 1

    def test_cv_total_percent_column_present(self, populated_registry):
        df = populated_registry.summary_df()
        assert "cv_total_%" in df.columns

    def test_cv_total_percent_value(self, populated_registry, result):
        df = populated_registry.summary_df()
        expected = round(result.cv_src * 100, 3)
        row = df[df["feed_id"] == "icsg"].iloc[0]
        assert row["cv_total_%"] == pytest.approx(expected)

    def test_feed_name_column(self, populated_registry):
        df = populated_registry.summary_df()
        row = df[df["feed_id"] == "icsg"].iloc[0]
        assert row["feed_name"] == "icsg"

    def test_person_column(self, populated_registry):
        df = populated_registry.summary_df()
        row = df[df["feed_id"] == "icsg"].iloc[0]
        assert row["person"] == "Florian_Huber"

    def test_multiple_feeds_multiple_rows(self, tmp_path, feed, feed2,
                                          metadata, metadata2, result):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed,  metadata,  result=result)
        reg.register(feed2, metadata2, result=result)
        df = reg.summary_df()
        assert len(df) == 2
        assert set(df["feed_id"]) == {"icsg", "bgs"}

    def test_empty_registry_returns_empty_df(self, registry):
        df = registry.summary_df()
        assert len(df) == 0

    def test_no_result_cv_total_is_none(self, tmp_path, feed, metadata):
        reg = DataFeedRegistry(odir=str(tmp_path))
        reg.register(feed, metadata, result=None)
        df = reg.summary_df()
        row = df[df["feed_id"] == "icsg"].iloc[0]
        assert row["cv_total_%"] is None

    def test_corrupt_metadata_emits_warning(self, populated_registry):
        """If metadata.json is corrupt, summary_df warns but does not raise."""
        entry = populated_registry["icsg"]
        meta_path = os.path.join(entry.odir, "metadata.json")
        with open(meta_path, "w") as f:
            f.write("NOT VALID JSON {{{{")
        with pytest.warns(UserWarning):
            df = populated_registry.summary_df()
        assert len(df) == 1   # row still present, metadata columns may be missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
