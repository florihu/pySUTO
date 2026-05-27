"""Tests for ``pysuto.config.load_config``."""
import textwrap

import pytest

from pysuto.config import Config, load_config


_VALID_YAML = textwrap.dedent("""\
    pipeline:
      acc_id: copper
""")


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_roundtrip(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert isinstance(cfg, Config)
    assert cfg.pipeline.acc_id == "copper"


def test_unknown_key_raises(tmp_path):
    bad = _VALID_YAML.replace("acc_id: copper", "acc_id: copper\n  foo: 99")
    with pytest.raises(ValueError, match="foo"):
        load_config(_write(tmp_path, bad))


def test_missing_key_raises(tmp_path):
    bad = "pipeline: {}\n"
    with pytest.raises(ValueError, match="acc_id"):
        load_config(_write(tmp_path, bad))


def test_frozen(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    with pytest.raises(Exception):
        cfg.pipeline.acc_id = "other"  # type: ignore[misc]
