"""Tests for ``pysuto.config.load_config``."""
import textwrap

import pytest

from pysuto.config import Config, load_config


_VALID_YAML = textwrap.dedent("""\
    pipeline:
      acc_id: copper
      imputation:
        bindings:
          - {params_id: eff, builder: ParametricImputer}
      constraints:
        bindings:
          - {params_id: conc, builder: RatioConstraintBuilder}
    propagation:
      p: 1.3
      q: 3.0
      alpha_min: 0.05
    imputers:
      flat_prior:
        default_cv: 0.3
        cv_lo_supply: 0.04
        cv_hi_supply: 0.3
        cv_lo_use: 0.2
        cv_hi_use: 0.5
        row_inflation: 1.5
      time:
        window: 3
        min_obs: 2
        gap_inflation: 1.05
        method: linear
        interpolate_only: true
    constraints:
      sigma_a_cv_min: 0.05
      sigma_a_abs_floor: 1.0e-6
      sigma_c_cv_min: 0.05
      sigma_c_abs_floor: 1.0e-3
""")


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_roundtrip(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert isinstance(cfg, Config)
    assert cfg.pipeline.acc_id       == "copper"
    assert cfg.pipeline.imputation.bindings[0].params_id == "eff"
    assert cfg.pipeline.constraints.bindings[0].builder  == "RatioConstraintBuilder"
    assert cfg.propagation.p         == 1.3
    assert cfg.propagation.q         == 3.0
    assert cfg.propagation.alpha_min == 0.05
    assert cfg.imputers.flat_prior.default_cv == 0.3
    assert cfg.imputers.time.window  == 3
    assert cfg.imputers.time.method  == "linear"
    assert cfg.imputers.time.interpolate_only is True
    assert cfg.constraints.sigma_a_abs_floor == pytest.approx(1e-6)


def test_unknown_key_raises(tmp_path):
    bad = _VALID_YAML.replace("p: 1.3", "p: 1.3\n  foo: 99")
    with pytest.raises(ValueError, match="foo"):
        load_config(_write(tmp_path, bad))


def test_missing_key_raises(tmp_path):
    bad = _VALID_YAML.replace("  alpha_min: 0.05\n", "")
    with pytest.raises(ValueError, match="alpha_min"):
        load_config(_write(tmp_path, bad))


def test_frozen(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    with pytest.raises(Exception):
        cfg.propagation.p = 2.0  # type: ignore[misc]
