"""Typed loader for ``config/config.yaml``.

Init scripts call :func:`load_config` once and pass the resolved values as
kwargs to module constructors.  Modules keep their existing kwarg signatures
and defaults — they do not import this module.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union

import yaml


@dataclass(frozen=True)
class PropagationConfig:
    p:         float
    q:         float
    alpha_min: float


@dataclass(frozen=True)
class FlatPriorConfig:
    default_cv: float


@dataclass(frozen=True)
class TimeImputerConfig:
    window:           int
    min_obs:          int
    gap_inflation:    float
    method:           str
    interpolate_only: bool


@dataclass(frozen=True)
class ImputersConfig:
    flat_prior: FlatPriorConfig
    time:       TimeImputerConfig


@dataclass(frozen=True)
class ConstraintsConfig:
    sigma_a_cv_min:    float
    sigma_a_abs_floor: float
    sigma_c_cv_min:    float
    sigma_c_abs_floor: float


@dataclass(frozen=True)
class Config:
    propagation: PropagationConfig
    imputers:    ImputersConfig
    constraints: ConstraintsConfig


def _build(cls, payload):
    if not isinstance(payload, dict):
        raise ValueError(f"{cls.__name__}: expected mapping, got {type(payload).__name__}")
    expected = {f.name for f in fields(cls)}
    extra    = set(payload) - expected
    if extra:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(extra)}")
    missing = expected - set(payload)
    if missing:
        raise ValueError(f"{cls.__name__}: missing keys {sorted(missing)}")
    kwargs = {}
    for f in fields(cls):
        v = payload[f.name]
        if hasattr(f.type, "__dataclass_fields__"):
            kwargs[f.name] = _build(f.type, v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: Union[str, Path] = "config/config.yaml") -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return _build(Config, {
        "propagation": raw["propagation"],
        "imputers":    raw["imputers"],
        "constraints": raw["constraints"],
    })
