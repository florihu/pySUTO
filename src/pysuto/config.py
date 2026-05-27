"""Typed loader for ``config/config.yaml``.

Init scripts call :func:`load_config` once and pass the resolved values as
kwargs to module constructors.  Modules keep their existing kwarg signatures
and defaults — they do not import this module.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Union, get_args, get_origin

import yaml


@dataclass(frozen=True)
class PropagationConfig:
    p:         float
    q:         float
    alpha_min: float


@dataclass(frozen=True)
class FlatPriorConfig:
    default_cv:    float
    cv_lo_supply:  float
    cv_hi_supply:  float
    cv_lo_use:     float
    cv_hi_use:     float
    row_inflation: float


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
class Binding:
    params_id: str
    builder:   str


@dataclass(frozen=True)
class ImputationPipeline:
    bindings: List[Binding]


@dataclass(frozen=True)
class ConstraintsPipeline:
    bindings: List[Binding]


@dataclass(frozen=True)
class PipelineConfig:
    acc_id:      str
    imputation:  ImputationPipeline
    constraints: ConstraintsPipeline


@dataclass(frozen=True)
class Config:
    pipeline:    PipelineConfig
    propagation: PropagationConfig
    imputers:    ImputersConfig
    constraints: ConstraintsConfig


def _build_field(typ, value):
    if hasattr(typ, "__dataclass_fields__"):
        return _build(typ, value)
    if get_origin(typ) is list:
        (item_type,) = get_args(typ)
        if hasattr(item_type, "__dataclass_fields__"):
            return [_build(item_type, v) for v in value]
    return value


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
    kwargs = {f.name: _build_field(f.type, payload[f.name]) for f in fields(cls)}
    return cls(**kwargs)


def load_config(path: Union[str, Path] = "config/config.yaml") -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return _build(Config, {
        "pipeline":    raw["pipeline"],
        "propagation": raw["propagation"],
        "imputers":    raw["imputers"],
        "constraints": raw["constraints"],
    })
