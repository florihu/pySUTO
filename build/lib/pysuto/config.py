"""Typed loader for ``config/config.yaml``.

Init scripts call :func:`load_config` once and read ``cfg.pipeline.acc_id``
to wire the AccountingStructure they should be running against. All other
stage knobs (propagation exponent, imputer windows, σ floors, builder
bindings, …) live inline in the driver scripts themselves.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, get_args, get_origin

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    acc_id: str


@dataclass(frozen=True)
class Config:
    pipeline: PipelineConfig


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
    return _build(Config, {"pipeline": raw["pipeline"]})
