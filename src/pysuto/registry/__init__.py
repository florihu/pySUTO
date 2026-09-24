"""
registry — harmonized output registries for Stages 1–4.

Public surface (re-exported here so call sites can keep using
``from pysuto.registry import X``):

* :class:`BaseOutputRegistry`     — shared scaffolding
* :class:`DataFeedRegistry`       — Stage 1
* :class:`ImputationRunRegistry`  — Stage 2
* :class:`ConstraintRunRegistry`  — Stage 3
* :class:`OptimRunRegistry`       — Stage 4
* :class:`UncPropRunRegistry`     — Stage 5
* :class:`PsutRunRegistry`        — Stage 6
* :class:`_RegistryEntry`         — trimmed index row
* :func:`make_run_id`             — ``{stage}_{ts}_{code}`` minter
* :func:`build_sidecar`           — per-run ``metadata.json`` payload builder

All four output registries share one run-id pattern and one sidecar schema,
defined in :class:`BaseOutputRegistry`:

* run-id literal format ``{stage}_{YYYYMMDD_HHMMSS}_{8-char-uuid4-hex}``
* per-run sidecar ``metadata.json`` with ``parents.inputs`` and
  ``parents.upstream``, plus free-form ``stage_config`` / ``stage_summary``.
* top-level ``registry.json`` index trimmed to four fields per entry.
"""

from pysuto.meta   import DataFeedMetadata
from ._sidecar     import (
    _NpEncoder,
    _RegistryEntry,
    _json_dump,
    _json_load,
    build_sidecar,
    make_run_id,
)
from .base         import BaseOutputRegistry
from .feed         import DataFeedRegistry
from .imputation   import ImputationRunRegistry
from .constraint   import ConstraintRunRegistry
from .optim        import OptimRunRegistry
from .uncprop      import UncPropRunRegistry
from .psut         import PsutRunRegistry

__all__ = [
    "BaseOutputRegistry",
    "DataFeedRegistry",
    "ImputationRunRegistry",
    "ConstraintRunRegistry",
    "OptimRunRegistry",
    "UncPropRunRegistry",
    "PsutRunRegistry",
    "DataFeedMetadata",
    "_RegistryEntry",
    "_NpEncoder",
    "_json_dump",
    "_json_load",
    "make_run_id",
    "build_sidecar",
]
