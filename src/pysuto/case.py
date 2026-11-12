"""
case.py
=======
A **case** as a typed declaration (GLOSSARY.md "Case", #26 §3, #120): the
:class:`CaseSpec` a case module ``pysuto.cases.<id>`` instantiates and the Stage 2
and Stage 3 drivers build from.

A case declares one section per stage. A section is a frozen dataclass whose
UPPER_CASE fields are the stage's **knobs** (GLOSSARY.md "Knob") at the case's
values, and whose lowercase fields are the case's selectors and builders — plain
functions over a :class:`CaseContext` — and its other, non-knob values. The case
module subclasses :class:`ImpSection` and :class:`ConstSection` to declare its knob
fields; the library names none of them. ``--set KNOB=VALUE`` replaces a knob field
(:meth:`Section.with_knobs`), and a run records its section in its sidecar
(:meth:`CaseSpec.describe`).

The library never imports ``pysuto.cases``: a case module sits at the top of the
import direction, beside the drivers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Callable, Mapping, Optional

from pysuto.knobs import check_type

if TYPE_CHECKING:
    from pysuto.acc import AccountingStructure
    from pysuto.agg import AggregationResult
    from pysuto.input_registry import InputRegistries


@dataclass(frozen=True)
class Section:
    """One stage's part of a case: knobs are its UPPER_CASE fields."""

    def knobs(self) -> dict:
        """``{KNOB: value}`` for every UPPER_CASE field, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name.isupper()}

    def with_knobs(self, overrides: Mapping[str, object]) -> "Section":
        """This section with ``overrides`` replacing its knob fields. A name that is
        not a knob raises ``KeyError``; a value is checked, never coerced, by the
        rules of ``--set`` (:func:`pysuto.knobs.check_type`), else ``TypeError``."""
        knobs = self.knobs()
        bad = [k for k in overrides if k not in knobs]
        if bad:
            raise KeyError(f"{bad} are not knobs of {type(self).__name__}; its knobs are "
                           f"{', '.join(knobs)}")
        for name, value in overrides.items():
            check_type(name, value, knobs[name])
        return replace(self, **overrides)

    def describe(self) -> dict:
        """Every field for a sidecar: values by value, functions by qualified name."""
        return {f.name: _describe(getattr(self, f.name)) for f in fields(self)}


def _describe(value):
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


@dataclass(frozen=True)
class ImpSection(Section):
    """The Stage 2 section (``collect_aggregate_impute.py``).

    ``imputers(ctx)`` returns the imputer cascade in order, terminal fillers
    included: a :class:`Cascade` for the fixpoint group, then each terminal imputer.
    ``pre_impute(ctx)`` runs after aggregation and before the cascade is built,
    ``post_impute(ctx)`` after the imputation; each may change ``ctx.agg_result`` in
    place. ``None`` is no hook. ``stage_config(ctx)`` returns the dict of the case's
    knob values that the imp sidecar records beside the generic keys. ``diag_year`` is
    the year the Stage 2 diagnostics focus on (``None``: the first year of the acc's
    time range). The knob ``P_PROP`` (the Lp exponent of the aggregation) is read by
    the stage body, so every case's section declares it.
    """
    imputers:     Callable
    pre_impute:   Optional[Callable] = None
    post_impute:  Optional[Callable] = None
    stage_config: Optional[Callable] = None
    diag_year:    Optional[str] = None


@dataclass(frozen=True)
class ConstSection(Section):
    """The Stage 3 section (``constraints.py``).

    ``builders(ctx)`` returns the constraint builders in registration order.
    ``nodata_anchor_exempt(ctx)`` returns ``(mask, n)``: the per-edge exemption
    from the no-data σ anchor that ``ConstraintEngine`` takes (``None`` keeps its
    sector rule) and the number of region-years it spares, recorded in the
    sidecar; ``None`` is ``(None, 0)``. ``stage_config(ctx)`` returns the dict of the
    case's knob values that the const sidecar records beside the generic keys
    (``ctx.n_anchor_exempt`` is that count). ``diag_year`` is the year of the Stage 3
    diagnostics (``None``: the first timestep). The knobs the stage body hands to
    ``ConstraintEngine`` are ``P_PROP``, ``SIGMA_A_CV_MIN``, ``SIGMA_A_ABS_FLOOR``,
    ``SIGMA_C_CV_MIN``, ``SIGMA_C_ABS_FLOOR``, ``PRIOR_BOX_K``, ``NODATA_ANCHOR_CV``,
    ``NODATA_ANCHOR_EXCLUDE``, ``PIN_FLAT_PRIOR_ZEROS`` and ``DIAG_PLOT``; every case's
    section declares them.
    """
    builders:             Callable
    nodata_anchor_exempt: Optional[Callable] = None
    stage_config:         Optional[Callable] = None
    diag_year:            Optional[str] = None


@dataclass(frozen=True)
class CaseSpec:
    """One case: its id, the accounting structure it is built on (an acc registry
    id) and its Stage 2 and Stage 3 sections.

    ``driver_knobs`` is ``{stage: {KNOB: value}}`` for the drivers of Stages 4–6 (stage
    ``"opt"`` or ``"psut"``): the case's values of the driver constants that differ
    from the driver's own, which the driver applies before ``--set``. Those drivers
    hold no case policy; an empty mapping leaves them as they are.
    """
    id:           str
    acc_id:       str
    imp:          ImpSection
    const:        ConstSection
    driver_knobs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def with_knobs(self, stage: str, overrides: Mapping[str, object]) -> "CaseSpec":
        """This case with ``overrides`` replacing knob fields of section ``stage``."""
        return replace(self, **{stage: getattr(self, stage).with_knobs(overrides)})

    def ref(self) -> dict:
        """The case as a Stage 4–6 run records it (``stage_config["case"]``)."""
        return {"id": self.id, "acc_id": self.acc_id}

    def describe(self, stage: str) -> dict:
        """The case as a ``stage`` run records it (``stage_config["case"]``): the case
        id, the acc id and that stage's section."""
        return {"id": self.id, "acc_id": self.acc_id,
                stage: getattr(self, stage).describe()}


@dataclass(frozen=True)
class Cascade:
    """The fixpoint group of a Stage 2 cascade (``ImputationEngine.register_cascade``):
    its imputers in the order they run and the cap on its passes."""
    imputers: list
    max_iter: int


@dataclass
class CaseContext:
    """What a case's selectors and builders read.

    Attributes
    ----------
    case : the case, with the run's knob overrides applied.
    acc : the run's ``AccountingStructure``.
    inputs : the run's ``InputRegistries``.
    data_dir : the data root holding ``proc/`` and ``input/``.
    agg_result : Stage 2: the aggregation, imputed in place; Stage 3: the imp run.
    logger : the driver's logger.
    param_refs : the params entries the run records in ``parents.inputs.params``;
        builders append what they read.
    n_anchor_exempt : Stage 3: the region-years ``nodata_anchor_exempt`` spares.
    reports : name to DataFrame; the case's screens put their reports here and the
        Stage 2 driver writes each to ``<run>/diag/<name>.csv``.
    """
    case:       CaseSpec
    acc:        AccountingStructure
    inputs:     InputRegistries
    data_dir:   str
    agg_result: AggregationResult
    logger:     logging.Logger
    param_refs: list = field(default_factory=list)
    reports:    dict = field(default_factory=dict)
    n_anchor_exempt: int = 0
