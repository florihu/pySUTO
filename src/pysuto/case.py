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

from dataclasses import dataclass, field, fields, replace
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class Section:
    """One stage's part of a case: knobs are its UPPER_CASE fields."""

    def knobs(self) -> dict:
        """``{KNOB: value}`` for every UPPER_CASE field, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name.isupper()}

    def with_knobs(self, overrides: Mapping[str, object]) -> "Section":
        """This section with ``overrides`` replacing its knob fields; a name that is
        not a knob raises ``KeyError``. Values are not checked here: the command line
        checks them through :func:`pysuto.knobs.parse_overrides`."""
        knobs = self.knobs()
        bad = [k for k in overrides if k not in knobs]
        if bad:
            raise KeyError(f"{bad} are not knobs of {type(self).__name__}; its knobs are "
                           f"{', '.join(knobs)}")
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
    place. ``None`` is no hook.
    """
    imputers:    Callable
    pre_impute:  Optional[Callable] = None
    post_impute: Optional[Callable] = None


@dataclass(frozen=True)
class ConstSection(Section):
    """The Stage 3 section (``constraints.py``).

    ``builders(ctx)`` returns the constraint builders in registration order.
    ``nodata_anchor_exempt(ctx)`` returns ``(mask, n)``: the per-edge exemption
    from the no-data σ anchor that ``ConstraintEngine`` takes (``None`` keeps its
    sector rule) and the number of region-years it spares, recorded in the
    sidecar; ``None`` is ``(None, 0)``.
    """
    builders:             Callable
    nodata_anchor_exempt: Optional[Callable] = None


@dataclass(frozen=True)
class CaseSpec:
    """One case: its id, the accounting structure it is built on (an acc registry
    id) and its Stage 2 and Stage 3 sections."""
    id:     str
    acc_id: str
    imp:    ImpSection
    const:  ConstSection

    def with_knobs(self, stage: str, overrides: Mapping[str, object]) -> "CaseSpec":
        """This case with ``overrides`` replacing knob fields of section ``stage``."""
        return replace(self, **{stage: getattr(self, stage).with_knobs(overrides)})

    def describe(self, stage: str) -> dict:
        """The case as a ``stage`` run records it (``stage_config["case"]``): the case
        id, the acc id and that stage's section."""
        return {"id": self.id, "acc_id": self.acc_id,
                stage: getattr(self, stage).describe()}


@dataclass(frozen=True)
class Cascade:
    """The fixpoint group of a Stage 2 cascade (``ImputationEngine.register_cascade``):
    its imputers by id, the order they run in — a permutation of the ids (#104) —
    and the cap on its passes."""
    imputers: Mapping[str, object]
    order:    tuple
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
    time_scope : Stage 2: the time interpolator's edge scope, computed by the driver
        until #121 moves its selector into the case.
    scale_total_resolver : Stage 3: the scale totals' σ resolver, built by the driver
        until #121 moves it into the case.
    """
    case:       CaseSpec
    acc:        object
    inputs:     object
    data_dir:   str
    agg_result: object
    logger:     object
    param_refs: list = field(default_factory=list)
    time_scope: Optional[object] = None
    scale_total_resolver: Optional[Callable] = None
