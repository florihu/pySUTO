"""
case.py
=======
A **case** as a typed declaration (GLOSSARY.md "Case", #26 §3, #120): the
:class:`CaseSpec` a case module instantiates (``CASE``) and the Stage 2 and Stage 3
drivers build from.

A case declares one section per stage. A section is a frozen dataclass whose
UPPER_CASE fields are the stage's **knobs** (GLOSSARY.md "Knob") at the case's
values, and whose lowercase fields are the case's selectors and builders — plain
functions over a :class:`CaseContext` — and its other, non-knob values. The case
module subclasses :class:`ImpSection` and :class:`ConstSection` to declare its knob
fields; the library names none of them. ``--set KNOB=VALUE`` replaces a knob field
(:meth:`Section.with_knobs`), and a run records its section in its sidecar
(:meth:`CaseSpec.describe`).

The library never imports ``pysuto.tutorial`` by name: a case module sits at the top of
the import direction, beside the drivers. :func:`resolve_case` finds a case from the
string ``--case`` carries (#187) and imports a bundled alias's module only at call time,
through :data:`ALIASES`, a name-to-module table, so the pipeline, the drivers and a
step's child share one resolver without the library importing a case.
The module also hosts the drivers' ``--case`` argparse helpers, :func:`add_case_arg` and
:func:`case_from_args`; the latter ends a driver through ``parser.error`` on an unknown
reference.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional

from pysuto._provenance import sha256_file
from pysuto.impute import FillImputer
from pysuto.knobs import check_type
from pysuto.relation.sources import PAIR_ACCEPTS, TWIN_ACCEPTS, Fill, ParamSource

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


def _qualname(fn) -> Optional[str]:
    return None if fn is None else f"{fn.__module__}.{fn.__qualname__}"


@dataclass(frozen=True)
class FillEntry:
    """A cascade entry that is one :class:`~pysuto.relation.sources.Fill` of a params
    relation source (GLOSSARY.md "Fill", ADR 0011): ``params_id`` names the params
    family, ``twin`` picks the source's reading (``True``: the conc twins, ``False``:
    the per-pair rows), ``rule`` and ``sigma_rule`` the direction and σ rule of the
    :class:`~pysuto.impute.FillImputer`, and ``scope`` the writer scope as a function of
    the accounting structure (``None``: every endpoint of the source's relations). The
    entry is data until :meth:`build` reads the params registry through the context."""
    id:         str
    params_id:  str
    rule:       str
    sigma_rule: str                = "parametric"
    twin:       bool               = False
    scope:      Optional[Callable] = None

    def build(self, ctx: "CaseContext") -> FillImputer:
        """The imputer of this entry: its source resolved on ``ctx.acc`` from the
        registered params file, which the run records as a parent input."""
        spec = ctx.case.imp
        path = ctx.use_params(self.params_id)
        source = ParamSource.from_params(path, ctx.acc, twin=self.twin)
        fill = Fill(source, self.rule,
                    None if self.scope is None else self.scope(ctx.acc), id=self.id)
        kw = {"p": spec.P_PROP}
        if self.sigma_rule == "parametric":
            kw.update(q=spec.Q_REG, alpha_min=spec.ALPHA_MIN_REG)
        ctx.logger.info("C  built FillImputer id=%s (params=%s, %d relations)",
                        self.id, self.params_id, len(source))
        return FillImputer(fill, self.sigma_rule, **kw)

    def describe(self) -> dict:
        """The entry for a sidecar: its source (params id and the param types read),
        rule, σ rule and scope name."""
        types = TWIN_ACCEPTS if self.twin else PAIR_ACCEPTS
        return {"kind": "fill", "id": self.id,
                "source": {"params_id": self.params_id, "param_types": sorted(types)},
                "rule": self.rule, "sigma_rule": self.sigma_rule,
                "scope": _qualname(self.scope)}


@dataclass(frozen=True)
class CascadeStep:
    """A cascade entry that is not a fill, e.g. the network's recipe traversal:
    ``build(ctx)`` returns the imputer, whose ``id`` is the entry's."""
    id:    str
    build: Callable

    def describe(self) -> dict:
        return {"kind": "step", "id": self.id, "build": _qualname(self.build)}


@dataclass(frozen=True)
class ImpSection(Section):
    """The Stage 2 section (``collect_aggregate_impute.py``).

    ``cascade`` declares the fixpoint group as data: its entries (:class:`FillEntry`,
    :class:`CascadeStep`) in declaration order; the knob ``CASCADE_ORDER`` (empty:
    declaration order) is the order they run in and must be a permutation of their ids
    (:meth:`ordered`), and ``CASCADE_MAX_ITER`` caps the group's passes. ``terminal(ctx)``
    returns the terminal imputers, run once after the cascade in the order listed (time,
    share, flat prior; ADR 0011). ``pre_impute(ctx)`` runs after aggregation and before
    the cascade is built, ``post_impute(ctx)`` after the imputation; each may change
    ``ctx.agg_result`` in place. ``None`` is no hook. ``stage_config(ctx)`` returns the
    dict of the case's knob values that the imp sidecar records beside the generic keys.
    ``diag_year`` is the year the Stage 2 diagnostics focus on (``None``: the first year
    of the acc's time range). The knob ``P_PROP`` (the Lp exponent of the aggregation) is
    read by the stage body, and ``Q_REG`` and ``ALPHA_MIN_REG`` by a parametric fill, so
    every case's section declares them. ``MIN_FLOW_KT`` is the coverage threshold of the
    feed collector (kt): a feed leg below it is not aggregated (``feedprep/reexport.py``
    documents its coupling to the feed weights); the imp sidecar records it with the
    case's other knobs (#279).
    """
    cascade:      tuple
    terminal:     Callable
    pre_impute:   Optional[Callable] = None
    post_impute:  Optional[Callable] = None
    stage_config: Optional[Callable] = None
    diag_year:    Optional[str] = None
    CASCADE_ORDER:    tuple = ()
    CASCADE_MAX_ITER: int   = 10
    MIN_FLOW_KT:      float = 0.01

    def ordered(self) -> list:
        """The cascade entries in ``CASCADE_ORDER``, which must be a permutation of
        their ids (#104); anything else — an unknown, repeated or missing id — raises
        ``ValueError``."""
        by_id = {e.id: e for e in self.cascade}
        order = tuple(self.CASCADE_ORDER) or tuple(by_id)
        if len(by_id) != len(self.cascade) or sorted(order) != sorted(by_id):
            raise ValueError(f"CASCADE_ORDER {order} must be a permutation of "
                             f"{tuple(e.id for e in self.cascade)}")
        return [by_id[k] for k in order]

    def describe(self) -> dict:
        """:meth:`Section.describe` with the cascade as its entries' records."""
        return {**super().describe(), "cascade": [e.describe() for e in self.cascade]}

    def validate(self) -> None:
        """Raise ``ValueError`` for knob values the case cannot run: a
        ``CASCADE_ORDER`` that is no permutation of the cascade's ids."""
        self.ordered()

    def imputers(self, ctx: "CaseContext") -> list:
        """The imputers in the order they register: a :class:`Cascade` of the entries'
        imputers in ``CASCADE_ORDER`` (none if the case declares no entries), then each
        terminal imputer. The entries are built in declaration order."""
        order = self.ordered()
        built = {e.id: e.build(ctx) for e in self.cascade}
        out = []
        if order:
            out.append(Cascade([built[e.id] for e in order], self.CASCADE_MAX_ITER))
        return out + list(self.terminal(ctx))


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
    ``NODATA_ANCHOR_EXCLUDE``, ``PIN_FLAT_PRIOR_ZEROS``, ``PIN_EXCLUDE_SECTORS`` and
    ``DIAG_PLOT``; every case's section declares them.
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

    ``stage1`` is the case's Stage 1 hook, ``stage1(data_dir) -> {feed_id: feed_run_id}``,
    which writes the case's feeds into the feed registry of ``data_dir`` (the ``feed``
    step of ``python -m pysuto run``); ``None`` is a case with no hook.

    ``case_source`` and ``case_sha256`` are where the case came from (``alias:toy``,
    ``module:mylab.zinc`` or ``path:<abs>``) and the sha256 of that module's source
    file. Only :func:`resolve_case` sets them, and a run records them in
    ``stage_config["case"]``; they are left out of equality (provenance, not the case);
    a case built in Python has neither and records neither. ``with_knobs`` keeps them (a
    run records its knobs); code that changes a resolved case structurally, with
    ``dataclasses.replace`` on builders, selectors or sections, must clear both
    (``replace(case, case_source=None, case_sha256=None)``) so that the run does not claim
    the unmodified file. Start from the module's own ``CASE``, which carries neither.
    """
    id:           str
    acc_id:       str
    imp:          ImpSection
    const:        ConstSection
    driver_knobs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    stage1:       Optional[Callable] = None
    case_source:  Optional[str] = field(default=None, compare=False)
    case_sha256:  Optional[str] = field(default=None, compare=False)

    def with_knobs(self, stage: str, overrides: Mapping[str, object]) -> "CaseSpec":
        """This case with ``overrides`` replacing knob fields of section ``stage``."""
        return replace(self, **{stage: getattr(self, stage).with_knobs(overrides)})

    def driver_knobs_for(self, stage: str, knobs: Iterable[str],
                         namespace: Mapping[str, object]) -> dict:
        """The case's ``driver_knobs[stage]``, checked by the knob protocol: a name
        outside ``knobs`` raises ``KeyError``, a value of another type than the driver
        constant's current value in ``namespace`` raises ``TypeError``."""
        values = dict(self.driver_knobs.get(stage, {}))
        knobs = tuple(knobs)
        bad = [k for k in values if k not in knobs]
        if bad:
            raise KeyError(f"case {self.id!r} sets {bad}, which are not knobs of the "
                           f"{stage} driver; its knobs are {', '.join(knobs)}")
        for name, value in values.items():
            check_type(name, value, namespace[name])
        return values

    def ref(self) -> dict:
        """The case as a Stage 4–6 run records it (``stage_config["case"]``): the id, the
        acc id and, for a resolved case, its source and the sha256 of its file."""
        ref = {"id": self.id, "acc_id": self.acc_id}
        if self.case_source is not None:
            ref.update(case_source=self.case_source, case_sha256=self.case_sha256)
        return ref

    def describe(self, stage: str) -> dict:
        """The case as a ``stage`` run records it (``stage_config["case"]``): the case
        id, the acc id, the provenance of :meth:`ref` and that stage's section."""
        return {**self.ref(), stage: getattr(self, stage).describe()}


def check_case(sidecar: dict, case: CaseSpec) -> None:
    """Raise ``ValueError`` when the run of ``sidecar`` records another case than
    ``case``. A run made before the case id was recorded names none and passes."""
    recorded = (sidecar.get("stage_config", {}).get("case") or {}).get("id")
    if recorded is not None and recorded != case.id:
        raise ValueError(f"run {sidecar.get('run_id')!r} was made by case {recorded!r}, "
                         f"but this driver runs --case {case.id}")


# ======================================================================================
# Resolving a ``--case`` reference (#187)
# ======================================================================================
# The bundled cases, name to module; the modules are imported at call time only. A case of
# your own is not bundled: it is a file or a module the caller names (ADR 0047).
ALIASES = {"toy":           "pysuto.tutorial.toy",
           "toy_coproduct": "pysuto.tutorial.toy_coproduct"}


def _is_path(reference: str) -> bool:
    return reference.endswith(".py") or "/" in reference or os.sep in reference


def case_reference(reference: str, cwd: Optional[str] = None) -> str:
    """``reference`` as a step's child must receive it: a file path made absolute against
    ``cwd`` (default the working directory), an alias or a dotted module unchanged."""
    if _is_path(reference):
        return os.path.abspath(os.path.join(cwd or os.getcwd(), reference))
    return reference


def _load_file(path: str):
    if not os.path.isfile(path):
        raise ValueError(f"unknown case '{path}': read as a file path (it ends in .py or "
                         f"holds a path separator), and no such file exists")
    folder = os.path.dirname(path)
    if folder not in sys.path:      # as a script run by Python: siblings are importable
        sys.path.insert(0, folder)
    name = "_pysuto_case_" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # dataclasses and pickling look the module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def resolve_case(reference: str) -> CaseSpec:
    """The :class:`CaseSpec` ``CASE`` of the module ``reference`` names, stamped with
    where it came from.

    ``reference`` is a bundled alias (:data:`ALIASES`), a file path (a string ending in
    ``.py`` or holding a path separator, relative to the working directory; its folder is
    put first on ``sys.path`` if absent, before the file runs, so that it can import the
    modules next to it as a script run by Python can) or a dotted
    module (``mylab.zinc``, found on ``sys.path``). The returned case has ``case_source``
    (``alias:<name>``, ``path:<abs>`` or ``module:<dotted>``) and ``case_sha256``, the
    sha256 of the module's source file.

    Raises
    ------
    ValueError
        Nothing of that name is found (the message names the forms tried), or the module
        declares no ``CASE`` that is a :class:`CaseSpec`. An import error inside the
        module propagates.
    """
    if reference in ALIASES:
        module, source = importlib.import_module(ALIASES[reference]), f"alias:{reference}"
    elif _is_path(reference):
        path = os.path.abspath(reference)
        module, source = _load_file(path), f"path:{path}"
    else:
        try:
            module = importlib.import_module(reference)
        except ModuleNotFoundError as exc:
            if exc.name is None or not (reference == exc.name
                                        or reference.startswith(exc.name + ".")):
                raise                       # a dependency of the module is missing
            raise ValueError(
                f"unknown case {reference!r}: not a bundled alias "
                f"({', '.join(sorted(ALIASES))}), and no module of that name imports "
                f"({exc}); a file path ends in .py or holds a path separator") from exc
        source = f"module:{reference}"
    case = getattr(module, "CASE", None)
    if not isinstance(case, CaseSpec):
        raise ValueError(f"case {reference!r}: module {module.__name__} declares no "
                         f"CASE that is a CaseSpec")
    if not getattr(module, "__file__", None):
        raise ValueError(f"case {reference!r}: module {module.__name__} has no "
                         f"source file")
    return replace(case, case_source=source, case_sha256=sha256_file(module.__file__))


def add_case_arg(parser, default: Optional[str] = None) -> None:
    """Add ``--case <reference>`` to a driver's parser: ``default`` is the case it runs
    without the option; ``None`` makes the option required, which
    :func:`case_from_args` enforces (exit 2, listing the aliases)."""
    parser.add_argument("--case", default=default, metavar="CASE",
                        help="Case: a bundled alias (" + ", ".join(sorted(ALIASES))
                             + "), a dotted module or a file path"
                             + (" (required)." if default is None
                                else f" (default {default})."))


def case_from_args(parser, args) -> CaseSpec:
    """The case ``--case`` names; an unknown reference ends the driver through
    ``parser.error`` (exit 2, naming what was tried)."""
    if args.case is None:
        parser.error("--case is required: a bundled alias (" + ", ".join(sorted(ALIASES))
                     + "), a dotted module or a file path")
    try:
        return resolve_case(args.case)
    except ValueError as exc:
        parser.error(str(exc))


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

    :meth:`use_params` is how a Stage 2 entry reads a params file.
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

    def use_params(self, params_id: str) -> str:
        """The path of the registered ``params_id`` CSV; the run records the entry once
        in ``param_refs``."""
        entry = self.inputs.params.entry(params_id)
        if entry not in self.param_refs:
            self.param_refs.append(entry)
        return str(self.inputs.params.resolve(params_id) / "params.csv")
