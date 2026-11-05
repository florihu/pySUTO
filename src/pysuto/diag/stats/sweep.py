"""
sweep.py
========
:class:`SweepStatsDiagnostic` — the tables behind :class:`pysuto.diag.SweepDiag` (#107,
#117).

A sweep's points are scored against its **base chain** (GLOSSARY.md "Sweep",
"Point", "Outcome"). Every point gets

* the **cross-evaluated objective** — ``J_base_at_point = Σ_t J_base(a*_point(t)) −
  Σ_t J_base(a*_base(t))``: each point's answer scored under the base QP
  (:meth:`pysuto.solver.ClarabelSolver.objective_at`), ≥ 0 while the point is feasible
  for the base's hard rows (``max_hard_violation_base`` says whether it is);
* the **posterior distance** ``mass_moved = Σ_years Σ |a*_point − a*_base|`` (kt), in
  total and per valuation, and per category in the category tables with the **pooled
  retention** ``Σ_t Σa* / Σ_t Σa_prior`` (GLOSSARY.md "Pooled retention"); a category
  is ``sector_from→sector_to valuation``, and one with a signed edge (structural lower
  bound below 0) or a zero prior has no retention;
* every declared :class:`Outcome` and :class:`MaxAbs` — any edge set over a window of
  years, three measures — with a **year profile** for every retention outcome, and an
  :class:`EveryCategory` that declares one retention outcome per category;
* every declared :class:`Criterion` judged: a **verdict** per criterion.

Only the base's const run and the points' opt runs are read, so a sweep that kept only
Stage 4 (#108) is scored like any other.

:func:`sensitivity_table` ranks the axes per outcome and sizes their effect: in a random
sweep the standardised rank regression coefficient (SRRC) and Spearman's ρ of each
numeric axis, for every axis the **swing** (GLOSSARY.md "Swing"), and in a full grid
the two-way interaction range; :func:`sensitivity_by_year` does so per year of the year
profiles.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from pysuto.acc import AccountingStructure
from pysuto.param_relation import expand_edge_index
from pysuto.registry import OptimRunRegistry, SweepRunRegistry
from pysuto.solver import ClarabelSolver

FAMILIES = ("edges", "sum_rows", "soft_ratios", "other_soft")
MEASURES = ("mass_moved", "retention", "total")
# Below this R² the rank regression does not explain the ranks additively, and the
# ranking is read off Spearman's ρ instead of the SRRC (#108's reporting rule).
R2_MIN = 0.5
# #108's registered influence rule: an axis is influential on an outcome when |SRRC|
# reaches this and the rank regression's R² reaches R2_MIN.
SRRC_MIN = 0.2


# =============================================================================
# Outcomes and criteria
# =============================================================================

@dataclass(frozen=True)
class Outcome:
    """A named scalar of a point's answer against the base's (GLOSSARY.md "Outcome").

    Parameters
    ----------
    name : the column it becomes in :meth:`SweepStatsDiagnostic.points_frame`.
    measure : over the selected edges, the years of the window pooled —
        ``"mass_moved"`` ``Σ|a*_p − a*_b|`` (kt);
        ``"retention"`` ``Σa*_p/Σa_prior,p − Σa*_b/Σa_prior,b``, the change in pooled
        retention against the base (refused over a signed edge);
        ``"total"`` ``Σa*_p/Σa*_b − 1``, the relative change of the posterior, which
        also sees a knob that moves prior and posterior together.
    select : the edges — a dict ``column → value or list of values`` over the edge table
        (``sector_from``, ``sector_to``, ``region_from``, ``region_to``, ``valuation``,
        ``time``, ``category``), a callable ``edges → bool mask``, or None for every edge.
    window : ``(first year, last year)``, both included; None for the base's full span.
    """
    name:    str
    measure: str
    select:  Any = None
    window:  Optional[tuple] = None

    def __post_init__(self) -> None:
        if self.measure not in MEASURES:
            raise ValueError(f"outcome {self.name!r}: measure {self.measure!r}; expected "
                             f"one of {MEASURES}")
        if self.window is not None and len(self.window) != 2:
            raise ValueError(f"outcome {self.name!r}: window {self.window!r}; expected "
                             f"(first year, last year)")


@dataclass(frozen=True)
class MaxAbs:
    """The largest |value| among ``parts`` — the worst of several outcomes. Scored and
    judged by a :class:`Criterion`, never ranked: the |·| folds a one-way effect into a
    V that no additive rank regression fits."""
    name:  str
    parts: tuple


@dataclass(frozen=True)
class EveryCategory:
    """One retention outcome ``d_retention[<category> <valuation>]`` per category of
    ``valuation`` that has a retention, in chain order, expanded when the sweep is
    scored; :meth:`SweepStatsDiagnostic.category_table` names the categories without
    one."""
    valuation: str
    window:    Optional[tuple] = None


@dataclass(frozen=True)
class Criterion:
    """A limit on one or more outcomes, fixed before the sweep is read (GLOSSARY.md
    "Criterion"). A point meets it when every |outcome| (the parts of a
    :class:`MaxAbs`) is below ``limit``; it holds when every done point does.

    Parameters
    ----------
    item : the name it is cited by (e.g. ``"A6"``).
    outcomes : :class:`Outcome` and :class:`MaxAbs` objects; scored even when not
        declared as outcomes.
    limit : > 0, in the outcomes' unit.
    """
    item:     str
    outcomes: tuple
    limit:    float

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError(f"criterion {self.item!r} names no outcome")
        if not self.limit > 0:
            raise ValueError(f"criterion {self.item!r}: limit {self.limit!r}; expected "
                             f"> 0")


def _parts(o) -> tuple:
    return o.parts if isinstance(o, MaxAbs) else (o,)


def _once(outcomes) -> list:
    """Each outcome once, in first-seen order; two different ones of one name raise."""
    out: dict = {}
    for o in outcomes:
        if out.setdefault(o.name, o) != o:
            raise ValueError(f"two different outcomes are named {o.name!r}")
    return list(out.values())


def _mask(name: str, select, edges: pd.DataFrame) -> np.ndarray:
    if select is None:
        return np.ones(len(edges), dtype=bool)
    if callable(select):
        m = np.asarray(select(edges), dtype=bool)
    else:
        m = np.ones(len(edges), dtype=bool)
        for col, v in select.items():
            if col not in edges.columns:
                raise KeyError(f"selector column {col!r} is not in the edge table "
                               f"{list(edges.columns)}")
            vals = [str(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])]
            m &= edges[col].astype(str).isin(vals).to_numpy()
    if m.shape != (len(edges),):
        raise ValueError(f"outcome {name!r}: its selector returned shape {m.shape}, "
                         f"expected ({len(edges)},)")
    if not m.any():
        raise ValueError(f"outcome {name!r}: its selector {select!r} matches no edge")
    return m


def _max_abs(values) -> float:
    """max |v|, NaN when any value is NaN."""
    return float(np.max(np.abs(np.asarray(values, dtype=float))))


def _ratio(num, den):
    """num / den, NaN where den is 0: a retention over a zero prior is undefined."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, np.nan)


def _window_label(first: str, last: str) -> str:
    return first if first == last else f"{first}–{last}"


def _measure(o: Outcome, m: np.ndarray, F_p, P_p, F_b, P_b) -> float:
    if o.measure == "mass_moved":
        return float(np.abs(F_p[m] - F_b[m]).sum())
    if o.measure == "retention":
        return float(_ratio(F_p[m].sum(), P_p[m].sum()) - _ratio(F_b[m].sum(), P_b[m].sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(F_p[m].sum() / F_b[m].sum() - 1.0)


# =============================================================================
# Sensitivity statistics (pure)
# =============================================================================

def axis_label(axis: dict) -> str:
    """An axis's name: its keys joined by ``=`` (a tied axis names every key)."""
    return "=".join(axis["keys"])


def _is_numeric_axis(axis: dict) -> bool:
    if "distribution" in axis:
        return True
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in axis["values"])


def _levels(axis: dict, column: pd.Series) -> Optional[np.ndarray]:
    """Level index per row for a list axis (values compared as JSON), else None."""
    if "values" not in axis:
        return None
    canon = [json.dumps(v) for v in axis["values"]]
    return np.array([canon.index(json.dumps(v)) for v in column], dtype=np.int64)


def _standardised_ranks(x: np.ndarray) -> Optional[np.ndarray]:
    r = rankdata(x)
    sd = r.std()
    return None if sd == 0 else (r - r.mean()) / sd


def _swing(axis: dict, column: pd.Series, y: np.ndarray) -> float:
    """The outcome's mean over the top third of the axis's values minus its mean over
    the bottom third; for a list axis the spread of its level means."""
    lev = _levels(axis, column)
    if lev is not None:
        means = pd.Series(y).groupby(lev).mean()
        return float(means.max() - means.min()) if len(means) > 1 else np.nan
    k = len(y) // 3
    if k == 0:
        return np.nan
    order = np.argsort(column.to_numpy(dtype=float), kind="stable")
    return float(y[order[-k:]].mean() - y[order[:k]].mean())


def sensitivity_table(frame: pd.DataFrame, axes: Sequence[dict],
                      outcomes: Sequence[str], mode: str) -> pd.DataFrame:
    """Sensitivity of each outcome column to each axis.

    ``frame`` holds one row per scored point with a column per axis (named by the
    axis's first key) and the outcome columns; ``axes`` are the declaration's axis
    dicts (``keys`` plus ``values`` or ``distribution``). Rows with a non-finite
    outcome are dropped per outcome. Returns ``outcome, axis, statistic, value, n``
    with statistic ``srrc`` / ``spearman`` (numeric axes and two-level list axes, ranked
    in their declared order, random mode) and the rank regression's ``r2`` (axis
    ``(rank regression)``), ``swing`` (every axis, in the outcome's unit: top-third
    minus bottom-third mean of a numeric axis, spread of the level means of a list
    axis) and ``interaction_range`` (pairs of list axes in a full grid, axis ``a×b``);
    NaN where a statistic is undefined. A low ``r2`` says the ranks are not additive in
    the axes: interactions or non-monotone effects, where Spearman's ρ is the safer read.
    """
    rows = []
    for out in outcomes:
        y_all = pd.to_numeric(frame[out], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y_all)
        sub, y = frame[ok], y_all[ok]
        n = int(ok.sum())
        cols = {axis_label(ax): sub[ax["keys"][0]] for ax in axes}

        if mode == "random":
            numeric = [ax for ax in axes if _is_numeric_axis(ax)]
            z_y = _standardised_ranks(y) if n > 2 else None
            zs = {axis_label(ax):
                  _standardised_ranks(cols[axis_label(ax)].to_numpy(dtype=float))
                  for ax in numeric}
            usable = [k for k, z in zs.items() if z is not None]
            # A nominal list axis enters as level indicators, or its effect would leak
            # into the others'. Two levels have an order (second vs first), so their one
            # standardised indicator is the axis's standardised rank and gets an SRRC;
            # more levels have none.
            dummies, ranked = [], {}
            for ax in axes:
                if ax in numeric:
                    continue
                lev = _levels(ax, cols[axis_label(ax)])
                for k in np.unique(lev)[1:]:
                    d = (lev == k).astype(float)
                    if d.std() > 0:
                        dummies.append((d - d.mean()) / d.std())
                        if len(ax["values"]) == 2:
                            ranked[axis_label(ax)] = (len(dummies) - 1, lev)
            coef, r2 = {}, np.nan
            k = len(usable) + len(dummies)
            if z_y is not None and k and n > k + 1:
                X = np.column_stack([zs[k] for k in usable] + dummies)
                beta = np.linalg.lstsq(X, z_y, rcond=None)[0]
                coef = dict(zip(usable, beta))
                coef.update({lab: beta[len(usable) + j] for lab, (j, _) in ranked.items()})
                r2 = float(1.0 - np.sum((z_y - X @ beta) ** 2) / np.sum(z_y ** 2))
            rows.append((out, "(rank regression)", "r2", r2, n))
            for ax in axes:
                lab = axis_label(ax)
                x = (cols[lab].to_numpy(dtype=float) if zs.get(lab) is not None
                     else ranked[lab][1] if lab in ranked else None)
                rho = (float(spearmanr(x, y).correlation)
                       if x is not None and z_y is not None else np.nan)
                rows.append((out, lab, "srrc", float(coef.get(lab, np.nan)), n))
                rows.append((out, lab, "spearman", rho, n))

        for ax in axes:
            rows.append((out, axis_label(ax), "swing", _swing(ax, cols[axis_label(ax)], y),
                         n))

        if mode == "grid":
            lists = [ax for ax in axes if "values" in ax]
            for a, b in itertools.combinations(lists, 2):
                la, lb = _levels(a, cols[axis_label(a)]), _levels(b, cols[axis_label(b)])
                cell = pd.Series(y).groupby([la, lb]).mean()
                full = len(cell) == len(a["values"]) * len(b["values"])
                if not full:
                    val = np.nan
                else:
                    ma = pd.Series(y).groupby(la).mean()
                    mb = pd.Series(y).groupby(lb).mean()
                    g = float(np.mean(y))
                    resid = [cell[(i, j)] - ma[i] - mb[j] + g for i, j in cell.index]
                    val = float(np.max(resid) - np.min(resid))
                rows.append((out, f"{axis_label(a)}×{axis_label(b)}", "interaction_range",
                             val, n))
    return pd.DataFrame(rows, columns=["outcome", "axis", "statistic", "value", "n"])


def sensitivity_by_year(frame: pd.DataFrame, axes: Sequence[dict],
                        profiles: pd.DataFrame, mode: str) -> pd.DataFrame:
    """:func:`sensitivity_table` on each year's slice of the year profiles.

    ``frame`` holds one row per scored point (``index`` and the axis columns),
    ``profiles`` the year profiles (``index, outcome, year, value``). Returns
    ``year, outcome, axis, statistic, value, n``, the years in their order in
    ``profiles``."""
    out = []
    for year, g in profiles.groupby("year", sort=False):
        wide = g.pivot(index="index", columns="outcome", values="value")
        t = sensitivity_table(frame.join(wide, on="index"), axes,
                              list(dict.fromkeys(g["outcome"])), mode)
        t.insert(0, "year", year)
        out.append(t)
    if not out:
        return pd.DataFrame(columns=["year", "outcome", "axis", "statistic", "value", "n"])
    return pd.concat(out, ignore_index=True)


def rank_statistic(sens: pd.DataFrame, outcome: str) -> tuple:
    """The statistic that ranks the axes for ``outcome`` and the rank regression's R²:
    ``"srrc"`` where R² reaches R2_MIN, ``"spearman"`` below it, ``"swing"`` where there
    is no rank regression (a grid)."""
    r2 = sens[(sens["outcome"] == outcome) & (sens["statistic"] == "r2")]["value"]
    if not len(r2):
        return "swing", np.nan
    r2 = float(r2.iloc[0])
    return ("srrc" if r2 >= R2_MIN else "spearman"), r2


def influential(sens: pd.DataFrame, outcome: str) -> set:
    """The axes #108's rule calls influential on ``outcome``: |SRRC| ≥ SRRC_MIN with the
    rank regression's R² ≥ R2_MIN."""
    if rank_statistic(sens, outcome)[0] != "srrc":
        return set()
    t = sens[(sens["outcome"] == outcome) & (sens["statistic"] == "srrc")]
    return set(t[t["value"].abs() >= SRRC_MIN]["axis"])


def top_axis(sens: pd.DataFrame, outcome: str, r2_min: float = R2_MIN) -> Optional[tuple]:
    """The axis that moves ``outcome`` most: by |SRRC| where the rank regression's R²
    reaches ``r2_min``, else by |Spearman's ρ|, by |swing| in a grid. Returns ``(axis,
    statistic, value)``, None when no axis has that statistic."""
    t = sens[sens["outcome"] == outcome]
    r2 = t[t["statistic"] == "r2"]["value"]
    stat = ("swing" if not len(r2) else "srrc" if r2.iloc[0] >= r2_min else "spearman")
    t = t[t["statistic"] == stat].dropna(subset=["value"])
    if not len(t):
        return None
    row = t.loc[t["value"].abs().idxmax()]
    return row["axis"], stat, float(row["value"])


# =============================================================================
# SweepStatsDiagnostic
# =============================================================================

class SweepStatsDiagnostic:
    """Score a sweep's points against its base chain.

    Parameters
    ----------
    sweep_run_id : the ``sweep`` run.
    root : the ``data/proc`` directory holding the sweep, const and opt registries.
    edges : the edge table, one row per global edge (row order = edge id), with at
        least ``sector_from``, ``sector_to``, ``category`` and ``valuation`` plus any
        column a selector reads; default from the base const run's accounting structure
        (the expanded edge index, its ``time``, and ``category`` =
        ``sector_from→sector_to``). Categories run in chain order: by ``sector_from``,
        then ``sector_to``, in the structure's declared sector order (in a given table,
        the order the sectors first appear).
    outcomes : the declared :class:`Outcome`, :class:`MaxAbs` and :class:`EveryCategory`.
    criteria : the declared :class:`Criterion` objects.
    """

    def __init__(self, sweep_run_id: str, root: str = os.path.join("data", "proc"),
                 edges: Optional[pd.DataFrame] = None,
                 outcomes: Sequence[Union[Outcome, MaxAbs, EveryCategory]] = (),
                 criteria: Sequence[Criterion] = ()) -> None:
        reg = SweepRunRegistry(os.path.join(root, "sweep"))
        self.sweep_run_id = sweep_run_id
        self.root = root
        self.manifest = reg.load_manifest(sweep_run_id)
        self.sidecar  = reg.load_sidecar(sweep_run_id)
        self.declaration = self.manifest["declaration"]
        up = self.sidecar["parents"]["upstream"]
        self.base_const, self.base_opt = up["const_run_id"], up["opt_run_id"]
        self._edges = edges
        self._sector_order: Optional[list] = None
        self.outcome_defs = tuple(outcomes)
        self.criteria = tuple(criteria)
        self.base_max_hard_violation: Optional[float] = None
        self._cache: dict = {}

    # ------------------------------------------------------------------
    @property
    def axes(self) -> list:
        return self.declaration["axes"]

    @property
    def mode(self) -> str:
        return self.declaration["mode"]

    def _solver(self) -> ClarabelSolver:
        return ClarabelSolver(const_dir=os.path.join(self.root, "const"),
                              opt_dir=os.path.join(self.root, "opt"))

    def _opt(self, opt_run_id: str) -> dict:
        return OptimRunRegistry(os.path.join(self.root, "opt")).load(opt_run_id)

    def _edge_table(self, const_run) -> pd.DataFrame:
        if self._edges is None:
            acc = AccountingStructure.load(const_run.acc_id,
                                           storage_dir=os.path.join(self.root, "acc"))
            ei = expand_edge_index(acc.edge_index, acc.node_index).sort_values("edge_id")
            cols = ["sector_from", "sector_to", "region_from", "region_to", "valuation",
                    "time"]
            self._edges = ei[cols].astype(str).reset_index(drop=True)
            self._edges["category"] = (self._edges["sector_from"] + "→"
                                       + self._edges["sector_to"])
            self._sector_order = list(dict.fromkeys(acc.root_nodes["sector"].astype(str)))
        if self._sector_order is None:
            self._sector_order = list(dict.fromkeys(
                self._edges[["sector_from", "sector_to"]].astype(str).to_numpy().ravel()))
        return self._edges

    # ------------------------------------------------------------------
    def _category_table(self, edges, cat, cats, cat_code, P_b, signed) -> pd.DataFrame:
        """Per category in chain order: its sectors, valuation and why it has no
        retention (``"signed"``, ``"zero prior"``, or ``""``)."""
        first = edges.assign(_key=cat).drop_duplicates("_key").set_index("_key").loc[cats]
        n_signed = np.bincount(cat_code, weights=signed.astype(float), minlength=len(cats))
        prior = np.bincount(cat_code, weights=P_b, minlength=len(cats))
        pos = {s: k for k, s in enumerate(self._sector_order)}
        vpos = {v: k for k, v in enumerate(dict.fromkeys(edges["valuation"].astype(str)))}
        t = pd.DataFrame({
            "category":      cats,
            "sector_from":   first["sector_from"].astype(str).to_numpy(),
            "sector_to":     first["sector_to"].astype(str).to_numpy(),
            "valuation":     first["valuation"].astype(str).to_numpy(),
            "no_retention":  np.where(n_signed > 0, "signed",
                                      np.where(prior == 0, "zero prior", "")),
            "edge_category": first["category"].astype(str).to_numpy(),   # the selector's
            "code":          np.arange(len(cats)),                       # np.unique's
        })
        key = [(pos[a], pos[b], vpos[v]) for a, b, v in
               zip(t["sector_from"], t["sector_to"], t["valuation"])]
        return t.iloc[sorted(range(len(t)), key=key.__getitem__)].reset_index(drop=True)

    def _declared(self, ctab: pd.DataFrame) -> list:
        """The declared outcomes with every :class:`EveryCategory` expanded, then the
        criteria's outcomes, each once."""
        flat = []
        for o in self.outcome_defs + tuple(q for c in self.criteria for q in c.outcomes):
            if not isinstance(o, EveryCategory):
                flat.append(o)
                continue
            rows = ctab[(ctab["valuation"] == o.valuation) & (ctab["no_retention"] == "")]
            if not len(rows):
                raise ValueError(f"no category of valuation {o.valuation!r} has a "
                                 f"retention")
            flat += [Outcome(f"d_retention[{r.category}]", "retention",
                             {"category": r.edge_category, "valuation": r.valuation},
                             o.window)
                     for r in rows.itertuples()]
        return _once(flat)

    def _span(self, o: Outcome, years: list) -> tuple:
        """The positions of an outcome's first and last year among the base's years."""
        if o.window is None:
            return 0, len(years) - 1
        pos = []
        for y in o.window:
            if str(y) not in years:
                raise ValueError(f"outcome {o.name!r}: window year {y!r} is not a year of "
                                 f"the base ({years[0]}–{years[-1]})")
            pos.append(years.index(str(y)))
        if pos[0] > pos[1]:
            raise ValueError(f"outcome {o.name!r}: window {o.window!r} runs backwards")
        return pos[0], pos[1]

    def _compute(self) -> None:
        solver = self._solver()
        base_run = solver.load_run(self.base_const)
        base_res = self._opt(self.base_opt)
        edges = self._edge_table(base_run)
        n_e = base_run.G.shape[1]
        if len(edges) != n_e:
            raise ValueError(f"the edge table has {len(edges)} rows, the base run "
                             f"{n_e} edges")
        years = list(base_res)
        year_of = np.full(n_e, -1, dtype=np.int64)
        for k, t in enumerate(years):
            year_of[base_res[t].cols_t] = k

        def full(res, attr):
            v = np.zeros(n_e)
            for t, r in res.items():
                v[r.cols_t] = getattr(r, attr)
            return v

        def by(code, n, w):
            return np.bincount(code, weights=w, minlength=n)

        F_b, P_b = full(base_res, "F_opt"), full(base_res, "a_prior")
        signed = np.asarray(base_run.l, dtype=float) < 0
        cat = (edges["category"].astype(str) + " "
               + edges["valuation"].astype(str)).to_numpy()
        cats, cat_code = np.unique(cat, return_inverse=True)
        cat_size = np.bincount(cat_code, minlength=len(cats))
        ctab = self._category_table(edges, cat, cats, cat_code, P_b, signed)
        chain = ctab["code"].to_numpy()
        undefined = np.zeros(len(cats), dtype=bool)
        undefined[chain] = (ctab["no_retention"] != "").to_numpy()

        declared = self._declared(ctab)
        leaves = _once(q for o in declared for q in _parts(o))
        span, idx, rows_ot = {}, {}, []
        for o in leaves:
            span[o.name] = self._span(o, years)
            lo, hi = span[o.name]
            m = _mask(o.name, o.select, edges) & (year_of >= lo) & (year_of <= hi)
            if not m.any():
                raise ValueError(f"outcome {o.name!r}: its selector matches no edge in "
                                 f"{_window_label(years[lo], years[hi])}")
            if o.measure == "retention" and signed[m].any():
                raise ValueError(f"outcome {o.name!r}: {int(signed[m].sum())} of its edges "
                                 f"are signed (structural lower bound below 0), where a "
                                 f"retention is undefined; measure its mass moved or total")
            idx[o.name] = np.flatnonzero(m)
            codes = np.unique(cat_code[m])
            whole = (len(codes) == 1 and (lo, hi) == (0, len(years) - 1)
                     and m.sum() == cat_size[codes[0]])
            vals = np.unique(edges["valuation"].astype(str).to_numpy()[m])
            limits = [c for c in self.criteria
                      if o.name in {q.name for p in c.outcomes for q in _parts(p)}]
            rows_ot.append((o.name, o.measure, _window_label(years[lo], years[hi]),
                            vals[0] if len(vals) == 1 else "mixed",
                            cats[codes[0]] if whole else "",
                            min((c.limit for c in limits), default=np.nan),
                            ", ".join(c.item for c in limits)))
        retention = [o for o in leaves if o.measure == "retention"]
        ret_year_b = {o.name: _ratio(by(year_of[idx[o.name]], len(years), F_b[idx[o.name]]),
                                     by(year_of[idx[o.name]], len(years), P_b[idx[o.name]]))
                      for o in retention}

        vals, val_code = np.unique(edges["valuation"].astype(str).to_numpy(),
                                   return_inverse=True)
        J_b = {t: solver.objective_at(base_run, t, base_res[t].F_opt, base_res[t].a_prior)
               for t in years}
        self.base_max_hard_violation = max(J_b[t]["max_hard_violation"] for t in years)
        ret_b = _ratio(by(cat_code, len(cats), F_b), by(cat_code, len(cats), P_b))
        ret_b[undefined] = np.nan
        cy = cat_code * len(years) + year_of
        ret_by_b = _ratio(by(cy, len(cats) * len(years), F_b),
                          by(cy, len(cats) * len(years), P_b)).reshape(len(cats), len(years))
        ret_by_b[undefined] = np.nan
        reason = np.empty(len(cats), dtype=object)
        reason[chain] = ctab["no_retention"].to_numpy()

        point_rows, cat_rows, year_rows, prof_rows = [], [], [], []
        for pt in self.manifest["points"]:
            row = {"index": pt["index"], "status": pt["status"],
                   **{f"run_{s}": r for s, r in pt["runs"].items()}}
            for ax in self.axes:
                for k in ax["keys"]:
                    row[k] = pt["overrides"].get(k)
            opt_id = pt["runs"].get("opt")
            if pt["status"] != "done" or opt_id is None:
                point_rows.append(row)
                continue
            res = self._opt(opt_id)
            for t in years:
                if not np.array_equal(res[t].cols_t, base_res[t].cols_t):
                    raise ValueError(f"point {pt['index']} ({opt_id}) has other edges than "
                                     f"the base in {t}; it cannot be compared")
            F_p, P_p = full(res, "F_opt"), full(res, "a_prior")
            moved = np.abs(F_p - F_b)
            row["mass_moved_total"] = float(moved.sum())
            for v, m in zip(vals, by(val_code, len(vals), moved)):
                row[f"mass_moved_{v}"] = float(m)
            at_base = {t: solver.objective_at(base_run, t, res[t].F_opt,
                                              base_res[t].a_prior)
                       for t in years}
            row["J_base_at_point"] = sum(at_base[t]["total"] - J_b[t]["total"]
                                         for t in years)
            for f in FAMILIES:
                row[f"J_base_at_point_{f}"] = sum(at_base[t][f] - J_b[t][f] for t in years)
            row["max_hard_violation_base"] = max(at_base[t]["max_hard_violation"]
                                                 for t in years)
            leaf = {o.name: _measure(o, idx[o.name], F_p, P_p, F_b, P_b) for o in leaves}
            row.update(leaf)
            for o in declared:
                if isinstance(o, MaxAbs):
                    row[o.name] = _max_abs([leaf[q.name] for q in o.parts])
            for o in retention:
                i = idx[o.name]
                rp = _ratio(by(year_of[i], len(years), F_p[i]),
                            by(year_of[i], len(years), P_p[i])) - ret_year_b[o.name]
                lo, hi = span[o.name]
                prof_rows += [(pt["index"], o.name, years[k], float(rp[k]))
                              for k in range(lo, hi + 1)]

            m_cat = by(cat_code, len(cats), moved)
            ret_p = _ratio(by(cat_code, len(cats), F_p), by(cat_code, len(cats), P_p))
            ret_p[undefined] = np.nan
            ret_by_p = _ratio(by(cy, len(cats) * len(years), F_p),
                              by(cy, len(cats) * len(years), P_p)).reshape(len(cats),
                                                                           len(years))
            ret_by_p[undefined] = np.nan
            for j in chain:
                cat_rows.append((pt["index"], cats[j], float(m_cat[j]), float(ret_p[j]),
                                 float(ret_b[j]), reason[j]))
                year_rows += [(pt["index"], cats[j], years[k], float(ret_by_p[j, k]),
                               float(ret_by_b[j, k])) for k in range(len(years))]
            point_rows.append(row)

        self._cache = {
            "points": pd.DataFrame(point_rows),
            "categories": pd.DataFrame(cat_rows, columns=[
                "index", "category", "mass_moved", "retention", "retention_base",
                "no_retention"]),
            "years": pd.DataFrame(year_rows, columns=[
                "index", "category", "year", "retention", "retention_base"]),
            "profiles": pd.DataFrame(prof_rows, columns=["index", "outcome", "year",
                                                         "value"]),
            "category_table": ctab.drop(columns=["edge_category", "code"]),
            "outcome_table": pd.DataFrame(rows_ot, columns=[
                "outcome", "measure", "window", "valuation", "category", "limit",
                "criteria"]),
        }
        self._max_abs = [o.name for o in declared if isinstance(o, MaxAbs)]
        self._windows = {r[0]: r[2] for r in rows_ot}
        self._full_window = _window_label(years[0], years[-1])

    def _ensure(self, refresh: bool = False) -> None:
        if refresh or not self._cache:
            self._compute()

    def _get(self, key: str, refresh: bool = False) -> pd.DataFrame:
        self._ensure(refresh)
        return self._cache[key].copy()

    # ------------------------------------------------------------------
    def points_frame(self, refresh: bool = False) -> pd.DataFrame:
        """One row per point: status, run ids, axis values, scalar outcomes."""
        return self._get("points", refresh)

    def category_frame(self, refresh: bool = False) -> pd.DataFrame:
        """Per done point and category, in chain order: mass moved, pooled retention and
        the base's (NaN for a category with a ``no_retention`` reason)."""
        return self._get("categories", refresh)

    def retention_by_year(self, refresh: bool = False) -> pd.DataFrame:
        """Per done point, category and year: retention and the base's."""
        return self._get("years", refresh)

    def year_profiles(self, refresh: bool = False) -> pd.DataFrame:
        """Per done point, retention outcome and year of its window: the outcome
        evaluated over that year alone (GLOSSARY.md "Year profile")."""
        return self._get("profiles", refresh)

    def category_table(self) -> pd.DataFrame:
        """Per category in chain order: ``sector_from``, ``sector_to``, ``valuation`` and
        ``no_retention`` (``"signed"``, ``"zero prior"`` or ``""``)."""
        return self._get("category_table")

    def outcome_table(self) -> pd.DataFrame:
        """Per scored outcome (every :class:`MaxAbs` part, no :class:`MaxAbs`): measure,
        window, valuation (``"mixed"`` across several), the category whose pooled edges
        it is (``""`` for any other set), the smallest limit of the criteria that judge
        it and their items."""
        return self._get("outcome_table")

    def outcomes(self) -> list:
        """The scalar outcome columns of :meth:`points_frame` that are ranked: every
        numeric column but the run's bookkeeping and the :class:`MaxAbs` outcomes."""
        df = self.points_frame()
        skip = ({"index", "status"} | {k for ax in self.axes for k in ax["keys"]}
                | set(self._max_abs))
        return [c for c in df.columns if c not in skip and not c.startswith("run_")
                and pd.api.types.is_numeric_dtype(df[c])]

    def window_of(self, outcome: Optional[str]) -> str:
        """An outcome's window as ``first–last`` (one year: ``first``); the base's full
        span for None or a column that is not a declared outcome."""
        self._ensure()
        return self._windows.get(outcome, self._full_window)

    def _done(self) -> pd.DataFrame:
        df = self.points_frame()
        return df[df["status"] == "done"].reset_index(drop=True)

    def sensitivity(self) -> pd.DataFrame:
        """:func:`sensitivity_table` over every ranked outcome of the done points, with
        the outcome's window."""
        t = sensitivity_table(self._done(), self.axes, self.outcomes(), self.mode)
        t.insert(1, "window", [self.window_of(o) for o in t["outcome"]])
        return t

    def sensitivity_by_year(self) -> pd.DataFrame:
        """:func:`sensitivity_by_year` over the year profiles of the done points."""
        keys = ["index"] + [k for ax in self.axes for k in ax["keys"]]
        return sensitivity_by_year(self._done()[keys], self.axes, self.year_profiles(),
                                   self.mode)

    def criterion_frame(self) -> pd.DataFrame:
        """Per declared criterion and done point: the largest |outcome| (``value``), the
        outcome that set it and whether the point meets the limit (a NaN never does)."""
        done = self._done()
        rows = []
        for c in self.criteria:
            names = list(dict.fromkeys(q.name for o in c.outcomes for q in _parts(o)))
            v = np.abs(done[names].to_numpy(dtype=float))
            for i in range(len(done)):
                finite = np.isfinite(v[i])
                j = int(np.argmax(np.where(finite, v[i], -np.inf))) if finite.any() else 0
                val = float(v[i, j]) if finite.all() else np.nan
                rows.append((c.item, int(done["index"].iloc[i]), val, names[j],
                             bool(val < c.limit)))
        return pd.DataFrame(rows, columns=["item", "index", "value", "outcome", "meets"])

    def verdicts(self) -> pd.DataFrame:
        """One row per declared criterion: its limit, the done points that meet it, and
        the worst point, the outcome that set it, that outcome's window and value."""
        cf = self.criterion_frame()
        rows = []
        for c in self.criteria:
            t = cf[cf["item"] == c.item]
            w = t.loc[t["value"].idxmax()] if t["value"].notna().any() else None
            rows.append((c.item, c.limit, int(t["meets"].sum()), len(t),
                         None if w is None else int(w["index"]),
                         None if w is None else w["outcome"],
                         None if w is None else self.window_of(w["outcome"]),
                         np.nan if w is None else float(w["value"])))
        return pd.DataFrame(rows, columns=[
            "item", "limit", "points_meeting", "points_done", "worst_point",
            "worst_outcome", "worst_window", "worst_value"])

    def write(self, odir: str) -> list:
        """Write points.csv, categories.csv, retention_by_year.csv, sensitivity.csv,
        year_profiles.csv, sensitivity_by_year.csv and verdicts.csv."""
        os.makedirs(odir, exist_ok=True)
        out = []
        pts = self.points_frame()
        for c in pts.columns:
            if pts[c].map(lambda v: isinstance(v, (list, tuple, dict))).any():
                pts[c] = pts[c].map(repr)
        for name, df in (("points.csv", pts), ("categories.csv", self.category_frame()),
                         ("retention_by_year.csv", self.retention_by_year()),
                         ("sensitivity.csv", self.sensitivity()),
                         ("year_profiles.csv", self.year_profiles()),
                         ("sensitivity_by_year.csv", self.sensitivity_by_year()),
                         ("verdicts.csv", self.verdicts())):
            path = os.path.join(odir, name)
            df.to_csv(path, index=False)
            out.append(path)
        return out
