"""
sweep.py
========
:class:`SweepStatsDiagnostic` — the tables behind :class:`pysuto.diag.SweepDiag` (#107).

A sweep's points are scored against its **base chain** (GLOSSARY.md "Sweep",
"Point", "Outcome"). Every point gets

* the **cross-evaluated objective** — ``J_base_at_point = Σ_t J_base(a*_point(t)) −
  Σ_t J_base(a*_base(t))``: each point's answer scored under the base QP
  (:meth:`pysuto.solver.ClarabelSolver.objective_at`), ≥ 0 while the point is feasible
  for the base's hard rows (``max_hard_violation_base`` says whether it is);
* the **posterior distance** ``mass_moved = Σ_years Σ |a*_point − a*_base|`` (kt), in
  total and per valuation, and per category in the category tables with the
  **retention** ``Σa*/Σa_prior`` (GLOSSARY.md "Retention"); a category is
  ``sector_from→sector_to valuation``;
* every declared :class:`Outcome` and :class:`MaxAbs` — any edge set, three measures.

Only the base's const run and the points' opt runs are read, so a sweep that kept only
Stage 4 (#108) is scored like any other.

:func:`sensitivity_table` ranks the axes per outcome: in a random sweep the
standardised rank regression coefficient (SRRC) and Spearman's ρ of each numeric axis,
for every list axis the main-effect range (spread of the outcome's mean per level), and
in a full grid the two-way interaction range.
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


# =============================================================================
# Outcomes
# =============================================================================

@dataclass(frozen=True)
class Outcome:
    """A named scalar of a point's answer against the base's (GLOSSARY.md "Outcome").

    Parameters
    ----------
    name : the column it becomes in :meth:`SweepStatsDiagnostic.points_frame`.
    measure : over the selected edges, every year pooled —
        ``"mass_moved"`` ``Σ|a*_p − a*_b|`` (kt);
        ``"retention"`` ``Σa*_p/Σa_prior,p − Σa*_b/Σa_prior,b``, the change in retention
        against the base;
        ``"total"`` ``Σa*_p/Σa*_b − 1``, the relative change of the posterior, which
        also sees a knob that moves prior and posterior together.
    select : the edges — a dict ``column → value or list of values`` over the edge table
        (``sector_from``, ``sector_to``, ``region_from``, ``region_to``, ``valuation``,
        ``time``, ``category``), a callable ``edges → bool mask``, or None for every edge.
    """
    name:    str
    measure: str
    select:  Any = None

    def __post_init__(self) -> None:
        if self.measure not in MEASURES:
            raise ValueError(f"outcome {self.name!r}: measure {self.measure!r}; expected "
                             f"one of {MEASURES}")


@dataclass(frozen=True)
class MaxAbs:
    """The largest |value| among ``parts`` — the worst of several outcomes."""
    name:  str
    parts: tuple


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


def _measure(o: Outcome, m: np.ndarray, F_p, P_p, F_b, P_b) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        if o.measure == "mass_moved":
            return float(np.abs(F_p[m] - F_b[m]).sum())
        if o.measure == "retention":
            return float(F_p[m].sum() / P_p[m].sum() - F_b[m].sum() / P_b[m].sum())
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


def sensitivity_table(frame: pd.DataFrame, axes: Sequence[dict],
                      outcomes: Sequence[str], mode: str) -> pd.DataFrame:
    """Sensitivity of each outcome column to each axis.

    ``frame`` holds one row per scored point with a column per axis (named by the
    axis's first key) and the outcome columns; ``axes`` are the declaration's axis
    dicts (``keys`` plus ``values`` or ``distribution``). Rows with a non-finite
    outcome are dropped per outcome. Returns ``outcome, axis, statistic, value, n``
    with statistic ``srrc`` / ``spearman`` (numeric axes and two-level list axes, ranked
    in their declared order, random mode) and the rank regression's ``r2`` (axis
    ``(rank regression)``), ``main_effect_range`` (list axes)
    and ``interaction_range`` (pairs of list axes in a full grid, axis ``a×b``); NaN
    where a statistic is undefined. A low ``r2`` says the ranks are not additive in
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
            lev = _levels(ax, cols[axis_label(ax)])
            if lev is None:
                continue
            means = pd.Series(y).groupby(lev).mean()
            spread = float(means.max() - means.min()) if len(means) > 1 else np.nan
            rows.append((out, axis_label(ax), "main_effect_range", spread, n))

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


def top_axis(sens: pd.DataFrame, outcome: str, r2_min: float = R2_MIN) -> Optional[tuple]:
    """The axis that moves ``outcome`` most in a random sweep: by |SRRC| where the rank
    regression's R² reaches ``r2_min``, else by |Spearman's ρ|. Returns ``(axis,
    statistic, value)``, None when no axis has that statistic."""
    t = sens[sens["outcome"] == outcome]
    r2 = t[t["statistic"] == "r2"]["value"]
    stat = "srrc" if len(r2) and r2.iloc[0] >= r2_min else "spearman"
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
        least ``category`` and ``valuation`` plus any column a selector reads; default
        from the base const run's accounting structure (the expanded edge index, its
        ``time``, and ``category`` = ``sector_from→sector_to``).
    outcomes : the declared :class:`Outcome` and :class:`MaxAbs` columns.
    """

    def __init__(self, sweep_run_id: str, root: str = os.path.join("data", "proc"),
                 edges: Optional[pd.DataFrame] = None,
                 outcomes: Sequence[Union[Outcome, MaxAbs]] = ()) -> None:
        reg = SweepRunRegistry(os.path.join(root, "sweep"))
        self.sweep_run_id = sweep_run_id
        self.root = root
        self.manifest = reg.load_manifest(sweep_run_id)
        self.sidecar  = reg.load_sidecar(sweep_run_id)
        self.declaration = self.manifest["declaration"]
        up = self.sidecar["parents"]["upstream"]
        self.base_const, self.base_opt = up["const_run_id"], up["opt_run_id"]
        self._edges = edges
        self.outcome_defs = tuple(outcomes)
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
        return self._edges

    # ------------------------------------------------------------------
    def _compute(self) -> None:
        solver = self._solver()
        base_run = solver.load_run(self.base_const)
        base_res = self._opt(self.base_opt)
        edges = self._edge_table(base_run)
        n_e = base_run.G.shape[1]
        if len(edges) != n_e:
            raise ValueError(f"the edge table has {len(edges)} rows, the base run "
                             f"{n_e} edges")
        masks = {o.name: _mask(o.name, o.select, edges) for o in self._leaf_outcomes()}
        cat = (edges["category"].astype(str) + " "
               + edges["valuation"].astype(str)).to_numpy()
        cats, cat_code = np.unique(cat, return_inverse=True)
        vals, val_code = np.unique(edges["valuation"].astype(str).to_numpy(),
                                   return_inverse=True)
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

        def ratio(num, den):
            with np.errstate(divide="ignore", invalid="ignore"):
                return num / den

        F_b, P_b = full(base_res, "F_opt"), full(base_res, "a_prior")
        J_b = {t: solver.objective_at(base_run, t, base_res[t].F_opt, base_res[t].a_prior)
               for t in years}
        self.base_max_hard_violation = max(J_b[t]["max_hard_violation"] for t in years)
        ret_b = ratio(by(cat_code, len(cats), F_b), by(cat_code, len(cats), P_b))
        cy = cat_code * len(years) + year_of
        ret_by_b = ratio(by(cy, len(cats) * len(years), F_b),
                         by(cy, len(cats) * len(years), P_b))

        point_rows, cat_rows, year_rows = [], [], []
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
            leaf = {o.name: _measure(o, masks[o.name], F_p, P_p, F_b, P_b)
                    for o in self._leaf_outcomes()}
            for o in self.outcome_defs:
                row[o.name] = (_max_abs([leaf[q.name] for q in o.parts])
                               if isinstance(o, MaxAbs) else leaf[o.name])

            m_cat = by(cat_code, len(cats), moved)
            ret_p = ratio(by(cat_code, len(cats), F_p), by(cat_code, len(cats), P_p))
            for c, m, rp, rb in zip(cats, m_cat, ret_p, ret_b):
                cat_rows.append((pt["index"], c, float(m), float(rp), float(rb)))
            ret_by_p = ratio(by(cy, len(cats) * len(years), F_p),
                             by(cy, len(cats) * len(years), P_p))
            for j in range(len(cats) * len(years)):
                year_rows.append((pt["index"], cats[j // len(years)], years[j % len(years)],
                                  float(ret_by_p[j]), float(ret_by_b[j])))
            point_rows.append(row)

        self._cache = {
            "points": pd.DataFrame(point_rows),
            "categories": pd.DataFrame(cat_rows, columns=[
                "index", "category", "mass_moved", "retention", "retention_base"]),
            "years": pd.DataFrame(year_rows, columns=[
                "index", "category", "year", "retention", "retention_base"]),
        }

    def _leaf_outcomes(self) -> list:
        """The declared outcomes and the parts of every :class:`MaxAbs`, once each."""
        out: dict = {}
        for o in self.outcome_defs:
            for q in (o.parts if isinstance(o, MaxAbs) else (o,)):
                if out.setdefault(q.name, q) != q:
                    raise ValueError(f"two different outcomes are named {q.name!r}")
        return list(out.values())

    def _get(self, key: str, refresh: bool) -> pd.DataFrame:
        if refresh or not self._cache:
            self._compute()
        return self._cache[key].copy()

    # ------------------------------------------------------------------
    def points_frame(self, refresh: bool = False) -> pd.DataFrame:
        """One row per point: status, run ids, axis values, scalar outcomes."""
        return self._get("points", refresh)

    def category_frame(self, refresh: bool = False) -> pd.DataFrame:
        """Per done point and category: mass moved, retention, the base's retention."""
        return self._get("categories", refresh)

    def retention_by_year(self, refresh: bool = False) -> pd.DataFrame:
        """Per done point, category and year: retention and the base's."""
        return self._get("years", refresh)

    def outcomes(self) -> list:
        """The scalar outcome columns of :meth:`points_frame`."""
        df = self.points_frame()
        skip = {"index", "status"} | {k for ax in self.axes for k in ax["keys"]}
        return [c for c in df.columns if c not in skip and not c.startswith("run_")
                and pd.api.types.is_numeric_dtype(df[c])]

    def sensitivity(self) -> pd.DataFrame:
        """:func:`sensitivity_table` over every outcome of the done points."""
        df = self.points_frame()
        done = df[df["status"] == "done"].reset_index(drop=True)
        return sensitivity_table(done, self.axes, self.outcomes(), self.mode)

    def write(self, odir: str) -> list:
        """Write points.csv, categories.csv, retention_by_year.csv, sensitivity.csv."""
        os.makedirs(odir, exist_ok=True)
        out = []
        pts = self.points_frame()
        for c in pts.columns:
            if pts[c].map(lambda v: isinstance(v, (list, tuple, dict))).any():
                pts[c] = pts[c].map(repr)
        for name, df in (("points.csv", pts), ("categories.csv", self.category_frame()),
                         ("retention_by_year.csv", self.retention_by_year()),
                         ("sensitivity.csv", self.sensitivity())):
            path = os.path.join(odir, name)
            df.to_csv(path, index=False)
            out.append(path)
        return out
