"""
parametric.py
=============
CSV-driven + lambda-based parametric imputer.

* :class:`ParametricRelation` — lambda directed fill: ``f_target = Σ coeff_i × f_source_i``.
* :class:`ParametricImputer`  — runs CSV bidirectional pairs + lambda directed
                                 relations, deduplicating by specificity.

Module-local helper :func:`_reg_sensitivity` (single caller —
:class:`ParametricImputer`'s backward branch) lives here too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from pysuto.logging import get_logger
from pysuto.lp import lp_combine
from pysuto.param_relation import (
    ParamRelationResolver,
    expand_edge_index,
    float_or,
    sigma_from_row,
)

from .base import BaseImputer, _apply_cv_cap, _log

logger = get_logger(__name__)


_AGG_TRUE = {"true", "1", "yes", "y", "t"}


def _parametric_consumes(row) -> bool:
    """True if ParametricImputer should build per-pair fills from this params row.

    The ``agg`` flag is the **relation-type marker** that partitions the shared
    params CSV between consumers (the single place this is decided): an
    ``agg=true`` row is a *pooled, set-scoped recipe relation* (wildcarded a-side,
    ``Σb = r·Σa`` per bucket) owned by ``NetworkImputer`` (Stage 2 prior) and the
    agg ratio-constraint builder (Stage 3); a non-agg row is a *per-pair relation*
    consumed here and by the per-pair ratio builder. ParametricImputer therefore
    skips agg rows (and the ``stripping_ratio`` family) — resolving a pooled
    wildcard row per-pair would, e.g., backfill a spurious ``Scrap = Matte / eff``
    from a broadened ``a_sector_from=*`` eff row. The eff/conc per-pair rows still flow.
    """
    if str(row.get("agg", "")).strip().lower() in _AGG_TRUE:
        return False
    if str(row.get("param_type", "")).strip() == "stripping_ratio":
        return False
    return True


def _reg_sensitivity(coeff: float, q: float, alpha_min: float) -> float:
    """
    Bounded effective sensitivity used in ParametricImputer σ propagation.

        s(α) = |α| / (|α|^q + α_min^q)^(1/q)

    Asymptotes to 1 for |α| >> α_min (caps coefficient amplification of σ),
    and to |α|/α_min for |α| << α_min (linearly damps very small coefficients).
    q controls the knee sharpness (q=3 is a moderate default).
    """
    a = abs(float(coeff))
    am = abs(float(alpha_min))
    if a == 0.0 and am == 0.0:
        return 0.0
    return a / (a ** q + am ** q) ** (1.0 / q)


# ---------------------------------------------------------------------------
# ParametricRelation  (lambda-based, directed)
# ---------------------------------------------------------------------------

@dataclass
class ParametricRelation:
    """
    Directed fill: f_target = Σ coeff_i × f_source_i

    target      Predicate over expanded edge_index (ei_exp) → bool mask.
                Identifies which edges to fill.
    sources     Predicate over expanded ei_exp → bool mask.
                Identifies candidate source edges.
                Within each target-edge context, sources are filtered to
                edges with the same region_from and time as the target.
    coefficients list[float] — one coefficient per matched source edge.
                If len == 1, the single coefficient is applied to the sum
                of all matched sources.
    epsilon     Relative model uncertainty added in quadrature.
    param_id    Optional string (logged as param_id in ImputationLog).
    """

    description: str
    target: Callable
    sources: Callable
    coefficients: list[float]
    epsilon: float = 0.0
    param_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ParametricImputer
# ---------------------------------------------------------------------------

class ParametricImputer(BaseImputer):
    """
    Two instantiation modes:

    1. from_csv(path, acc_struct)  — precomputes bidirectional (a_id, b_id, coeff, σ)
       pairs from a params-template CSV.  Each pair is filled in whichever
       direction the missing mask requires: f_b = coeff·f_a or f_a = f_b/coeff.

    2. ParametricImputer(relations=[...])  — directed fills from ParametricRelation
       lambda objects.  Sources are matched per target by region_from and time.
    """

    def __init__(
        self,
        relations: Optional[list[ParametricRelation]] = None,
        _csv_pairs: Optional[list] = None,
        _source_path: Optional[str] = None,
        cv_cap: float = float("inf"),
        p: float = 1.3,
        q: float = 3.0,
        alpha_min: float = 0.05,
        id: Optional[str] = None,
    ) -> None:
        self.id          = id or type(self).__name__
        self._relations  = relations or []
        # Each tuple: (a_id, b_id, coeff, sigma_param, epsilon, param_id, desc)
        # Note: per-row `bound_lower / bound_upper` from the CSV are *ratio bounds*
        # (consumed by RatioBoundBuilder in constraints.py); they are NOT absolute
        # caps on the filled target flow and are deliberately not read here.
        self._csv_pairs  = _csv_pairs or []
        self._source_path: Optional[str] = _source_path
        self._ei_exp:    Optional[pd.DataFrame] = None  # lazy cache for lambda relations
        self.cv_cap     = float(cv_cap)
        self.p          = float(p)
        self.q          = float(q)
        self.alpha_min  = float(alpha_min)

    @property
    def source_paths(self) -> list[str]:
        return [self._source_path] if self._source_path else []

    def config(self) -> dict:
        return {
            "cv_cap":         self.cv_cap,
            "p":              self.p,
            "q":              self.q,
            "alpha_min":      self.alpha_min,
            "source_path":    self._source_path,
            "n_csv_pairs":    len(self._csv_pairs),
            "n_relations":    len(self._relations),
            "relations_desc": [r.description for r in self._relations],
        }

    @classmethod
    def from_csv(
        cls,
        path: str,
        acc_struct,
        cv_cap: float = float("inf"),
        p: float = 1.3,
        q: float = 3.0,
        alpha_min: float = 0.05,
        id: Optional[str] = None,
    ) -> "ParametricImputer":
        """
        Load a params-template CSV (semicolon-delimited) and precompute
        (edge_a_id, edge_b_id) pairs for bidirectional imputation.

        Comment lines (starting with #) and rows with blank param_id are skipped.
        """
        df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype=str)
        df = df.dropna(subset=["param_id"])
        df = df[~df["param_id"].str.startswith("#")]

        resolver = ParamRelationResolver(acc_struct)

        # dict[(a_id, b_id)] → (pair_tuple, specificity)
        # The resolver yields every (a, b) match; this imputer keeps only
        # the most specific row per pair so a wildcard row never overrides
        # a deliberately narrow one.
        best: dict[tuple, tuple] = {}

        for _, row in df.iterrows():
            if not _parametric_consumes(row):
                continue
            nominal = float(row["value_nominal"])
            try:
                sigma_p = sigma_from_row(row, nominal)
            except Exception as exc:
                logger.warning("param_id=%s: sigma error (%s) — skipped", row["param_id"], exc)
                continue
            epsilon  = float_or(row.get("epsilon"), 0.0)
            # bound_lower / bound_upper are *ratio* bounds — consumed by
            # RatioBoundBuilder for constraint building, not by this imputer.
            param_id = str(row["param_id"])
            desc     = str(row.get("description", ""))
            n_pairs  = 0

            for rr in resolver.resolve_row(row):
                n_pairs += 1
                key  = (rr.a_edge_id, rr.b_edge_id)
                pair = (rr.a_edge_id, rr.b_edge_id, nominal, sigma_p, epsilon, param_id, desc)
                if key not in best or rr.specificity > best[key][1]:
                    if key in best:
                        logger.debug(
                            "param_id=%s: replacing lower-specificity pair %s (spec %d→%d)",
                            param_id, key, best[key][1], rr.specificity,
                        )
                    best[key] = (pair, rr.specificity)

            if n_pairs == 0:
                logger.warning("param_id=%s: no A-edges matched", param_id)

        pairs = [v[0] for v in best.values()]
        logger.info("from_csv: loaded %d bidirectional pairs from %s", len(pairs), path)
        return cls(
            _csv_pairs=pairs,
            _source_path=str(path),
            cv_cap=cv_cap,
            p=p,
            q=q,
            alpha_min=alpha_min,
            id=id,
        )

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        entries: list[dict] = []
        cv_cap  = self.cv_cap
        p       = self.p
        q       = self.q
        am      = self.alpha_min

        # ── CSV pairs (bidirectional) ──────────────────────────────────
        for a_id, b_id, coeff, sig_p, epsilon, param_id, desc in self._csv_pairs:
            a_nan = np.isnan(f[a_id])
            b_nan = np.isnan(f[b_id])

            if not a_nan and b_nan:
                f_b = float(coeff * f[a_id])
                # Forward (A→B): bare coefficient. σ_b = Lp(|coeff|·σ_a, σ_param, ε·f_b)
                s_b   = lp_combine(
                    [coeff * sigma[a_id], sig_p, epsilon * f_b], p,
                )
                s_b, capped = _apply_cv_cap(s_b, f_b, cv_cap)
                f[b_id]     = f_b
                sigma[b_id] = s_b
                entries.append(_log(
                    b_id, step,
                    "parametric_capped" if capped else "parametric",
                    [a_id], [coeff], f_b, s_b, desc, param_id,
                ))

            elif a_nan and not b_nan and coeff != 0.0:
                # coeff == 0 (e.g. concwruse: WR is copperless) makes the backward
                # inversion A = B/coeff undefined — A's Mass comes from elsewhere
                # (the Mining CO-split), never from inverting a zero ratio.
                inv = 1.0 / coeff
                f_a = float(inv * f[b_id])
                # Backward (B→A): regularized sensitivity caps σ amplification
                # when |inv| ≫ 1 (e.g. 1/ore_grade ≈ 100s).
                s_inv = _reg_sensitivity(inv, q, am)
                s_a   = lp_combine(
                    [s_inv * sigma[b_id], sig_p, epsilon * f_a], p,
                )
                s_a, capped = _apply_cv_cap(s_a, f_a, cv_cap)
                f[a_id]     = f_a
                sigma[a_id] = s_a
                entries.append(_log(
                    a_id, step,
                    "parametric_capped" if capped else "parametric",
                    [b_id], [inv], f_a, s_a, desc, param_id,
                ))

        # ── Lambda relations (directed: target ← sources) ─────────────
        if self._relations:
            if self._ei_exp is None:
                self._ei_exp = expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
            ei_exp = self._ei_exp
            for rel in self._relations:
                t_mask = rel.target(ei_exp)
                t_ids  = ei_exp[t_mask]["edge_id"].values

                for t_id in t_ids:
                    if not np.isnan(f[t_id]):
                        continue

                    t_row = ei_exp[ei_exp["edge_id"] == t_id].iloc[0]
                    t_rfrom = t_row["region_from"]
                    t_time  = t_row["time"]

                    # Sources filtered to same region_from and same time
                    s_mask = (
                        rel.sources(ei_exp)
                        & (ei_exp["region_from"].astype(str) == str(t_rfrom))
                        & (ei_exp["time"].astype(str) == str(t_time))
                    )
                    s_ids = ei_exp[s_mask]["edge_id"].values

                    if len(s_ids) == 0:
                        continue

                    # Resolve coefficients
                    if len(rel.coefficients) == 1:
                        coeffs = [rel.coefficients[0]] * len(s_ids)
                    elif len(rel.coefficients) == len(s_ids):
                        coeffs = list(rel.coefficients)
                    else:
                        logger.warning(
                            "ParametricRelation '%s': %d coefficients but %d sources "
                            "for target edge %d — skipped",
                            rel.description, len(rel.coefficients), len(s_ids), t_id,
                        )
                        continue

                    if any(np.isnan(f[s]) for s in s_ids):
                        continue

                    f_t = float(sum(c * f[s] for c, s in zip(coeffs, s_ids)))
                    # Lambda (target ← sources): bare coefficients (forward by definition)
                    # σ_t = ( Σ_l |α_l·σ_l|^p + |ε·f_t|^p )^(1/p)
                    terms = [c * sigma[s] for c, s in zip(coeffs, s_ids)]
                    terms.append(rel.epsilon * f_t)
                    s_t = lp_combine(terms, p)
                    s_t, capped = _apply_cv_cap(s_t, f_t, cv_cap)

                    f[t_id]     = f_t
                    sigma[t_id] = s_t
                    entries.append(_log(
                        t_id, step,
                        "parametric_capped" if capped else "parametric",
                        list(s_ids), coeffs, f_t, s_t,
                        rel.description, rel.param_id,
                    ))

        return entries


