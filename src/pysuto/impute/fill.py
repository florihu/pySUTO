"""
fill.py
=======
:class:`FillImputer` — one :class:`~pysuto.relation.sources.Fill` of the params relation
source as a cascade imputer (ADR 0010, 0011).

A pair relation ``b − r·a = 0`` is filled where one side is known and the other is not:
``forward`` writes ``b = r·a``, ``backward`` writes ``a = b/r``. The value and provenance
come from the relation; the σ of the filled edge is the fill's own rule (unifying them is
a measured, separate change):

* ``"parametric"`` — conc, ore-grade and efficiency pairs, every term in the filled edge's
  flow units (kt): forward ``Lp(|r|·σ_a, |f_a|·σ_param, ε·f_b)``, backward
  ``Lp(s(1/r)·σ_b, |f_a|·σ_param/|r|, ε·f_a)`` with the bounded sensitivity ``s``
  (:func:`~pysuto.impute.base._reg_sensitivity`); methods ``parametric`` /
  ``parametric_capped`` (#148).
* ``"trade"`` — the cross-region trade-seed of the CC leg from its BACI Mass twin, forward
  only: ``Lp(r·σ_m, |f_cc|·σ_param/r)``, no ε term; methods ``trade_seed`` /
  ``trade_seed_capped`` (ADR 0010).
"""

from __future__ import annotations

import numpy as np

from pysuto.lp import P_DEFAULT, lp_combine
from pysuto.relation.sources import Fill

from .base import BaseImputer, _apply_cv_cap, _reg_sensitivity

SIGMA_RULES = ("parametric", "trade")


class FillImputer(BaseImputer):
    """Run one :class:`~pysuto.relation.sources.Fill` with a σ rule (module docstring)."""

    def __init__(self, fill: Fill, sigma_rule: str = "parametric", p: float = P_DEFAULT,
                 q: float = 3.0, alpha_min: float = 0.05,
                 cv_cap: float = float("inf")) -> None:
        if sigma_rule not in SIGMA_RULES:
            raise ValueError(f"sigma_rule {sigma_rule!r} must be one of {SIGMA_RULES}")
        if sigma_rule == "trade" and fill.rule != "forward":
            raise ValueError("the trade σ rule is defined for a forward fill only")
        self.id         = fill.id or type(self).__name__
        self.fill       = fill
        self.sigma_rule = sigma_rule
        self.p          = float(p)
        self.q          = float(q)
        self.alpha_min  = float(alpha_min)
        self.cv_cap     = float(cv_cap)

    @property
    def source_paths(self) -> list[str]:
        return [self.fill.source.path] if self.fill.source.path else []

    def config(self) -> dict:
        return {
            "rule":        self.fill.rule,
            "sigma_rule":  self.sigma_rule,
            "cv_cap":      self.cv_cap,
            "p":           self.p,
            "q":           self.q,
            "alpha_min":   self.alpha_min,
            "source_path": self.fill.source.path,
            "n_relations": len(self.fill.source),
        }

    def writer_scope(self, acc) -> np.ndarray:
        return self.fill.writer_scope(acc.n_edges)

    def impute(self, ledger, step: int, acc_struct) -> None:
        f, sigma = ledger.f, ledger.sigma
        fill, p, cv_cap = self.fill, self.p, self.cv_cap
        fwd = fill.rule in ("forward", "both")
        bwd = fill.rule in ("backward", "both")
        trade = self.sigma_rule == "trade"
        name = "trade_seed" if trade else "parametric"
        for rel in fill.source.relations:
            b_id, a_id = int(rel.edges[0]), int(rel.edges[1])
            coeff   = float(-rel.coefficients[1])
            sig_p   = rel.meta["sigma_param"]
            epsilon = rel.meta["epsilon"]
            desc, param_id = ("", None) if trade else (rel.meta["description"], rel.source)
            a_nan = np.isnan(f[a_id])
            b_nan = np.isnan(f[b_id])

            if fwd and fill.allows(b_id) and not a_nan and b_nan:
                f_b = float(coeff * f[a_id])
                if trade:
                    # σ for CC = conc·Mass, the Mass twin (CC/conc) inverted:
                    # ∂CC/∂Mass = conc, ∂CC/∂conc = Mass = f_cc/conc.
                    s_b = lp_combine(
                        [coeff * sigma[a_id], abs(f_b) * sig_p / coeff], p,
                    )
                else:
                    # Forward: bare coefficient; σ_param's first-order image in flow
                    # units is |f_a|·σ_param, defined for coeff == 0 as well (#148).
                    s_b = lp_combine(
                        [coeff * sigma[a_id], abs(f[a_id]) * sig_p, epsilon * f_b], p,
                    )
                s_b, capped = _apply_cv_cap(s_b, f_b, cv_cap)
                ledger.write(
                    b_id, f_b, s_b, step, name + "_capped" if capped else name,
                    [a_id], [coeff], desc, param_id,
                )

            elif bwd and fill.allows(a_id) and a_nan and not b_nan and coeff != 0.0:
                # coeff == 0 (e.g. concwruse: WR is copperless) makes A = B/coeff
                # undefined — A's Mass comes from elsewhere, never from a zero ratio.
                inv = 1.0 / coeff
                f_a = float(inv * f[b_id])
                # Backward: the regularised sensitivity caps σ amplification when
                # |inv| ≫ 1 (e.g. 1/ore_grade ≈ 100s); |f_a|·σ_param/|coeff| is the
                # parameter term in A's flow units (#148).
                s_inv = _reg_sensitivity(inv, self.q, self.alpha_min)
                s_a   = lp_combine(
                    [s_inv * sigma[b_id], abs(f_a) * sig_p / abs(coeff),
                     epsilon * f_a], p,
                )
                s_a, capped = _apply_cv_cap(s_a, f_a, cv_cap)
                ledger.write(
                    a_id, f_a, s_a, step, name + "_capped" if capped else name,
                    [b_id], [inv], desc, param_id,
                )
