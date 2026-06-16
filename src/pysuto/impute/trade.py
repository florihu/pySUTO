"""
trade.py
========
Trade-seed imputer (ADR 0010).

:class:`TradeImputer` seeds the **copper-content (CC)** value of cross-region
``Flow→Process`` import legs from their already-aggregated BACI **Mass** twin:
``f_CC = conc·f_Mass``. It runs **first** in the cascade so the
:class:`~pysuto.impute.network.NetworkImputer`'s recipe gate (which treats
cross-region imports as 0-when-missing) can pool the covered import legs, and the
flow-balance can close the lone domestic leg from the upstream product.

Ownership: TradeImputer writes **only** cross-region (``from_region != to_region``)
``Flow→Process`` CC edges — the complement of the NetworkImputer's same-region
territory, so the two never contend for an edge. Routes with no BACI Mass
observation are left NaN (an honest flat-prior gap the solver can widen), never
fabricated. The Mass twin it reads is already in ``f_agg`` on covered legs, so
there is no new artifact and no feed access; the imputer is a no-op when conc or
BACI are absent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pysuto.logging import get_logger
from pysuto.lp import lp_combine
from pysuto.param_relation import ParamRelationResolver, conc_twin_map, get_edge_cache

from .base import BaseImputer, _apply_cv_cap, _log

logger = get_logger(__name__)


class TradeImputer(BaseImputer):
    """Seed cross-region ``Flow→Process`` CC legs from the BACI Mass twin."""

    def __init__(
        self,
        conc_csv: str,
        p: float = 1.3,
        cv_cap: float = float("inf"),
        id: Optional[str] = None,
    ) -> None:
        self.id = id or type(self).__name__
        self._conc_csv = conc_csv
        self.p = float(p)
        self.cv_cap = float(cv_cap)
        # populated by prepare()
        self._cc_to_mass: dict[int, tuple[int, float, float]] = {}
        self._seed_edges: list[int] = []  # cross-region Flow→Process CC edges with a twin

    @property
    def source_paths(self) -> list[str]:
        return [self._conc_csv]

    def prepare(self, acc) -> None:
        get_edge_cache(acc)  # warm the per-acc edge cache used by the resolver
        resolver = ParamRelationResolver(acc)
        # CC edge -> (Mass twin, conc, σ_conc); conc only (ore_grade is a Mining-CO
        # concept, irrelevant to traded Flow→Process products).
        self._cc_to_mass = conc_twin_map(self._conc_csv, resolver)
        # restrict to cross-region Flow→Process CC edges — the network's complement.
        em = acc.edge_meta
        eligible = set(
            int(e) for e in em.loc[
                (em["from_region"].to_numpy() != em["to_region"].to_numpy())
                & (em["from_entity"].to_numpy() == "Flow")
                & (em["to_entity"].to_numpy() == "Process")
                & (em["valuation"].to_numpy() == "CC"),
                "edge_id",
            ].to_numpy()
        )
        self._seed_edges = [cc for cc in self._cc_to_mass if cc in eligible]

    def impute(self, f, sigma, missing, step, acc_struct) -> list[dict]:
        entries: list[dict] = []
        p = self.p
        for cc in self._seed_edges:
            if not np.isnan(f[cc]):
                continue
            m, conc, conc_sig = self._cc_to_mass[cc]
            if np.isnan(f[m]):
                continue  # no BACI Mass observation on this route → leave NaN
            f_cc = float(conc * f[m])
            # σ for CC = conc·Mass, mirroring network._emit_twins (Mass = CC/conc)
            # inverted: ∂CC/∂Mass = conc, ∂CC/∂conc = Mass = f_cc/conc.
            s_cc = lp_combine([conc * sigma[m], abs(f_cc) * conc_sig / conc], p)
            s_cc, capped = _apply_cv_cap(s_cc, f_cc, self.cv_cap)
            f[cc] = f_cc
            sigma[cc] = s_cc
            entries.append(_log(
                int(cc), step,
                "trade_seed_capped" if capped else "trade_seed",
                [int(m)], [conc], f_cc, s_cc, "", None,
            ))
        return entries
