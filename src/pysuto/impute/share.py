"""
share.py
========
``ShareImputer`` — the net-stock seed. A signed stock-change leg (``Cathode→MESC``,
``Cathode→USC``, ``CP→PSC``) that no feed reports is centred at 0 and held there:
``f[net] = 0``, ``σ = max(seed_cv·|f[base]|, seed_abs_floor)`` — replacing the
``noisy_zero`` flat prior. The stock legs are the only signed edges of the account,
so any looseness makes them a sink for every contradiction of the cathode balance
(Round X, ADR 0040); the seed is therefore priced like a feed-backed level (ADR 0025's
scale-total rule): 1 % of the co-located base flow, floored at the feeds' 0.3 kt
resolution. The empirical p95 band (p95 × a 100 kt floor) that priced the legs before
gave the exchange-location legs σ ≈ 553 kt and the solver +9.7 Mt of phantom exchange
stock over 30 years.

base = the co-located ``Cathode(i)→FU(j)`` consumption edge for MESC/USC (a no-trade
pair is pinned at the floor) and the same region's ``CP→Cathode`` product for PSC
(Round Y, ADR 0041). The share transfer that once priced PSC from its reported years is
retired: the recipe pre-empted it on every chain, and carrying a signed stock change
across years is what ADR 0040 bars. A reported value is never touched; the Mass twin of
a reported CC value is the conc ParametricImputer's, not this imputer's.

Runs as a terminal imputer (after time interpolation, before the flat prior) so the
base flows are already covered.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pysuto.param_relation import expand_edge_index

from .base import BaseImputer, _log

# (sector_from, entity_from, sector_to, entity_to) selectors: net leg → its base.
_NET = {
    "psc":  ("CP",      "Process", "PSC",  "SB"),
    "mesc": ("Cathode", "Flow",    "MESC", "SB"),
    "usc":  ("Cathode", "Flow",    "USC",  "SB"),
}
_BASE = {
    # PSC base = the cathode-pool product, same region (Round Y, ADR 0041): the seed
    # prices the leg at 1 % of the producer's throughput in either valuation.
    "psc":  ("CP",      "Process", "Cathode", "Flow"),
    # MESC/USC base = final-use consumption, co-located by (region_from, region_to).
    "mesc": ("Cathode", "Flow",    "FU",      "SB"),
    "usc":  ("Cathode", "Flow",    "FU",      "SB"),
}


class ShareImputer(BaseImputer):
    """Hold every unreported stock-change leg at a zero-mean seed priced on its
    co-located base flow (ADR 0040; PSC since ADR 0041). Runs late (after time
    interpolation, before the flat-prior fallback) so the bases are covered."""

    def __init__(
        self,
        p: float = 1.3,
        id: Optional[str] = None,
        seed_cv: float = 0.01,
        seed_abs_floor: float = 0.3,
    ) -> None:
        self.id             = id or "share"
        self.p              = float(p)
        # seed σ = max(seed_cv·|base|, seed_abs_floor) — kt (ADR 0040)
        self.seed_cv        = float(seed_cv)
        self.seed_abs_floor = float(seed_abs_floor)

    def config(self) -> dict:
        return {"p": self.p, "seed_cv": self.seed_cv,
                "seed_abs_floor": self.seed_abs_floor}

    # ------------------------------------------------------------------
    def impute(self, f, sigma, missing, step, acc_struct) -> list[dict]:
        ei = expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        entries: list[dict] = []
        for ind in _NET:
            entries += self._seed(ind, f, sigma, ei, step)
        return entries

    # ------------------------------------------------------------------
    @staticmethod
    def _sel(ei, spec):
        s_from, e_from, s_to, e_to = spec
        return ei[(ei["sector_from"] == s_from) & (ei["entity_from"] == e_from)
                  & (ei["sector_to"] == s_to) & (ei["entity_to"] == e_to)]

    # ---- zero-mean seed per co-located edge, held at 1 % / 0.3 kt ----
    def _seed(self, ind, f, sigma, ei, step) -> list[dict]:
        keys = ["region_from", "region_to", "time", "valuation"]
        net  = self._sel(ei, _NET[ind])[["edge_id"] + keys]
        base = self._sel(ei, _BASE[ind])[["edge_id"] + keys].rename(
            columns={"edge_id": "base_id"})
        m = net.merge(base, on=keys, how="inner")
        if m.empty:
            return []

        net_ids  = m["edge_id"].to_numpy(dtype=int)
        base_ids = m["base_id"].to_numpy(dtype=int)

        fill = (np.isnan(f[net_ids]) & ~np.isnan(f[base_ids]))
        e_fill, b_fill = net_ids[fill], base_ids[fill]
        s_new = np.maximum(self.seed_cv * np.abs(f[b_fill]), self.seed_abs_floor)
        f[e_fill], sigma[e_fill] = 0.0, s_new
        desc = f"{ind.upper()} zero-mean, σ=max(cv·|base|, floor)"
        return [_log(int(e), step, "share_seed", [int(b)], [0.0], 0.0, float(s), desc,
                     f"seed_{ind}")
                for e, b, s in zip(e_fill, b_fill, s_new)]
