"""
share.py
========
``ShareImputer`` — impute net-stock-change (Δstock) edges as a share of a
well-covered *base* flow, so cold runs get smooth, throughput-scaled priors
instead of ``noisy_zero`` flat priors. The flat prior (f=0, σ=1e6) leaves the
Part-A soft-bound σ_c under-scaled (its den_scale reads the *prior* throughput)
and jags the cold time series; a throughput-scaled prior fixes both.

Two per-indicator modes:

* ``"transfer"`` (PSC): ``share(t) = f[net(t)] / f[base(t)]`` computed where the
  net edge is already covered, interpolated across the region's time series and
  extrapolated to the boundaries (so 1995 — an ICSG baseline year with no NAS —
  is seeded), then ``f[net(t)] = share(t)·f[base(t)]`` for the still-missing
  years. Data-backed (real per-region PSC observations). base = CSXEW+CER inputs
  to the CP pool (= total cathode production; NOT CP→Cathode, which is
  production−PSC and circular).

* ``"seed"`` (MESC / USC): the net change is centred at 0 and held there —
  ``f[net]=0``, ``σ = max(seed_cv·|f[base_colocated]|, seed_abs_floor)`` — replacing
  ``noisy_zero``. MESC/USC are 47×47 source→dest matrices whose signed central
  share ≈ 0, so a zero-mean seed is the honest informative prior, and a stock
  leg is the one signed edge of the account: any looseness makes it a sink for
  every contradiction of the cathode balance (Round X; ADR 0040). The seed is
  therefore priced like a feed-backed level (ADR 0025's scale-total rule, 1 % of
  the co-located Cathode(i)→FU(j) flow, floored at the feeds' 0.3 kt resolution),
  not by the empirical p95 band — the band (p95 × a 100 kt floor) gave the hub
  legs σ ≈ 553 kt and the solver +9.7 Mt of phantom exchange stock over 30 years.
  base = the co-located Cathode(i)→FU(j) consumption edge; a no-trade pair is
  pinned at the floor.

PSC keeps ``σ = (p95/Z95)·base`` (the empirical band, consistent with the Part-A
soft bound); ``p95`` (value_max) is read per (indicator, region) from the
stock_bounds params CSV. MESC/USC read no p95 since Round X.
"""

from __future__ import annotations

import bisect
from typing import Optional

import numpy as np
import pandas as pd

from pysuto.logging import get_logger
from pysuto.param_relation import expand_edge_index, float_or

from .base import BaseImputer, _log

logger = get_logger(__name__)

#: p95 ≈ Z95·σ — read the symmetric ±p95 half-band as the ratio-space σ (matches
#: Z95_STOCK in constraints/ratio.py so the imputed prior and the soft bound agree).
Z95 = 1.645

# (sector_from, entity_from, sector_to, entity_to) selectors.
_NET = {
    "psc":  ("CP",      "Process", "PSC",  "SB"),
    "mesc": ("Cathode", "Flow",    "MESC", "SB"),
    "usc":  ("Cathode", "Flow",    "USC",  "SB"),
}
_BASE = {
    # PSC base = the cathode-pool product CP→Cathode, same region (Round Y, ADR 0041):
    # the seed prices the leg at 1 % of the producer's throughput in either valuation.
    "psc":  [("CP", "Process", "Cathode", "Flow")],
    # MESC/USC base = final-use consumption, co-located by (region_from, region_to).
    "mesc": [("Cathode", "Flow", "FU", "SB")],
    "usc":  [("Cathode", "Flow", "FU", "SB")],
}
_MODE = {"psc": "seed", "mesc": "seed", "usc": "seed"}


class ShareImputer(BaseImputer):
    """Impute Δstock edges as share×base (PSC) or a zero-mean throughput-scaled
    seed (MESC/USC). Runs late in the cascade (after time interpolation, before
    the flat-prior fallback) so the base flows are already covered."""

    def __init__(
        self,
        p95: dict,
        scale_abs_floor: float = 100.0,
        p: float = 1.3,
        id: Optional[str] = None,
        seed_cv: float = 0.01,
        seed_abs_floor: float = 0.3,
    ) -> None:
        self.id              = id or "share"
        self._p95            = p95           # dict[(indicator, region_to)] -> p95 (PSC)
        self.scale_abs_floor = float(scale_abs_floor)
        self.p               = float(p)
        # MESC/USC seed σ = max(seed_cv·|FU leg|, seed_abs_floor) — kt (ADR 0040).
        self.seed_cv         = float(seed_cv)
        self.seed_abs_floor  = float(seed_abs_floor)

    # ------------------------------------------------------------------
    @classmethod
    def from_stock_bounds_csv(
        cls, path: str, scale_abs_floor: float = 100.0,
        p: float = 1.3, id: Optional[str] = None,
        seed_cv: float = 0.01, seed_abs_floor: float = 0.3,
    ) -> "ShareImputer":
        """Read the per-(indicator, region) p95 (value_max) from the stock_bounds
        params CSV — the same file that defines the Part-A soft bounds, so the
        imputed PSC band and the constraint band are the same number. The MESC/USC
        seed does not use p95 (see ``seed_cv`` / ``seed_abs_floor``)."""
        df = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
        p95: dict = {}
        for _, row in df.iterrows():
            if str(row.get("param_type", "")).strip() != "stock_bound":
                continue
            pid    = str(row.get("param_id", "")).lower()
            region = str(row.get("b_region_to", "")).strip()
            vmax   = float_or(row.get("value_max", "nan"), np.nan)
            if np.isnan(vmax) or not region:
                continue
            for ind in ("mesc", "usc", "psc"):     # mesc/usc before psc (substring safety)
                if f"bound{ind}" in pid:
                    p95[(ind, region)] = abs(vmax)
                    break
        logger.info("ShareImputer.from_stock_bounds_csv: %d (indicator,region) p95 entries", len(p95))
        return cls(p95, scale_abs_floor=scale_abs_floor, p=p, id=id,
                   seed_cv=seed_cv, seed_abs_floor=seed_abs_floor)

    def config(self) -> dict:
        return {"scale_abs_floor": self.scale_abs_floor, "p": self.p,
                "n_p95": len(self._p95),
                "seed_cv": self.seed_cv, "seed_abs_floor": self.seed_abs_floor}

    # ------------------------------------------------------------------
    def impute(self, f, sigma, missing, step, acc_struct) -> list[dict]:
        ei = expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        entries: list[dict] = []
        for ind in ("psc", "mesc", "usc"):
            if _MODE[ind] == "transfer":
                entries += self._transfer(ind, f, sigma, ei, step)
            else:
                entries += self._seed(ind, f, sigma, ei, step)
        return entries

    # ------------------------------------------------------------------
    def _sel(self, ei, spec):
        s_from, e_from, s_to, e_to = spec
        return ei[(ei["sector_from"] == s_from) & (ei["entity_from"] == e_from)
                  & (ei["sector_to"] == s_to) & (ei["entity_to"] == e_to)]

    def _floor_sigma(self, p95, base_val):
        """σ = (p95/Z95)·|base|, floored at (p95/Z95)·scale_abs_floor so a tiny /
        collapsed base still yields a finite, soft prior (mirrors Part-A A2)."""
        k = p95 / Z95
        return max(k * abs(base_val), k * self.scale_abs_floor)

    # ---- PSC: share = net/base transferred across the region's years ----
    def _transfer(self, ind, f, sigma, ei, step) -> list[dict]:
        entries: list[dict] = []
        net  = self._sel(ei, _NET[ind])
        base_sel = [self._sel(ei, s) for s in _BASE[ind]]

        for (region, val), grp in net.groupby(["region_to", "valuation"]):
            p95 = self._p95.get((ind, region))
            if p95 is None:
                continue
            net_ser = {str(t): int(e) for t, e in zip(grp["time"], grp["edge_id"])}

            # base[time] = list of edge_ids (CSXEW + CER) for this (region, val)
            base_ser: dict[str, list[int]] = {}
            for bsel in base_sel:
                b = bsel[(bsel["region_to"] == region) & (bsel["valuation"] == val)]
                for t, e in zip(b["time"], b["edge_id"]):
                    base_ser.setdefault(str(t), []).append(int(e))

            def base_of(t):
                ids = base_ser.get(t)
                if not ids:
                    return None
                vals = [f[i] for i in ids]
                if any(np.isnan(v) for v in vals):
                    return None
                return float(sum(vals)), ids

            years = sorted(net_ser, key=lambda t: int(t))
            share = {}
            for t in years:
                e = net_ser[t]
                if np.isnan(f[e]):
                    continue
                bv = base_of(t)
                if bv is None or abs(bv[0]) < 1e-9:
                    continue
                share[t] = float(f[e]) / bv[0]
            if not share:                        # no observed share → leave to flat-prior
                continue
            share_full = _interp_share(share, years)

            for t in years:
                e = net_ser[t]
                if not np.isnan(f[e]):
                    continue
                bv = base_of(t)
                sh = share_full.get(t)
                if bv is None or sh is None:
                    continue
                base_val, base_ids = bv
                f_new = sh * base_val
                s_new = self._floor_sigma(p95, base_val)
                f[e], sigma[e] = f_new, s_new
                entries.append(_log(
                    e, step, "share_transfer", base_ids, [sh] * len(base_ids),
                    f_new, s_new, f"{ind.upper()} = share·(CSXEW+CER)",
                    f"share_{ind}_{region}",
                ))
        return entries

    # ---- MESC/USC: zero-mean seed per co-located edge, held at 1 % / 0.3 kt ----
    def _seed(self, ind, f, sigma, ei, step) -> list[dict]:
        keys = ["region_from", "region_to", "time", "valuation"]
        net  = self._sel(ei, _NET[ind])[["edge_id"] + keys]
        base = self._sel(ei, _BASE[ind][0])[["edge_id"] + keys].rename(
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


def _interp_share(share: dict, years: list) -> dict:
    """Linear-interpolate the observed share across interior years; carry the
    nearest observed share to the boundaries (NOCB before the first obs → seeds
    1995, LOCF after the last). Share is smoother/more stationary than the raw
    signed Δstock level, so this is the temporal-series interpolation of the plan."""
    obs = sorted(share, key=lambda t: int(t))
    yi  = [int(t) for t in obs]
    yv  = [share[t] for t in obs]
    out: dict = {}
    for t in years:
        ti = int(t)
        if t in share:
            out[t] = share[t]
        elif ti <= yi[0]:
            out[t] = yv[0]
        elif ti >= yi[-1]:
            out[t] = yv[-1]
        else:
            k   = bisect.bisect_right(yi, ti)
            t0, t1, v0, v1 = yi[k - 1], yi[k], yv[k - 1], yv[k]
            lam = (ti - t0) / (t1 - t0)
            out[t] = (1 - lam) * v0 + lam * v1
    return out
