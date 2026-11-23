"""
share.py
========
``ShareImputer`` — impute net-stock-change (Δstock) edges as a share of a
well-covered *base* flow, so cold runs get smooth, throughput-scaled priors
instead of ``noisy_zero`` flat priors. The flat prior (f=0, σ=1e6) leaves the
Part-A soft-bound σ_c under-scaled (its den_scale reads the *prior* throughput)
and jags the cold time series; a throughput-scaled prior fixes both.

Each stock-change leg is declared as a ``StockLeg`` (its stock sector, its mode, the net
edge's selector and its base selectors), so the sector labels are the caller's;
the toy's instances are in ``pysuto.tutorial.toy``, the copper ones in
``scripts/copper.py``. Two per-leg modes:

* ``"transfer"`` (copper: PSC): ``share(t) = f[net(t)] / f[base(t)]`` computed where the
  net edge is already covered, interpolated across the region's time series and
  extrapolated to the boundaries (so 1995 — an ICSG baseline year with no NAS —
  is seeded), then ``f[net(t)] = share(t)·f[base(t)]`` for the still-missing
  years. Data-backed (real per-region PSC observations). base = CSXEW+CER inputs
  to the CP pool (= total cathode production; NOT CP→Cathode, which is
  production−PSC and circular).

* ``"seed"`` (copper: MESC / USC): the net change is centred at 0 and held there —
  ``f[net]=0``, ``σ = max(seed_cv·|f[base_colocated]|, seed_abs_floor)`` — replacing
  ``noisy_zero``. MESC/USC are 47×47 source→dest matrices whose signed central
  share ≈ 0, so a zero-mean seed is the honest informative prior, and a stock
  leg is the one signed edge of the account: any looseness makes it a sink for
  every contradiction of the cathode balance (Round X; ADR 0040). The seed is
  therefore priced like a feed-backed level (ADR 0025's scale-total rule, 1 % of
  the co-located Cathode(i)→FU(j) flow, floored at the feeds' 0.3 kt resolution),
  not by the empirical p95 band — the band (p95 × a 100 kt floor) gave the transit-region
  legs σ ≈ 553 kt and the solver +9.7 Mt of phantom exchange stock over 30 years.
  base = the co-located Cathode(i)→FU(j) consumption edge; a no-trade pair is
  pinned at the floor.

A transfer leg keeps ``σ = (p95/Z95)·base`` (the empirical band, consistent with the
Part-A soft bound); ``p95`` (value_max) is read per (stock sector, region) from the
stock_bounds params CSV, the stock sector being the row's ``b_sector_to`` selector —
declared in the row, not parsed from its ``param_id``. Seed legs read no p95 since
Round X.

The seed floor ``seed_abs_floor`` (kt) is applied in each leg's own valuation unit, kt for
Mass and kt CC for content, to Mass and content net-stock legs alike.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pysuto.logging import get_logger
from pysuto.relation.param_relation import expand_edge_index
from pysuto.relation.params import WILDCARD, read_params

from .base import BaseImputer

logger = get_logger(__name__)

#: The param_types read from the stock_bounds file.
ACCEPTS = frozenset({"stock_bound"})

#: p95 ≈ Z95·σ — read the symmetric ±p95 half-band as the ratio-space σ (matches
#: Z95_STOCK in constraints/ratio.py so the imputed prior and the soft bound agree).
Z95 = 1.645

_MODES = ("transfer", "seed")


@dataclass(frozen=True)
class StockLeg:
    """One stock-change indicator of the account.

    sector  The stock sector the net edge points to (``net[2]``); the key of the p95
            table and, lower-cased, of the log's ``share_<sector>_<region>`` tags.
    mode    ``"transfer"`` (share of a base carried across years) or ``"seed"``
            (zero-mean, σ priced on the co-located base).
    net     ``(sector_from, entity_from, sector_to, entity_to)`` of the net edge.
    base    The base selectors, same layout: any number for ``"transfer"`` (summed,
            intra-region), exactly one for ``"seed"`` (co-located by region pair).
    """

    mode: str
    net:  tuple
    base: tuple

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"StockLeg {self.net!r}: mode {self.mode!r} not in {_MODES}")
        if len(self.net) != 4 or not self.base or any(len(b) != 4 for b in self.base):
            raise ValueError(f"StockLeg {self.net!r}: selectors are 4-tuples "
                             "(sector_from, entity_from, sector_to, entity_to)")
        if self.mode == "seed" and len(self.base) != 1:
            raise ValueError(f"StockLeg {self.net!r}: a seed leg takes one base selector")

    @property
    def sector(self) -> str:
        return self.net[2]


class ShareImputer(BaseImputer):
    """Impute Δstock edges, per declared ``StockLeg``, as share×base (transfer) or a
    zero-mean throughput-scaled seed (seed). Runs late in the cascade (after time
    interpolation, before the flat-prior fallback) so the base flows are already
    covered.

    Parameters
    ----------
    p95 : dict
        `(stock sector, region_to)` -> p95, read by the transfer legs.
    legs : sequence of StockLeg
        The stock-change indicators, imputed in this order."""

    def __init__(
        self,
        p95: dict,
        legs: Sequence[StockLeg],
        scale_abs_floor: float = 100.0,
        p: float = 1.3,
        id: Optional[str] = None,
        seed_cv: float = 0.01,
        seed_abs_floor: float = 0.3,
    ) -> None:
        self.id              = id or "share"
        self._p95            = p95           # (sector, region_to) -> p95, transfer legs
        self.legs            = tuple(legs)
        self.scale_abs_floor = float(scale_abs_floor)
        self.p               = float(p)
        # Seed σ = max(seed_cv·|base leg|, seed_abs_floor) — kt (ADR 0040).
        self.seed_cv         = float(seed_cv)
        self.seed_abs_floor  = float(seed_abs_floor)

    # ------------------------------------------------------------------
    @classmethod
    def from_stock_bounds_csv(
        cls, path: str, legs: Sequence[StockLeg], scale_abs_floor: float = 100.0,
        p: float = 1.3, id: Optional[str] = None,
        seed_cv: float = 0.01, seed_abs_floor: float = 0.3,
    ) -> "ShareImputer":
        """Read the per-(stock sector, region) p95 (value_max) from the stock_bounds
        params CSV — the same file that defines the Part-A soft bounds, so the
        imputed transfer band and the constraint band are the same number. A row
        belongs to the leg whose sector is its ``b_sector_to`` selector; rows of any
        other sector are skipped. The seed legs do not use p95 (see ``seed_cv`` /
        ``seed_abs_floor``)."""
        sectors = {leg.sector for leg in legs}
        p95: dict = {}
        for row in read_params(path, ACCEPTS):
            sector = row.selectors["b_sector_to"]
            region = row.selectors["b_region_to"]
            vmax   = row.value_max
            if np.isnan(vmax) or region in WILDCARD or sector not in sectors:
                continue
            p95[(sector, region)] = abs(vmax)
        logger.info("ShareImputer.from_stock_bounds_csv: %d (sector,region) p95 entries",
                    len(p95))
        return cls(p95, legs, scale_abs_floor=scale_abs_floor, p=p, id=id,
                   seed_cv=seed_cv, seed_abs_floor=seed_abs_floor)

    def config(self) -> dict:
        return {"scale_abs_floor": self.scale_abs_floor, "p": self.p,
                "n_p95": len(self._p95),
                "seed_cv": self.seed_cv, "seed_abs_floor": self.seed_abs_floor}

    # ------------------------------------------------------------------
    def writer_scope(self, acc) -> np.ndarray:
        """The stock-change legs: every declared net-change edge."""
        ei = expand_edge_index(acc.edge_index, acc.node_index)
        scope = np.zeros(acc.n_edges, dtype=bool)
        for leg in self.legs:
            scope[self._sel(ei, leg.net)["edge_id"].to_numpy(dtype=np.int64)] = True
        return scope

    def impute(self, ledger, step, acc_struct) -> None:
        ei = expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        for leg in self.legs:
            if leg.mode == "transfer":
                self._transfer(leg, ledger, ei, step)
            else:
                self._seed(leg, ledger, ei, step)

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

    # ---- transfer: share = net/base carried across the region's years ----
    def _transfer(self, leg, ledger, ei, step) -> None:
        f = ledger.f
        ind  = leg.sector
        net  = self._sel(ei, leg.net)
        base_sel = [self._sel(ei, s) for s in leg.base]
        base_txt = "+".join(s[0] for s in leg.base)

        for (region, val), grp in net.groupby(["region_to", "valuation"]):
            p95 = self._p95.get((ind, region))
            if p95 is None:
                continue
            net_ser = {str(t): int(e) for t, e in zip(grp["time"], grp["edge_id"])}

            # base[time] = list of edge_ids (the base selectors) for this (region, val)
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
                ledger.write(
                    e, f_new, s_new, step, "share_transfer", base_ids,
                    [sh] * len(base_ids), f"{ind.upper()} = share·({base_txt})",
                    f"share_{ind.lower()}_{region}",
                )

    # ---- seed: zero-mean per co-located edge, held at seed_cv / seed_abs_floor ----
    def _seed(self, leg, ledger, ei, step) -> None:
        f = ledger.f
        ind  = leg.sector
        keys = ["region_from", "region_to", "time", "valuation"]
        net  = self._sel(ei, leg.net)[["edge_id"] + keys]
        base = self._sel(ei, leg.base[0])[["edge_id"] + keys].rename(
            columns={"edge_id": "base_id"})
        m = net.merge(base, on=keys, how="inner")
        if m.empty:
            return

        net_ids  = m["edge_id"].to_numpy(dtype=np.int64)
        base_ids = m["base_id"].to_numpy(dtype=np.int64)

        fill = (np.isnan(f[net_ids]) & ~np.isnan(f[base_ids]))
        e_fill, b_fill = net_ids[fill], base_ids[fill]
        s_new = np.maximum(self.seed_cv * np.abs(f[b_fill]), self.seed_abs_floor)
        desc = f"{ind.upper()} zero-mean, σ=max(cv·|{leg.base[0][2]}|, floor)"
        ledger.write_many(
            e_fill, np.zeros(e_fill.size), s_new, step, "share_seed", desc,
            f"seed_{ind.lower()}", sources=[[int(b)] for b in b_fill],
            coeffs=[[0.0]] * e_fill.size,
        )


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
