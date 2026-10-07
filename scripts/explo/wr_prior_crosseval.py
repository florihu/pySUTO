"""
wr_prior_crosseval.py
=====================
Waste-rock prior measurement (ADR 0038 amendment 2026-10-07): the prior carries
WR Mass = k·Ore Mass at the flat σ. Reads the per-year tables of two opt runs whose
QPs share G, c and σ_c and differ only in a₀ / σ_a, and prints:

    W1  WR Mass prior and posterior per year (Mt), WR retention Σa*/Σa_prior
        (category Mining→WR Mass) per year, and posterior ÷ 2·Ore
    W2  δ(F) = J_new(F) − J_base(F) at both posteriors, split into the WR fills
        (a₀ changed) and the edges whose σ_a alone changed (the no-data anchor)
    W3  each posterior under both QPs: J(F) = Σ((F − a₀)/σ_a)² + Σ_soft((GF − c)/σ_c)²
        reproduces Clarabel's cost_primal after the constant Σa₀²/σ_a², so a
        negative J_new(F_base) − J_new(F_new) means the new solve stopped at a worse
        point of its own QP than the base posterior

    python scripts/explo/wr_prior_crosseval.py <opt_base> <opt_new>

Needs the per-year tables (``scripts/solver_stats.py <opt> --years all``) of both runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT  = Path("data/proc/opt")
YEARS = range(1995, 2025)
COLS  = ["edge_id", "from_sector", "to_sector", "valuation", "a_prior", "a_post", "sigma_a"]


def _year(base, new, y):
    eb = pd.read_csv(ROOT / base / "diag" / f"edges_{y}.csv", usecols=COLS)
    en = pd.read_csv(ROOT / new / "diag" / f"edges_{y}.csv",
                     usecols=["edge_id", "a_prior", "a_post", "sigma_a"])
    e = eb.merge(en, on="edge_id", suffixes=("_b", "_n"))
    soft = {}
    for lab, r in (("b", base), ("n", new)):
        c = pd.read_csv(ROOT / r / "diag" / f"constraints_{y}.csv",
                        usecols=["cone_type", "residual", "sigma_c"])
        free = c[c.cone_type == "free"]
        soft[lab] = float(((free.residual / free.sigma_c) ** 2).sum())
    return e, soft


def _prior_term(F, a0, s):
    return float((((F - a0) / s) ** 2).sum())


def main(base, new):
    w1, w2, w3 = [], [], []
    for y in YEARS:
        e, soft = _year(base, new, y)
        mass = e.valuation == "Mass"
        wr  = e[mass & (e.from_sector == "Mining") & (e.to_sector == "WR")]
        ore = e[mass & (e.from_sector == "Mining") & (e.to_sector == "Ore")]
        w1.append((y, wr.a_prior_b.sum() / 1e3, wr.a_prior_n.sum() / 1e3,
                   wr.a_post_b.sum() / 1e3, wr.a_post_n.sum() / 1e3,
                   wr.a_post_n.sum() / wr.a_prior_n.sum(),
                   wr.a_post_n.sum() / (2 * ore.a_post_n.sum())))

        ch = e[(e.a_prior_b != e.a_prior_n) | (e.sigma_a_b != e.sigma_a_n)]
        fill = ch.a_prior_b != ch.a_prior_n
        row = [y, int(fill.sum()), int((~fill).sum())]
        for F in (ch.a_post_b, ch.a_post_n):
            d = (((F - ch.a_prior_n) / ch.sigma_a_n) ** 2
                 - ((F - ch.a_prior_b) / ch.sigma_a_b) ** 2)
            row += [float(d[fill].sum()), float(d[~fill].sum())]
        w2.append(row)

        Jbb = _prior_term(e.a_post_b, e.a_prior_b, e.sigma_a_b) + soft["b"]
        Jbn = _prior_term(e.a_post_n, e.a_prior_b, e.sigma_a_b) + soft["n"]
        Jnn = _prior_term(e.a_post_n, e.a_prior_n, e.sigma_a_n) + soft["n"]
        Jnb = _prior_term(e.a_post_b, e.a_prior_n, e.sigma_a_n) + soft["b"]
        w3.append((y, Jbb, Jbn - Jbb, Jnn, Jnb - Jnn))

    pd.set_option("display.width", 200)
    t1 = pd.DataFrame(w1, columns=["year", "prior_base_Mt", "prior_new_Mt", "post_base_Mt",
                                   "post_new_Mt", "retention_new", "post_new/2Ore"])
    print("== W1 Mining→WR Mass per year")
    print(t1.round(3).to_string(index=False))
    tot = t1[["prior_base_Mt", "prior_new_Mt", "post_base_Mt", "post_new_Mt"]].sum()
    print(f"   1995–2024: prior {tot.prior_base_Mt:,.1f} → {tot.prior_new_Mt:,.1f} Mt, "
          f"posterior {tot.post_base_Mt:,.1f} → {tot.post_new_Mt:,.1f} Mt, "
          f"retention (new) {tot.post_new_Mt / tot.prior_new_Mt:.3f}")

    t2 = pd.DataFrame(w2, columns=["year", "n_fill", "n_sigma_only",
                                   "dWR@Fb", "dSigma@Fb", "dWR@Fn", "dSigma@Fn"])
    print("\n== W2 δ(F) = J_new(F) − J_base(F)")
    print(t2.round(3).to_string(index=False))
    d = t2[["dWR@Fb", "dSigma@Fb", "dWR@Fn", "dSigma@Fn"]]
    print(f"   max |δ| {d.abs().max().round(3).to_dict()}; max |δ(F_base) − δ(F_new)| "
          f"{((t2['dWR@Fb'] + t2['dSigma@Fb']) - (t2['dWR@Fn'] + t2['dSigma@Fn'])).abs().max():.3f}")

    t3 = pd.DataFrame(w3, columns=["year", "J_base(F_base)", "J_base(F_new)-J_base(F_base)",
                                   "J_new(F_new)", "J_new(F_base)-J_new(F_new)"])
    print("\n== W3 cross-evaluation")
    print(t3.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
