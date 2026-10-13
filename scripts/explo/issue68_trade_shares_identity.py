"""
issue68_trade_shares_identity.py
================================
#68 measurement (#66): a trade_shares version rebuilt by the raw-usage builder against a
registered base version, and the builder's diagnostics under both usage sources.

Pass rule, fixed on #68 before the rebuild: the same rows, and every weight equal to the
base's within 1e-9 relative.  Applied as |Δw| ≤ 1e-9 · |w_base|, which asks for an exact
0 where the base weight is 0.

    python scripts/explo/issue68_trade_shares_identity.py <base_version> <new_version>
    python scripts/explo/issue68_trade_shares_identity.py --diag <feed_icsg_run> \
                                                                 <feed_baci_run>

The first form prints the row and group counts, the keys only one side carries, the
largest absolute and relative weight difference, where the differing weights sit, and
the verdict.  The second recomputes the builder's ADR 0024 and ADR 0018 lines in memory
and writes and registers nothing.  It runs once with usage from the registered ICSG input
(#66) and once with the pre-#66 rule: the given ICSG feed run's fan total, and the raw
rows only where that total is 0.  It splits the ADR 0024 sum into flow and stock legs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import build_trade_shares as B                          # noqa: E402
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402

ROOT  = Path("data/input/params/trade_shares")
FEEDS = Path("data/proc/feed/registry")
KEY   = ["root_sector_from", "root_sector_to", "root_region_from", "root_region_to",
         "root_time"]
GROUP = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
RTOL  = 1e-9


# ---------------------------------------------------------------------------
# Identity of two registered versions
# ---------------------------------------------------------------------------

def load(version: int) -> pd.DataFrame:
    """One registered snapshot's weights; years stay strings."""
    return pd.read_csv(ROOT / f"v{version}" / "params.csv", dtype={"root_time": str})


def identity(base_v: int, new_v: int) -> bool:
    base, new = load(base_v), load(new_v)
    for v, df in ((base_v, base), (new_v, new)):
        print(f"trade_shares v{v}: {len(df):,} rows, {df.groupby(GROUP).ngroups:,} groups")

    merged = base.merge(new, on=KEY, how="outer", suffixes=("_base", "_new"),
                        indicator=True, validate="one_to_one")
    only = merged[merged["_merge"] != "both"]
    print(f"keys in one version only: {len(only):,}")
    if len(only):
        print(only[KEY + ["_merge"]].to_string(index=False))

    both = merged[merged["_merge"] == "both"].copy()
    if both[["weight_base", "weight_new"]].isna().any().any():
        raise ValueError("a NaN weight cannot be compared; the builder emitted one")
    w_base, w_new = both["weight_base"].to_numpy(), both["weight_new"].to_numpy()
    diff = np.abs(w_new - w_base)
    bad  = diff > RTOL * np.abs(w_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(w_base != 0.0, diff / np.abs(w_base), np.nan)
    print(f"max |Δweight|: {diff.max():.3e}   "
          f"max relative (base ≠ 0): {np.nanmax(rel):.3e}")
    print(f"base weights at 0: {int((w_base == 0.0).sum()):,}, of which nonzero in "
          f"v{new_v}: {int(((w_base == 0.0) & (w_new != 0.0)).sum()):,}")

    # where the differing weights sit
    both["diff"], both["rel"] = diff, rel
    moved = both[both["diff"] > 0.0]
    leg = np.where(moved["root_region_from"] == moved["root_region_to"],
                   "domestic", "import")
    kind = np.where(moved["root_sector_to"].isin(B.SIGNED_SECTORS_TO), "stock", "flow")
    print(f"weights differing at all: {len(moved):,} in "
          f"{moved[GROUP].drop_duplicates().shape[0]:,} groups; by leg: "
          f"{pd.Series(list(zip(leg, kind))).value_counts().to_dict()}")
    top = both.loc[both["rel"].idxmax()]
    group_total = both.groupby(GROUP)["weight_base"].transform("sum")[top.name]
    print(f"largest relative difference: {tuple(top[KEY])}, v{base_v} weight "
          f"{top['weight_base']:.6g} kt CC = {top['weight_base'] / group_total:.3g} of its "
          f"v{base_v} group total")

    print(f"weights beyond {RTOL:g} relative: {int(bad.sum()):,}")
    if bad.any():
        print(both.loc[bad].groupby(GROUP).size().rename("n_weights").to_string())
    ok = not len(only) and not bad.any()
    print(f"verdict: {'PASS' if ok else 'FAIL'} (same rows; every |Δw| ≤ {RTOL:g}·|w| "
          f"of v{base_v})")
    return ok


# ---------------------------------------------------------------------------
# The builder's diagnostics under both usage sources
# ---------------------------------------------------------------------------

def _froot(run: str) -> np.ndarray:
    return np.load(FEEDS / run / "result.npz")["F_root"]


def _feed_usage(acc, icsg_froot: np.ndarray) -> pd.DataFrame:
    """Per fan group the ICSG feed run's fan total, the pre-#66 builder's usage."""
    ei = B._decoded_edges(acc)
    cc = ei[ei["root_sector_from"].isin(B.SECTORS) & (ei["valuation"] == "CC")]
    tot = cc[GROUP].copy()
    tot["usage_raw"] = icsg_froot[cc["edge_id"].to_numpy()]
    return tot.groupby(GROUP, sort=False)["usage_raw"].sum().reset_index()


def _adr24_line(free: pd.DataFrame, bound: pd.DataFrame) -> str:
    both = free.merge(bound, on=KEY, suffixes=("_free", "_bound"))
    dom = both[both["root_region_from"] == both["root_region_to"]]
    moved = dom["weight_free"] - dom["weight_bound"]
    stock = dom["root_sector_to"].isin(B.SIGNED_SECTORS_TO)
    return (f"{int((moved > 1e-9).sum()):,} domestic legs bounded, "
            f"{moved.sum():,.0f} kt CC 'moved' = flow legs {moved[~stock].sum():,.0f} "
            f"+ {int(stock.sum()):,} stock legs {moved[stock].sum():,.0f}")


def _adr18_line(shares: pd.DataFrame, pre: pd.DataFrame) -> str:
    grp = (shares[GROUP].drop_duplicates()
           .merge(pre[GROUP].drop_duplicates(), how="left", indicator=True))
    rec = grp.loc[grp["_merge"] == "left_only", GROUP]
    dom = shares[shares["root_region_from"] == shares["root_region_to"]]
    kt = dom.merge(rec, on=GROUP)["weight"].sum()
    return f"{len(rec):,} groups recovered, {kt:,.0f} kt CC"


def diag(icsg_run: str, baci_run: str) -> None:
    acc   = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                     storage_dir=B.ACC_DIR)
    conc  = B._conc_lookup()
    baci  = _froot(baci_run)
    edges = B.build_edges_frame(acc, baci, B.SECTORS)
    obs   = B.baci_observed_regions(acc, baci)
    held  = B.imports_exhausted_groups(edges)
    prod  = B.icsg_raw_production(B.BOUND_DOMESTIC_SECTORS, edges)
    raw   = B.icsg_raw_usage(B.SECTORS)
    feed  = _feed_usage(acc, _froot(icsg_run))
    # pre-#66: the feed run's fan total, the raw rows only where that total is 0
    old = feed.merge(raw, on=GROUP, how="left", suffixes=("", "_rows"))
    old["usage_raw"] = np.where(old["usage_raw"] > 0, old["usage_raw"],
                                old["usage_raw_rows"].fillna(0.0))

    sources = {"raw ICSG input (#66)": (raw, raw),
               f"pre-#66 rule on {icsg_run}": (old[GROUP + ["usage_raw"]], feed)}
    shares = {}
    for label, (usage, usage_pre) in sources.items():
        s    = B.compute_trade_share_weights(edges, usage, conc, obs, held, prod)
        free = B.compute_trade_share_weights(edges, usage, conc, obs, held, None)
        pre  = B.compute_trade_share_weights(edges, usage_pre, conc, None)
        print(f"{label}:\n   ADR 0024: {_adr24_line(free, s)}"
              f"\n   ADR 0018: {_adr18_line(s, pre)}")
        shares[label] = s

    a, b = shares.values()
    m = a.merge(b, on=KEY, how="outer", indicator=True)
    print(f"weights, one source only: {int((m['_merge'] != 'both').sum()):,}   "
          f"max |Δweight| between the sources: "
          f"{(m['weight_x'] - m['weight_y']).abs().max():.3e}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--diag":
        diag(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        sys.exit(0 if identity(int(sys.argv[1]), int(sys.argv[2])) else 1)
    else:
        sys.exit(__doc__)
