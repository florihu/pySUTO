"""
issue68_trade_shares_identity.py
================================
#68 measurement (#66): a trade_shares version rebuilt by the raw-usage builder against a
registered base version.  Pass rule, fixed on #68 before the rebuild: the same rows, and
every weight equal to the base's within 1e-9 relative.  A weight the base holds at
exactly 0 must be exactly 0 in the new version, since a relative difference is undefined
there.

    python scripts/explo/issue68_trade_shares_identity.py <base_version> <new_version>

Prints the row and group counts, the keys only one side carries, the largest absolute
and relative weight difference, the fan groups holding a weight beyond the tolerance,
and the verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT  = Path("data/input/params/trade_shares")
KEY   = ["root_sector_from", "root_sector_to", "root_region_from", "root_region_to",
         "root_time"]
GROUP = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
RTOL  = 1e-9


def load(version: int) -> pd.DataFrame:
    """One registered snapshot's weights; years stay strings."""
    return pd.read_csv(ROOT / f"v{version}" / "params.csv", dtype={"root_time": str})


def main(base_v: int, new_v: int) -> bool:
    base, new = load(base_v), load(new_v)
    print(f"trade_shares v{base_v}: {len(base):,} rows, "
          f"{base.groupby(GROUP).ngroups:,} groups")
    print(f"trade_shares v{new_v}: {len(new):,} rows, "
          f"{new.groupby(GROUP).ngroups:,} groups")

    m = base.merge(new, on=KEY, how="outer", suffixes=("_base", "_new"),
                   indicator=True, validate="one_to_one")
    only = m[m["_merge"] != "both"]
    print(f"keys in one version only: {len(only):,}")
    if len(only):
        print(only[KEY + ["_merge"]].to_string(index=False))

    both = m[m["_merge"] == "both"]
    b, n = both["weight_base"].to_numpy(), both["weight_new"].to_numpy()
    diff = np.abs(n - b)
    zero = b == 0.0
    rel  = np.where(zero, np.nan, diff / np.where(zero, 1.0, np.abs(b)))
    print(f"max |Δweight|: {diff.max():.3e}   "
          f"max relative (base ≠ 0): {np.nanmax(rel):.3e}")
    print(f"weights differing at all: {int((diff > 0.0).sum()):,}")
    print(f"base weights at 0: {int(zero.sum()):,}, of which nonzero in v{new_v}: "
          f"{int((zero & (n != 0.0)).sum()):,}")

    bad = (zero & (n != 0.0)) | (~zero & (rel > RTOL))
    print(f"weights beyond {RTOL:g} relative: {int(bad.sum()):,}")
    if bad.any():
        print(both.loc[bad].groupby(GROUP).size().rename("n_weights").to_string())

    ok = not len(only) and not bad.any()
    print(f"verdict: {'PASS' if ok else 'FAIL'} (same rows; every weight within "
          f"{RTOL:g} relative of v{base_v})")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(0 if main(int(sys.argv[1]), int(sys.argv[2])) else 1)
