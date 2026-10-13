# #68: the raw-usage trade shares reproduce v29 (#66)

**Run:** none. The rebuild passes, so no ICSG feed or Stage 2–4 run is made. The reference chain stays
`opt_20261012_135741_67851aa1` (REP
[`…issue65-icsg-stock-parse`](analysis_opt_20261012_135741_67851aa1_20261012_issue65-icsg-stock-parse.md)).
**Against:** trade_shares **v29** (`e5e541df`), the last build of the pre-#66 builder, which read usage
from `feed_icsg_20261012_130637_fa562177`. Also **v28** (`af1c4590`), the version that feed run consumed.
**Chain:** icsg v13 + `feed_baci_20261012_130305_1cdad15b` (baci v16) → `scripts/build_trade_shares.py`
at `79464de` → trade_shares **v30** (`387c5d68`). Nothing downstream is rebuilt.
**Feeds/params:** icsg v13, baci v16, conc v10, eff v4, acc copper v4, all the #65 chain's versions.
Only trade_shares v30 is new.
**Date:** 2026-10-13 · **Branch:** `diag/raw-usage-trade-shares-measure` · **Issues:** #68, #66 ·
**ADR:** 0018 and 0002, amended by #66; none amended here

**Verdict: passes.** v30 carries exactly v29's 146,311 weights over 3,113 fan groups. Every weight
equals v29's within 1.6e-11 relative, against the 1e-9 the pass rule allows. The reference chain stays
`opt_20261012_135741_67851aa1`.

## 1. Change

#66, merged in `79464de`, makes the trade-share builder read every fan group's usage from the raw `Use`
rows of the registered ICSG input. Before, it read the fan total of the latest ICSG feed run and used
the raw rows only where that total was 0. That feed run is built from the trade shares. After an ICSG
change the weights were therefore built on the previous ICSG data until a second pass (REP
`…issue65-icsg-stock-parse` §6). On the #65 chain the feed run and the raw rows agree, so the change
should be inert here. This measurement checks that.

## 2. Pass rule (fixed on #68 before the rebuild)

- The new trade shares have exactly v29's rows.
- Every weight equals v29's to ≤ 1e-9 relative.

The script applies the second item as |Δw| ≤ 1e-9 · |w_v29|. Where v29 holds a weight at exactly 0,
that asks for an exact 0.

## 3. Result

| v30 against | v29 | v28 |
|---|---|---|
| rows / fan groups | 146,311 / 3,113 | 146,311 / 3,113 |
| keys in one version only | 0 | 0 |
| weights differing at all | 810 | 686 |
| max absolute difference, raw weight in kt CC | 3.6e-12 | 3.6e-12 |
| max relative difference | 1.6e-11 | 1.6e-11 |
| weights at 0 in the base, nonzero in v30 | 0 of 113,055 | 0 of 113,055 |
| weights beyond 1e-9 relative | 0 | 0 |

- **The differences are round-off.** The raw usage totals reproduce the fan totals of
  `feed_icsg_20261012_130637_fa562177` to 3.5e-16 relative (measured for #66). The 810 weights that
  differ from v29 sit in 177 fan groups: 98 domestic legs and 712 import legs. The import legs are
  rescaled by ADR 0024's bound, which divides by the usage.
- **The largest relative difference is a near-zero residual.** It is Bulgaria's domestic concentrate
  leg in 2015: 0.0035 kt CC in v29, 1.2e-5 of its v29 group total. The residual subtracts two
  near-equal totals and amplifies the round-off.
- **No stock leg differs.** A stock leg's weight reads only the sign of its usage (ADR 0043).
- **For scale:** v29 differs from v28 in 947 weights at 3.1e-12 relative to v28, the round-off #65
  reported.

## 4. The builder's diagnostics are unchanged

#66 made usage an input of the weight computation, so the pre-#66 rule can be replayed with the
current code. The script's `--diag` mode does that in memory, on the same inputs, with usage from
`feed_icsg_20261012_130637_fa562177` and the raw rows only where its fan total is 0. Its weights equal
the raw-usage weights within 1.8e-12 kt CC absolute, on the same rows. Both print the same diagnostic
lines, which are also what the v30 build printed.

| builder line | pre-#66 rule | raw usage (#66) |
|---|---|---|
| ADR 0024: domestic legs bounded | 442 | 442 |
| ADR 0024: "kt CC moved" | −304,203 | −304,203 |
| ADR 0018: groups recovered | 189 | 189 |
| ADR 0018: kt CC recovered | 18,992 | 18,992 |

#66 also switched the ADR 0018 line's reference build to raw usage. On a chain where the feed run and
the raw rows agree, that changes nothing.

## 5. A pre-existing defect in the ADR 0024 diagnostic

The ADR 0024 line's "kt CC moved" is negative. That predates #66.

- **The cause.** The line sums free minus bound weight over every domestic leg, stock legs included.
  Since ADR 0043 a stock leg's domestic weight is its region's own production when production is given,
  and 0 when it is not.
- **What the stock legs add.** The 402 domestic stock legs add −324,738 kt CC to the sum. That is
  their regions' own cathode production, counted once per leg as a supply-mix weight. It is not
  tonnage moved.
- **The real figure.** The 442 bounded flow legs moved +20,536 kt CC from the domestic residual onto
  import legs.

The fix restricts the sum to flow legs and changes a printed line only. It is not made here, because
#68 is a measure ticket.

## 6. Consequences

- **trade_shares v30 is the registered version.** The next ICSG feed build reads it. It equals v28,
  which the reference feed run consumed, within 1.6e-11 relative.
- **One pass is exact from #66 on.** After an ICSG change, run `build_trade_shares.py`, then
  `icsg_feed.py` (ADR 0018 amendment).
- **Still open:** nothing detects a skipped pass. TODO.md's Round T entry "nothing enforces the
  documented build order" stands.

## Reproduce

```
python scripts/build_trade_shares.py                          # the rebuild; registered as v30
python scripts/explo/issue68_trade_shares_identity.py 29 30   # §3, the pass rule
python scripts/explo/issue68_trade_shares_identity.py 28 30   # §3
python scripts/explo/issue68_trade_shares_identity.py 28 29   # §3, #65's round-off, for scale
python scripts/explo/issue68_trade_shares_identity.py --diag \
    feed_icsg_20261012_130637_fa562177 feed_baci_20261012_130305_1cdad15b   # §4 and §5
```

The rebuild is deterministic. Re-running it on the same inputs writes the same file, which the
params registry does not version again.
