# #283: the Stage-3 build-time checks that now raise pass on the reference const, and the final review's check chain confirms every const-to-psut array identical

**Run:** no pipeline run was made on this branch. The reference const's stored arrays were read once, read-only, against the new check logic (`ConstraintEngine._check`, `RatioConstraintBuilder.from_csv`).
**Against:** `const_20261119_125651_ff7bfc9c` (the final-review reference, #155).
**Chain:** reference chain `psut_20261119_133504_4c783a88` / `uncprop_20261119_133344_08dfa15b`. The check chain is the final review's wave 2 (`analysis_opt_20261123_171426_e335ec99_20261123_final-review-check-chain.md`): from Stage 1 on `b08d860`, which holds this fix; `const_20261123_170806_4991b908` built under the raising checks, and every stored array from the feeds to psut is identical to the reference (const 303, opt 240, uncprop 150, psut 366).
**Feed+params versions:** `stock_bounds` current at read time (47 rows, all `stock_bound`).
**Date:** 2026-11-23
**Branch:** `fix/u3-findings-283`
**ADR:** 0013 (path of `nodata_exempt_mask` updated), 0006/0025 (unchanged)

**Verdict:**
- The new raises do not fire on the reference const (static evidence below). They change no stored array: no builder, σ rule, row order or id was touched.
- Confirmed: the check chain's `compare` finds every const-to-psut array identical (1,059; 1,102 with the feeds and the imp).

## Evidence on `const_20261119_125651_ff7bfc9c`

| Check (now a raise) | Reference value |
|---|---|
| G width = n_edges | 854,460 columns, max column index 854,459 |
| `u >= l` | 0 edges with `u < l` |
| Hard rows: σ_c = 0, every year | 0 violations in 30 years |
| Soft rows: σ_c finite and > 0, every year | 0 violations in 30 years (461,170 soft rows) |
| Structural `l > 0` (FlowBoundsBuilder) | 0 edges with `l > 0` in the stored `l.npy` |
| `stock_bound` rows with no `value_max`, valuation mixing | 0 of 47 rows lack `value_max`; all are Mass-or-CC on both sides (U3 audit) |

- The mass-balance count check passes (39,480 rows) and the ratio per-year uniformity check warns in the reference log by construction (431,814 `ratio_eq` and 598,994 `ratio_bound` rows, 14,379–14,408 and 19,948–19,990 per year), so both stay advisory.
