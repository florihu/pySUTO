# Secondary-smelter copper enters the chain once

**Status:** accepted at the feed level, **measurement pending** — Round O, 2026-09-17. The feed
gate passed before any solver run (below); the chain result is recorded in the REP named at the
end and summarised in *Measured*. Amends [ADR 0021](0021-icsg-refined-production-anchors-anode-into-er.md)
(two of its statements, see its amendment) and the production recipe that ADR introduced; the
[ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md) netting and the
[ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md) bound consume the corrected `P`
unchanged in form. Coupled: `scrap_share` v2. Baseline `opt_20260916_124721_c4d1d944`.

## Context

The copper-metabolism insights analysis
([REP](../rep/analysis_opt_20260916_124721_c4d1d944_20260917_copper-metabolism-insights.md))
found the model reporting a 30 % secondary share of refinery-side input against ICSG's 17 %, and
traced it to `scripts/parse/icsg_parse/icsg_clean.py`: ICSG publishes two secondary *production*
series and the parse converted both into *inputs* —

```
Smelter Second.   → Scrap→Smelting = S_sm / eff_Sm
Refinery Second.  → Scrap→CFR      = S_ref / (eff_CFR · eff_ER)
```

— but ICSG's secondary refined production **contains** the copper that reached the refinery as
blister or anode from a secondary smelter (ICSG's own description: secondary smelters "reflect
anode production capacity … the blister is fire refined to anode copper for electro-refining").
The smelter route entered twice.

**The evidence, over all 503 region-years carrying both series:**

| class | region-years | what it is |
|---|---|---|
| S_ref > S_sm·k — a visible direct route | 250 | Austria (60 kt smelter inside 123 kt refined = 123 kt total), China (ratio 0.72), Germany (0.70), Belgium (0.75), Spain (0.19) |
| **identical published figures** (\|Δ\| < 0.5 %) | **131, in 15 regions** | Brazil, Mexico, Sweden, Poland, Russia, India, Korea … — the secondary refined figure *is* the smelter figure; the direct route is nil |
| S_sm·k > S_ref — excess | 107 (1,945 kt over 30 y) | Japan (median ratio 1.13, no anode exports) and Iran (1.25, none) are definitional mismatches; Bulgaria (2.41; exports 113 kt/yr of anode against 35 kt/yr secondary smelter output) is genuinely additive |

with `k = eff_CFR·eff_ER`. Austria is the definitional proof — a 100 %-secondary plant whose booked
inputs (62 + 125 = 187 kt) exceeded its entire output (123 kt) by 48 %, and whose smelter scrap the
solver deleted from 62 to 0.1 kt. The fifteen identical-series countries are the sharper one.

**Why it was the mine-chain deletion.** ADR 0021 anchors the anode entering ER on total refined
production and lets `Scrap→CFR` carry `S_ref`, so the anode balance reduces to
`Matte = P/(eff_CFR·eff_ER)` with `P = total − S_ref`: the matte may carry primary copper only, and
every tonne of smelter scrap displaces a tonne of concentrate. Both-booked, the world's anode
anchor could absorb **14,758 kt** of concentrate in 2024 against **18,207 kt** mined — 3,450 kt with
nowhere to go; nested, −464 kt. Over 1995-2024 the regional overlap `Σ min(S_sm·k, S_ref)` is
**70.0 Mt** against **52.9 Mt** of world MF→Concentrate deletion (ratio 0.75, r 0.67). The surplus
was not paid where it arose: China's smelters lost 14.9 Mt of imported concentrate over 2020-24
and took on 7.0 Mt of scrap up to their hard band ceiling (0.40); Chile's mines lost 9.1 Mt.
[ADR 0029](0029-the-concentrate-grade-stays-global.md)'s export-mass gap is the *regional
routing* of this world surplus, not an independent cause.

**A second instance, one node down.** `build_baci_reexport.icsg_intake_production` computed
`P_Cathode = anode_in·eff_ER + Refining/Scrap·eff_CFR·eff_ER + SXEW`. With `anode_in` total refined
since ADR 0021, secondary refined was counted twice at the cathode node — ~4.7 Mt (17 %) of world
cathode production in 2024 — in exactly the `P` that ADR 0027's cathode netting
(`E = max(0, X − P)`) and ADR 0024's cathode bound consume.

## Decision

1. **The parse books the direct route only.** Per ICSG country and year, on the production values
   and before any divisor,

   ```
   direct-to-refinery scrap  Scrap→CFR      = max(0, S_ref − S_sm · eff_CFR · eff_ER) / (eff_CFR · eff_ER)
   smelter scrap             Scrap→Smelting = S_sm / eff_Smelting                          (unchanged)
                             Anode→ER       = (P + S_ref) / eff_ER                         (unchanged)
   ```

   A region-year with `S_ref` but no smelter series is unchanged; one with a smelter series but no
   `S_ref` is unchanged (nothing is booked twice). The **excess** region-years clip to a zero direct
   route and are **listed**, never silently — `data/proc/feedprep/icsg_clean/icsg_secondary_excess.csv`
   (107 rows; Japan 19 y / 693 kt, Bulgaria 29 y / 588 kt, Iran 23 y / 352 kt, Sweden 3 y / 117 kt,
   Korea 11 y / 105 kt). Bulgaria's genuine direct route, ~15 kt/yr, is the known cost of having
   no hand-maintained exception list. `icsg_clean.py --to-feed` copies the clean file into the feed
   directory (what was a hand copy); the registry versioned it icsg v8 → **v9**: 611 rows changed,
   all `Refining/Scrap/Use`, world −78 %.
2. **The cathode recipe carries no second scrap term:** `P_Cathode = anode_in·eff_ER + SXEW`. The
   arithmetic moved into a pure `production_recipes(...)` so it is unit-tested (the term survived
   because it was not).
3. **`scrap_share` is regenerated (v2) as a coupled consequence.** Its generator derived the CFR
   band from a `Matte/Refining/Use` series ADR 0021 removed, so the shipped v1 CFR bands predated
   both changes and their hard **floors** sat far above the direct-route shares the fix leaves —
   China 0.160 vs 0.040, Germany 0.342 vs 0.126, Belgium 0.355 vs 0.065, Austria 0.95 vs 0.503 —
   which would have forced the double-counted scrap back in, or deleted the matte to satisfy the
   share. The CFR share is now direct-to-refinery scrap over the total CFR input inferred from
   the anode leaving it (`Refining/Anode/Use ÷ eff_CFR`); every Smelting band of v2 is asserted
   identical to v1; the feed is read through the registry. No intermediate run on v1 was made:
   it would have measured a wrong constraint, not the fix.
4. **Feed gate, before any solver run** (`scripts/explo/roundO_secondary_diag.py --feed-only`):
   only the `Refining/Scrap/Use` rows differ between v8 and v9 — PASS; Austria's booked inputs
   over its output **1.480 → 1.000** (30-y range 1.45–1.99 → 1.00–1.00); world concentrate gap
   2024 **+3,450 → −464 kt** (the nested arithmetic on the old feed, reproduced by the new feed's
   own numbers); `P_Cathode / (refined + SX-EW)` **1.000**.

`copper_efficiency.csv` (the parse's eff source) and the eff params v3 are numerically identical
and not synchronised by any code; the subtraction uses the parse's own `read_eff()` like the
existing divisors. Noted, not changed.

## Considered and rejected

- **Leaving the feed and widening the scrap or grade bands.** ADR 0028 measured what moving a wall
  does — it relocates the deletion — and the surplus here is an input, not a price.
- **An explicit additive exception list (Bulgaria).** Correct for ~15 kt/yr, but a hand-maintained
  set a future yearbook could silently invalidate; the excess csv keeps the evidence visible instead.
- **Leaving the excess region-years both-booked.** Wrong for Japan and Iran (no anode exports;
  their excess is a definitional mismatch) — it would keep ~360 kt/yr of Japanese smelter scrap
  counted twice and make Japan ~47 % secondary.
- **A soft `scrap_share` mode or v1 bands kept.** The floors bind (above); a soft centre would pull
  toward the wrong observations.
- **Deferring the cathode term to its own round.** Same defect family, one-line fix, separable by
  sector in the netting ledger; a second chain only if the sector split is ambiguous.
- **Fixing at feed-build time (the ADR 0027 pattern) instead of the parse.** The parse is where the
  production→input semantics are stated and where ADR 0021 was itself applied (parse + hand copy);
  a derived-file fix would have left the stated semantics wrong.

## Measured

*Pending — filled after run `opt_…` (label `roundO_secondary`); REP
`docs/rep/analysis_opt_<O>_20260917_roundO-secondary-double-count.md`.*

## Consequences

- The ICSG feed's `Refining/Scrap/Use` rows mean **direct-to-refinery scrap** from v9 on; anything
  that read them as "all secondary input" (none found in code; the copper-metabolism insights
  script computes both readings explicitly) must say which it means.
- `data/` is gitignored: on a fresh checkout, `python scripts/parse/icsg_parse/icsg_clean.py
  --to-feed` and `python scripts/explo/scrap_share_study.py` re-apply this decision.
- CONTEXT.md carries the vocabulary: *smelter scrap*, *direct-to-refinery scrap*, ICSG's two
  output classifications, *direct melt*.
