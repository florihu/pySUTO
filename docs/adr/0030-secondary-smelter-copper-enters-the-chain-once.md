# Secondary-smelter copper enters the chain once

**Status:** **accepted** — Round O, measured 2026-09-21; **merged** with Rounds N–P as `683ab31`
once [ADR 0031](0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md) anchored the
smelter-scrap *level* and the ADR 0010 amendment closed the pure-secondary recipe (common-set z²
+10.9 % → +7.4 %). Amends [ADR 0021](0021-icsg-refined-production-anchors-anode-into-er.md).
**Amended 2026-10-21 (#63):** imported anode refined beyond primary output leaves the direct
route; see the amendment at the end. Built, to be measured (#63's REP).

> ⚠ **This rule shipped incomplete and nothing caught it.** The subtraction below keys on
> `(Region, Year)` using the feed's **source** region label, so it silently did nothing for any
> territory whose two series are labelled differently — **19 of Belgium's 30 years, 2,434 kt**. Its
> tests passed and its feed gate passed. Completed by
> [ADR 0032](0032-one-territory-one-label.md), which took the arc to **−1.1 % vs base**.

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

Run `opt_20260917_121759_c57f252b` (label `roundO_secondary`, 30/30 Solved) against base
`opt_20260916_124721_c4d1d944`. Full evidence:
[`docs/rep/analysis_opt_20260917_121759_c57f252b_20260921_roundO-secondary-double-count.md`](../rep/analysis_opt_20260917_121759_c57f252b_20260921_roundO-secondary-double-count.md).

**The feed gate passed on all four checks** — 611 changed rows, all `Refining/Scrap/Use`; Austria's
both-booked ratio 1.480 → **1.000** (30-y 1.00–1.00); world concentrate gap +3,450 → **−464 kt**;
`P_Cathode`/(refined+SXEW) 1.000. 107 excess region-years (1,945 kt) listed, not silently dropped.

**What the chain bought.** World MF→Concentrate retention 0.873 → **0.894**; mine-chain deletion
52,871 → **44,197 kt** (−16.4 %). Split by the ADR 0029 detector, the *forced* term is flat
(25,913 → 26,421 kt) and the **discretionary** term — the deletion the model chooses — falls
26,958 → **17,776 kt (−34.1 %)**. Peru 0.884 → 0.923. That split is the strongest confirmation the
double count was real.

**What it cost.** Sum-row z² 1,719,806 → **1,949,396 (+13.3 %)**; Anode +87,939, MF +45,013,
Concentrate +40,714. Not a σ artifact: σ is *identical* on all 21,729 unchanged-anchor sum rows.
Slovakia's anode alone is +64,630 — its anchor never moved (57 kt, σ 0.97) while its posterior anode
collapses ~43 → ~6.5 kt in all 30 years. Austria and Belgium carry most of the rest, and 146 *new*
rows (+43,811) are phantom concentrate/MF totals in Germany (|z| ≈ 41) and Spain — countries with no
mine.

**Why, and what it means for this ADR.** The rule moves copper from a route the model anchors **by
level** (`Refining/Scrap`) to one it anchors **only by share** (`Smelting/Scrap`, a band
`0.95 ≤ Σ(Scrap→Smelting)/Σ(all→Smelting) ≤ 1.00`). A share band over an idle node is satisfied
trivially: ICSG has Slovakia's smelter eating ~28 kt/yr of scrap, the model routes 0.06 kt/yr — in
*both* runs. Where the smelter node has real throughput the substitution works and anode output is
preserved (Germany +0 %, Japan −2 %, China −8 %); where it does not, the copper is lost (Belgium
−21 %, Austria −27 %, Slovakia −79 %). Compounding it, `scrap_share` v2 — regenerated from the
post-fix series as the plan required — flipped Slovakia's CFR band from [0.950, 1.000] to
[0.000, 0.280], closing the other route too.

**Acceptance: 0 and a′, d′, f, h pass; a partial; b, c, d, e, g fail.** Posterior secondary share
0.300 → 0.257 against a prior the fix moved to 0.164 — the feed is now right and the model still
overrides it. China's smelters went *harder* against their 0.402 ceiling (0.379 → 0.398). **Chile
did not recover** (0.557 → 0.561, hard-capped 19/30): its deletion is the forced, BACI-export-mass
kind, which this round cannot touch — the Round O thesis about Chile is not supported.

**Decision: the parse rule stands, the round is held.** The defect is not in the rule; it is the
level/share asymmetry the rule exposed. Round P: make ICSG `Smelting/Scrap` a soft scale total on
`Σ(Scrap→Smelting)` alongside its band, and stop deriving a band from a series the same commit
rewrote. Re-measure with the same script before merging either.

## Consequences

- The ICSG feed's `Refining/Scrap/Use` rows mean **direct-to-refinery scrap** from v9 on; anything
  that read them as "all secondary input" (none found in code; the copper-metabolism insights
  script computes both readings explicitly) must say which it means.
- `data/` is gitignored: on a fresh checkout, `python scripts/parse/icsg_parse/icsg_clean.py
  --to-feed` and `python scripts/explo/scrap_share_study.py` re-apply this decision.
- GLOSSARY.md carries the vocabulary: *smelter scrap*, *direct-to-refinery scrap*, ICSG's two
  output classifications, *direct melt*.

## Amendment — 2026-10-21 (#63): imported anode is not direct-to-refinery scrap

ICSG's secondary refined production also holds the imported blister and anode a region refines
beyond its primary output, and the rule above booked that copper as direct-to-refinery scrap.
Austria refines about 60 kt CC a year of it (ICSG table 12, BACI HS 740200), so its 2019 direct
route was 128.1 − 60.0·0.9845 = 69.0 kt CC (70.1 kt CC of `Scrap→CFR`). The matte prior then fell to
a fabricated pin (3.33, σ 0.057), the `scrap_share` ceiling bound, and the solver cut smelter scrap
62 → 4.5 kt CC (`opt_20261015_093646_ddd9d3d9`). On `opt_20261019_161937_fd301f58`, Austria still
carries a sum-row z² of 68,713 (5.97 % of 1,150,207), led by smelter scrap 2019–2024 and
direct-refinery scrap. The Round O feed gate's "Austria 1.000" closed only because the imports
were counted as scrap.

```
A_imp     = min(max(0, eff_ER · N − P), S_ref)
Scrap→CFR = max(0, S_ref − S_sm · eff_CFR · eff_ER − A_imp) / (eff_CFR · eff_ER)
```

The terms of the rule:

- **`N` is BACI's net cross-region anode imports** in kt CC: HS 740200 legs into the region minus
  legs out of it, with root self-trade dropped (ADR 0014). Gross mass is converted with the conc
  params' Anode nominal, keyed by exporter.
- **`P` is ICSG primary refined production.** The cap `S_ref` is refined output less primary
  output.
- **Imports are refined as primary first.** Only the part beyond `P` can sit in the secondary
  series.
- **Some cases contribute 0:** net exports, a region-year with no BACI row, and an ICSG territory
  mapped to ROW. ROW's trade belongs to 186 economies, not to the territory (ADR 0032).

`icsg_clean.py --baci-version N` reads the **gross** BACI parse vN through
`InputRegistries.load()`, which is the version `build_baci_reexport.py --source-version N` nets
next. A netted version raises, for two reasons:

- The netting re-attributes anode legs, 10,901 of them in baci v29.
- The netting reads this parse's output through `icsg_intake_production`.

The baci and conc versions are recorded in `parents.json` beside the feed's `values.csv`. The
registry hashes that file, so a parse against another BACI version is another icsg version, and
#53's check then forces the trade-share rebuild. The new build order is: BACI parse → register →
`icsg_clean.py --baci-version N --to-feed` → `build_baci_reexport.py --source-version N` → … as
before. `icsg_imported_anode.csv` lists every region-year that loses copper this way.

**Dry estimate** (read-only: baci v28, conc v10, the parse input behind icsg v15). Over 1995–2024
the direct route falls by 770 kt CC world-wide:

- Austria −616 kt CC, in 27 of 30 years; 2019 goes from 69.0 to 13.2 kt CC.
- Belgium −95 kt CC, in 3 years.
- Italy −54 kt CC, in 5 years.
- Slovakia −3.5 and the United Kingdom −1.9 kt CC.

Without imports the branch reproduces icsg v15's 783 direct-route rows exactly. The triage's
Belgium −145 and Italy −83 were gross imports; on v28 that gives −143 and −89.

**Knock-on, not changed here.** `production_recipes` (ADR 0027's netting) takes a converter's
matte intake as `max(P_matte, anode_in/eff_CFR − scrap_cfr)`, which assumes every anode refined
at home was also converted there. With the smaller direct route, Austria's matte intake rises from
60 to about 117 kt CC in 2019. Only the matte netting reads that number; #63's REP checks it.

**Rejected:** filling `P` from the region's own primary smelter production `Sm_prim` first,
`A_imp = max(0, eff_ER·N − (P − Sm_prim·eff_CFR·eff_ER))`. The anode balance makes it exact, but
only if ICSG's smelter and refined series close. They do not: in 2010–2024 China's primary smelter
output is 88–102 % of its primary refined output, while it imports 0.36–0.76 Mt CC of anode a
year. That rule would move China by −4,723, Germany by −284 and Korea by −234 kt CC over 30 years,
on statistical gaps.

**Status:** built, to be measured (#63's REP) by the Round-E stop rule.
