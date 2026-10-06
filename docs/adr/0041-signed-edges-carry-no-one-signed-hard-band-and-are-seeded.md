# A signed edge carries no one-signed hard ratio band, and the producer stock change is seeded like the other stock legs

**Status:** proposed — Round Y, 2026-10-06 (diagnosis of Round X.0 `opt_20261005_134720_ee69cda9`,
chain `imp_20261005_125928_e9ffa8cb` → `const_20261005_130418_a5fb94a6`; spec #41; the Y.0 / Y.1
measure tickets supply the run ids and the P1–P10 / Round-E verdict, after which this line becomes
accepted or rejected on measurement). Amends [ADR 0040](0040-stock-change-legs-are-held-at-their-prior.md)
(PSC joins the seeded legs; the PSC share transfer is retired) and
[ADR 0039](0039-flow-lower-bounds-are-enforced-in-the-qp.md) (the signed legs' negative half was still
forbidden, by the concentration band). Closes #36 item 2; feed follow-up #40.

On X.0 the producer stock change `CP→PSC` was flat at 0 in Mass and "varying" in CC for three reasons.
No concentration row matched `*→PSC` and ICSG reports PSC in CC only, so no imputer could derive a Mass
prior: all 1,410 Mass region-years were the flat prior (0, σ 1e6), outside the no-data pin and anchor
(ADR 0013), a free signed sink that absorbed ≈ 1,380 kt over 30 years (22–87 kt/yr; China 2013
−32 kt). In CC the Cathode Pool has no efficiency row, so the recipe wrote `PSC = (1 − 1.0)·Σin = 0`
with σ 0 on 872 region-years (875 pinned once the three zeros the interpolator copied from them are
counted) and Stage 3 floored it to σ 1e-6 — a hard zero on a leg ADR 0013 says never to pin; only
the 187 ICSG region-years varied. And `conccathodesupply` matches every edge leaving
Cathode, so its hard band `bl·denom ≤ num ≤ bu·denom` already sat on the signed `Cathode→USC` and
`Cathode→MESC` legs; a band with `bl ≥ 0` is satisfiable for a negative pair only at zero, so no
USC/MESC edge-year was negative, prior or posterior, in 265,080 — the one-directional +16 kt/yr
residual ADR 0040 recorded without a cause. We decide that the hard ratio-bound builder emits no
one-signed band (lower bound ≥ 0, or an upper bound alone) on a pair in which either edge has a
negative structural lower bound, leaving the soft ratio equality to tie the valuations there (zero-
spanning bands, the stock bands, stay hard); that PSC gets an ordinary cathode-concentration row
(`concpscuse`, 0.9975, band 0.995–1.0: cathode held at the producer), so a reported CC value has a
Mass twin and both valuations are tied softly; and that PSC is a stock-change leg under ADR 0040's
rule in both valuations — barred from the recipe, its twin and the time interpolator, and seeded at
`0 ± max(0.01·|CP→Cathode|, 0.3 kt)` where no feed reports it — with the share transfer that once
priced it from its reported years retired (the recipe pre-empted it on every chain, and carrying a
signed stock change across years is what ADR 0040 bars). Rejected: a concentration row for PSC with
no band (`sd` only) — dodges the sign problem for one leg and leaves USC/MESC one-sided; the p95 band
as the seed price (≈ 3 % of production, the retired transfer's pricing) — a wider sink than ADR 0040
accepted for MESC/USC; keeping the eff-less zero as intended — contradicts ADR 0013 and hides the pin;
waiting for the ICSG parse fix (#35) — the mechanism is independent of the US values, which the
measurement reports separately. Consequences: USC and MESC may now go negative under the soft tie, so
the +16 kt/yr residual becomes a measurable two-sided quantity (P10) once the feed stops dropping the
negative rows (#40); ≈ 268k of the 844k hard concentration rows disappear; a PSC leg keeps the flat
prior only where the Cathode Pool has no product prior (≈ 348 region-years per valuation), still
outside the anchor; the garbled US PSC 2006–2012 (#35) now carries a Mass twin at its feed σ; the
`stock_bounds` params are a Stage-3 input only; the manuscript SI's share-transfer row is stale
until the Results are re-baselined; retention on PSC is quoted beside the mass moved, as for the other
signed legs.
