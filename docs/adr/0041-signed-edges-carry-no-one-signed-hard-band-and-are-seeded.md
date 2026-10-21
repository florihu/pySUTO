# A signed edge carries no one-signed hard ratio band, and the producer stock change is seeded like the other stock legs

**Status:** split on measurement — Round Y, 2026-10-06 (spec #41; REP
[`…psc-mass-prior`](../rep/analysis_opt_20261005_134720_ee69cda9_20261006_psc-mass-prior.md)). The band
rule and the concentration row are **accepted** on Y.0 `opt_20261006_105904_61c5efe9` vs X.0
`opt_20261005_134720_ee69cda9`: Mass–CC gap on PSC 22–87 → 0–6 kt in every year, common-set sum-row z²
−0.1 %, Round-E within 0.008, 133,950 one-signed pairs skipped and USC/MESC two-signed for the first
time (23,800 negative edge-years each, drift +16/+13 → −6/−8 kt/yr). The PSC net-stock seed at 1 % of
`CP→Cathode` is **rejected on measurement** on Y.1 `opt_20261006_120237_643c4b4a`: the 870 seeded legs
move 7,943 kt CC over 30 years (272 kt/yr, limit 100; 2.65σ on average, 6.2σ at p90 — China −1,546 kt,
Zambia +857 kt at σ 3.8, Austria +356 kt at σ 0.9), the PSC stock-bound rows go from max |z| 4.48 to
22.6, CP→Cathode retention moves 0.028 in Chile, and the common-set z² falls 10.3 % because forced
contradictions became discretionary stock changes — ADR 0040's sink, not a fit. The eff-less hidden pin
(H5) therefore stays open; the no-trade floor alone (σ 0.3 kt) is the candidate price for a later
round, and the two seed commits (485aac4, bd41e29) are reverted so that the branch ships Y.0. Amends
[ADR 0040](0040-stock-change-legs-are-held-at-their-prior.md) (the band that made its seed one-sided is
gone) and [ADR 0039](0039-flow-lower-bounds-are-enforced-in-the-qp.md) (the signed legs' negative half
was still forbidden, by the concentration band). Closes #36 item 2 for the band and the row only; feed
follow-up #40.
**Amended 2026-10-21 (#36 item 2):** the eff-less hidden pin is priced at the no-trade floor alone.
Where a process has no `efficiency` row, the NetworkImputer writes the `0` it closes on a signed
output (structural l < 0, i.e. `CP→PSC`) at σ 0.3 kt, the feeds' resolution, instead of σ 0, which
Stage 3 floored to 1e-6: a zero closed by a missing parameter is not an observation. The prior stays 0 and
nothing is seeded; Mining's copperless `WR` CC, the other eff-less output, is unsigned and keeps its
exact zero. On `imp_20261021_100015_ea05007a` this reprices 857 CC `PSC` legs (38 regions, every
year; 3 of them time-interpolated copies) and the 857 Mass legs that follow them. To be measured.

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

**Measured 2026-10-21 (#36 item 2): rejected.** Priced at the 0.3 kt floor, the 857 eff-less CC PSC
legs and their Mass twins (`opt_20261021_115215_653c8677`, against `opt_20261021_111957_5c6d13bc`)
act as a sink again, as the seed did in Round Y: the near-pin PSC rows (σ_c < 0.01 kt) go from
Σz² 0.04 to 1,437 (Slovakia 2016 z +37.8, the United Kingdom 1996 +12.8) and 18 stock legs rise
above |z| 5 (Uzbekistan 5, −27.1 kt CC). `EFFLESS_STOCK_SIGMA_KT = 0.0` keeps the hidden pin, now
stated: CC PSC is held at zero wherever ICSG does not report it and CP has no `eff` row.
