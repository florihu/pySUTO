# A stock change is drawn from its region's cathode supply, and BACI's cathode trade carries no stock prior

**Status:** accepted — #40.1, 2026-10-12, shipped over a failed pre-registered item by the
maintainer's decision (run `opt_20261012_122124_40b1b370`, 30/30 Solved, against Round Z
`opt_20261008_092950_625a6c7e` and #40's run `opt_20261012_084619_3c11d751`: hub use totals z²
160,444 → 841, mixed-sign stock legs 1,250 → 0, common-set sum-row z² +0.13 % vs Z, Round-E held;
item 3 failed, the gross stock-leg adjustment exceeding Z's by > 100 kt in 10 of 30 years while the
net stayed within ±91 kt except 1995, −163.5 kt, which has no stock data; REPs
[`…issue40-signed-stock-fans`](../rep/analysis_opt_20261012_084619_3c11d751_20261012_issue40-signed-stock-fans.md),
[`…issue40-1-supply-mix-fans`](../rep/analysis_opt_20261012_122124_40b1b370_20261012_issue40-1-supply-mix-fans.md)).
Amends [ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md) for the signed legs and
[ADR 0040](0040-stock-change-legs-are-held-at-their-prior.md) for the stock legs' prior. Issues #40,
#36 item 1; follow-up #37.

ICSG reports a stock change only at the region holding the stock, so its MESC and USC rows fan over
origin regions like any single-endpoint row (ADR 0002). Two rules fanned them wrongly. The
trade-share builder emitted weights only for a positive change, so every draw-down stayed
undistributed. Fanning by magnitude then exposed the flow rule underneath: ADR 0024's domestic
residual put a hub's draw-down on cathode the hub never refined, wherever its imports could not
carry it. The UAE's −59.9 kt in 2003 is one case; 48 use totals at six exchange hubs went to
z ≈ 58.6. We decided that **a stock change is drawn from the region's cathode supply in
proportion**: own production on the domestic leg, each partner's cathode exports to the region on
its import leg, read from the route's Cathode→FU leg. A region with neither production nor imports
leaves the change to the net-stock seed. We also decided that **BACI's sector-pair cathode fan is
100/0/0** onto FU/USC/MESC. Its 1 % shares had given every stock leg a positive Mass prior beside
ICSG's signed CC (1,250 mixed-sign leg-years). They had also built +5.8 Mt of exchange and user stock
over 30 years, while ICSG's world stocks of every type rose 0.46 Mt.

**Considered options.**
- *Imports first with the production bound unconditional*, the first build of this round. Once the
  full cathode trade replaced the 1 % signal, it put China's exchange draw-downs wholly on imported
  cathode, although China refines about three times what it imports. Rejected before measurement.
- *Keep 98/1/1 and fix only the fan.* Restores the mixed signs and the +5.8 Mt build. Rejected.
- *Hold the change undistributed whenever imports cannot carry it.* Drops the hubs' largest
  draw-downs. Rejected.

**Consequence.** With the 2 % gone, the solver moves stock changes between legs about 2–3.6 times as
much as in Z, and in 1995 it draws on zero-prior legs. That is the net-stock seed's σ question,
filed on #37.
