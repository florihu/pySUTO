# Stock-change legs are held at their prior: reported or zero, one per cent or 0.3 kt

**Status:** accepted — Round X, 2026-10-05 (run `opt_20261005_134720_ee69cda9`, 30 years Solved:
world MESC posterior 15,657 → 4,259 kt CC over 30 years against a 3,869 kt prior, cumulative
drift +9.6 Mt → +0.39 Mt, common-set sum-row z² −0.4 % vs W.1, Round-E items held; the residual
+16 kt/yr on each signed family is recorded; REP
[`…mesc-usc-stock-sink`](../rep/analysis_opt_20260930_092111_e3e0677d_20261005_mesc-usc-stock-sink.md)).
Retires the zero-centred MESC/USC net-stock rows of the soft-net-stock-bounds fix
([REP 2026-09-01](../rep/analysis_opt_20260831_121315_645bbff1_20260901_soft-net-stock-bounds-fix.md));
leaves [ADR 0039](0039-flow-lower-bounds-are-enforced-in-the-qp.md) (signed stock legs have no
lower-bound row) and the PSC rows unchanged. Issue #34; follow-ups #35, #36. Amended by
[ADR 0041](0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md) (Round Y): PSC joins
the seeded legs with base `CP→Cathode`, the PSC share transfer is retired, and the one-signed hard
concentration band that made this seed one-sided on USC/MESC is no longer emitted on a signed edge.

The three stock-change legs are the only signed edges of the account, so whatever σ they carry is
the price of routing a contradiction of the cathode balance into a stock instead of surfacing it
as `z` on production, exports or use. On the W.1 reference run (`opt_20260930_092111_e3e0677d`)
the `Cathode(i)→MESC(j)` legs into the exchange locations (Netherlands, Singapore, United
Kingdom) were seeded at σ = (p95/1.645)·max(|FU(i→j)|, 100 kt) with a cluster p95 of 9.1 — a
ratio of a warehouse country's stock change to its own small semi use — i.e. **553 kt on 484
legs**; the matching Stage-3 rows had the same σ_c through a 100 kt scale floor that the driver
comment and REP 0611 reasoned in tonnes; and, before the seed, the NetworkImputer's Flow-node
balance wrote the cathode residual onto the domestic stock leg (51 fills, 2,780 kt) while the
time interpolator invented stock changes between reported years (96 fills, 3,034 kt). The
posterior MESC was +325…+898 kt every year, 15.7 Mt over 30 years against a 6.0 Mt prior;
Zambia 2010 moved 121 kt to a Dutch warehouse at z 0.22 while its use scale total sat at z 20.5.
We decide that a stock leg's prior is a reported value (ICSG exchange or user stocks at the
exchange location, the BACI-fanned share) or a zero-centred seed, that no imputer manufactures
one (the balance residual and the interpolator are barred from `MESC`/`USC`, and the domestic
USC/MESC residual formulas of the FormulaImputer — dormant on W.1 only because the balance wrote
the twin first, 7,290 kt of one-signed stock change once it did not — are retired, and with them
the FU residual, which the balance now reaches first (0 fills on X.0; its export term never
matched an edge, so where it did fire, W.1, it booked production/conc with no export subtracted:
28,740 kt on 226 legs the network writes at 8,845 kt); and the
balance never derives a node's supply from uses that include a stock leg, since with the stock
legs counted as zero it wrote `CP→Cathode` from the trade legs before the recipe could, 836 fills
and a fabricated German mine chain on the first chain), and that the seed
is priced like a feed-backed level, ADR 0025's rule, σ = max(0.01·|co-located FU flow|, 0.3 kt):
a no-trade pair is pinned, China→China may move 44 kt, Zambia→Netherlands 0.3 kt. The
zero-centred MESC/USC rows are retired — redundant once the edges are tight and, on a region
with a reported stock change, a pull of the data point toward zero (US: z ≈ 2.4) — while the PSC
rows stay (stock_bounds v5: PSC only, root labels, scale floor 0.1 kt). Rejected: a pooled-p95
band (≈ 6 % of the leg) plus an absolute ceiling — two new constants to encode what one existing
rule already says, and the big-consumer domestic legs would have stayed a 270 kt sink; a hard
band — went infeasible in 1995 under the zero pin; a constant σ of 0.3 kt on every leg — treats
a 4 Mt flow like a 1 kt one; an exchange-stock level anchor from LME/SHFE/COMEX data — the ICSG
series the feed already carries is that data, and 1995, 2005–2013 and 2015 are missing from it
(#35). Consequences: the cathode contradictions now land where the data disagree (Zambia:
production 527 vs uses 377 kt CC in 2010) and must be read there, not in a stock; retention on
`MESC`/`USC` is quoted only where the prior sum exceeds 50 kt and always beside the mass moved;
the manuscript's reference chain changes once the round ships, so the Results are re-baselined
in a separate step; the one-signed BACI 98/1/1 prior and the unanchored no-data PSC are the
next levers (#36).
