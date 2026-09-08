# Round E — contradicted flow nodes: a negative result

*Runs: `imp_20260908_093526_6d137357 → opt_20260908_095337_ee8877ed` (anchor-only) and
`imp_20260908_095815_30ed4291 → const_20260908_100308_bde730af → opt_20260908_101628_420a30dc`
(both σ's), cold, 30/30 Solved each. Baselines: `opt_20260907_112840_a8677803` (Round C,
shipped) and `opt_20260908_084500_724780b3` (Round D). Change:
[ADR 0019](../adr/0019-contradicted-flow-nodes-lose-anchor-authority.md) — **rejected as a
remedy; detector retained**.*

**Headline: the σ remedy does not work, and it fails in the specific way we said we would not
accept.** It returns the z² metric to baseline while leaving the model in Round D's state.
The detector it was built on is worth keeping.

## 1 · What was tried

Round D restored 29,648 kt CC of ICSG refining data and total sum-row z² rose 4.3 %. The
mechanism was pinned to **contradicted flow nodes**: a Matte node's mass balance is *hard*, and
two tight scale totals sit across it — inflow from ICSG concentrate use, outflow from the
restored ICSG refining feed. Because `σ_c = ANCHOR_CV·|c|` scales with the total, the smaller
side is the cheaper anchor to satisfy, so the solver collapses the node to it (Russia 1995:
honouring the 545 kt inflow costs z ≈ 574, collapsing to the 50.8 kt outflow costs z ≈ 53).

Round E detects those nodes (`pysuto.flow_node_balance`, outflow < 0.5 × inflow — **48 Matte
node-years**) and strips the outflow side's authority: σ on its covered edges (Stage 2) and on
its `usetotout` anchor (Stage 3, the first live use of ADR 0016's `sigma_cv_resolver`).

Two variants were run, the first by accident and usefully so — the Stage-2 screen initially ran
*before* imputation, where a Flow node's inflow edge (`Smelting→Matte`) carries no feed
observation and therefore reads 0, so nothing was flagged and only the Stage-3 anchors loosened.

## 2 · Results

| | Round C (shipped) | Round D | E anchor-only | **E both σ's** |
|---|---|---|---|---|
| sum-row z² | 3,781,532 | 3,943,708 | 3,842,019 | **3,788,641** |
| total soft z² | 3.957 M | 4.121 M | 4.019 M | **3.966 M** |
| soft z p99 | 10.39 | 10.58 | 10.38 | **10.36** |
| world 2015 Conc-CC retention | 0.758 | 0.729 | — | **0.728** |
| world 2015 Smelt-CC | 0.710 | 0.683 | — | **0.682** |
| world 2015 Cath-CC | 0.996 | 0.991 | — | **0.987** |
| tailings z² | 8.60 k | 9.05 k | — | 9.00 k |
| Cathode_China max \|z\| | 18.14 | 18.15 | — | 16.12 |
| solver | 30/30 | 30/30 | 30/30 | 30/30 |

On the metric it works: z² is back to +0.2 % of the shipped baseline. **On the model it does
nothing.**

| node (anchor) | Round C | Round D | E anchor-only | E both σ's |
|---|---|---|---|---|
| Iran Smelting 1996 posterior (c = 116.0) | 103.6 | 4.2 | 4.6 | **4.6** |
| Russia Cathode-use 1995 (c = 657.1) | 582.0 | 34.0 | 34.0 | **34.0** |
| Russia Smelting 1995 (c = 545.0) | 477.5 | 29.8 | 28.9 | **29.0** |

Attribution of the −155,067 z² between Round D and Round E: −65,229 on the 204 loosened rows
themselves, −89,838 elsewhere; posteriors moved more than 1 % on **1,150 of 26,190** rows, and
**Σ|Δposterior| = 441 kt against a 96.1 Mt total — 0.46 % of the model.** The retention numbers
confirm it independently: they sit at Round D's degraded level, unmoved.

## 3 · Why it fails

σ cannot resolve a contradiction in the input data; it can only stop charging for it.

The anchor-only variant behaved exactly as the arithmetic predicted — the outflow anchor went
quiet (Iran Matte `usetotout` z 6.5 → 0.2) while the node stayed collapsed. Loosening **both**
σ's did not help either, and the reason is not σ: **there is nowhere for the chain to settle.**
Iran 1995–96 reports smelting 126/114 kt CC with refining 4.1/3.8, no anode exports and
~10 kt of cathode exports. Freeing the refining datum lets the solver raise it in principle,
but the matte would then have to leave through a refinery, an anode export or a cathode export
that the data says does not exist. The model collapses smelting because that is the only
available resolution.

Reducing the penalty while the physics stands still is the "prices away the symptom" failure
Round C rejected for hub σ exceptions. Rejecting it here for the same reason.

## 4 · What the round is worth

- **The detector ships** (`src/pysuto/flow_node_balance.py`, run in both stages, reported to
  `diag/contradicted_node_screen.csv`), with the σ constants set to `None`. It is a
  data-quality instrument: a flag means *the source's own two series disagree for that
  region-year*, which is a question for the data, not the solver.
- **35 of the 48 contradicted node-years are pre-existing** — Bulgaria 1997–2007, Brazil,
  Zambia — and were already collapsing in the shipped Round-C model, unnoticed. Round D did not
  create this class; it enlarged it by 13.
- **A data-side remedy is now well sized.** Of ADR 0018's 165 restored groups only **16
  (1,716 kt of 29,648)** fall in a contradicted node-year. Withholding exactly those — the fan
  stays undistributed, precisely as before Round D — would keep **94 % of the recovery**
  (149 groups, 27,932 kt) while removing the collapse trigger. This is the "leave those fans
  undistributed" option declined at the Round-E design point; the evidence since then is that
  the σ alternative chosen instead does not work.

## 5 · Decision needed

Round D + E sit unmerged on `fix/masscc-round-d`. Three ways forward:

- **(a) Round F — withhold the 16 contradicted restored groups.** Keeps 94 % of Round D's data
  recovery, should remove the regression at source rather than repricing it. Predicted, not
  measured: those region-years return to exactly their Round-C behaviour.
- **(b) Merge D as-is** (with E's detector, σ off): 29.6 Mt more real data, +4.3 % z²,
  −3 pp Conc/Smelt retention, 13 new collapsed node-years.
- **(c) Revert D and E**, keeping both as documented diagnoses — the fan-drop discovery
  (45 Mt silently dropped), the contradicted-node detector, and the finding that ICSG's early
  Refining series is unreliable for Russia, Iran, Bulgaria and Zambia.

Recommendation: **(a)**, then re-measure against both baselines before any merge.
