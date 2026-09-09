# Round H — the anode trade was never reconciled, and one flag was why

*Runs analysed (all 30/30 Solved cold, `--prior-mode cold`):*

| run | id | change |
|---|---|---|
| **F** | `opt_20260908_114321_279e7979` | shipped baseline (main\@3c24a46) |
| **G** | `opt_20260909_084958_2742afa3` | ADR 0021 anchor move only |
| **I** | `opt_20260909_112935_9fb45d44` | ADR 0022 `effer` pooling only |
| **H** | `opt_20260909_094009_49eab1dc` | both |

*Branch `fix/masscc-round-g`. Suite 895 passed / 1 skipped.*

## 1 · The finding

Cross-region **anode** trade was retained at **4–7 %** in every year and in every run of the
project's history, while every other traded sector reconciled normally:

| cross-region CC retention | 2000 | 2015 | 2022 |
|---|---|---|---|
| Concentrate | 104.8 % | 82.1 % | 80.2 % |
| Matte | 91.5 % | 106.5 % | 111.4 % |
| **Anode** | **5.6 %** | **6.7 %** | **4.7 %** |
| Cathode | 112.1 % | 99.2 % | 95.9 % |

2,087 kt CC of 2022 anode trade survived as 99 kt. The deletion was **forced, not cheap**:
anode legs carry a median cv of 0.0496 — as tight as cathode's 0.0481 — yet a median |z| of
**16.9** against cathode's 0.0, with 76 legs paying 114,115 z².

**Cause.** `effer` was the only process-efficiency row without `agg = true`, so it was emitted
per endpoint pair and resolved to the same-region leg only:

```
ratio_effcfr_Belgium_CC_2015                                        ← aggregate, pools every input leg
ratio_effer_ER.Belgium-CER.Belgium_Anode.Belgium-ER.Belgium_CC_...  ← the DOMESTIC leg alone
```

A refinery's output was tied to its domestic anode input, so **imported anode was not an input
to the ER recipe**. It could not produce cathode, so the solver deleted it — and exporters'
anode was pushed back onto their own domestic `Anode→ER` leg (Chile 394 → 9, Zambia 319 → 1,
Bulgaria 112 → 0.02, each with its domestic leg rising to absorb it). Belgium 2015 shows the
receiving end: `ER`, `CER`, `CP` and `Cathode` all sit at ≈156 kt against anchors of ≈380, every
row at z ≈ −34.5 — short by almost exactly the 214 kt of deleted anode imports.

`effer` is the only non-pooled recipe whose input sector is traded; `effsxew` and
`effmfflotation` take `Ore`, which has no cross-region edges. This is the configuration ADR 0005
had already rejected as its option C, on a recipe that ADR's scope never reached.

## 2 · The 2×2

| metric | F (shipped) | G (anchor) | **I (`effer`)** | **H (both)** |
|---|---|---|---|---|
| sum-row z² | 3,834,168 | 4,017,200 (+4.8 %) | **2,821,249 (−26.4 %)** | **2,815,511 (−26.6 %)** |
| total soft z² | 4,010,004 | 4,164,430 | 2,974,804 | **2,943,059** |
| soft \|z\| p99 | 10.47 | 10.58 | 8.00 | **7.45** |
| rows \|z\|>10 | 4,835 | 4,906 | 3,727 | **3,478** |
| max \|z\| | 71.1 | 72.5 | 63.5 | **62.2** |
| tailings z² | 8,840 | 8,206 | 6,221 | **5,734** |
| **anode trade 2015 / 2022** | 6.7 / 4.7 % | 6.1 / 4.3 % | 94.1 / 61.0 % | **95.7 / 71.4 %** |
| contradicted Matte nodes | 36 | **9** | 36 | **9** |
| world 2015 CO Mass | 0.649 | 0.666 | 0.815 | **0.827** |
| world 2015 Conc-CC | 0.681 | 0.735 | 0.809 | **0.855** |
| world 2015 Smelt-CC | 0.806 | 0.837 | 0.882 | **0.932** |
| world 2015 Cath-CC | 0.994 | 0.954 | **0.964** | 0.954 |
| Chile conc exports 2015 / 2019 | 0.545 / 0.415 | 0.566 / 0.395 | **0.868 / 0.438** | 0.858 / 0.393 |
| deletion deciles d0→d9 | 0.12→0.08 | 0.09→0.09 | — | **0.01→0.04** |
| solver | 30/30 | 30/30 | 30/30 | 30/30 |

The constraint matrix is **identical in row count between G and H** (1,364,708), so the z² fall
is signal, not a smaller problem.

Two long-standing items move for the first time. **`Cathode_China` max |z| 18.1 → 3.5** — named
in the Round-B REP as "the one genuine cost" of ADR 0017 and open since 2026-09-03 — and the
**`Cathode_ROW` z ≈ 70 tail**, for which four separate REPs recommended a targeted σ exception,
falls to 61.6 with no σ change at all. Both were downstream of this defect.

## 3 · Reading the 2×2 — a correction to the Round-G REP

**ADR 0022 is the load-bearing change**: −26.4 of the −26.6 points on its own.

**ADR 0021's apparent +4.8 % regression was an artefact of the baseline.** Measured against a
model whose anode trade was structurally deleted, moving the refining anchor onto `Anode→ER`
could not pay off — the edge it newly anchors was in a network the solver was discarding. On top
of ADR 0022 the same change costs 0.2 pp of z² and adds Conc-CC +4.6 pp, Smelt-CC +5.0 pp, 2022
anode-trade retention +10.4 pp and p99 8.00 → 7.45. The Round-G REP's §6 recommendation to hold
it is therefore **withdrawn**; its measurement stands as the thing that pointed here.

The two are complementary and neither substitutes for the other: ADR 0022 alone leaves all 36
contradicted Matte node-years in place; ADR 0021 alone leaves anode trade at 6 %.

The one measure that prefers I over H is Cath-CC (0.964 vs 0.954), with Chile 2019 exports
0.438 vs 0.393. Both are small against what H gains elsewhere, but they are the honest cost of
ADR 0021 and are not explained away.

## 4 · What this does not fix

- **Anode trade in 2022 reconciles to 71 %, against 96 % in 2000 and 2015.** The late-period
  residual is unexplained and is the natural next question.
- **BACI anode-export priors still exceed regional production in places** (Zambia 2022: 838 kt
  CC of anode exports against 727 kt of smelter output), so part of the 2022 gap is
  over-statement rather than under-reconciliation — the transit round's territory.
- The 9 remaining contradicted Matte node-years (all Argentina, 1–22 kt inflow) are noise scale.
- **Belgium's ICSG series is split** across `Belgium` (2013–2024) and `Belgium-Luxembourg`
  (1995–2013), overlapping in 2013, both mapping to root `Belgium` — 2013 is double-counted.

## 5 · Reproducibility warning

Both changes live in `data/`, which is **gitignored**:

- `data/input/params/eff/current/params.csv` — `effer` must carry `agg = true`;
- `data/input/feed/icsg/current/conc.xlsx` — `sector_map` row `Refining_Anode → ER`.

On a fresh checkout neither is present, and the model silently returns to deleting anode trade
and mis-anchoring the refining series. Same wart ADR 0014 and ADR 0015 record, now load-bearing
for ~26 % of the model's tension.
