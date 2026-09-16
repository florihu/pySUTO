# Chile's mine chain — the deletion is trade-mediated, and the concentrate grade is a global constant

**Run:** `opt_20260915_131115_b8bf05a9` (the Round L shipped run) · const `const_20260915_125839_3818080a`
· imp `imp_20260915_125341_f79df7f6` · BACI input v8.
**Scope:** read-only diagnosis of the defect named in [ADR 0027](../adr/0027-re-exports-are-re-attributed-at-feed-time.md)'s
decision and in REP `…roundL1-baci-reexport` Addendum 2 — the model deletes ~31 % of Chile's
ICSG-reported mine output and ~60 % of its ore, in every run to date. All figures CC kt, pooled
2010–2019, unless stated. **No code changed. No experiment run yet.**

---

## 1. It is not Chile's defect. It is trade's defect, and Chile is the largest trader.

Mine concentrate production (`supplytotout MF CC`), anchor vs posterior, by region:

| region | anchor | posterior | shortfall | retention |
|---|---|---|---|---|
| **Chile** | 38,000 | 24,877 | **13,124** | 0.655 |
| China | 15,672 | 10,373 | 5,299 | 0.662 |
| ROW | 8,963 | 6,375 | 2,588 | 0.711 |
| Russia | 7,107 | 6,703 | 403 | 0.943 |
| United States | 7,626 | 7,627 | −1 | 1.000 |
| Peru | 16,960 | 17,257 | −297 | 1.018 |
| Australia | 9,012 | 9,584 | −572 | 1.063 |
| **world** | **152,236** | **133,132** | **19,104** | 0.875 |

Chile alone is **69 % of the world shortfall**. But the split is not exporter-versus-importer, it is
**self-sufficient versus trade-dependent**. Smelter concentrate intake, posterior ÷ prior:

| holds at its anchor | loses a fifth to a third |
|---|---|
| Russia 1.024 · Zambia 1.014 · United States 1.027 · Poland 0.954 | **Korea 0.717 · Germany 0.737 · China 0.748 · Japan 0.789 · Philippines 0.806 · Brazil 0.832** |

Every region whose copper never crosses a border keeps its chain. Every region whose copper must
travel loses a quarter of it, at both ends of the journey. Chile is the largest single number
attached to a trade-dependent node, so Chile pays most.

## 2. Nothing downstream is backing up — the loss is created at the mine→smelter junction

World CC, anchor → posterior, 2010–19:

| stage | anchor | posterior | retention |
|---|---|---|---|
| Mining (ore) | 242,691 | 209,219 | 0.862 |
| MF (concentrate produced) | 152,353 | 133,248 | **0.875** |
| Concentrate use (smelter intake) | 156,556 | 133,451 | **0.852** |
| Smelting (matte) | 171,992 | 160,112 | 0.931 |
| CFR (anode) | 185,248 | 184,108 | 0.994 |
| ER (CER) | 182,923 | 183,932 | 1.006 |
| CP (cathode) | 220,803 | 221,286 | 1.002 |

The deficit is 13–15 % at the top and gone by the anode. It is **healed, not transmitted** — by two
mechanisms that are worth naming because both are silent:

- **Scrap backfills.** World scrap into smelting +1,839; into China's smelters alone **+5,718**
  (15,329 → 21,047) against a concentrate intake that falls 13,299.
- **Implied process efficiency drifts up.** World smelting `matte ÷ (concentrate + scrap)`
  0.917 → **0.963**; refining `anode ÷ (matte + scrap)` 0.881 → **0.983**. Partly a composition
  shift (the priors carry implausible implied efficiencies for Japan 0.773, Korea 0.660, Germany
  0.638, and the solver pulls all of them to ~0.96), but it means copper deleted upstream never has
  to reappear downstream.

## 3. Why the chain must move at all: the priors are globally infeasible

Σ|Flow-node imbalance| at the prior = **43,841 kt CC** over 2010–19 (posterior 0 — these are hard
rows). Chile's Concentrate node is the third worst at **3,643**:

    ICSG mine            38,000
    − domestic smelting  13,428
    − BACI exports × 0.275  21,520
    = surplus             3,053     copper produced with nowhere in the data to go

So the hard balance *must* destroy something at Chile. The prior is 9.5 % over-determined at this
node. The posterior destroys **13,124** — 4.3× the infeasibility it was given.

## 4. The amplifier: one global concentrate grade

`conc_Concentrate` in `data/input/params/conc/v6/params.csv` is a **single global constant**:

    concconcentratesupply | a_region = * | value_nominal 0.275 | min 0.20 max 0.35
    concconcentrateuse    | a_region = * | value_nominal 0.275 | min 0.20 max 0.35

Every bilateral concentrate leg in the model, from every mine to every smelter, is priced at 27.5 %
Cu. The grade each region's own node would require to close is not 0.275 and is nowhere near
uniform:

| exporters — grade needed | importers — grade needed |
|---|---|
| Bulgaria 0.572 · ROW 0.442 · **Chile 0.325** · Congo 0.321 · United States 0.313 · Canada 0.267 · Brazil 0.266 | India 0.466 · Brazil 0.384 · Spain 0.371 · Germany 0.320 · Korea 0.315 · **China 0.292** · Japan 0.260 |

**Chile needs 0.325 — inside the parameter's own declared band [0.20, 0.35].** Raising the grade on
Chile's export legs to 0.325 would close its node exactly, and Chilean concentrates really do assay
around 28–32 % Cu. The solver went the other way, to the **floor**: see §5a — 35 % of Chile's export
leg-years sit exactly on 0.20, including 100 % of the Chile→Japan lane and 80 % of Chile→China.

This is the same shape of defect [ADR 0015](../adr/0015-in-situ-ore-grades-on-allocated-ore-basis.md) fixed for
**ore** grades — "region×year in-situ grades replace global 1 % rows" — and never fixed for
**concentrate** grade.

## 5. What Chile pays, row by row

Per year, 2010–19 (`σ_c` is 1.7 % of the anchor throughout):

| row | anchor c | posterior | z |
|---|---|---|---|
| `supplytotout Mining Mass` | 664,178 (2010) … 781,314 (2019) | 287,520 … 300,946 | **−33 … −38** |
| `supplytotout Mining CC` | 7,044 … 7,439 | 4,778 … 5,001 | −19 … −26 |
| `supplytotout MF CC` | 3,330 … 4,207 | 2,332 … 2,944 | −17 … −28 |
| `usetotout Concentrate CC` | 2,787 … 4,489 | 2,332 … 2,944 | −6 … −20 |
| `oregradeoresupplychile<year>` | — | — | **+13 … +15** |

The last row is the tell, and §5a explains it. The prior satisfies every one of these rows exactly
(`lhs_prior == c`), and no *box* bound (`l`/`u`) is active anywhere in the chain — but the hard
**ratio cones** are, and they are what shapes the answer.

**Why it can choose it.** After ADR 0017 damping, Chile's concentrate export legs carry
**cv 0.069**; its scale-total anchors carry **cv 0.017**. The leg is four times softer in relative
terms, so the disagreement between ICSG's two ends — its mine figure and its smelter figures — is
discharged through the trade that connects them. Both ends then have to follow the leg down.

## 5a. The bands are hard, and both of them are pinned — in opposite directions

`value_min` / `value_max` in the params CSVs are not prior shaping. `RatioBoundBuilder`
([`ratio.py:403`](../../src/pysuto/constraints/ratio.py)) turns each into a **hard non-negative
cone**, so every concentrate edge carries three rows — one soft ratio plus a hard lower and a hard
upper (`row_meta`: 66,270 soft + 66,270 lower + 66,270 upper for `concconcentratesupply`). The same
holds for `ore_grade`. Chile is pressed against one of each:

**Concentrate grade, at the FLOOR.** Posterior CC/Mass per edge, 2010–19:

| | legs at 0.20 | legs at 0.35 | median grade |
|---|---|---|---|
| all concentrate legs, world | 4.9 % | 4.5 % | 0.277 |
| **Chile's export legs** | **35.3 %** | 0.0 % | 0.231 |
| Chile → Japan | **100 %** | 0 % | 0.200 |
| Chile → China | 80 % | 0 % | 0.201 |
| Chile → Korea · Germany | 80 % | 0 % | 0.205 |

Chile's own domestic leg, by contrast, stays at 0.24–0.28 — the pressure is on the legs that cross
a border. The solver is not choosing 0.21; it is **held at 0.20 and would go lower**.

**Ore grade, at the CEILING.** Chile sits at `value_max` in **10 years out of 10**:

| year | nominal (ADR 0015, region×year) | `value_max` | posterior | at cap |
|---|---|---|---|---|
| 2010 | 0.01061 | 0.016617 | 0.016617 | ✔ |
| 2015 | 0.01016 | 0.016617 | 0.016617 | ✔ |
| 2019 | 0.00952 | 0.016617 | 0.016617 | ✔ |

Across the world only 5.9 % of region-years sit at the ore-grade cap; of the twelve biggest ore
producers **only Chile (10/10) and China (7/10)** do — and they are exactly the two whose ore mass
collapses (0.386, 0.555) while the United States (0.990), Peru (0.962), Zambia (1.014) and
Indonesia (0.990) hold.

**Why the cap is so permissive.** ADR 0015 made the ore-grade *nominal* region×year, and it is a
real declining series — Chile 1.51 % (1995) → 0.95 % (2019). The *band* was not: Chile's
`value_min`/`value_max` are constant across all 30 years at 0.9 × and 1.1 × the **period** extremes,
so the cap is 1.1 × Chile's **1995** grade. By 2019 the ceiling sits **75 % above** that year's own
nominal. In a region with a strong secular grade decline the band therefore licenses the solver to
price this year's ore at the grade of twenty-five years ago.

**Which gives the deletion an exact decomposition.** Chile's ore-mass retention is not one number
but the product of two:

    ore Mass retention 0.42  =  0.69  ×  0.61
                                 │       └─ rock discarded at the grade ceiling (g_prior ÷ g_cap)
                                 └───────── copper deleted (the trade story, §1–4)

So of Chile's "60 % of its ore", roughly half is the copper story and half is a **parameter-band
artefact**: once a tonne of copper is deleted, the ceiling decides how many tonnes of rock leave
with it. Widening the bands would make this worse; the fix is to make them year-local.

## 6. What this establishes, and what it does not

**Established.** The deletion is trade-mediated and systemic, not Chilean; it is created at the
mine→smelter junction and healed by scrap and by efficiency drift; the priors are genuinely
infeasible at Chile's node by 3,053 kt CC; the solver amplifies that 4.3×; and the single global
0.275 concentrate grade is demonstrably wrong for most regions, with the correction Chile needs
sitting inside the parameter's own band.

Added by §5a: the hard ratio cones are the instrument. Chile's export legs are pinned on the
concentrate floor (0.20) and its ore on the grade ceiling (0.016617, 1.1× its 1995 grade) in every
year, and the two together decompose the ore loss exactly into 0.69 copper × 0.61 rock.

**Not established.** That regionalising the grade *fixes* it. The amplification factor of 4.3×
says most of the damage is in how the system discharges infeasibility, not in the infeasibility
itself — so closing Chile's 3,053 may recover far more than 3,053, or may simply move the deletion
to whichever node becomes the softest. That is an experiment, not an inference.

## 7. Next

Two changes, and they are independent — measure them separately, in this order, because the second
is cheap and removes half the symptom without touching the physics argument.

1. **Year-local ore-grade bands.** `value_min`/`value_max` should bracket *that year's* nominal
   (ADR 0015 already produces a region×year series), not the region's 30-year extremes. Chile's
   ceiling is currently 1.1× its 1995 grade and binds in 10 years out of 10. This is a
   one-parameter-file change and it directly bounds how much rock can leave with each tonne of
   deleted copper. Expect Chile's ore retention to move from 0.39 toward ~0.65 without any change
   in the copper story.
2. **Region×year concentrate grade** — the ADR 0015 treatment applied to `conc_Concentrate`, which
   is still one global 0.275 with a global [0.20, 0.35] band. Chile's legs are pinned on that floor;
   a realistic Chilean band (~0.28–0.32) forbids emptying them and forces the imbalance to surface
   where it can be seen instead of being absorbed as grade. Sources: exporter-declared concentrate
   grades, or per-region derivation from the ICSG/BACI consistency condition in §4.
3. **Then re-read the amplification.** If Chile's node closes and the mine still deletes, the fault
   is in the discharge path (leg cv 0.069 versus anchor cv 0.017), an ADR 0016/0017/0025 question —
   related to the open `a_ref` item in [TODO](../TODO.md).

Throughout: do **not** read an exporter-retention change as a verdict on any round until this is
fixed; it is 36 % deleted at baseline and moves with anything that touches trade.
