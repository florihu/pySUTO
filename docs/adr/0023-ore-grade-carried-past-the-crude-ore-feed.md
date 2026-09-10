# Past the crude-ore feed's last year, the in-situ grade is carried forward — not the ore mass

**Status:** accepted — merged 2026-09-10 (Round I, REP
[`…roundI-ore-grade-carry`](../rep/analysis_opt_20260910_114200_7f4040d9_20260910_roundI-ore-grade-carry.md)).
Extends
[ADR 0015](0015-in-situ-ore-grades-on-allocated-ore-basis.md); retires the Stage 2
`time_crude` extrapolator.

## Context

The `crude_ore` feed (S&P allocated ore mass) ends in **2022**; ICSG mine production continues
through 2024. The model's Mass chain at the extraction front-end is
`CO→Mining→Ore→{MF, SXEW}→{Concentrate, CSXEW, Tailings}`, coupled to its CC twin by the
region×year in-situ grade that ADR 0015 derives from exactly those two feeds — so the grade rows
end in 2022 too.

Stage 2 handled the boundary with a scoped `TimeImputer` (`time_crude`) that carried the 2022
`CO→Mining` Mass value forward onto 2023–24. Measured on the shipped Round-H run
(`opt_20260909_094009_49eab1dc`) that was insufficient in three linked ways:

1. **It ran after the network cascade**, so only the boundary leg was filled. `Mining→Ore`,
   `Ore→MF`, `Ore→SXEW`, `MF→Tailings` Mass stayed flat-prior noisy zero (σ = 10⁶) in 2023–24.
2. **Therefore no scale totals.** `supplytotout_Mining` and `usetotout_Ore` (Mass) are built from
   the aggregated prior; with the downstream legs at zero the group sum is zero and
   `from_mask_anchor` drops the row. 31 + 30 rows per year through 2022, **none** in 2023–24.
3. **No grade rows either**, so nothing tied the carried Mass to the CC chain that ICSG does
   cover.

The solver then collapsed the unanchored ore chain to a few kt at no cost:

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| world ore Mass prior (kt) | 2,901,579 | 2,901,579 | 2,901,579 |
| world ore Mass posterior | 1,868,040 | **430,776** | **391,066** |
| Chile posterior | 229,564 | **3,289** | **2,629** |
| soft z² that year | 135,281 | 63,296 | 61,183 |

The halved z² in the last two years is the tell: it is not a better fit, it is two years of missing
rows.

## Decision

**Carry the grade, not the mass.** `scripts/build_ore_grade.py` now appends, for every region-year
that has ICSG production but lies after the region's last observed grade, a row with the last
observed in-situ grade (`src=carry`, `from=<year>` recorded in the comment) on a band widened from
±10 % to **±20 %** of the region's series range (`MARGIN_REL_CARRY`). Tailings rows for those years
are computed from the carried grade and that year's own production, exactly as for observed years
— they are not themselves carried. Forward only, from the last observed year; regions with no
observed grade stay uncovered.

Nothing else changes. With a grade present the `NetworkImputer` derives the whole Mass chain for
2023–24 through the existing `network_mass_twin` path (Mass = CC / g) from ICSG production; the
scale-total rows and the grade ratio rows appear on their own because the machinery that builds
them is unchanged. `time_crude` is retired from Stage 2: it would have nothing left to fill, and
where it did fire it disagreed with the grade-derived value (it repeated 2022's mass regardless of
2023's production).

The assumption is stated where it belongs: **the in-situ grade changes slowly; ore mass follows
production.** Chile 2023 is the worked example — production fell 1.6 % from 2022, so ore mass
falls from 735,040 to 723,198 kt instead of being held at 735,040.

## Considered options

- **Keep LOCF on the mass and move it before the cascade.** Fills the chain but pins ore mass to
  2022 whatever 2023 produced, and still needs grade rows for the CC coupling — two carries where
  one suffices, and the two disagree by the production change.
- **Carry the mass and the grade both.** Over-determined: Mass·g ≠ CC unless production is also
  flat. The network would then relax one of them silently.
- **Leave 2023–24 uncovered and drop them from the scope.** Loses two years of the trade and
  refining chain that *is* covered.
- **A separate `TimeImputer` on the grade params.** Params are not edges; the imputer has no
  purchase on them. The generator is the one place that owns the grade series (ADR 0015).

## Consequences

- 57 region-years carried (29 in 2023, 28 in 2024), all from 2022; 87 tailings rows derived for
  them. `ore_grade` params **v4** (3,894 rows; v3 3,636).
- `scripts/collect_aggregate_impute.py`: `CRUDE_ORE_TIMES` and the step-6 `time_crude`
  registration removed. The `TimeImputer` `scope_feed_id` feature stays.
- Rows for years ≤ 2022 carry identical values to v3 (the carried value equals an observed one,
  so it cannot extend a band); only the tailings rows' comment gained a `src=` tag.
- The carried band is an assumption about grade drift, not a measurement; revisit when the
  crude-ore feed is extended and the carried years become observed.

## Measured (Round I)

Run `opt_20260910_114200_7f4040d9` (imp `imp_20260910_112302_4040ad37`, const
`const_20260910_112841_87020ec3`, cold, **30/30 Solved**) against the shipped
`opt_20260909_094009_49eab1dc`; `scripts/explo/roundI_ore_carry_diag.py`.

| | 2022 | 2023 base → new | 2024 base → new |
|---|---|---|---|
| world ore Mass prior (kt) | 2,901,579 | 2,901,579 → 2,905,679 | 2,901,579 → 2,903,665 |
| world ore Mass posterior | 1,868,078 | **430,776 → 1,692,634** | **391,066 → 1,649,218** |
| Chile posterior | 229,562 | **3,289 → 183,719** | **2,629 → 184,670** |
| Mining/Ore Mass anchor rows | 61 | 0 → 58 | 0 → 56 |
| `oregrade` ratio rows | 504 | 0 → 480 | 0 → 465 |
| sum-row z² | 128,472 | 62,055 → 103,091 | 58,451 → 96,747 |

- **Headline sum-row z² rises 2.8 %** (2,815,511 → 2,893,508). All of it is 2023–24 acquiring
  the rows they lacked; 1995–2022 is 2,695,006 → 2,693,670 (the only ≤2022 effect is
  `time_interp` now seeing filled 2023 neighbours). The two boundary years now sit at 100–105k
  soft z², in line with 2020–22 (135–152k) rather than half of it. This is the honest number.
- 2.5 Mt/yr of ore mass (world) and 180 kt/yr (Chile) are restored in each of the two years —
  the stop rule's "moves mass" test is met by a wide margin.
- The largest 2023–24 ore rows are China (z ≈ −52, posterior 40–48 kt against a 350 kt prior)
  and ROW (−48): the same pattern China shows in 2022 on the base run (94,659 vs 364,410). That
  is the pre-existing mining-collapse thread, not something this round introduces.
- Prior-level check: Chile 2023 `CO→Mining` Mass = 723,197.6 = 6,735.6 / 0.00931361, i.e.
  production ÷ carried grade, as designed; the chain `Mining→Ore`, `Ore→MF`, `Ore→SXEW`,
  `MF→Tailings` is filled by `network_mass_twin` where it was flat-prior zero before.

## Discovered alongside: the input registry's snapshots were not immutable

Regenerating `ore_grade` overwrote **v1–v3 in place**: the 2026 migration materialised `v1/` as
hardlinks back to `current/`, each later snapshot linked "unchanged" files onward, so a truncating
write to `current/` (pandas `to_csv`, `open(…, "w")`) rewrote every linked snapshot.
`scripts/explo/registry_audit.py` finds **28 of 93** version files whose content no longer matches
the sha256 the registry recorded at materialisation — across `params` (conc v1–2, ore_grade v1–3),
`feed` (icsg v1–5 `values.csv`/`conc.xlsx`, baci and crude_ore `meta.xlsx`) and `acc`
(`rules.ods` v1–2). The recorded hashes are still the truth of what each run consumed; the retained
files are not. Fix in `src/pysuto/input_registry.py`: `initialize()` gives every file under
`current/` a private inode before hashing (`_privatise_current`), so the live copy can never share
one with a snapshot again. Regression test added. `ore_grade` v2 and v3 are exactly reproducible
(regenerated from the generator at `696027c` and `8ac5720`, both sha256-verified against the
registry); `scripts/explo/restore_ore_grade_versions.py` writes them into `v2/`/`v3/` only when
the regenerated sha256 equals the registry record — run by the owner, since it is a write into
history. v1 and the feed/acc mismatches are not reproducible from the repo.
