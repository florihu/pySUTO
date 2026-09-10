# Round I — the ore-mass chain past the crude-ore feed's last year (ADR 0023)

- run: `opt_20260910_114200_7f4040d9` (cold, 30/30 Solved) ← `const_20260910_112841_87020ec3`
  ← `imp_20260910_112302_4040ad37` ← feeds icsg `20260909_082747_a347d543`, baci
  `20260902_085153_de9e393f`, crude_ore `20260603_095651_558f7369`; params `ore_grade` **v4**
  (conc v6, eff v3, stock_bounds v4 unchanged)
- baseline: `opt_20260909_094009_49eab1dc` (shipped Round H)
- reproducer: `python scripts/explo/roundI_ore_carry_diag.py opt_20260909_094009_49eab1dc opt_20260910_114200_7f4040d9`
- branch `fix/ore-grade-carry-forward`; ADR [0023](../adr/0023-ore-grade-carried-past-the-crude-ore-feed.md)

## 1. What was wrong

The `crude_ore` feed ends in 2022. Stage 2 carried the 2022 `CO→Mining` Mass forward with a scoped
`TimeImputer` (`time_crude`) — but that step ran *after* the network cascade, so the rest of the
chain (`Mining→Ore`, `Ore→MF`, `Ore→SXEW`, `MF→Tailings` Mass) stayed at flat-prior zero, the
Mining/Ore Mass scale totals were dropped as empty groups, and the ADR 0015 grade rows (which come
from the same two feeds) ended in 2022 too. The solver then collapsed the unanchored chain:

| base run | 2022 | 2023 | 2024 |
|---|---|---|---|
| world ore Mass prior (kt) | 2,901,579 | 2,901,579 | 2,901,579 |
| world ore Mass posterior | 1,868,040 | 430,776 | 391,066 |
| Chile posterior | 229,564 | 3,289 | 2,629 |
| Mining/Ore Mass anchor rows | 61 | 0 | 0 |
| soft z² | 135,281 | 63,296 | 61,183 |

The halved z² in the boundary years was flattering the headline metric by ~75k.

## 2. The change

`scripts/build_ore_grade.py` carries each region's last observed in-situ grade into later years
that have ICSG production (`src=carry`, band ±20 %); tailings rows for those years are derived
from the carried grade and the year's own production. The NetworkImputer then fills the whole
Mass chain from production through the grade. `time_crude` is retired. 57 region-years carried
(29 + 28, all from 2022), 87 tailings rows; values for years ≤ 2022 identical to v3.

## 3. Measured

| | 2022 | 2023 base → new | 2024 base → new |
|---|---|---|---|
| world ore Mass prior | 2,901,579 | 2,901,579 → 2,905,679 | 2,901,579 → 2,903,665 |
| world ore Mass posterior | 1,868,078 | **430,776 → 1,692,634** | **391,066 → 1,649,218** |
| Chile posterior | 229,562 | **3,289 → 183,719** | **2,629 → 184,670** |
| Mining/Ore Mass anchor rows | 61 | 0 → 58 | 0 → 56 |
| `oregrade` ratio rows | 504 | 0 → 480 | 0 → 465 |
| sum-row z² | 128,472 | 62,055 → 103,091 | 58,451 → 96,747 |
| all soft z² | 135,279 | 63,296 → 110,162 | 61,183 → 104,948 |

| totals | base | new | Δ |
|---|---|---|---|
| sum-row z², 30 y | 2,815,511 | 2,893,508 | **+2.8 %** |
| sum-row z², 1995–2022 | 2,695,006 | 2,693,670 | −0.05 % |
| all soft z², 30 y | 2,943,059 | 3,032,263 | +3.0 % |

**Read the +2.8 % as the removal of an artifact, not a regression.** Every unit of it is 2023–24
acquiring 114 anchor rows and 945 grade rows they lacked; the pre-2023 years are untouched to
0.05 % (the residual is `time_interp` seeing filled 2023 neighbours). 2.5 Mt/yr of ore mass is
restored at world level, 180 kt/yr for Chile.

Largest 2023–24 ore-chain rows on the new run: China `supplytotout_Mining` z −52 (posterior
40–48 kt vs 350 kt prior) and ROW −48 — the same shape China already shows in 2022 on the base
run (94,659 vs 364,410). That is the paused mining-collapse thread, unchanged by this round.

## 4. Stop-rule check

Round E's signature was metric movement without mass movement. Here the metric moves *against*
the change and mass moves by 2.5 Mt/yr — the opposite signature. No constant was tuned.

## 5. Discovered alongside — registry snapshots were mutable

Regenerating `ore_grade` rewrote **v1–v3 in place**: the migration had materialised `v1/` as
hardlinks back to `current/`, and every later snapshot linked "unchanged" files onward, so
pandas' truncating `to_csv` on `current/params.csv` reached every version.
`python scripts/explo/registry_audit.py` → **28 of 93** version files no longer match their
recorded sha256 (params conc v1–2, ore_grade v1–3; feed icsg v1–5 `values.csv`/`conc.xlsx`,
baci and crude_ore `meta.xlsx`, baci v1 `weights.csv`; acc `rules.ods` v1–2). Recorded hashes
remain the truth of what each run consumed — every run's `metadata.json` carries them — but the
retained files are not that content.

Fixed in `src/pysuto/input_registry.py`: `initialize()` now gives every file under `current/` a
private inode before hashing (`_privatise_current`), so a live-copy write can never reach a
snapshot again; regression test added. The first pipeline run after merge repairs the 12 still-
shared `current/` files listed by the audit.

**Restorable with proof:** `ore_grade` v2 and v3 regenerate exactly from the generator at
`696027c` and `8ac5720` (sha256 `25910e35…` and `3ebf3b3d…`, both equal to the registry record).
Writing them into `v2/`/`v3/` is a write into history and was deliberately left to the owner.
`ore_grade` v1, and the feed/acc mismatches, are not reproducible from the repo alone.

## 6. Merge recommendation

Ship. Both parts are corrections of stated invariants (the chain is derived from the grade;
snapshots are immutable), and neither moves anything before 2023.
