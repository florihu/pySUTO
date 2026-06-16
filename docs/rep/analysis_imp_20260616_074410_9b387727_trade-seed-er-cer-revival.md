# Trade-seed + non-blocking recipe gate revives the ER→CER chain prior — imp_20260616_074410_9b387727

Verification of [ADR 0010](../adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)
(trade-first seeding + non-blocking `bilateral_import` recipe tier + impute consolidation).
Implements the fix designed in
[analysis_opt_…er-cer-prior-not-propagated.md](analysis_opt_20260611_140957_24eeecb0_20260615_er-cer-prior-not-propagated.md) §8.

- imp run (chain-3 retired): `imp_20260616_074410_9b387727` (label `trade-seed-chain3-retired`)
- imp run (chain-3 present): `imp_20260616_072703_963f435c` (label `trade-seed-verify`)
- feeds: `feed_icsg_20260611_131027_a7c03949`, `feed_baci_20260603_095116_12139840`,
  `feed_crude_ore_20260603_095651_558f7369` · acc: copper

---

## 0 — Executive summary

The all-or-nothing primary-input determinacy gate that left every deep-chain `agg=true` product at
`flat_prior_noisy_zero` is fixed. Cross-region `Flow→Process` import legs are now (a) **seeded** in CC
from their BACI Mass twin by the new `TradeImputer` and (b) treated as **0-when-missing, non-blocking**
by the network recipe gate and flow-balance. The deep refining chain forward-fills again and the
warm-start prior carries a real value.

| quantity | diagnosis baseline (opt …24eeecb0) | this run |
|---|---:|---:|
| `ER→CER [CC]` `network_forward` (of 1410) | **0** | **961** (889 + 72 capped) |
| `ER→CER [CC]` `flat_prior_noisy_zero` | 1349 | **445** |
| `ER→CER [CC]` 2020 prior sum (Σ 47 regions) | **1.83** | **20,822** (solved ≈ 13,551) |
| `trade_seed` fills (total) | — | **8,396** |

`CFR→Anode` and `Smelting→Matte` (100 % flat-prior in the baseline) likewise revive
(961 / 962 `network_forward` of 1410). The prior is now the same order of magnitude as the
solved value — the "rocket" is gone and the warm-start chain ([ADR 0009](../adr/0009-warm-start-prior-chain.md))
has something meaningful to carry.

---

## 1 — Chain method counts (CC), this run

| chain edge [CC] | network_forward | network_balance | flat_prior | trade_seed |
|---|---:|---:|---:|---:|
| `Smelting→Matte` (product) | 799 (+50) | 112 (+1) | 445 | — |
| `CFR→Anode` (product) | 921 (+40) | 1 | 445 | — |
| `ER→CER` (product) | 889 (+72) | — | 445 | — |
| `CER→CP` (flow out) | — | 933 (+28) | 445 | — |
| `Anode→ER` (inputs, 66,270) | — | 880 (+81) | 58,147 | 5,587 |

(`+n` = the `_capped`/`_clipped` variants.) Regions still flat-prior are those where the upstream
product is itself fully uncovered — strictly better than the baseline, and improved further by the
time-carry once any year closes.

## 2 — Chain-3 `*_domestic` retirement is provably a no-op

The hand-written `concentrate_/matte_/anode_domestic` Mass formulas are subsumed by the network's
`network_mass_twin` on the domestic legs (Anode→ER 874, Matte→CFR 861, Concentrate→Smelting 884
domestic Mass fills). **Across the whole log, `formula*` methods fired 0×.** Diffing the imp output
before vs after retiring the three relations:

- `f_agg` identical (`max|Δ| = 0.0`); every method count unchanged; total log entries equal (787,805).

So the retirement removes duplication (and the `anode_domestic` `entity_to=="Flow"` over-attribution
bug) with zero behaviour change. See ADR 0010 §C.

## 3 — Cathode FD-split trio: KEEP (not subsumable)

The Cathode flow node has **three** same-region domestic FD use legs (`Cathode→{FU,USC,MESC}`), beyond
the lone-gap determinacy of `_do_flow_balance`. Observed: `Cathode→USC` / `Cathode→MESC` Mass remain
**mostly flat-prior** (1278 / 1247 of 1410), confirming the network does not reach the 3-way split.
The `cathode_{fu,usc,mesc}_domestic` relations are retained. (They also fired 0× here — starved by
their mutual dependency when only one FD leg is unknown — a pre-existing latent condition, out of
scope for ADR 0010.)

## 4 — Ownership partition held

`diag.ownership_violations` over the run log returned **empty**: no cross-region edge filled by a
`network_*` method, no same-region edge filled by `trade_seed`. The Trade↔Network structural
partition (`from_region != to_region`) is intact.

## Artefact sources

- imp: `data/proc/imp/imp_20260616_074410_9b387727/{agg_result.npz, imputation_log.parquet}`
  (and `…_072703_963f435c` for the chain-3-present comparison)
- acc: `data/proc/acc/copper/{edge_index.csv, node_index.csv}`
- code: `src/pysuto/impute/{trade.py, network.py}`, `src/pysuto/param_relation.py` (`conc_twin_map`),
  `src/pysuto/diag/imp.py` (`ownership_violations`), `scripts/collect_aggregate_impute.py` (cascade)
- reproduction: read-only `numpy`/`pandas` loads of the above
- predecessor: `analysis_opt_20260611_140957_24eeecb0_20260615_er-cer-prior-not-propagated.md`
