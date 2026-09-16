# Single-endpoint raw rows fan out, they do not replicate

A raw feed row that classifies only **one** endpoint (e.g. ICSG `Use` rows: consumer known,
producer unknown) must resolve to **one** source edge whose unclassified endpoint is left open
(`"<any>"`) and distributed across the matching root edges by normalised weights — the
[`README`](../../README.md) glossary definition of **fan-out**. `ConcordanceTable.from_raw`
([`conc.py`](../../src/pysuto/conc.py#L456)) previously filled the unclassified **to**-side with
`"<any>"` (correct) but the unclassified **from**-side with the *concrete matched-edge*
sector/region. For `Use` rows that materialised one source edge per producing region, each
assigned the **full** raw value — **replication, not fan-out** (see [`CONTEXT.md`](../../CONTEXT.md)).
ICSG inflated total mass **36×** (5,576 / 7,926 raw rows replicated, up to 48×), which corrupted
both the per-edge prior `a₀` and the `from_mask_anchor` control totals and drove the Stage-4
collapse (REP [`analysis_opt_20260602_133908_46bc514e_20260603_feed-scale-collapse.md`](../rep/analysis_opt_20260602_133908_46bc514e_20260603_feed-scale-collapse.md)).

The fix makes the fallback **symmetric**: any endpoint the raw data did not classify falls back
to `"<any>"` for both sides. The classified (primary) side is always non-null, so the change is a
no-op for `Supply` rows and only affects single-endpoint `Use` rows — bringing the code into line
with the documented fan-out definition.

**Three coordinated changes are required** (each later one surfaced when the mass-conservation
guard rejected the previous, incomplete state):

1. **Symmetric `<any>` sector/region fallback** ([`conc.py`](../../src/pysuto/conc.py#L465)) —
   collapse the producer fan to one source edge.
2. **Entity follows `<any>`** ([`conc.py`](../../src/pysuto/conc.py#L470)) — when an endpoint's
   sector is `"<any>"`, its entity must be `"<any>"` too; otherwise an open sector that spans a
   `Flow` producer *and* the `SB` import source re-splits into two source edges (the
   `sector → entity` 1:1 coupling does **not** hold for `"<any>"`, see CONTEXT.md).
3. **`build_F_src` must de-duplicate the fan-out** ([`scripts/icsg_feed.py`](../../scripts/icsg_feed.py#L252)) —
   `ct.tab` has one row per *(src edge × root edge)*, so a single raw row that fans across N roots
   appears N times; the `groupby(SRC_COLS).sum()` must run over unique `(raw_id, src edge)` pairs
   or it re-counts each raw value once per root and re-inflates onto the single `<any>` edge.
   (BACI / crude_ore already de-duplicate; ICSG did not.)

## Considered options

- **A. Symmetric `"<any>"` fallback for sector + region on both endpoints. (chosen.)** One source
  edge, weights split 1/N (uniform) and mass is preserved (`ΣF_root = Σraw`). Routes `Use` rows
  through the same weighted-fan path BACI already uses.
- **B. Also make `src_entity_*` fall back to `"<any>"` when its sector is `"<any>"`. (adopted.)**
  Initially deferred on the assumption that `sector → entity` is 1:1, but the mass-conservation
  guard rejected the build: an **open** sector spans entities (`Flow` producer + `SB` import
  source), so the concrete entity fallback re-split into two source edges (residual 1.43×
  inflation). Entity now follows the `"<any>"` sector. `<any>` on entity works in the weight-matrix
  builder because root nodes are always resolved from concrete edge attributes, never from `src_*`.
- **C. Require feed authors to map the producer region explicitly (config, not code).** Rejected:
  for genuinely single-endpoint data (ICSG publishes national consumption with no producer origin)
  there is nothing to map; forcing a concrete region would assert information we do not have.
- **D. Weight the producer split by BACI trade shares. (adopted 2026-06-08.)** The uniform split
  (option A) fixes the mass inflation but leaves every ICSG-CC trade value a *fan artifact* that
  disagrees with the real BACI-Mass observation on the same edge, breaking the `conc_*_supply` ratio
  (both-observed z up to ~10⁶ — see REP
  [`…_icsg-baci-trade-share.md`](../rep/analysis_opt_20260603_111330_0c1a19b7_20260608_icsg-baci-trade-share.md)).
  Now implemented via the `user_weights_df` mechanism (see Implementation below).

## Consequences

- **Feed output changes for single-endpoint `Use` feeds (ICSG).** `F_src`/`F_root`/`σ_root` shrink
  to mass-correct values; feeds must be rebuilt. `Supply`-only feeds (crude_ore) and bilateral
  feeds (BACI) are unaffected (their unclassified side was already `"<any>"` or fully classified).
- **A mass-conservation guard** (`|ΣF_root − Σraw| / Σraw ≤ tol`, hard error) is added at feed
  read-in as a feed-agnostic backstop — it would have caught the 36× inflation, and covers the
  option-B entity fragility.
- **Relies on the `sector → entity` 1:1 invariant** now recorded in CONTEXT.md.
- **Validation (verified in-process).** Rebuilding the ICSG concordance + `F_src`: raw rows mapping
  to multiple distinct source edges drops **5,576 (max 48×) → 0**; `ΣF_src` and `ΣF_root` both equal
  the matched raw total **2,069,585 kt** (was 74.7M — `36.08× → 1.0000×`); `F_root` max drops
  7393.1 → 4282.7. The mass-conservation guard passes.

## Implementation of option D — BACI-trade-share weighting (2026-06-08)

*Amended 2026-09-16 by [ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md): the
`BACI_Mass` the fan weights read is the **netted** bilateral matrix — re-exports re-attributed
to origin and destination — not the gross one. The domestic leg's "needs no ICSG production
data" property was already withdrawn by [ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md),
whose bound now covers Cathode and Anode as well as Concentrate.*

The producer fan is reweighted by BACI bilateral import shares for the four **traded** ICSG-fanned
sectors — **Cathode, Concentrate, Anode, Matte** (Scrap is an ICSG `Use` type but is **not traded** and
is excluded). Per (sector_from, sector_to, consumer Y, year t), in copper-content units:

    import X≠Y : raw weight = conc_sector · BACI_Mass_Froot(X→Y, t)
    domestic   : raw weight = max(0, ICSG_usage_CC(Y, t) − Σ_X conc_sector · BACI_Mass(X→Y, t))

`ConcordanceTable._impute_weights` normalises each src fan's raw weights to sum 1, so
`F_root_CC(import) = conc·BACI_Mass` (the `conc_*_supply` ratio then holds by construction) and
`F_root_CC(domestic) = usage − imports` (national usage conserved). The domestic leg is needed because
BACI has **no self-trade**; its weight falls out of `usage − imports` and needs no ICSG production data.
*(Amended by [ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md): for the bounded sectors
the residual is capped at the region's own production and the remainder goes to the import legs —
the "no production data" property let the fan book copper a region never produced.)*
The split is computed in **CC** (apply `conc` to BACI Mass) so low-conc sectors are unit-consistent.

- **Cross-feed dependency (new).** ICSG's fan weights now derive from BACI. Feeds stay loosely coupled
  via a versioned artifact: `scripts/build_trade_shares.py` reads the BACI feed `F_root`, the ICSG feed
  `F_root` (for usage — weight-invariant), and the `conc` param, and emits the **`trade_shares`**
  params-kind input (`params.csv`, so the params registry content-versions it). `scripts/icsg_feed.py`
  consumes it as `user_weights_df` (region+time-keyed) and records its entry in the feed sidecar; absent
  artifact ⇒ uniform fan (unchanged).
- **Time-variant.** Weights are per-year (`root_time` key); `_impute_weights` already normalises per
  `src_time`.
- **Fallback.** Per-consumer-group complete: covered groups emit weights for all fan targets (0 for
  untraded producers); groups BACI does not cover get no rows ⇒ uniform.
- **Validation (verified in-process, opt `opt_20260608_064938_44faa1f1`).** `conc_*_supply` both-observed
  z>1e3 @2010: cathode **75.9% → 10.0%**; per-family overall cathode 4.33%→0.57%, concentrate
  2.13%→0.09%, anode 1.20%→0.09%. ICSG mass still conserved (`ΣF_root = 2,069,585 kt`). The remaining
  residual is dominated by **transit-hub** trade edges (Switzerland/Netherlands), where BACI gross trade
  legitimately exceeds ICSG consumption (usage-vs-trade totals gap; imports compress, domestic clips to 0),
  plus a few stale-BACI domestic self-trade artifacts — both separate follow-ups, not the uniform-fan bug.
