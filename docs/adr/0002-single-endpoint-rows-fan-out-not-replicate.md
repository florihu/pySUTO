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
- **D. Weight the producer split by BACI trade shares immediately.** Deferred (follow-up): the
  uniform split fixes the mass inflation; trade-weighting improves accuracy and is a separate change
  via the existing `user_weights_df` mechanism.

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
