# Function tags for paper-skeleton atoms

The `function` field on each atom is a drafting hint, not a strict template — it tells the LLM what rhetorical mode the paragraph operates in, so the prose comes out in the right register. Eleven tags, grouped by where they typically appear.

## Quick reference

| Tag | Where it lives | What it does |
|---|---|---|
| `motivation`     | Intro, sometimes Discussion | Sets the stakes |
| `state_of_art`   | Intro, Methods, Discussion  | Positions vs prior work |
| `research_gap`   | Intro                       | Names what's missing |
| `contribution`   | Intro                       | Headline novelty claim |
| `synthesis`      | Discussion, Conclusion      | Pulls earlier claims together |
| `claim`          | Methods, Results            | Asserts a specific result/fact |
| `definition`     | Methods                     | Pins notation or terminology |
| `derivation`     | Methods                     | Multi-step algebraic chain |
| `evidence`       | Results                     | Data/numbers supporting a claim |
| `justification`  | Methods, Discussion         | Defends a choice — "why X not Y" |
| `qualification`  | Methods, Discussion         | Caveat, scope limit |
| `transition`     | Anywhere (sparingly)        | Bridges sections |

---

## Argument-setting

Mostly Intro / Discussion / Conclusion.

### `motivation`
Why the problem matters; sets the stakes.
**Drafting:** one-paragraph hook, no method or solution yet.
**Example:** an opening atom — "global Cu accounts are assembled from disagreeing sources with no unified σ treatment."

### `state_of_art`
What has been done; positioning vs prior work.
**Drafting:** named tools/papers with brief characterisation, no critique yet.
**Example:** `meth-arch-02` — "Relative to ODYM, AISHA, PIOLab, Lenzen 2009 …".

### `research_gap`
What is missing in the existing literature/tools.
**Drafting:** one specific lack, not a general complaint.
**Example:** an intro atom stating "no existing framework composes pedigree-derived σ propagation with a global QP."

### `contribution`
The headline novelty claim.
**Drafting:** explicit "we propose / introduce / extend." Reserved for *one* atom per paper.

### `synthesis`
Pulling together claims established earlier.
**Drafting:** no new sources, only callbacks to prior atom ids. Used in the conclusion and sometimes at the end of the discussion.

---

## Argument-load-bearing

Mostly Methods / Results.

### `claim`
Asserts a specific result, formula, or design fact and supports it briefly.
**Drafting:** state the claim, give the supporting math or code reference, stop.
**Example:** `meth-s2-01` — aggregation as one matrix op. Workhorse tag — perhaps 60 % of methods atoms.

### `definition`
Introduces a symbol, term, or notation that downstream atoms use.
**Drafting:** pin the meaning unambiguously, give dimensions/types, do not editorialise.
**Example:** an atom that introduces ε as a free decision variable acting as residual slack.

### `derivation`
Shows the algebraic or logical chain that produces a result, in multiple steps.
**Drafting:** equations linked by `=` / `⟹` and brief connective sentences; cite the standard form being adapted. Use sparingly — most methods atoms are `claim`, not `derivation`.
**Example:** a full Lagrangian derivation of the (P*) optimality conditions.

### `evidence`
Presents data, output, or computation that supports a claim.
**Drafting:** numbers come from `data_refs.yaml` or a diagnostics notebook; no claims not backed by a number. Dominates the Results section.
**Example:** a Results atom reporting κ(Σ_a^{−1}) ≈ 10⁶ with ZeroCone vs ≈ 10¹² with σ_c → 0.

### `justification`
Defends a design choice; the "why X, not Y" atom.
**Drafting:** state choice, name the alternative rejected, give one or two reasons. Pre-empts reviewer questions.
**Example:** `meth-s4-03` — why ZeroCone over a small-σ proxy.

### `qualification`
Caveat, scope limit, or boundary condition on an earlier claim.
**Drafting:** "this holds for p ≥ 1", "the framework assumes static admissibility", etc.
**Example:** `meth-lp-02` — the diversification invariant requires p ≥ 1.

---

## Connective

### `transition`
Bridges between sections or sub-arguments.
**Drafting:** one or two sentences; no new content. Use sparingly — most transitions belong in the human stitching pass, not as standalone atoms.

---

## Usage rules

- `contribution` and `research_gap` appear **exactly once** per paper. Two of either means the argument has split and needs consolidating.
- `qualification` requires an explicit `depends_on` pointing at the atom it qualifies. A standalone caveat is a sign the qualified claim is missing.
- `claim` is the safe default; reach for the more specific tags only when the atom clearly does one of the other jobs.

---

## Rhetorical balance — a sanity-check before drafting

- Methods section heavy on `claim` and thin on `justification` → reads as asserted rather than defended.
- Results section heavy on `claim` and thin on `evidence` → overclaiming.
- Intro with three `state_of_art` atoms and zero `research_gap` → unbalanced; the gap is the point.

A quick scan of the function-tag distribution per section is a cheap sanity check before any prose is written.