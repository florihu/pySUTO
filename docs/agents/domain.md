# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is **single-context**. There is no separate `CONTEXT.md`; the canonical
domain/spec source is **`README.md`** at the repo root.

## Before exploring, read these

- **`README.md`** — the architecture and spec doc. Use its §-numbered project map:
  - §2 — architecture and the 5-stage pipeline (feed → agg/imp → const → solver → uncprop)
  - §3 — module reference (where each class lives)
  - §5 — file structure and the one-way import direction
  - §6 — test coverage map
  - **`## Glossary`** — canonical domain vocabulary (e.g. *period slice* `a*(t)` vs *complete a\**, *cross-year placeholder*, *no cross-time coupling*).
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If `docs/adr/` doesn't exist yet, **proceed silently**. Don't flag its absence;
don't suggest creating it upfront. `/grill-with-docs` creates ADRs lazily when a
real architectural decision gets resolved, and maintains the README `## Glossary`.

## File structure

```
/
├── README.md          ← architecture, spec, and `## Glossary`
├── CLAUDE.md          ← behavioral guidelines + Agent skills config
├── docs/
│   ├── adr/           ← architectural decision records (created lazily)
│   ├── rep/           ← analysis reports
│   └── agents/        ← this skill's config
└── src/pysuto/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal,
a hypothesis, a test name), use the term as defined in the README `## Glossary`.
Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either
you're inventing language the project doesn't use (reconsider) or there's a real
gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0007 — but worth reopening because…_
