# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is **single-context**: one `GLOSSARY.md` at the repo root, one `docs/adr/`.

## Before exploring, read these

- **`GLOSSARY.md`** — the project's single glossary (e.g. *period slice* `a*(t)` vs *complete a\**,
  *cross-year placeholder*, *no cross-time coupling*).
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.
- **`docs/architecture.md`** — the architecture and spec doc. Use its sections:
  - *Architecture* — the six-stage pipeline (feed → agg/imp → const → solver → uncprop → psut)
  - *Module reference* — where each class lives
  - *Import direction* — the one-way import direction and its layer table
- **`docs/workflow.md`** — run ids, sidecars, input registries, knobs, sweeps.
- **`scripts/README.md`** — the copper application folder.
- **`README.md`** — install, quickstarts and where things are.

## File structure

```
/
├── GLOSSARY.md        ← the glossary
├── README.md          ← install, quickstarts, where things are
├── CLAUDE.md          ← behavioral guidelines + Agent skills config
├── docs/
│   ├── architecture.md ← stages, module reference, import direction
│   ├── workflow.md    ← run ids, sidecars, registries, knobs
│   ├── adr/           ← architectural decision records
│   ├── rep/           ← analysis reports
│   └── agents/        ← this skill's config
└── src/pysuto/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal,
a hypothesis, a test name), use the term as defined in `GLOSSARY.md`.
Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either
you're inventing language the project doesn't use (reconsider) or there's a real
gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0007 — but worth reopening because…_
