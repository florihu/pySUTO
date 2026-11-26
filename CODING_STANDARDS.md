# Coding standards

The judgement calls a reviewer makes from a diff (`mattpocock-skills:code-review`, Standards
axis). The conventions every agent needs while it writes code are in [CLAUDE.md](CLAUDE.md); a
rule a test or `ruff` can check belongs in that check, not here.

## Documentation in the same branch

Four documents describe the library and the copper case: [README.md](README.md) (install, quickstarts, where things are), [docs/architecture.md](docs/architecture.md) (the stages, the module reference, the import direction), [docs/workflow.md](docs/workflow.md) (run ids, sidecars, registries, knobs, sweeps) and [scripts/README.md](scripts/README.md) (the application folder). A diff that changes what one of them describes updates it in the same branch:

| The diff changes | The document and section to update |
|---|---|
| a stage's math, classes, key parameters or use case | `docs/architecture.md`, that stage's section under *Architecture* |
| a subcommand, a driver option, the knob protocol or the sweep | `docs/workflow.md`, *Pipeline and knobs* |
| a module's role, its public names, or a module added, moved or removed | `docs/architecture.md`, *Module reference*; the layer table of *Import direction* |
| a script, builder, parser or report of the application | `scripts/README.md` |
| the install, a quickstart or a top-level folder | `README.md` |
| a modelling choice or an assumption | the section that states it, with its ADR link |

Knob values live in the code (`docs/architecture.md` *Stage 3*: "values live there, not here"). A new value of
an existing knob needs no documentation edit. A new knob family or a new meaning does.

`tests/test_readme_names.py` finds a name that the four documents keep after the code drops it. This rule
finds the meaning: a sentence that names live code but no longer says what the code does. Flag
each such sentence with its section and the hunk that made it stale.
