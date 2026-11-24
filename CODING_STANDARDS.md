# Coding standards

The judgement calls a reviewer makes from a diff (`mattpocock-skills:code-review`, Standards
axis). The conventions every agent needs while it writes code are in [CLAUDE.md](CLAUDE.md); a
rule a test or `ruff` can check belongs in that check, not here.

## README in the same branch

[README.md](README.md) is the documentation of the library and of the copper case. A diff that
changes what a README section describes updates that section in the same branch:

| The diff changes | The README section to update |
|---|---|
| a stage's math, classes, key parameters or use case | that stage's section under *Architecture* |
| a subcommand, a driver option, the knob protocol or the sweep | *Workflow*, *Pipeline and knobs* |
| a module's role, its public names, or a module added, moved or removed | *Module reference*, *File structure* |
| a modelling choice or an assumption | the section that states it, with its ADR link |

Knob values live in the code (README *Stage 3*: "values live there, not here"). A new value of
an existing knob needs no README edit. A new knob family or a new meaning does.

`tests/test_readme_names.py` finds a name that README keeps after the code drops it. This rule
finds the meaning: a sentence that names live code but no longer says what the code does. Flag
each such sentence with its section and the hunk that made it stale.
