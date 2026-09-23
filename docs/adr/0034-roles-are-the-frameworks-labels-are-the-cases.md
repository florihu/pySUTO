# Roles are the framework's, labels are the case's

**Status:** proposed (2026-09-22; accepted when [issue #26](https://github.com/florihu/pySUTO/issues/26) §1 lands)

pySUTO is meant to run any metal's supply-use table, yet nine library modules compare the copper
case's *labels* — `Process`/`Flow`/`SB`, `Nature`/`SEM`/`FD`, `CC`/`Mass` — as string literals
(`acc.py:796-800`, `param_relation.py:163`, `impute/network.py:98,237`, three copies of
`_BALANCE_ENTITIES`, …), and `AccountingStructure.load()` discards the rule-derived masks and
rebuilds them from those literals. A case whose content valuation is not called `CC` breaks before
any imputer runs, and nothing validates the vocabulary a case supplies.

**Decision.** The framework owns a small `Enum` of **roles**; the case binds its labels to them in
`root.ods` via a `role` column, and the library compares roles through `AccountingStructure`
accessors, never labels. The role vocabulary is deliberately the glossary's existing one: boundary
role `nature | secondary | final_demand` (a `secondary` node is an input on a use edge and a residual
output on a supply edge — the direction rule the code already applies) and valuation role
`content | gross`. The entity kinds `Process | Flow | SB` stay **reserved framework words**: they are
the meta-model itself, used by every ADR and the `constraint_id` grammar, and renaming them buys a
metal case nothing. Validation runs on build **and** on load (sector→entity 1:1, exactly one
`content` and one `gross` valuation, every `SB` node has a boundary role), and the masks built from
`rules.ods` are persisted so `load()` returns what `__init__` built.

## Considered options

- **Fixed reserved labels, validation only.** Cheapest, but a tin case would have to call its
  content valuation `CC` and its scrap `SEM`; the literal scatter across nine modules stays.
- **Roles and labels both in a separate case yaml.** One more file per case and two sources of
  truth for the node vocabulary next to `root.ods`.
- **Splitting `secondary` into `secondary_input` / `residual_output`.** Contradicts the single
  `SEM` category the glossary defines (which also carries `PSC`) and forces every case to assign a
  fourth role that edge direction already determines.
