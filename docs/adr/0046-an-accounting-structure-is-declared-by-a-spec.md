# An accounting structure is declared by a spec; ODS and JSON are two formats of it, each with a reader

**Status:** proposed (2026-11-11; #57, groundwork for #26 §1). The copper account built from the
files of `acc/copper` v4 by the code before (`afef59c`) and after the change (`feat/acc-spec`)
is byte-identical in `edge_index.csv`, `edge_index_all.csv`, `node_index.csv`,
`root_nodes.csv`, `bounds_l.npy`, `bounds_u.npy` and `masks_user.npz` (854,460 admissible
edges); `acc/copper` v4 is not re-versioned.

The copper accounting structure could be declared only by its four files, `root.ods`,
`rules.ods`, `bounds.ods` and `masks.csv`, which `AccountingStructure` parsed sheet by sheet.
Its Python-object path took a different, procedural schema: callable rules that built no
`RuleSet`, so every edge was non-tradable and the supply and use masks fell back to entity-name
literals. A second case (#59's toy, then a second metal) has to declare a structure without a
spreadsheet. We decide that a structure is declared by one in-memory **acc spec**, a dict of
tables: sectors with their entity and optional sub, regions, valuations, times, supply and use
adjacency lists, tradable flows, entity-pair accounting rows, and optional bound and mask
records. `AccountingStructure.from_spec` is its one constructor. The four files and a single
`acc.json` are two formats of the spec, each with its reader (`read_acc_spec_ods`,
`read_acc_spec_json`) that produces the dict, and `AccountingStructure(root_file_path=…,
rules_file_path=…)` reads the files into a spec and builds through `from_spec`. A version
holding `acc.json` beside any of the four files is refused, not resolved by precedence. The
spec is validated, never coerced: a missing or unknown key at the top level or in a `sectors`
or `accounting` entry, a non-string label, a non-bool `same_region`, and a rule naming an
undeclared sector or entity all raise. The input registry hashes `acc.json` where present, as
it does `parents.json` (#53). Hashes are compared as whole dicts, so an `"acc.json": ""` key
on the ids without one would re-version every one of them. When #26 §1 lands, its `role` is
one more key on the `sectors` and `valuations` entries.

## Considered options

- **0/1 matrices or adjacency lists** for supply and use. Lists: the in-memory form a reader of
  code expects ("Mining supplies Ore and WR"), and the ODS matrix converts to them losslessly. In
  JSON a matrix is mostly zeros, with row and column labels kept in step by hand.
- **YAML or JSON.** JSON: the standard library reads it and a dict is literally JSON, so no
  dependency is added. YAML allows comments; it can be added as a third reader if a case author
  needs them.
- **One document or four files.** One: the optional parts are omitted keys, not empty files, and
  a structure is reviewed in one diff. The four files stay, as the copper case's reader.
- **#26 §3 rejected a YAML case file** because every selector in a case (which feed fills which
  edges, which rows a σ policy covers) needs a mini-language. That objection does not apply here.
  An acc spec declares the node universe and the admissibility rules, which are tables, not
  selectors. It is the `acc` registry kind, which a case references by id (GLOSSARY "Case").
