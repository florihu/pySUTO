# scripts/explo: the exploration and measurement scripts behind the REPs

This folder holds a frozen record. Each script produced numbers for one REP, ADR or issue.

- Each script ran against the commit that last changed it. The first line of the script names that commit.
- A script may not import on `main`. The library moved on after the script ran. The module paths, the class names and the stored layouts can differ.
- To rerun a script, check out the commit named in its header. Run it there, from the repository root. Its docstring gives the arguments.
- Do not port a script to `main`. A ported script does not reproduce the numbers of its REP, because those numbers came from the old code.
- The folder is exempt from the comment rule and from lint (`ruff check` skips it).
- `scripts/explo/prune_runs.py` is tooling, not a record. It has no header, and issue #303 moves it to `tools/`.
