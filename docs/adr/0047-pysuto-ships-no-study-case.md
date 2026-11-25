# pySUTO ships no study case: a case is caller code that `--case` resolves, and copper lives in `scripts/`

The package is the general method: the six stages, the case protocol (`CaseSpec`, `resolve_case`), the toy
cases and the tutorial. The copper case (`copper.py`, its Stage 2 and 3 model, and `copper_feed.py`, its
feed-side labels and Stage 1 hook) was a module of `pysuto.cases`, so the wheel carried a study that a second
user cannot use, and `src/` named it as the default of every driver. We decide that pySUTO ships no study
case: a case is a file or a module that the caller names and `--case` resolves (`pysuto.case.resolve_case`), and
copper is `scripts/copper.py` and `scripts/copper_feed.py` of this repository, which already is the copper
application. The `scripts/` drivers default `--case` to `scripts/copper.py`, `resolve_case` puts a case
file's folder first on `sys.path` so that the file imports its siblings as a script run by Python does
(`copper` imports `copper_feed`, one way); `copper` leaves the
bundled aliases, which would point at a module the wheel does not hold. The alternative was a `studies/` layer
inside the package; it was rejected because a second user keeps their case in their own project, and the layer
would be a second place for the copper application beside `scripts/`. The price: the callables a copper run
records in `stage_config` carry the module name `copper` (or the path-loaded name), not `pysuto.cases.copper`; the
arrays, `row_meta`, the imputation log and `parents.inputs` do not depend on it, and a copper run now needs a
checkout of the repository (or its release archive), not only the wheel.

**Status:** accepted (2026-11-18; #185, #189). The move to `scripts/` reproduces the chain of
`opt_20261116_134050_b54f65dc` in every stored array and in `parents.inputs`, Stages 2–6, through
`python -m pysuto run --case scripts/copper.py`: `imp_20261118_090806_71969db4`, `const_20261118_091159_576e7b1f`,
`opt_20261118_091818_403319a5`, `uncprop_20261118_095441_3322e9e3`, `psut_20261118_095601_3b3c6f5b`
(`case_sha256` `2a6861ee…`, the committed `scripts/copper.py`). The drivers' default chain reproduced it as
well; those verification runs were pruned.

**Amended 2026-11-25 (#302):** feed preparation (`feedprep/`: re-export netting, trade shares, ore-grade carry) is application code from this ticket on, `scripts/feedprep/`, and comes back into the library with a second metal as its test.
