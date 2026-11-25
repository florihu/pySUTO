# Measure tickets

A **measure ticket** decides whether one change ships: a σ rule, builder, imputer, feed or knob,
measured on the copper chain against the reference chain it would replace. It ends in a
`measure(...)` commit, a REP and a closed issue. The build lands first on its own `fix/` or `feat/`
branch ("to be measured" in its merge title); the measurement runs on a `diag/<slug>-measure` branch
in the main checkout.

## Steps

1. **Pre-register.** Comment on the issue with the run plan (drivers, base run), the numbered
   decision items (each a rule with its threshold, against the base) and the predictions.
   *Done when* the comment is posted before the first stage driver of the measurement starts.
2. **Run** the chain, one stage driver at a time (CLAUDE.md "Pipeline"). Set a knob for one run
   through the driver's `--set KNOB=VALUE` (Stages 2–4, #100). For Stage 5, until #100 covers it,
   override the knob in memory under the driver's stage lock (#101); the sidecar records the
   value and the driver file stays as committed:

   ```bash
   python - <<'EOF'
   import sys; sys.path.insert(0, "scripts")
   import uncertainty_propagation as U
   import pysuto.stages as S
   from pysuto.case import resolve_case
   from pysuto.registry import stage_lock
   S.EXCLUDE_SCALE_TOTALS = True       # the stage body reads it from pysuto.stages
   with stage_lock("uncertainty_propagation", root="data/proc"):
       U.main("<opt_run_id>", "<label>", data_dir="data",
              case=resolve_case("scripts/copper.py"))
   EOF
   ```

   *Done when* every year returns Solved, or the failure is itself the result.
3. **Measure** with a read-only `scripts/explo/issue<N>_<slug>_diag.py <base> <new>` that prints
   every decision item. *Done when* each item has a number from the two runs.
4. **REP** at `docs/rep/analysis_<run_id>_<YYYYMMDD>_issue<N>-<slug>.md`, opening with Run / Against /
   Chain / Feed+params versions / Date / Branch / ADR / Verdict, then the items table with each
   pre-registered rule beside its result and the predictions scored right or wrong. A σ or
   repricing change is judged by the Round-E stop rule (ADR 0025). *Done when* every item and
   prediction is scored and every number names its run.
5. **ADR.** The amendment that says "to be measured (#N)" now states the outcome: shipped or
   rejected on measurement, with the run id and the headline number. *Done when*
   `python scripts/issue_audit.py` lists no ADR line for #N.
6. **Commit and merge.** `measure(<scope>): <finding> (ADR …, #N)`, the round's bookkeeping in
   the milestone's issues, then `git merge --no-ff`. A miss at the numerical floor ships only by the
   maintainer's decision, which the REP states. *Done when* the merge is on main.
7. **Close** with `gh issue close <N> --comment` naming the REP, the verdict and the new reference
   chain. *Done when* the issue is closed and the comment links the REP.

## Executable-doc tickets

A ticket for a notebook or other executable doc states a headline effect as an acceptance criterion
only after a run has measured it on the case the doc ships. Until then it is a **hypothesis** with a
report-if-not path: the ticket says what the chapter and the closing comment carry when the number
does not appear (quote the measured case from the real data, or recast the effect). #61 shows both
outcomes: experiment 3 could not hold on the toy and the chapter quotes copper's measurement
instead; experiment 1 had to be recast.
