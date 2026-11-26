# Contributing to pySUTO

This guide is for researchers who model their own material, users with a bug or a feature
request, and paper reviewers. It tells you how to report, set up, change, test and submit. It
gives only the rules that a newcomer is likely to break; [CLAUDE.md](CLAUDE.md) holds all.

## 1. Ask or report

Open a [GitHub issue](https://github.com/florihu/pySUTO/issues/new) for a bug, a question or a
request. Give it one label: `bug`, `enhancement`, `documentation` or `question`. The maintainer
then sets its triage state.

A bug report gives:

- the pySUTO version: `python -c "import pysuto; print(pysuto.__version__)"`;
- the operating system and the Python version: `python --version`;
- the case, as you gave it to `--case`;
- the command you ran, and the error or the log under `<data-dir>/logs/`;
- the run id of the failed run, with its sidecar `<data-dir>/proc/<stage>/<run_id>/metadata.json`.

## 2. Before a pull request

Open an issue first. Wait until the maintainer triages it. The issue is the request; the
pull request is the work for it. A typo fix can skip the issue. The project asks for bug
fixes, documentation, and library features in `src/pysuto/` that are generic for every case.

A case for another metal stays in your own repository: pySUTO runs it with
`--case ./my_case.py` ([ADR 0047](docs/adr/0047-pysuto-ships-no-study-case.md)). The last
tutorial chapter, [10_your_case.ipynb](src/pysuto/tutorial/10_your_case.ipynb), writes one.
Copper's input data take no changes from outside.

## 3. Set up

Fork the repository on GitHub. Then clone your fork:

```bash
git clone https://github.com/<your-account>/pySUTO.git
cd pySUTO
```

Install the package in editable mode with the development tools:

```bash
pip install -e ".[dev]"
```

With conda, `conda env create -f environment.yml` makes the same install in the environment
`pysuto`. Then run `conda activate pysuto`. Use Python 3.9 or newer.

The git hooks are optional, because CI runs the same checks. Before a commit, they run the
tests and `ruff check`, and they refuse an issue-closing keyword. They need the environment
with pysuto to be active:

```bash
git config core.hooksPath .githooks
```

## 4. Make the change

- Write code that runs on Python 3.9: no `match` statement, no `zip(strict=...)`, no
  `itertools.pairwise`. Write `X | None` only in annotations, under
  `from __future__ import annotations`.
- Do not run black or `ruff format`. The code is aligned by hand.
- Write numpy-style docstrings.
- A comment in `src/` or `scripts/` states the reason or the invariant, in the present tense.
  Put issue numbers in the commit message, not in a comment. Only a workaround for an open
  issue cites its issue in a comment.
- Library code in `src/pysuto/` compares no case label (a metal, sector or region name). A test,
  `tests/test_acc_roles.py`, refuses such a label as a string literal.
- Update the documentation in the same branch as the code. The table in
  [CODING_STANDARDS.md](CODING_STANDARDS.md#documentation-in-the-same-branch) names the
  document for each kind of change.
- A bug fix ships with the test that reproduces it.

For the other rules (dtypes, numerics, logging, test placement), read [CLAUDE.md](CLAUDE.md).

## 5. Test

Run these commands from the repository root before you open the pull request:

```bash
python -m pytest tests/      # about 5 min
ruff check
```

If the change can alter the method's numbers, run the toy chain on `main` and on your branch.
Each run takes about 25 s and prints the run id of every stage:

```bash
python -m pysuto run --case toy --data-dir toy-data
```

Compare the two chains by their psut and uncprop run ids:

```bash
python -m pysuto compare --data-dir toy-data --base <psut_id> <uncprop_id> --new <psut_id> <uncprop_id>
```

Give the result in the pull request. The stored arrays are bit-identical only in the
reference environment, [environment-reference.yml](environment-reference.yml) (linux-64).
Thus compare your two chains with each other, not with the reference.

CI runs the suite and the lint again on the pull request, and the toy chain on Windows.
Before the merge, the maintainer runs the copper check chain and writes its report (REP).

## 6. Commit and pull request

- Name the branch `feat/…`, `fix/…` or `diag/…`.
- Write each commit message as `type(scope): finding as a sentence (#NN)`. The type is `feat`,
  `fix`, `docs`, `measure`, `chore`, `test` or `refactor`; the scope is a module or an
  artefact; `#NN` is the issue.
- Never write `Closes #NN`, `Fixes #NN` or a similar keyword in a commit message or a pull
  request body. An issue closes only through the maintainer's closing comment.
- Open the pull request against `main` and name the issue in its body.
- The maintainer can ask you to reword a commit.

## 7. Licence

You agree that your contribution has the same licence as the project: code under the
[MIT License](LICENSE), and a change to a CC BY table under [CC BY 4.0](LICENSE-DATA).
