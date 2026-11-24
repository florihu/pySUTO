# Parallel worktrees

When to split work into lanes, and the rules for an agent that works in its own git worktree
(`.claude/worktrees/<name>`, a `feat/`, `fix/` or `diag/` branch) while other agents or the
maintainer use the main checkout. `<wt>` is the worktree's absolute path, `<main>` the main
checkout's.

## When to start lanes

A **lane** is one stream of work that runs beside the others: a background agent, in its own
worktree when it edits code. With two or more tickets, or a read across three or more modules,
sessions or documents, write the lane plan before the first edit and send the independent lanes
in one message. Each brief names the goal, the files, the completion criterion and the *Report*
below.

- **A lane of its own**: each code-only ticket (no stage-driver run), each review, each read or
  audit of a separate area.
- **One serial lane in `<main>`**: every stage-driver run and measure ticket, one at a time, in
  blocker order (*Pipeline runs* below).
- **The same lane**: tickets that edit the same file, or where one needs another's output.
- While one lane waits on a solve or the suite, work on another; run the wait in the background.

## Environment

- Activate the `pysuto` env and give **every** `python`, `pytest`, `git commit` (the hook runs the
  suite) and notebook-kernel start an absolute `PYTHONPATH=<wt>/src`. The env's editable install
  points at `<main>`: without the prefix you import and test main's code. A relative path breaks
  as soon as the cwd changes.
- Install and build tests in a throwaway venv in the scratchpad, deleted afterwards; the
  `pysuto` env stays as the maintainer left it.
- A worktree has no `data/proc`. A suite run there leaves gitignored files in the worktree's own
  `data/`: remove them before the next full run.
- Work only inside your worktree: `<main>` and the other worktrees are read-only to you, and the
  branch stays local (no push, no tag).

## Pipeline runs

- Copper runs go to `<main>` only (`--data-dir <main>/data`), one at a time, under
  `<main>/data/proc/.stage.lock`; a held lock means wait, never delete it. Check `df -h` first.
- When the code may change during a run, run from a frozen copy: `git archive <rev> | tar -x -C
  <scratch>/frozen`, then drive that copy.
- A toy chain needs no `<main>` data: `python -m pysuto run --case toy --data-dir <scratch>`.
- A measurement is a run id and a number (CLAUDE.md, ADR 0028); report every run id created.

## Tutorial notebooks

- Notebook outputs change only through `scripts/refresh_tutorial.py`, which executes them and writes
  them free of home and temp paths. The kernel inherits the absolute `PYTHONPATH=<wt>/src`.
- `tests/test_tutorial_notebooks.py` checks the shipped notebooks statically; run it after any edit.

## Git

- Stage files by explicit path, so `data/`, `logs/` and `leg/` stay out.
- Describe the issue as `(#NN)` in the commit message; the hook refuses issue-closing keywords. Write
  the message to a file and run `git commit -F <file>`.
- Merging a branch that untracks files deletes them from the receiving working tree. Restore them
  from the pre-merge commit `<sha>`: list the names with `git ls-tree -r -z --name-only <sha> --
  <paths>` (`-z`, because non-ASCII names come back quoted otherwise), write each with
  `git cat-file -p <sha>:<path> > <path>`, and compare `git hash-object <path>` with the blob id
  from `git ls-tree <sha> -- <path>`.

## Report

The last message is all the orchestrator sees, under 400 words: branch, commit hashes and files
changed; the test result line; run ids created and the bit-identity result with what was compared;
deviations from the ticket and why; open questions for the maintainer.
