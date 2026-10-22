"""
issue_audit.py
==============
Read-only audit of the issue tracker against the conventions of ``docs/agents/``. Run it when a
round opens and before it closes; it changes nothing and exits 1 when it finds anything.

    R  rounds: open issues in a closed milestone — every issue of a closed round was closed,
       moved to the next round or moved out, each with a comment (issue-tracker.md "Rounds")
    T  triage: open issues without exactly one triage-state label (triage-labels.md)
    A  ADR paragraphs still "to be measured" (or "not yet measured") whose issues have all
       closed — the measure commit replaces the phrase with the outcome (measure-tickets.md)

    python scripts/issue_audit.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TRIAGE_DOC = Path("docs/agents/triage-labels.md")
ADR_DIR    = Path("docs/adr")
TO_MEASURE = re.compile(r"to be measured|not yet measured", re.IGNORECASE)
ISSUE_REF  = re.compile(r"#(\d+)")


@dataclass
class Finding:
    number:    int
    title:     str
    states:    list = field(default_factory=list)
    milestone: str = ""


@dataclass
class AdrFinding:
    path:   str
    line:   int
    issues: list
    text:   str


def state_labels(md: str) -> list:
    """The triage-state labels: the tracker column of triage-labels.md's table."""
    out = []
    for row in md.splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[1].startswith("`") and cells[1].endswith("`"):
            out.append(cells[1].strip("`"))
    if not out:
        raise ValueError("no triage-state labels found in the triage-labels table")
    return out


def _labels(issue: dict) -> list:
    return [lab["name"] for lab in issue.get("labels") or []]


def _open(issue: dict) -> bool:
    return str(issue["state"]).upper() == "OPEN"


def triage_findings(issues: list, states: list) -> list:
    """Open issues whose triage-state labels are not exactly one."""
    out = []
    for i in sorted(issues, key=lambda x: x["number"]):
        found = [n for n in _labels(i) if n in states]
        if _open(i) and len(found) != 1:
            out.append(Finding(i["number"], i["title"], states=found))
    return out


def round_findings(issues: list, rounds: dict) -> list:
    """Open issues in a milestone whose state is closed (``rounds``: title → state)."""
    out = []
    for i in sorted(issues, key=lambda x: x["number"]):
        title = (i.get("milestone") or {}).get("title")
        if _open(i) and title and rounds.get(title) == "closed":
            out.append(Finding(i["number"], i["title"], milestone=title))
    return out


def adr_findings(texts: dict, closed: set) -> list:
    """Lines saying "to be measured" whose paragraph names issues that have all closed.

    The paragraph, not the line: an amendment names its issue in its opening words and says
    "to be measured" lines later. An open issue in the paragraph may still be the measurement.
    """
    out = []
    for path in sorted(texts):
        lines = texts[path].splitlines()
        start = 0
        for n, line in enumerate(lines):
            if not line.strip():
                start = n + 1
                continue
            if not TO_MEASURE.search(line):
                continue
            end = next((k for k in range(n, len(lines)) if not lines[k].strip()), len(lines))
            refs = sorted({int(k) for k in ISSUE_REF.findall("\n".join(lines[start:end]))})
            if refs and all(k in closed for k in refs):
                out.append(AdrFinding(path, n + 1, refs, line.strip()))
    return out


def _gh(*args: str):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main() -> int:
    issues = _gh("issue", "list", "--state", "all", "--limit", "2000",
                 "--json", "number,title,state,labels,milestone")
    rounds = {m["title"]: m["state"] for m in
              _gh("api", "repos/{owner}/{repo}/milestones?state=all&per_page=100")}
    closed = {i["number"] for i in issues if not _open(i)}
    texts  = {p.name: p.read_text(encoding="utf-8") for p in sorted(ADR_DIR.glob("*.md"))}

    leftovers = round_findings(issues, rounds)
    untriaged = triage_findings(issues, state_labels(TRIAGE_DOC.read_text(encoding="utf-8")))
    stale     = adr_findings(texts, closed)
    print(f"== R open issues in a closed round: {len(leftovers)}")
    for f in leftovers:
        print(f"   #{f.number} [{f.milestone}] {f.title[:90]}")
    print(f"== T open issues without exactly one triage state: {len(untriaged)}")
    for f in untriaged:
        print(f"   #{f.number} {f.states or '(none)'} {f.title[:90]}")
    print(f"== A ADR lines still to be measured, every issue of their paragraph closed: {len(stale)}")
    for f in stale:
        print(f"   {f.path}:{f.line} {['#%d' % k for k in f.issues]} {f.text[:110]}")
    return 1 if (leftovers or untriaged or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
