"""
milestone_rates.py
==================
Read-only base rate for a due date or a time estimate (issue-tracker.md "Due dates"): for each
round (GitHub milestone), the issues closed, the maintainer's working days (Monday to Thursday)
from its first to its last close, and the issues closed per working day. An open round shows
its rate so far beside its due date.

    python scripts/milestone_rates.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

WORK_DAYS = (0, 1, 2, 3)  # Monday to Thursday


@dataclass
class RoundRate:
    title     : str
    state     : str
    closed    : int
    open      : int
    first     : dt.date
    last      : dt.date
    work_days : int
    due       : str

    @property
    def rate(self) -> float:
        return self.closed / self.work_days


def working_days(first: dt.date, last: dt.date) -> int:
    """Count the Monday-to-Thursday days from ``first`` to ``last``, both in, at least 1."""
    span = (last - first).days + 1
    days = sum((first + dt.timedelta(days=k)).weekday() in WORK_DAYS for k in range(span))
    return max(days, 1)


def round_rates(issues: list, milestones: list) -> list:
    """One ``RoundRate`` per milestone with at least one closed issue, by first close.

    Parameters
    ----------
    issues : list
        ``gh issue list --json closedAt,milestone`` records of closed issues.
    milestones : list
        ``gh api .../milestones`` records (``title``, ``state``, ``open_issues``, ``due_on``).
    """
    closes = defaultdict(list)
    for issue in issues:
        if issue.get("milestone") and issue.get("closedAt"):
            day = dt.date.fromisoformat(issue["closedAt"][:10])
            closes[issue["milestone"]["title"]].append(day)
    rows = []
    for m in milestones:
        dates = closes.get(m["title"])
        if not dates:
            continue
        first, last = min(dates), max(dates)
        rows.append(RoundRate(m["title"], m["state"], len(dates), m["open_issues"], first,
                              last, working_days(first, last), (m.get("due_on") or "")[:10]))
    return sorted(rows, key=lambda r: r.first)


def _gh(*args: str):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main() -> int:
    issues     = _gh("issue", "list", "--state", "closed", "--limit", "2000",
                     "--json", "closedAt,milestone")
    milestones = _gh("api", "repos/{owner}/{repo}/milestones?state=all&per_page=100")
    print(f"{'round':24s} {'state':6s} {'closed':>6s} {'open':>4s} {'first':>10s} "
          f"{'last':>10s} {'wd':>3s} {'per wd':>6s} {'due':>10s}")
    for r in round_rates(issues, milestones):
        print(f"{r.title[:24]:24s} {r.state:6s} {r.closed:6d} {r.open:4d} {r.first!s:>10s} "
              f"{r.last!s:>10s} {r.work_days:3d} {r.rate:6.1f} {r.due:>10s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
