"""The milestone rates count a round's span in the maintainer's working days, Monday to
Thursday, from its first to its last issue close. These tests feed toy records; only ``main``
calls ``gh``."""
from __future__ import annotations

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import milestone_rates as M  # noqa: E402

MON = dt.date(2026, 11, 16)
THU = dt.date(2026, 11, 19)
FRI = dt.date(2026, 11, 20)
SUN = dt.date(2026, 11, 22)


def _closed(title, day):
    return {"closedAt": f"{day.isoformat()}T10:00:00Z", "milestone": {"title": title}}


def _milestone(title, state="closed", open_issues=0, due_on="2026-11-26T23:59:59Z"):
    return {"title": title, "state": state, "open_issues": open_issues, "due_on": due_on}


def test_a_week_has_four_working_days_and_a_weekend_none():
    assert M.working_days(MON, SUN) == 4
    assert M.working_days(FRI, SUN) == 1  # a span with no working day counts as one


def test_a_round_rate_is_its_closed_issues_per_working_day_of_its_close_span():
    issues = [_closed("a", MON), _closed("a", THU), _closed("a", THU), _closed("b", FRI),
              {"closedAt": THU.isoformat() + "T10:00:00Z", "milestone": None}]
    rows = M.round_rates(issues, [_milestone("b", "open", 2, None), _milestone("a"),
                                  _milestone("empty")])
    assert [r.title for r in rows] == ["a", "b"]
    assert (rows[0].closed, rows[0].work_days, rows[0].due) == (3, 4, "2026-11-26")
    assert rows[0].rate == pytest.approx(0.75)
    assert (rows[1].state, rows[1].open, rows[1].due) == ("open", 2, "")
