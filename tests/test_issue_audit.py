"""The issue audit reads the tracker against docs/agents/: rounds closed with open issues,
open issues without exactly one triage state, and ADR amendments still "to be measured" by an
issue that has closed. These tests feed it toy issues and texts; only ``main`` calls ``gh``."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import issue_audit as A  # noqa: E402

TRIAGE_MD = """
| Label in mattpocock/skills | Label in our tracker | Meaning |
| -------------------------- | -------------------- | ------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified |
| `wontfix`                  | `wontfix`            | Will not be actioned |
"""


def _issue(number, labels=(), milestone=None, state="OPEN"):
    return {"number": number, "title": f"issue {number}", "state": state,
            "labels": [{"name": n} for n in labels],
            "milestone": {"title": milestone} if milestone else None}


def test_state_labels_are_the_tracker_column_of_the_table():
    assert A.state_labels(TRIAGE_MD) == ["needs-triage", "ready-for-agent", "wontfix"]


def test_an_open_issue_needs_exactly_one_state_label():
    states = ["needs-triage", "ready-for-agent", "wontfix"]
    issues = [_issue(1, ["ready-for-agent", "pipeline"]), _issue(2, ["enhancement"]),
              _issue(3, ["needs-triage", "ready-for-agent"]), _issue(4, [], state="CLOSED")]
    found = A.triage_findings(issues, states)
    assert [(f.number, f.states) for f in found] == [(2, []), (3, ["needs-triage", "ready-for-agent"])]


def test_an_open_issue_in_a_closed_round_is_a_leftover():
    issues = [_issue(1, milestone="manuscript ready 3"), _issue(2, milestone="manuscript ready"),
              _issue(3, milestone="manuscript ready", state="CLOSED"), _issue(4)]
    rounds = {"manuscript ready": "closed", "manuscript ready 3": "open"}
    assert [f.number for f in A.round_findings(issues, rounds)] == [2]


def test_an_amendment_to_be_measured_by_a_closed_issue_is_stale():
    texts = {"0030-x.md": "route. Built, to be measured (#63's REP).\nMeasured: ships.\n",
             "0040-x.md": "Narrowed 2026-10-21 (#38, to be measured): the balance\n",
             "0041-x.md": "the 857 Mass legs that follow them. To be measured.\n",
             "0014-x.md": "It maps to Taiwan; to be measured by #79.\n"}
    found = A.adr_findings(texts, closed={63, 79})
    assert [(f.path, f.line, f.issues) for f in found] == [("0014-x.md", 1, [79]),
                                                           ("0030-x.md", 1, [63])]
