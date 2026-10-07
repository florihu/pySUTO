"""
input_status.py
===============
Read-only: list the inputs whose ``data/input/<kind>/<id>/current/`` differs from the
latest registered snapshot — edits the next stage driver will register as a new version.
Writes nothing (``InputRegistry.initialize`` is what bumps).

    python scripts/input_status.py

Exit status 1 when anything is unregistered, 0 otherwise.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pysuto.input_registry import InputRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_ROOT = os.path.join("data", "input")
KINDS      = ("feed", "acc", "params", "mask")
# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------


def main() -> int:
    n = 0
    for kind in KINDS:
        for id, v in InputRegistry(kind, root=INPUT_ROOT).pending().items():
            print(f"{kind}/{id}: current/ differs from "
                  f"{'v' + str(v) if v else 'no registered version'}")
            n += 1
    if not n:
        print("every input's current/ matches its latest registered version")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
