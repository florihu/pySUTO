"""test_case_resolution_copper.py
=============================
A fresh interpreter resolves ``scripts/copper.py`` by path and reaches its Stage 1
imports. Split from ``tests/test_case_resolution.py`` (#304).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class TestForms:

    def test_a_fresh_interpreter_resolves_copper_and_reaches_its_stage1_imports(
            self, tmp_path):
        repo = Path(__file__).resolve().parents[2]
        code = ("from pysuto.case import resolve_case\n"
                f"case = resolve_case({str(repo / 'scripts' / 'copper.py')!r})\n"
                "import copper_feed\n"
                "assert case.id == 'copper' and case.stage1 is not None\n"
                "assert case.acc_id == copper_feed.ACC_ID\n")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(repo / "src")})
        assert proc.returncode == 0, proc.stderr
