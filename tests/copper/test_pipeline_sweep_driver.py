"""test_pipeline_sweep_driver.py
=============================
``scripts/sweep.py --dry-run`` lists the points and writes nothing. Split from
``tests/test_pipeline.py`` (#304); the fake checkout fixture comes from there.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_pipeline
from test_pipeline import _runs, repo  # noqa: E402,F401


def test_the_sweep_driver_dry_run_lists_the_points_and_writes_nothing(repo):
    import os
    import sys
    from pathlib import Path
    real = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, str(real / "scripts" / "sweep.py"), "--dry-run",
                        "--base", repo.base.opt], cwd=repo.root, capture_output=True,
                       text=True, env=dict(os.environ, PYTHONPATH=str(real / "src")),
                       timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "p003" in r.stdout and "plan:" in r.stdout and "GB" in r.stdout
    assert not (repo.root / "data" / "proc" / "sweep").exists()
    assert len(_runs(repo, "imp")) == 1
