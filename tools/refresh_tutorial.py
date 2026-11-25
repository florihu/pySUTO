"""refresh_tutorial.py
=====================
The one way to refresh the executed outputs of the tutorial notebooks (#188). It copies
the eleven chapters out with ``python -m pysuto tutorial`` into a temporary folder,
executes them in order on the ``python3`` kernel of the running interpreter, rewrites the
machine-specific prefixes of the outputs (the temporary folder, the installed or
checked-out ``pysuto`` package directory, the home directory) to stable relative forms,
and writes the notebooks back into ``src/pysuto/tutorial/``. It fails, writing nothing,
if a ``/home/``, the temporary folder or the home directory survives in any notebook.

Needs ``pip install -e ".[tutorial]"`` (nbclient comes with jupyter). Not shipped.

    python tools/refresh_tutorial.py [--keep-tmp]
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO     = Path(__file__).resolve().parents[1]
PKG_DEST = REPO / "src" / "pysuto" / "tutorial"
TIMEOUT  = 900


def _variants(path) -> list:
    """``path`` as given and resolved, longest first."""
    return sorted({str(path), str(Path(path).resolve())}, key=len, reverse=True)


def normalise(text: str, work: Path, tmp: Path, pkg: Path) -> str:
    """``text`` with the chapters' folder ``work``'s ``runs/`` as ``runs/``, the rest of
    the temporary folder ``tmp`` without its prefix, the ``pysuto`` package directory as
    ``pysuto``, and the home directory as ``~``. A ``path:<work>/`` provenance string
    (``case_source``) is the exception: it keeps its shape as ``path:/path/to/tutorial/``,
    still an absolute path."""
    for p in _variants(work):
        text = text.replace("path:" + p + "/", "path:/path/to/tutorial/")
    for p in _variants(work):
        text = text.replace(p + "/", "")
    for p in _variants(tmp):
        text = text.replace(p + "/runs/", "runs/").replace(p + "/", "")
    for p in _variants(pkg):
        text = text.replace(p + "/", "pysuto/").replace(p, "pysuto")
    for p in _variants(Path.home()):
        text = text.replace(p, "~")
    return text


def leaks(text: str, tmp: Path) -> list:
    """What machine-specific strings survive in ``text``."""
    needles = ["/home/"] + _variants(tmp) + _variants(Path.home())
    return sorted({n for n in needles if n in text})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep-tmp", action="store_true", help="Keep the temporary folder.")
    args = ap.parse_args(argv)
    spec = importlib.util.find_spec("pysuto")
    if spec is None or spec.origin is None:
        raise SystemExit("pysuto is not installed: pip install -e '.[tutorial]'")
    pkg = Path(spec.origin).resolve().parent
    tmp = Path(tempfile.mkdtemp(prefix="pysuto-tutorial-")).resolve()
    work = tmp / "tut"
    subprocess.run([sys.executable, "-m", "pysuto", "tutorial", str(work)], cwd=tmp,
                   check=True)
    t0, out = time.time(), {}
    for path in sorted(work.glob("[01][0-9]_*.ipynb")):
        t = time.time()
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3",
                       resources={"metadata": {"path": str(work)}}).execute()
        out[path.name] = normalise(nbformat.writes(nb), work, tmp, pkg)
        print(f"{path.name:26s}{time.time() - t:6.1f} s", flush=True)
    if len(out) != 11:
        raise SystemExit(f"expected eleven chapters, executed {len(out)}")
    bad = {n: leaks(t, tmp) for n, t in out.items() if leaks(t, tmp)}
    if bad:
        raise SystemExit(f"machine-specific paths survive, nothing written: {bad}")
    for name, text in out.items():
        (PKG_DEST / name).write_text(text, encoding="utf-8")
    print(f"eleven chapters in {time.time() - t0:.1f} s; no /home/, temp-dir or home path "
          f"in any of them; written to {PKG_DEST}")
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
