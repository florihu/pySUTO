"""
tutorial
========
The tutorial of pySUTO: the toy case and the eleven notebooks that run it through
the six stages and then run a case of your own. ``python -m pysuto tutorial [DEST]``
copies the notebooks out of the installed package.

* :mod:`pysuto.tutorial.toy` — the **toy case**: a synthetic three-region orium
  case generated from a planted ground truth; the alias ``toy``.
* :mod:`pysuto.tutorial.toy_coproduct` — the toy whose smelter has a second Flow product
  (``Dust``), the co-product case; the alias ``toy_coproduct``.
* :mod:`pysuto.tutorial.toy_ext` — the extras of chapters ``08_break_it`` and
  ``09_bridge`` and the helpers of ``10_your_case``.
* ``params_template.csv`` — the schema and two worked examples of a params family,
  copied out beside the notebooks.

Import direction: ``tutorial`` sits at the top of the stack, beside the
stage drivers and the ``python -m pysuto`` entry point. It imports the library; the
library never imports it (:func:`pysuto.case.resolve_case` reaches the toy aliases only
by name, at call time).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

#: The notebooks, their README and the params template: package data beside this module.
PACKAGE_DIR = Path(__file__).resolve().parent

__all__ = ["PACKAGE_DIR", "copy_tutorial", "toy", "toy_coproduct", "toy_ext"]


def copy_tutorial(dest) -> list:
    """Copy the eleven notebooks, ``README.md`` and ``params_template.csv`` into ``dest``,
    creating it.

    Parameters
    ----------
    dest : str or path-like
        A folder that does not exist or is empty; the copies are writable.

    Returns
    -------
    list of pathlib.Path
        The files written, sorted.

    Raises
    ------
    FileExistsError
        ``dest`` exists and is not an empty directory; nothing is written. A copy that
    fails midway leaves ``dest`` as it was: the files are written into a temporary
    sibling folder, which replaces ``dest`` only once all of them are in.
    """
    dest = Path(dest)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"{dest} exists and is not an empty directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        shipped = [PACKAGE_DIR / "README.md", PACKAGE_DIR / "params_template.csv"]
        for src in sorted(PACKAGE_DIR.glob("*.ipynb")) + shipped:
            shutil.copyfile(src, stage / src.name)
            (stage / src.name).chmod(0o644)
        stage.chmod(0o755)
        if dest.exists():
            dest.rmdir()
        os.replace(stage, dest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return sorted(dest / p.name for p in dest.iterdir())
