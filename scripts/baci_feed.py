"""baci_feed.py
============
Build the BACI copper-trade feed: a thin driver over ``pysuto.feedpipe.run_feed``.

The feed's policy (loaders, concordance tables, the ROW region and the cv-scale files) is
the copper case's ``BACI_FEED`` in ``scripts/copper_feed.py``; the acc id is
``config/config.yaml``'s ``pipeline.acc_id``. Run it from the repo root, once the inputs
it reads are registered (README *Workflow*, the build order of Stage 1).

Reads
-----
``data/input/feed/baci/v<N>/`` at the registered version in force: ``values.csv`` (the
bilateral trade flows, netted by build_baci_reexport.py), ``conc.xlsx`` (the concordance
workbook) and ``meta.xlsx`` (the pedigree sidecar); ``data/proc/acc/<acc_id>/`` (the
accounting structure).

Writes
------
``data/proc/feed/registry/<run_id>/`` with ``feed.pkl`` (the weight matrix and the
concordance), ``result.npz`` (F_root, sigma_root, cv_root and cv_src) and
``metadata.json`` (the sidecar, whose ``parents`` record the input versions);
``data/proc/feed/registry/registry.json`` indexes the runs. Its log is
``data/logs/baci_<timestamp>.log``. Takes ``data/proc/.stage.lock`` (#101).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# The feed's policy (loaders, concordance tables, ROW region, the weights file, the
# cv-scale files) is the copper case's BACI_FEED, scripts/copper_feed.py; the acc id is
# config/config.yaml's pipeline.acc_id, read inside main().

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copper_feed import BACI_FEED
from pysuto.config import load_config
from pysuto.feedpipe import run_feed
from pysuto.logging import start_run_log
from pysuto.registry import stage_lock

FEED_ID = BACI_FEED.feed_id


def main() -> None:
    start_run_log("data", FEED_ID)
    run_feed(BACI_FEED, acc_id=load_config("config/config.yaml").pipeline.acc_id)


if __name__ == "__main__":
    with stage_lock("baci_feed"):   # registries have no locking (#101)
        main()
