"""crude_ore_feed.py
=================
Build the crude-ore extraction feed: a thin driver over ``pysuto.feedpipe.run_feed``.

The feed's policy (loaders, concordance tables, the ROW region) is the copper case's
``CRUDE_ORE_FEED`` in ``scripts/copper_feed.py``; the acc id is ``config/config.yaml``'s
``pipeline.acc_id``. Run it from the repo root, once the inputs it reads are registered
(README *Workflow*, the build order of Stage 1).

Reads
-----
``data/input/feed/crude_ore/v<N>/`` at the registered version in force: ``values.csv``
(the copper crude ore extraction of crude_ore_parse_feed.py), ``conc.xlsx`` (the
concordance workbook) and ``meta.xlsx`` (the pedigree sidecar);
``data/proc/acc/<acc_id>/`` (the accounting structure).

Writes
------
``data/proc/feed/registry/<run_id>/`` with ``feed.pkl`` (the weight matrix and the
concordance), ``result.npz`` (F_root, sigma_root, cv_root and cv_src) and
``metadata.json`` (the sidecar, whose ``parents`` record the input versions);
``data/proc/feed/registry/registry.json`` indexes the runs. Its log is
``data/logs/crude_ore_<timestamp>.log``. Takes ``data/proc/.stage.lock`` (#101).
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# The feed's policy (loaders, concordance tables, ROW region) is the copper case's
# CRUDE_ORE_FEED, scripts/copper_feed.py; the acc id is config/config.yaml's
# pipeline.acc_id, read inside main().

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copper_feed import CRUDE_ORE_FEED
from pysuto.config import load_config
from pysuto.feedpipe import run_feed
from pysuto.logging import start_run_log
from pysuto.registry import stage_lock

FEED_ID = CRUDE_ORE_FEED.feed_id


def main() -> None:
    start_run_log("data", FEED_ID)
    run_feed(CRUDE_ORE_FEED, acc_id=load_config("config/config.yaml").pipeline.acc_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the crude-ore feed (one feed run in data/proc/feed/registry/); it is "
                    "a Stage 1 feed build independent of the BACI-to-ICSG chain (ADR 0024, 0027, 0030 am.). Reads config/config.yaml relative to the working "
                    "directory, so run it from the repo root; takes no other options.")
    parser.parse_args()          # before the stage lock: --help builds and writes nothing
    with stage_lock("crude_ore_feed"):   # registries have no locking (#101)
        main()
