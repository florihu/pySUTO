"""icsg_feed.py
============
Build the ICSG copper-flow feed: a thin driver over ``pysuto.feedpipe.run_feed``.

The feed's policy (loaders, concordance tables, the BACI-trade-share weights, which #53
refuses when stale, and the observed zeros) is the copper case's ``ICSG_SPEC`` in
``scripts/copper_feed.py``; the acc id is ``config/config.yaml``'s ``pipeline.acc_id``.
Run it from the repo root, once the inputs it reads are registered (README *Workflow*, the
build order of Stage 1).

Reads
-----
``data/input/feed/icsg/v<N>/`` at the registered version in force: ``values.csv`` (the
cleaned yearbook flows of icsg_clean.py), ``conc.xlsx`` (the concordance workbook) and
``meta.xlsx`` (the pedigree sidecar); ``data/proc/acc/<acc_id>/`` (the accounting
structure).

Writes
------
``data/proc/feed/registry/<run_id>/`` with ``feed.pkl`` (the weight matrix and the
concordance), ``result.npz`` (F_root, sigma_root, cv_root, cv_src and observed_zero) and
``metadata.json`` (the sidecar, whose ``parents`` record the input versions);
``data/proc/feed/registry/registry.json`` indexes the runs. Its log is
``data/logs/icsg_<timestamp>.log``. Takes ``data/proc/.stage.lock`` (#101).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# The feed's policy is the copper case's complete ICSG spec, ``copper_feed.
# ICSG_SPEC``: the loaders there, the BACI-trade-share weights
# (#53 refuses stale ones) and the observed zeros (#76–#97, their knobs OBSERVED_ZERO_CV
# and OBSERVED_ZERO_FLOOR_KT live there). The acc id is config/config.yaml's
# pipeline.acc_id, read inside main() by run_feed.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copper_feed import ICSG_SPEC as SPEC
from pysuto.config import load_config
from pysuto.feedpipe import run_feed
from pysuto.logging import start_run_log
from pysuto.registry import stage_lock

FEED_ID = SPEC.feed_id


def main() -> None:
    start_run_log("data", FEED_ID)
    run_feed(SPEC, acc_id=load_config("config/config.yaml").pipeline.acc_id)


if __name__ == "__main__":
    with stage_lock("icsg_feed"):   # registries have no locking (#101)
        main()
