"""__init__.py
============
feedprep: the feed-side algorithms of a case, labels as parameters (#147).

``trade_shares``  BACI-trade-share weights of ICSG's producer fan (ADR 0002, 0018, 0024)
``reexport``      the re-export netting and its waterfill (ADR 0027)
``ore_grade``     the in-situ ore-grade triangle (ADR 0015, 0023)
``builder_lock``  the stage lock of the build scripts run by hand (#281)

The case binds the labels and reads the registered inputs: ``scripts/copper_feed.py``.
"""

__all__ = ["builder_lock", "ore_grade", "reexport", "trade_shares"]
