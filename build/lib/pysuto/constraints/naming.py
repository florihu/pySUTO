"""
naming.py
=========
Single home for the ``constraint_id`` grammar and the canonical ``row_meta``
schema shared by every constraint builder.

Grammar::

    constraint_id = {type}_{variant}_{specs...}_{year}

* **type** ∈ {``massbal``, ``sum``, ``ratio``, ``bound``} — the constraint family.
* **variant** — the role within the family (``massbal``, ``usefulsupply``,
  ``subentity``, ``anchor``, a user csv label, ``ratioeq``, ``bound``, ``lower``).
* **specs** — type-specific human-readable fields (sector/region/valuation, or a
  rendered edge), never integer node/edge ids.
* **year** — always the final token; ``na`` when the row is time-agnostic.

Every slot token (type, name, each spec, year) and every sector/region/valuation
value must be underscore-free so the id is splittable on ``_``. A build-time
guard (:func:`_safe`) raises :class:`ConstraintIdError` otherwise.

``constraint_id`` is a unique human-readable **display key** — downstream code
must read structured fields from the :data:`ROW_META_COLS` columns, never split
the id.
"""

from __future__ import annotations

import numpy as np

TYPE_MASSBAL = "massbal"
TYPE_SUM     = "sum"
TYPE_RATIO   = "ratio"
TYPE_BOUND   = "bound"


class ConstraintIdError(ValueError):
    """Raised when a constraint-id slot/value token would break the grammar."""


def _safe(tok, *, field: str) -> str:
    """Validate one slot/value token is underscore-free.

    ``None`` / NaN collapse to ``"na"``. Any other value is stringified and must
    contain no ``_`` (else :class:`ConstraintIdError`).
    """
    if tok is None or (isinstance(tok, float) and np.isnan(tok)):
        return "na"
    s = str(tok)
    if s in ("nan", "None", ""):
        return "na"
    if "_" in s:
        raise ConstraintIdError(
            f"{field} token {s!r} contains '_', which is reserved as the "
            f"constraint_id slot separator. Use an underscore-free value."
        )
    return s


def build_constraint_id(type_: str, variant: str, *specs, year=None) -> str:
    """Join ``type_``, ``variant``, each spec and ``year`` with ``_``.

    Every token is validated underscore-free. ``year=None`` renders as ``na``.
    """
    parts = [_safe(type_, field="type"), _safe(variant, field="variant")]
    parts += [_safe(s, field="spec") for s in specs]
    parts.append(_safe(year, field="year"))
    return "_".join(parts)


def render_edge(sector_from, region_from, sector_to, region_to) -> str:
    """Render one edge as an underscore-free ``from→to`` token.

    Form: ``{sFrom}.{rFrom}-{sTo}.{rTo}`` — uses ``.`` (sector↔region) and ``-``
    (from↔to), never ``_``. Each component is validated underscore-free.
    """
    sf = _safe(sector_from, field="sector_from")
    rf = _safe(region_from, field="region_from")
    st = _safe(sector_to,   field="sector_to")
    rt = _safe(region_to,   field="region_to")
    return f"{sf}.{rf}-{st}.{rt}"


# Canonical, uniform column order every builder's row_meta must emit, so the
# engine's concat produces one consistent schema (no silently-introduced NaN
# columns). Fields that do not apply to a row are written as None.
ROW_META_COLS = [
    "constraint_id", "type", "variant", "is_hard", "cone_type",
    "time", "valuation", "sector", "region", "entity", "entity_sub",
    "ratio_kind", "bound_dir",
    "num_sector_from", "num_region_from", "num_sector_to", "num_region_to", "num_valuation",
    "denom_sector_from", "denom_region_from", "denom_sector_to", "denom_region_to", "denom_valuation",
]


def empty_row_meta_dict() -> dict:
    """A row_meta dict with every :data:`ROW_META_COLS` key set to None.

    Builders start from this and overwrite the keys they populate, guaranteeing
    a uniform column set across blocks.
    """
    return {col: None for col in ROW_META_COLS}
