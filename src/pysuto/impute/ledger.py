"""
ledger.py
=========
``PriorLedger`` — the one place a Stage-2 prior is written.

Every imputer reads ``ledger.f`` / ``ledger.sigma`` and writes through
:meth:`PriorLedger.write` (or :meth:`PriorLedger.write_many`). The ledger enforces,
at write time, the three rules a prior obeys:

* **no overwrite** — an edge that already holds a value is never rewritten, so the
  first writer in cascade order keeps it;
* **finite and non-negative** — a written value is finite and its σ finite and ≥ 0,
  or the write raises ``ValueError`` (a NaN value would leave the edge missing yet
  logged; a NaN or negative σ would enter ``sigma_agg``);
* **writer scope** — an imputer writes only inside the scope it declares
  (:meth:`BaseImputer.writer_scope`); scopes may overlap, the cascade order decides
  precedence (GLOSSARY.md "Writer scope").

A write outside scope raises :class:`ScopeError` under ``strict_scope=True`` (tests)
and is logged at WARNING and dropped otherwise (production). The check is at write
time, not post hoc, because a post-hoc drop cannot undo a value an imputer already
propagated into its own later fills.

The ledger also stamps the imputer id and the cascade pass onto each log entry.
"""

from __future__ import annotations

import numpy as np

from pysuto.logging import get_logger

from .base import BaseImputer, _log

logger = get_logger(__name__)


class ScopeError(RuntimeError):
    """An imputer wrote an edge outside its declared writer scope."""


def _check_written(vals: np.ndarray, sigs: np.ndarray, imputer_id: str) -> None:
    """Raise ``ValueError`` unless every value is finite and every σ finite and ≥ 0."""
    bad_v = ~np.isfinite(vals)
    with np.errstate(invalid="ignore"):
        bad_s = ~np.isfinite(sigs) | (sigs < 0)
    if bad_v.any() or bad_s.any():
        raise ValueError(
            f"imputer {imputer_id!r} wrote {int(bad_v.sum())} non-finite value(s) and "
            f"{int(bad_s.sum())} non-finite or negative sigma(s)"
        )


class PriorLedger:
    """
    The prior under construction: ``f``, ``sigma`` and the entries written to them.

    ``f`` and ``sigma`` are held by reference and filled in place.

    Parameters
    ----------
    f, sigma : numpy.ndarray
        Float64 prior value and σ, shape ``(n_edges,)``; NaN = not yet written.
    strict_scope : bool
        Raise :class:`ScopeError` on an out-of-scope write instead of dropping it.
    """

    def __init__(self, f: np.ndarray, sigma: np.ndarray,
                 strict_scope: bool = False) -> None:
        self.f = f
        self.sigma = sigma
        self.strict_scope = bool(strict_scope)
        self._scope: np.ndarray | None = None
        self._imputer_id = ""
        self._pass: int | None = None
        self._entries: list[dict] = []

    # -- writer lifecycle ---------------------------------------------------

    def open(self, imputer_id: str, pass_no: int | None, scope: np.ndarray) -> None:
        """Start a step: subsequent writes belong to ``imputer_id`` inside ``scope``."""
        scope = np.asarray(scope)
        if scope.dtype != bool or scope.shape != self.f.shape:
            raise ValueError(
                f"writer scope of {imputer_id!r} must be a bool array of shape "
                f"{self.f.shape}; got {scope.dtype} {scope.shape}"
            )
        self._scope = scope
        self._imputer_id = imputer_id
        self._pass = pass_no
        self._entries = []

    def close(self) -> list[dict]:
        """End the step and return its log entries, stamped with imputer id and pass."""
        entries, self._entries = self._entries, []
        self._scope = None
        return entries

    def in_scope(self, e: int) -> bool:
        """Whether the open writer may write edge ``e`` (a query; it records nothing)."""
        return bool(self._scope[e])

    # -- writes -------------------------------------------------------------

    def write(
        self, e: int, val: float, sig: float, step: int, method: str,
        sources, coeffs, desc: str = "", param_id: str | None = None,
        relaxed: bool = False, assumed_absent=(),
    ) -> bool:
        """
        Write one edge; True when written, False when the edge already held a value
        (checked first, as no-overwrite) or (non-strict) lay outside scope.

        ``relaxed`` / ``assumed_absent`` tag a relaxed recipe close and the primary
        edges it counted as 0 (provenance only; the arrays are unaffected).
        """
        if not np.isnan(self.f[e]):
            return False
        if not self._scope[e]:
            self._reject(1, e)
            return False
        _check_written(np.asarray([val]), np.asarray([sig]), self._imputer_id)
        self.f[e] = val
        self.sigma[e] = sig
        self._append(_log(int(e), step, method, sources, coeffs, val, sig, desc,
                          param_id, relaxed, list(assumed_absent)))
        return True

    def write_many(
        self, edges: np.ndarray, vals: np.ndarray, sigs: np.ndarray, step: int,
        method: str, desc: str = "", param_id: str | None = None,
        sources: list | None = None, coeffs: list | None = None,
    ) -> int:
        """
        Write many edges at once (vectorised); returns the number written.

        ``sources`` / ``coeffs`` are per-edge lists, empty for every edge when omitted.
        Edges already holding a value are skipped; out-of-scope ones raise or drop as
        in :meth:`write`.
        """
        edges = np.asarray(edges, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        sigs = np.asarray(sigs, dtype=np.float64)
        free = np.isnan(self.f[edges])
        ok = self._scope[edges]
        if not (ok | ~free).all():
            bad = free & ~ok
            self._reject(int(bad.sum()), int(edges[bad][0]))
        keep = ok & free
        if not keep.all():
            edges, vals, sigs = edges[keep], vals[keep], sigs[keep]
            if sources is not None:
                sources = [s for s, k in zip(sources, keep) if k]
            if coeffs is not None:
                coeffs = [c for c, k in zip(coeffs, keep) if k]
        _check_written(vals, sigs, self._imputer_id)
        self.f[edges] = vals
        self.sigma[edges] = sigs
        for i, (e, v, s) in enumerate(zip(edges, vals, sigs)):
            self._append(_log(
                int(e), step, method,
                [] if sources is None else sources[i],
                [] if coeffs is None else coeffs[i],
                v, s, desc, param_id,
            ))
        return int(edges.size)

    # -- internals ----------------------------------------------------------

    def _append(self, entry: dict) -> None:
        entry["imputer_id"] = self._imputer_id
        entry["pass"] = self._pass
        self._entries.append(entry)

    def _reject(self, n: int, first_edge: int) -> None:
        msg = (f"imputer {self._imputer_id!r} wrote {n} edge(s) outside its writer "
               f"scope (first: edge {first_edge})")
        if self.strict_scope:
            raise ScopeError(msg)
        logger.warning("%s; dropped", msg)


def run_imputer(imputer: BaseImputer, f: np.ndarray, sigma: np.ndarray, step: int,
                acc, pass_no: int | None = None,
                strict_scope: bool = True) -> list[dict]:
    """
    Run one prepared imputer once on ``f`` / ``sigma``; return its log entries.

    The single-imputer entry point of tests and one-off scripts (the engine shares
    one ledger across its steps). Strict by default, unlike the engine: a script
    that runs one imputer wants a scope breach to stop it.

    Parameters
    ----------
    imputer : BaseImputer
        Already prepared (and attached, where it needs the aggregation).
    f, sigma : numpy.ndarray
        The prior, filled in place.
    step, pass_no : int
        Logged on each entry.
    acc : AccountingStructure
        Passed to ``imputer.writer_scope`` and ``imputer.impute``.
    strict_scope : bool
        Raise :class:`ScopeError` instead of dropping an out-of-scope write.
    """
    ledger = PriorLedger(f, sigma, strict_scope=strict_scope)
    ledger.open(imputer.id, pass_no, imputer.writer_scope(acc))
    imputer.impute(ledger, step, acc)
    return ledger.close()
