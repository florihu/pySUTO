"""
logging.py
==========
Named loggers with a console handler, and the per-module and per-run log files a driver
turns on (:func:`configure_file_logs`, :func:`start_run_log`, :func:`add_run_handler`).
Importing this module writes nothing (#190). Imports nothing from pysuto.
"""

from __future__ import annotations

import logging
import sys
import os
import threading
from datetime import datetime


_LOG_DIR = None          # set by configure_file_logs(); None until a run asks for files
_NAMES: list = []        # every logger get_logger returned, in first-call order
_LOCK = threading.Lock() # guards _LOG_DIR, _NAMES and the file-handler swap (threads)


class _ConsoleHandler(logging.StreamHandler):
    """A stream handler that replaces a character its stream cannot encode.

    A pipe on Windows encodes with the ANSI code page (cp1252), which lacks ``≈`` and ``→``;
    the plain handler then drops the whole record with a "Logging error".
    """

    def format(self, record):
        text = super().format(record)
        enc = getattr(self.stream, "encoding", None)
        return text.encode(enc, "replace").decode(enc) if enc else text


def utf8_stdio_env(env=None) -> dict:
    """A copy of ``env`` (default: this process's) in which a child Python writes its
    stdout and stderr as UTF-8, whatever the locale (#293).

    ``PYTHONIOENCODING`` sets the streams only; ``PYTHONUTF8`` would also change the
    default encoding of every ``open()`` of the child, which the pipeline does not want.
    """
    out = dict(os.environ if env is None else env)
    out["PYTHONIOENCODING"] = "utf-8"
    return out


def use_utf8_stdio() -> None:
    """Make this process's stdout and stderr UTF-8 (#293), for a driver run directly with
    redirected output: a pipe or file on Windows encodes with cp1252, which lacks the
    ``σ`` and ``→`` of the diag summaries. A stream without ``reconfigure`` is left."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and getattr(stream, "encoding", "") != "utf-8":
            stream.reconfigure(encoding="utf-8")


def get_logger(name="system"):
    """
    Return a named logger with a console handler and no file handler.

    Importing pySUTO writes nothing to disk (#190); the per-module file
    ``<log_dir>/<name>.log`` appears only after :func:`configure_file_logs` names a
    directory. Calling get_logger with the same name twice is idempotent
    (handlers are not duplicated).
    """
    logger = logging.getLogger(name)
    with _LOCK:
        if not logger.handlers:  # no duplicate handlers; own ones only
            logger.setLevel(logging.DEBUG)
            ch = _ConsoleHandler(sys.stdout)
            ch.setFormatter(_formatter())
            logger.addHandler(ch)
            _NAMES.append(name)
            if _LOG_DIR is not None:
                _add_file_handler(logger, name)

    return logger


def configure_file_logs(log_dir) -> None:
    """
    Send the per-module logs to ``<log_dir>/<name>.log`` from now on.

    Adds a file handler to every logger :func:`get_logger` has returned and to
    every one it returns later. A driver calls it once with ``<data-dir>/logs``;
    the directory is created here, not at import. Calling it again with the same
    directory is a no-op; with another one, the loggers move to the new directory.
    """
    global _LOG_DIR
    log_dir = os.fspath(log_dir)
    with _LOCK:
        if log_dir == _LOG_DIR:
            return
        _LOG_DIR = log_dir
        os.makedirs(log_dir, exist_ok=True)
        for name in _NAMES:
            logger = logging.getLogger(name)
            for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
                logger.removeHandler(h)
                h.close()
            _add_file_handler(logger, name)


def start_run_log(data_dir, name: str) -> str:
    """
    Start the logs of one driver run under ``<data_dir>/logs`` and return that directory.

    Makes stdout and stderr UTF-8 (:func:`use_utf8_stdio`), calls :func:`configure_file_logs` for the per-module files and attaches the
    timestamped run handler ``<data_dir>/logs/<name>_<YYYYmmdd_HHMMSS>.log`` to the
    root logger (:func:`add_run_handler`).
    """
    use_utf8_stdio()
    log_dir = os.path.join(os.fspath(data_dir), "logs")
    configure_file_logs(log_dir)
    add_run_handler(os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"))
    return log_dir


def _add_file_handler(logger: logging.Logger, name: str) -> None:
    fh = logging.FileHandler(os.path.join(_LOG_DIR, f"{name}.log"), encoding="utf-8")
    fh.setFormatter(_formatter())
    logger.addHandler(fh)


def add_run_handler(log_path: str) -> logging.FileHandler:
    """
    Attach a FileHandler for *this run* to the root logger.

    Because all named loggers propagate to root by default, a single call
    here funnels every logger in the process (feedpipe, conc, baci, …)
    into one timestamped file — without changing per-module log files.

    Returns the handler so the caller can remove it later if needed.

    Usage (top of an init script)::

        from pysuto.logging import add_run_handler
        add_run_handler(f"{data_dir}/logs/baci_v0_{datetime.now():%Y%m%d_%H%M%S}.log")
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(_formatter())
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    return fh


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
