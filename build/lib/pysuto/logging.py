# src/pysuto/logging.py
import logging
import sys
import os


def get_logger(name="system"):
    """
    Return a named logger with console + per-module file handler.

    The file is written to logs/<name>.log (created on first call).
    Calling get_logger with the same name twice is idempotent (handlers
    are not duplicated).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent duplicate handlers; check own handlers only (not ancestors')
        logger.setLevel(logging.DEBUG)
        formatter = _formatter()

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", f"{name}.log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def add_run_handler(log_path: str) -> logging.FileHandler:
    """
    Attach a FileHandler for *this run* to the root logger.

    Because all named loggers propagate to root by default, a single call
    here funnels every logger in the process (feedpipe, conc, baci, …)
    into one timestamped file — without changing per-module log files.

    Returns the handler so the caller can remove it later if needed.

    Usage (top of an init script)::

        from pysuto.logging import add_run_handler
        add_run_handler(f"logs/baci_v0_{datetime.now():%Y%m%d_%H%M%S}.log")
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(_formatter())
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    return fh


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
