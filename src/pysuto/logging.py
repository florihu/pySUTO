# src/logger.py
import logging
import sys
import os

def get_logger(name="system"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # prevent duplicate handlers
        logger.setLevel(logging.DEBUG)  # configurable level
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional: file handler
        os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
        path = os.path.join("logs", f"{name}.log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger