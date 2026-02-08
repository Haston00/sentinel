"""
SENTINEL — Structured logging.
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_DIR


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger that writes to console and file."""
    logger = logging.getLogger(f"sentinel.{name}")

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_file = LOG_DIR / "sentinel.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
