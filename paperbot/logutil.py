"""Structured logging to console + rotating file."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(log_dir: Path, level: str = "INFO", filename: str = "paperbot.log") -> None:
    """Configure root logging once: console + 5MB rotating file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. repeated CLI calls in tests)
        return
    root.setLevel(level.upper())
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMAT))
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / filename, maxBytes=5_000_000, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(console)
    root.addHandler(file_handler)
    # ccxt is chatty at DEBUG; keep third-party noise down.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
