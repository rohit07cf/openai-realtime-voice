"""Structured logging setup for the voice agent.

Responsibility: Configure Python logging with a consistent format
    across all modules.
Pattern: Module-level factory — call once at startup.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a human-readable format.

    Parameters
    ----------
    level:
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(numeric)
    # Avoid duplicate handlers on repeated calls (e.g. Streamlit re-runs)
    if not root.handlers:
        root.addHandler(handler)
