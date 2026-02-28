"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app.core.manager import RealtimeManager


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test gets a fresh RealtimeManager singleton."""
    RealtimeManager.reset_singleton()
    yield
    RealtimeManager.reset_singleton()
