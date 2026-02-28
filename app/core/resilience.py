"""Resilience primitives — Circuit Breaker and exponential-backoff retry.

Responsibility: Protect the system from cascading failures when the
    OpenAI Realtime WebSocket becomes unreachable.
Pattern: Circuit Breaker (GoF-adjacent stability pattern) with three
    states — CLOSED (normal), OPEN (failing, cooldown), HALF_OPEN
    (probe).  Combined with exponential-backoff retry for transient errors.
Why: A naive "reconnect forever" loop can saturate logs, burn API quota,
    and block the UI.  The circuit breaker gives the remote service time
    to recover and gives the user clear feedback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Observable states of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for WebSocket reconnection.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures before tripping OPEN.
    cooldown_seconds:
        How long to stay OPEN before allowing a HALF_OPEN probe.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for cooldown expiry."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker → HALF_OPEN (cooldown expired)")
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def record_success(self) -> None:
        """Call after a successful connection / operation."""
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit breaker → CLOSED (success)")
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Call after a connection failure."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker → OPEN after %d failures (cooldown %.0fs)",
                self._failure_count,
                self._cooldown_seconds,
            )
        else:
            logger.info(
                "Circuit breaker failure %d/%d",
                self._failure_count,
                self._failure_threshold,
            )

    def allow_request(self) -> bool:
        """Return True if a connection attempt is permitted."""
        state = self.state  # triggers cooldown check
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True
        return False

    def reset(self) -> None:
        """Force-reset to CLOSED (e.g. manual reconnect button)."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        logger.info("Circuit breaker manually reset → CLOSED")


async def retry_with_backoff(
    coro_factory,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    circuit_breaker: CircuitBreaker | None = None,
) -> bool:
    """Retry an async callable with exponential backoff.

    Parameters
    ----------
    coro_factory:
        A zero-arg callable that returns an awaitable (e.g. ``lambda: connect()``).
    max_retries:
        Maximum number of attempts.
    base_delay:
        Initial delay in seconds (doubled each retry).
    max_delay:
        Cap on the delay.
    circuit_breaker:
        Optional circuit breaker that gates attempts.

    Returns
    -------
    bool
        True if the coroutine succeeded, False if retries exhausted.
    """
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        if circuit_breaker and not circuit_breaker.allow_request():
            logger.warning("Circuit is OPEN — skipping attempt %d", attempt)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue

        try:
            await coro_factory()
            if circuit_breaker:
                circuit_breaker.record_success()
            return True
        except Exception as exc:
            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, exc)
            if circuit_breaker:
                circuit_breaker.record_failure()
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    logger.error("All %d retry attempts exhausted", max_retries)
    return False
