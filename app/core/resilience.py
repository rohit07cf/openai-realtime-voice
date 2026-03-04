"""Resilience primitives — Circuit Breaker and exponential-backoff retry.

Responsibility: Protect the system from cascading failures when the
    OpenAI Realtime WebSocket becomes unreachable.
Pattern: Circuit Breaker (GoF-adjacent stability pattern) with three
    states — CLOSED (normal), OPEN (failing, cooldown), HALF_OPEN
    (probe).  Combined with exponential-backoff retry for transient errors.
Why: A naive "reconnect forever" loop can saturate logs, burn API quota,
    and block the UI.  The circuit breaker gives the remote service time
    to recover and gives the user clear feedback.

OpenAI Realtime API Resilience:
In real-time voice applications using the OpenAI Realtime API, network
instability or server issues can disrupt WebSocket connections, interrupting
ongoing conversations. This module provides fault tolerance to maintain
real-time operation by preventing excessive reconnection attempts and allowing
graceful recovery.

WebSockets and Real-Time Stability:
WebSocket connections for the Realtime API must remain stable for continuous
audio streaming and event exchange. The circuit breaker prevents rapid-fire
reconnections that could overwhelm the network or API limits, while exponential
backoff spaces out retries to give servers time to recover. This ensures
persistent, low-latency communication essential for voice interactions.

WebRTC Integration Resilience:
While WebRTC handles client-side audio capture/playback, this resilience
layer protects the backend WebSocket to OpenAI. Failures here could cascade
to WebRTC streams, so stable reconnection prevents audio dropouts in real-time
conversations.

Real-Time Operation Impact:
Without resilience, a temporary network glitch could halt voice streaming,
breaking the real-time flow. These patterns allow automatic recovery without
user intervention, maintaining seamless AI conversations even under adverse
conditions.
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

    Circuit Breaker for Real-Time Connections:
    Protects the OpenAI Realtime API WebSocket from cascading failures.
    In CLOSED state, connections proceed normally. After repeated failures
    (e.g., network timeouts), it opens to prevent further attempts, giving
    the service time to recover. This prevents log spam, API quota exhaustion,
    and UI blocking during real-time voice sessions.
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
        """Call after a successful connection / operation.

        Real-Time Recovery:
        Upon successful WebSocket reconnection or operation, resets the circuit
        to CLOSED, allowing normal real-time traffic to resume immediately.
        This ensures that once the OpenAI Realtime API is reachable again,
        voice streaming and event exchange can continue without delay.
        """
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit breaker → CLOSED (success)")
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Call after a connection failure.

        Failure Handling for Real-Time Systems:
        Increments failure count and potentially opens the circuit after
        threshold failures. This prevents further WebSocket connection attempts
        to the OpenAI Realtime API, avoiding resource waste and log noise.
        In real-time voice apps, this gives the system breathing room to recover
        from network issues without interrupting ongoing conversations abruptly.
        """
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

    Exponential Backoff for Real-Time Resilience:
    Used for retrying WebSocket connections to the OpenAI Realtime API.
    Starts with short delays, exponentially increasing to avoid overwhelming
    the server during outages. Combined with the circuit breaker, it prevents
    aggressive retries that could disrupt real-time voice streaming or consume
    API resources. In real-time apps, this allows graceful recovery without
    long pauses in conversation.

    WebSockets and Real-Time Impact:
    Ensures that temporary WebSocket disconnections (e.g., due to network
    issues) are retried intelligently, minimizing downtime in voice
    interactions. The backoff prevents rapid reconnections that could
    exacerbate problems.

    Integration with WebRTC:
    Stable backend connections ensure that WebRTC audio streams remain
    functional, as failures here could propagate to client-side audio
    capture/playback.
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
