"""Async event dispatcher — Observer pattern for decoupled event routing.

Responsibility: Accept typed ServerEvent objects and fan them out to
    registered async callbacks, grouped by event type string.
Pattern: Observer / Pub-Sub — listeners register interest in specific
    event types; the dispatcher invokes them when matching events arrive.
Why: Without a dispatcher, the receive loop would need to import and
    call every handler directly, creating a tightly-coupled monolith.
    The Observer pattern lets each subsystem (audio pipeline, transcript
    UI, debug log) register itself independently.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from app.models.events import _ServerBase

logger = logging.getLogger(__name__)

# Type alias for an async event handler
EventHandler = Callable[[_ServerBase], Awaitable[None]]


class EventDispatcher:
    """Async event dispatcher implementing the Observer pattern.

    Thread-safe for registration (though typically done at startup).
    Dispatch is fully async and concurrent — all handlers for a given
    event type are launched as tasks and awaited together.

    Usage::

        dispatcher = EventDispatcher()
        dispatcher.register("response.audio.delta", my_audio_handler)
        dispatcher.register("error", my_error_handler)
        await dispatcher.dispatch(some_server_event)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []

    def register(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* to be called for events matching *event_type*."""
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler %s for %r", handler.__name__, event_type)

    def register_global(self, handler: EventHandler) -> None:
        """Register *handler* to be called for *every* event (e.g. debug logging)."""
        self._global_handlers.append(handler)
        logger.debug("Registered global handler %s", handler.__name__)

    def unregister(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def dispatch(self, event: _ServerBase) -> None:
        """Route *event* to all matching handlers concurrently.

        Errors in individual handlers are logged but do not prevent
        other handlers from executing.
        """
        event_type: str = getattr(event, "type", "unknown")
        specific = self._handlers.get(event_type, [])
        all_handlers = specific + self._global_handlers

        if not all_handlers:
            logger.debug("No handlers for event type %r", event_type)
            return

        results = await asyncio.gather(
            *(self._safe_call(h, event) for h in all_handlers),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handler_name = all_handlers[i].__name__
                logger.error(
                    "Handler %s raised %s: %s",
                    handler_name,
                    type(result).__name__,
                    result,
                )

    @staticmethod
    async def _safe_call(handler: EventHandler, event: _ServerBase) -> Any:
        """Invoke handler, converting sync exceptions into return values."""
        return await handler(event)

    @property
    def registered_types(self) -> list[str]:
        """Return list of event types that have at least one handler."""
        return [t for t, h in self._handlers.items() if h]

    def clear(self) -> None:
        """Remove all handlers (useful in tests)."""
        self._handlers.clear()
        self._global_handlers.clear()
