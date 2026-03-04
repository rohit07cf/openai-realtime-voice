"""Async event dispatcher — Observer pattern for decoupled event routing.

Responsibility: Accept typed ServerEvent objects and fan them out to
    registered async callbacks, grouped by event type string.
Pattern: Observer / Pub-Sub — listeners register interest in specific
    event types; the dispatcher invokes them when matching events arrive.
Why: Without a dispatcher, the receive loop would need to import and
    call every handler directly, creating a tightly-coupled monolith.
    The Observer pattern lets each subsystem (audio pipeline, transcript
    UI, debug log) register itself independently.

OpenAI Realtime API Integration:
This dispatcher is crucial for handling events received from the OpenAI Realtime API
via WebSocket connections. Events include audio deltas, transcripts, errors, and
control messages. By routing these events to appropriate handlers (e.g., audio playback,
UI updates), it enables real-time processing of AI responses and user interactions.

WebSockets and Real-Time Event Flow:
Events originate from the WebSocket connection established in connection.py.
As messages arrive asynchronously from OpenAI's servers, they are parsed and
dispatched here. WebSockets ensure low-latency delivery, and the dispatcher's
concurrent execution allows multiple handlers to process events simultaneously,
maintaining real-time responsiveness.

WebRTC Context (if applicable):
While not directly implemented here, this dispatcher can handle events containing
audio data captured via WebRTC in the client-side UI (e.g., microphone input).
WebRTC enables peer-to-peer audio streaming in browsers, which can be integrated
with this system for seamless voice input/output.

Real-Time Operation:
The dispatch method launches all relevant handlers concurrently using asyncio.gather,
ensuring that events like audio deltas are processed immediately upon arrival.
This supports features like live audio streaming, instant transcript updates, and
dynamic AI responses, all within the real-time constraints of voice conversations.
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
        """Register *handler* to be called for events matching *event_type*.

        Event Handler Registration:
        Handlers are functions that process specific types of events from the OpenAI Realtime API.
        For example, registering a handler for "response.audio.delta" allows processing of
        streaming audio data as it arrives in real-time. This setup enables modular,
        decoupled handling of different event types (audio, transcripts, errors, etc.),
        supporting the real-time nature of voice conversations.
        """
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

        Real-Time Event Processing:
        This method is the heart of real-time operation in the OpenAI Realtime API integration.
        When an event arrives (e.g., from the WebSocket in connection.py), it is immediately
        dispatched to all registered handlers for that event type. Using asyncio.gather,
        handlers run concurrently, ensuring low-latency processing of time-sensitive events
        like audio deltas or user interruptions.

        OpenAI Realtime API Events:
        Events can include:
        - response.audio.delta: Incremental audio data from AI responses
        - response.audio.transcript.delta: Live transcription updates
        - error: Connection or processing errors
        - session.created/updated: Configuration changes
        Dispatching these promptly enables seamless, real-time voice interactions.

        WebSockets Integration:
        Events are typically received via the WebSocket connection, parsed into typed objects,
        and then dispatched here. The persistent WebSocket ensures events arrive asynchronously
        and are handled without delay.

        Concurrency and Fault Tolerance:
        By launching handlers as concurrent tasks, the system can process multiple aspects
        (e.g., audio playback and UI updates) simultaneously. Errors in one handler don't
        block others, maintaining system stability during real-time operations.
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
