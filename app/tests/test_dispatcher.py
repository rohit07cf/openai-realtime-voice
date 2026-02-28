"""Tests for the async EventDispatcher (Observer pattern).

Verifies that events are routed to the correct handlers, that errors
in one handler don't block others, and that global handlers receive
every event.
"""

from __future__ import annotations

import pytest

from app.core.dispatcher import EventDispatcher
from app.models.events import (
    ErrorEvent,
    ResponseAudioDelta,
    SessionCreatedEvent,
    parse_server_event,
)


@pytest.fixture
def dispatcher():
    return EventDispatcher()


class TestEventDispatcher:
    """Core dispatcher behavior tests."""

    @pytest.mark.asyncio
    async def test_handler_receives_matching_event(self, dispatcher: EventDispatcher):
        received = []

        async def handler(event):
            received.append(event)

        dispatcher.register("session.created", handler)

        event = SessionCreatedEvent(session={"id": "s1"})
        await dispatcher.dispatch(event)

        assert len(received) == 1
        assert isinstance(received[0], SessionCreatedEvent)

    @pytest.mark.asyncio
    async def test_handler_ignores_non_matching_event(self, dispatcher: EventDispatcher):
        received = []

        async def handler(event):
            received.append(event)

        dispatcher.register("session.created", handler)

        event = ErrorEvent(error={"message": "test"})
        await dispatcher.dispatch(event)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self, dispatcher: EventDispatcher):
        results = []

        async def handler_a(event):
            results.append("a")

        async def handler_b(event):
            results.append("b")

        dispatcher.register("error", handler_a)
        dispatcher.register("error", handler_b)

        await dispatcher.dispatch(ErrorEvent(error={"message": "test"}))

        assert "a" in results
        assert "b" in results

    @pytest.mark.asyncio
    async def test_global_handler_receives_all_events(self, dispatcher: EventDispatcher):
        received = []

        async def global_handler(event):
            received.append(event.type)

        dispatcher.register_global(global_handler)

        await dispatcher.dispatch(SessionCreatedEvent())
        await dispatcher.dispatch(ErrorEvent(error={"message": "test"}))

        assert "session.created" in received
        assert "error" in received

    @pytest.mark.asyncio
    async def test_handler_error_does_not_block_others(self, dispatcher: EventDispatcher):
        results = []

        async def failing_handler(event):
            raise ValueError("boom")

        async def good_handler(event):
            results.append("ok")

        dispatcher.register("error", failing_handler)
        dispatcher.register("error", good_handler)

        await dispatcher.dispatch(ErrorEvent(error={"message": "test"}))

        assert "ok" in results

    @pytest.mark.asyncio
    async def test_unregister(self, dispatcher: EventDispatcher):
        received = []

        async def handler(event):
            received.append(True)

        dispatcher.register("error", handler)
        dispatcher.unregister("error", handler)

        await dispatcher.dispatch(ErrorEvent(error={"message": "test"}))
        assert len(received) == 0

    def test_registered_types(self, dispatcher: EventDispatcher):
        async def noop(event):
            pass

        dispatcher.register("error", noop)
        dispatcher.register("session.created", noop)

        types = dispatcher.registered_types
        assert "error" in types
        assert "session.created" in types

    @pytest.mark.asyncio
    async def test_dispatch_no_handlers(self, dispatcher: EventDispatcher):
        # Should not raise
        await dispatcher.dispatch(SessionCreatedEvent())

    def test_clear(self, dispatcher: EventDispatcher):
        async def noop(event):
            pass

        dispatcher.register("error", noop)
        dispatcher.register_global(noop)
        dispatcher.clear()

        assert len(dispatcher.registered_types) == 0

    @pytest.mark.asyncio
    async def test_dispatch_parsed_event(self, dispatcher: EventDispatcher):
        received = []

        async def handler(event):
            received.append(event)

        dispatcher.register("response.audio.delta", handler)

        raw = {"type": "response.audio.delta", "delta": "AAAA"}
        event = parse_server_event(raw)
        await dispatcher.dispatch(event)

        assert len(received) == 1
        assert isinstance(received[0], ResponseAudioDelta)
