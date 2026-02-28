# OpenAI Realtime Voice Agent

A production-ready implementation of OpenAI's Realtime API for deploying high-performance, low-latency AI voice agents. Built with **Streamlit**, **Pydantic v2**, **asyncio**, and **websockets**.

## What This Project Is

This project demonstrates a clean, event-driven architecture for building voice agents on top of the OpenAI Realtime WebSocket API. It prioritizes **correctness**, **type safety**, and **resilience** over feature count — making it a strong reference for how to structure real-time streaming applications in Python.

Key capabilities:
- **Validated configuration** — every session parameter is checked before hitting the wire
- **Discriminated-union event parsing** — raw JSON becomes typed Python objects or fails loudly
- **Observer-pattern event routing** — subsystems register for events independently
- **Thread-safe audio buffering** — async producer, sync consumer, zero corruption
- **Circuit-breaker resilience** — automatic backoff and recovery on disconnects
- **Demo mode** — test the full pipeline via text input (no microphone required)

## Demo — How to Run

### Prerequisites

- Python 3.11+
- An OpenAI API key with Realtime API access

### Quickstart

```bash
# Clone and install
git clone https://github.com/rohit07cf/openai-realtime-voice.git
cd openai-realtime-voice
pip install -e ".[dev]"

# Run tests
pytest

# Launch the UI
streamlit run app/main.py
```

1. Enter your OpenAI API key in the sidebar
2. Adjust voice, temperature, and turn-detection settings
3. Click **Connect**
4. Use **Demo Mode** to send text messages — or connect a microphone for full voice interaction

### Environment Variables (optional)

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
REALTIME_MODEL=gpt-4o-realtime-preview-2024-12-17
VOICE=alloy
TEMPERATURE=0.8
LOG_LEVEL=INFO
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Config   │  │ Transcript│  │  Audio   │  │  Debug Log   │  │
│  │  Panel    │  │   View    │  │  Status  │  │   Panel      │  │
│  └────┬─────┘  └─────▲─────┘  └────▲─────┘  └──────▲───────┘  │
│       │               │             │               │          │
└───────┼───────────────┼─────────────┼───────────────┼──────────┘
        │               │             │               │
   ┌────▼───────────────┴─────────────┴───────────────┴────┐
   │                  VoiceAgentBridge                      │
   │              (Adapter / Bridge Pattern)                │
   │   ┌──────────────┐          ┌──────────────────┐      │
   │   │ AudioBuffer   │◄────────│  Event Handlers  │      │
   │   │ (thread-safe) │         │  (async → sync)  │      │
   │   └──────────────┘          └────────▲─────────┘      │
   └──────────┬───────────────────────────┼────────────────┘
              │ send()                    │ dispatch()
   ┌──────────▼───────────────────────────┴────────────────┐
   │                  RealtimeManager                      │
   │                  (Singleton Pattern)                   │
   │  ┌─────────────────┐    ┌──────────────────────┐      │
   │  │ RealtimeConnection│   │  EventDispatcher     │      │
   │  │ (Adapter Pattern) │   │  (Observer Pattern)  │      │
   │  │                   │   │                      │      │
   │  │  WebSocket ←──────┼───┤  register(type, fn)  │      │
   │  │  send / recv      │   │  dispatch(event)     │      │
   │  └─────────────────┘    └──────────────────────┘      │
   │  ┌──────────────────────────────────────────────┐      │
   │  │         CircuitBreaker + RetryPolicy         │      │
   │  │         (Circuit Breaker Pattern)            │      │
   │  └──────────────────────────────────────────────┘      │
   └────────────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────┐
   │  OpenAI Realtime API   │
   │  wss://api.openai.com  │
   └────────────────────────┘
```

### Data Flow

1. **Config Panel** → `RealtimeConfig` (Pydantic validated) → `session.update` event
2. **User Input** (text or audio) → `ClientEvent` → WebSocket → OpenAI
3. **OpenAI** → raw JSON → `parse_server_event()` → typed `ServerEvent`
4. **EventDispatcher** → routes to registered handlers (audio, transcript, error)
5. **AudioBuffer** (thread-safe) → Streamlit UI reads for playback status
6. **Transcript** accumulation → rendered in chat-message format

## Design Patterns Used

### Singleton — `RealtimeManager`
Ensures exactly one WebSocket connection per process. Critical because Streamlit re-executes the entire script on every widget interaction; without a singleton, each click would leak a connection.

### Observer / Pub-Sub — `EventDispatcher`
Decouples event producers (the receive loop) from consumers (audio pipeline, transcript UI, debug log). Each subsystem registers interest in specific event types. A failing handler cannot block other handlers.

### Adapter / Bridge — `VoiceAgentBridge`, `RealtimeConnection`
- `RealtimeConnection` wraps the `websockets` library behind a minimal interface, making it swappable.
- `VoiceAgentBridge` translates the async engine's interface into synchronous calls that Streamlit can make, and converts raw audio deltas into buffered PCM16 data.

### Circuit Breaker — `CircuitBreaker`
Three-state machine (CLOSED → OPEN → HALF_OPEN) that prevents the system from hammering a failing endpoint. After N consecutive failures, reconnect attempts are paused for a cooldown period, giving the remote service time to recover.

### Discriminated Union — `ServerEvent`, `ClientEvent`
Pydantic's `Literal` discriminator on the `type` field turns raw JSON into the exact Python subclass — or raises a clear `ValidationError`. Unrecognized event types fall back to `GenericServerEvent` so the pipeline never drops data silently.

## Design Document: Pydantic for Data Integrity in Streaming

### The Problem

The OpenAI Realtime API transmits dozens of event types over a single WebSocket. Each event has a different schema. A `response.audio.delta` carries a base64-encoded audio chunk; an `error` carries a nested `error` object with `message`, `code`, and `param` fields. Session configuration includes floats with specific ranges, enums with specific values, and nullable sub-objects.

In a fast-moving streaming pipeline, **silent data corruption is the worst outcome**. A malformed config might produce garbled audio. A mistyped event field might crash the pipeline minutes later in an unrelated handler.

### Why Pydantic Matters Here

**1. Validation at the Boundary**

Every piece of external data — environment variables, user-entered config, and server events — passes through a Pydantic model before entering the system:

```python
class RealtimeConfig(BaseModel):
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.8
    voice: Voice = Voice.ALLOY  # StrEnum — only valid voices accepted
```

An invalid temperature or voice is caught *before* the `session.update` event is sent, not after the server rejects it with a cryptic error.

**2. Discriminated Unions for Event Routing**

Instead of `if data["type"] == "response.audio.delta"` scattered through the codebase:

```python
ServerEvent = Annotated[
    SessionCreatedEvent | ErrorEvent | ResponseAudioDelta | ...,
    Field(discriminator="type"),
]
```

`parse_server_event()` resolves the type once, at the edge, and hands downstream code a fully-typed object with IDE autocomplete, docstrings, and guaranteed field presence.

**3. Preventing Invalid State at Runtime**

The `TurnDetectionConfig` model enforces that `threshold` is between 0 and 1, `silence_duration_ms` is non-negative, and `type` is one of the documented values. These constraints are impossible to violate after construction — a property that static types alone cannot guarantee.

**4. Performance**

Pydantic v2's Rust-based core (`pydantic-core`) validates models in microseconds. On the `response.audio.delta` hot path (tens of events per second), the overhead is negligible compared to base64 decoding and network I/O.

### Design Decision

We validate *at the boundary* (event parsing, config construction) and trust *within the core*. Internal functions receive typed models, not `dict[str, Any]`. This keeps the interior fast and the surface safe.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                          # Streamlit entrypoint
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enums.py                     # Voice, Modality, ConnectionState
│   │   ├── config.py                    # RealtimeConfig, AppSettings
│   │   └── events.py                    # ServerEvent, ClientEvent unions
│   ├── core/
│   │   ├── __init__.py
│   │   ├── connection.py                # WebSocket transport adapter
│   │   ├── manager.py                   # RealtimeManager (Singleton)
│   │   ├── dispatcher.py                # EventDispatcher (Observer)
│   │   └── resilience.py                # CircuitBreaker, retry_with_backoff
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── bridge.py                    # VoiceAgentBridge (Adapter)
│   │   └── components.py               # Streamlit rendering functions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio.py                     # AudioBuffer, PCM16 conversions
│   │   └── logging.py                   # Logging setup
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_config_validation.py    # 16 tests
│       ├── test_event_parsing.py        # 14 tests
│       ├── test_dispatcher.py           # 10 tests
│       └── test_audio_buffer.py         # 15 tests
├── README.md
├── pyproject.toml
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml
```

## Testing Strategy

**60 tests** across four test modules:

| Module | Tests | What it validates |
|--------|-------|-------------------|
| `test_config_validation.py` | 16 | Config ranges, enum constraints, session payload serialization |
| `test_event_parsing.py` | 14 | Discriminated union dispatch, fallback for unknown events, client event construction |
| `test_dispatcher.py` | 10 | Handler routing, error isolation, global handlers, unregister, clear |
| `test_audio_buffer.py` | 15 | Buffer operations, ring-buffer eviction, thread safety, PCM16 conversions, circuit breaker states |

Run all tests:

```bash
pytest -v
```

## Limitations & Future Work

### Known Limitations

- **No browser audio capture yet** — `streamlit-webrtc` integration is stubbed but not wired. The demo uses text-input mode.
- **Audio playback** — The UI shows buffer depth but does not auto-play audio through the browser. A WebRTC or JavaScript bridge is needed.
- **Single session** — The singleton manager supports one concurrent session. Multi-tenant deployments would need a session-per-user registry.
- **No function calling** — Tool/function definitions are accepted in config but the handler pipeline is not implemented.

### Next Steps

1. **streamlit-webrtc integration** — Capture mic audio in the browser and stream to `input_audio_buffer.append`
2. **JavaScript audio player** — Use `st.components.v1.html` to play PCM16 audio chunks in real time
3. **Function calling** — Parse `response.function_call_arguments.done` and invoke registered Python functions
4. **Observability** — Structured JSON logging, OpenTelemetry spans for event latency
5. **Ephemeral tokens** — Use the `/v1/realtime/client_secrets` endpoint for browser-safe auth

## License

MIT
