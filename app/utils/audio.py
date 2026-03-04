"""Thread-safe audio buffer and PCM16 conversion utilities.

Responsibility: Provide a safe container for audio bytes that is
    written by the async event loop and read by the Streamlit/playback
    thread without corruption.
Pattern: Producer-Consumer via a bounded queue with a simple bytes
    interface; plus pure-function audio converters.
Why: Streamlit runs in a different thread from asyncio.  Without a
    thread-safe buffer, audio playback will glitch, skip, or crash
    with race-condition corruption.
"""

from __future__ import annotations

import base64
import logging
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# OpenAI Realtime PCM16 default
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


class AudioBuffer:
    """Thread-safe audio buffer for cross-thread producer/consumer usage.

    The async receive loop *appends* decoded PCM16 bytes.
    The playback / UI thread *reads* frames without blocking the producer.

    Implementation uses a ``deque`` protected by a ``threading.Lock``
    so it works across asyncio ↔ sync boundaries.
    """

    def __init__(self, max_bytes: int = 5 * SAMPLE_RATE * SAMPLE_WIDTH) -> None:
        """
        Parameters
        ----------
        max_bytes:
            Approximate ceiling for buffered data (default ~5 s at 24 kHz/16-bit).
            When exceeded, oldest data is silently dropped (ring-buffer behavior).
        """
        self._lock = threading.Lock()
        self._buffer = deque[bytes]()
        self._total_bytes = 0
        self._max_bytes = max_bytes

    def append(self, data: bytes) -> None:
        """Append PCM16 audio bytes (producer side)."""
        with self._lock:
            self._buffer.append(data)
            self._total_bytes += len(data)
            # Ring-buffer eviction
            while self._total_bytes > self._max_bytes and self._buffer:
                evicted = self._buffer.popleft()
                self._total_bytes -= len(evicted)

    def read_all(self) -> bytes:
        """Drain the entire buffer and return concatenated bytes (consumer side)."""
        with self._lock:
            if not self._buffer:
                return b""
            chunks = list(self._buffer)
            self._buffer.clear()
            self._total_bytes = 0
        return b"".join(chunks)

    def read_frames(self, n_frames: int) -> bytes:
        """Read exactly *n_frames* samples (2 bytes each) from the front.

        Returns fewer bytes if the buffer doesn't have enough.
        """
        needed = n_frames * SAMPLE_WIDTH
        with self._lock:
            joined = b"".join(self._buffer)
            self._buffer.clear()
            self._total_bytes = 0

        out = joined[:needed]
        leftover = joined[needed:]
        if leftover:
            with self._lock:
                self._buffer.appendleft(leftover)
                self._total_bytes = len(leftover)
        return out

    def clear(self) -> None:
        """Discard all buffered audio."""
        with self._lock:
            self._buffer.clear()
            self._total_bytes = 0

    @property
    def depth_bytes(self) -> int:
        """Current buffer size in bytes."""
        with self._lock:
            return self._total_bytes

    @property
    def depth_seconds(self) -> float:
        """Current buffer duration in seconds."""
        return self.depth_bytes / (SAMPLE_RATE * SAMPLE_WIDTH)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._total_bytes == 0


def decode_audio_delta(b64_data: str) -> bytes:
    """Decode a base64 audio delta string into raw PCM16 bytes."""
    return base64.b64decode(b64_data)


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM16 (little-endian signed 16-bit) to float32 in [-1, 1].

    Useful for waveform visualization or further DSP.
    """
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - 1]
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float32 samples [-1, 1] back to PCM16 bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)
    return int_samples.tobytes()


def encode_pcm16_to_base64(pcm_bytes: bytes) -> str:
    """Encode raw PCM16 bytes for ``input_audio_buffer.append``."""
    return base64.b64encode(pcm_bytes).decode("ascii")


def pcm16_to_wav_bytes(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono data in a WAV container for ``st.audio`` playback."""
    import io
    import wave

    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()


def wav_bytes_to_pcm16_24k(wav_bytes: bytes) -> bytes:
    """Convert WAV audio bytes to PCM16 24 kHz mono for the OpenAI Realtime API.

    Handles arbitrary sample rates, channel counts, and bit depths
    coming from the browser's ``st.audio_input`` widget.
    """
    import io
    import wave

    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            raw_frames = wf.readframes(wf.getnframes())
            src_rate = wf.getframerate()
            src_channels = wf.getnchannels()
            src_width = wf.getsampwidth()

    # Decode raw frames to int16
    if src_width == 2:
        samples = np.frombuffer(raw_frames, dtype=np.int16)
    elif src_width == 4:
        samples = (np.frombuffer(raw_frames, dtype=np.int32) >> 16).astype(np.int16)
    elif src_width == 1:
        samples = ((np.frombuffer(raw_frames, dtype=np.uint8).astype(np.int16) - 128) * 256).astype(
            np.int16
        )
    else:
        raise ValueError(f"Unsupported WAV sample width: {src_width}")

    # Mix to mono
    if src_channels > 1:
        samples = samples.reshape(-1, src_channels).mean(axis=1).astype(np.int16)

    # Resample to 24 kHz
    if src_rate != SAMPLE_RATE:
        float_samples = samples.astype(np.float32)
        new_length = int(len(float_samples) * SAMPLE_RATE / src_rate)
        indices = np.linspace(0, len(float_samples) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(float_samples)), float_samples)
        samples = resampled.astype(np.int16)

    return samples.tobytes()
