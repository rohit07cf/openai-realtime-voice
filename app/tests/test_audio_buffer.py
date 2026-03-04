"""Tests for the thread-safe AudioBuffer and PCM16 conversion utilities.

Verifies correctness of the ring-buffer eviction, cross-thread safety,
and format-conversion round-trips.
"""

from __future__ import annotations

import threading

import numpy as np

from app.utils.audio import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    AudioBuffer,
    decode_audio_delta,
    encode_pcm16_to_base64,
    float32_to_pcm16,
    pcm16_to_float32,
)


class TestAudioBuffer:
    """Test the thread-safe AudioBuffer."""

    def test_append_and_read_all(self):
        buf = AudioBuffer()
        buf.append(b"\x00\x01")
        buf.append(b"\x02\x03")
        data = buf.read_all()
        assert data == b"\x00\x01\x02\x03"

    def test_read_all_drains_buffer(self):
        buf = AudioBuffer()
        buf.append(b"\x00\x01")
        buf.read_all()
        assert buf.is_empty
        assert buf.read_all() == b""

    def test_clear(self):
        buf = AudioBuffer()
        buf.append(b"\x00" * 100)
        buf.clear()
        assert buf.is_empty
        assert buf.depth_bytes == 0

    def test_depth_bytes(self):
        buf = AudioBuffer()
        buf.append(b"\x00" * 48)
        assert buf.depth_bytes == 48

    def test_depth_seconds(self):
        buf = AudioBuffer()
        one_second = SAMPLE_RATE * SAMPLE_WIDTH  # 24000 * 2 = 48000 bytes
        buf.append(b"\x00" * one_second)
        assert abs(buf.depth_seconds - 1.0) < 0.01

    def test_ring_buffer_eviction(self):
        max_bytes = 100
        buf = AudioBuffer(max_bytes=max_bytes)
        # Append more than max
        buf.append(b"\x01" * 60)
        buf.append(b"\x02" * 60)
        # Should have evicted the first chunk
        assert buf.depth_bytes <= max_bytes
        data = buf.read_all()
        assert b"\x02" in data

    def test_read_frames(self):
        buf = AudioBuffer()
        # 4 samples = 8 bytes (PCM16)
        buf.append(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        data = buf.read_frames(2)  # 2 frames = 4 bytes
        assert len(data) == 4
        assert data == b"\x00\x01\x02\x03"
        # Remaining 4 bytes still in buffer
        assert buf.depth_bytes == 4

    def test_read_frames_insufficient_data(self):
        buf = AudioBuffer()
        buf.append(b"\x00\x01")
        data = buf.read_frames(100)  # Request more than available
        assert data == b"\x00\x01"

    def test_thread_safety(self):
        """Concurrent producers and consumers should not corrupt data."""
        buf = AudioBuffer(max_bytes=100_000)
        errors = []

        def producer():
            try:
                for _ in range(1000):
                    buf.append(b"\xaa" * 10)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(500):
                    buf.read_all()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestPCM16Conversion:
    """Test audio format conversion utilities."""

    def test_pcm16_to_float32_round_trip(self):
        # Create known PCM16 samples
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        pcm_bytes = samples.tobytes()

        float_samples = pcm16_to_float32(pcm_bytes)
        assert float_samples.dtype == np.float32
        assert len(float_samples) == 5
        assert abs(float_samples[0]) < 0.001  # zero
        assert float_samples[1] > 0.0  # positive
        assert float_samples[2] < 0.0  # negative

    def test_float32_to_pcm16_clipping(self):
        # Values outside [-1, 1] should be clipped
        samples = np.array([2.0, -2.0], dtype=np.float32)
        pcm = float32_to_pcm16(samples)
        int_samples = np.frombuffer(pcm, dtype=np.int16)
        assert int_samples[0] == 32767
        assert int_samples[1] == -32767  # -1.0 * 32767

    def test_decode_encode_round_trip(self):
        original = b"\x00\x10\x20\x30\x40\x50"
        encoded = encode_pcm16_to_base64(original)
        decoded = decode_audio_delta(encoded)
        assert decoded == original

    def test_pcm16_to_float32_odd_bytes(self):
        # Odd number of bytes should be truncated to even
        odd_bytes = b"\x00\x01\x02"
        result = pcm16_to_float32(odd_bytes)
        assert len(result) == 1  # Only 2 bytes used


class TestCircuitBreakerResilience:
    """Integration-style test for the circuit breaker."""

    def test_circuit_breaker_states(self):
        from app.core.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # 1 < 2

        cb.record_failure()
        assert cb.state == CircuitState.OPEN  # 2 >= 2
        assert not cb.allow_request()

        # Wait for cooldown
        import time

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()

        # Success resets
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_reset(self):
        from app.core.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
