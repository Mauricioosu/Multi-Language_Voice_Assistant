
"""Unit tests for performance monitoring."""

from unittest.mock import patch

import pytest

from voice_assistant.metrics import PerformanceMetrics


def test_initial_metrics():
    metrics = PerformanceMetrics()

    assert metrics.timings == {}
    assert metrics.total_time == 0.0
    assert metrics.response_latency == 0.0


def test_measure_execution_time():
    metrics = PerformanceMetrics()

    with patch(
        "voice_assistant.metrics.perf_counter",
        side_effect=[10.0, 12.5],
    ):
        with metrics.measure("stt"):
            pass

    assert metrics.timings["stt"] == pytest.approx(2.5)


def test_response_latency():
    metrics = PerformanceMetrics()

    metrics.timings = {
        "recording": 3.0,
        "stt": 1.0,
        "llm": 2.0,
        "tts_synthesis": 0.5,
        "playback": 4.0,
    }

    assert metrics.response_latency == pytest.approx(3.5)


def test_total_time():
    metrics = PerformanceMetrics()

    metrics.timings = {
        "recording": 3.0,
        "stt": 1.0,
        "llm": 2.0,
        "tts_synthesis": 0.5,
        "playback": 4.0,
    }

    assert metrics.total_time == pytest.approx(10.5)


def test_measure_records_time_on_exception():
    metrics = PerformanceMetrics()

    with patch(
        "voice_assistant.metrics.perf_counter",
        side_effect=[5.0, 6.0],
    ):
        with pytest.raises(RuntimeError):
            with metrics.measure("llm"):
                raise RuntimeError("Inference failed")

    assert metrics.timings["llm"] == pytest.approx(1.0)


def test_performance_report(caplog):
    metrics = PerformanceMetrics()

    metrics.timings = {
        "stt": 1.0,
        "llm": 2.0,
        "tts_synthesis": 0.5,
    }

    with caplog.at_level("INFO"):
        metrics.report("en")

    assert "PERFORMANCE REPORT [en]" in caplog.text
    assert "Response latency" in caplog.text
    assert "3.500 s" in caplog.text
    assert "Total measured time" in caplog.text
