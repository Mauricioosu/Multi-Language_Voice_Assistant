
"""Unit tests for assistant configuration."""

import pytest

from voice_assistant.config import Settings


@pytest.mark.parametrize(
    "rate,duration,expected",
    [
        (16000, 30, 480),
        (16000, 20, 320),
        (8000, 30, 240),
    ],
)
def test_chunk_calculation(rate, duration, expected):
    settings = Settings(
        rate=rate,
        chunk_duration_ms=duration,
    )

    assert settings.chunk == expected


def test_piper_language_mapping():
    settings = Settings()

    assert settings.piper_models == {
        "pt": settings.piper_model,
        "en": settings.piper_model_en,
    }
