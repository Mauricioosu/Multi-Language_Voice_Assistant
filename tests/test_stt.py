
"""Unit tests for multilingual speech recognition."""

import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from voice_assistant.config import Settings
from voice_assistant.stt import SpeechRecognizer


def create_wav(path, rate=16000, frames=1600):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frames)

    return path


@pytest.fixture
def recognizer():
    with patch(
        "voice_assistant.stt.WhisperModel"
    ):
        yield SpeechRecognizer(Settings())


@pytest.fixture
def audio_file(tmp_path):
    return create_wav(tmp_path / "input.wav")


@pytest.mark.parametrize(
    "language,text",
    [
        ("pt", "Olá, tudo bem?"),
        ("en", "Hello, how are you?"),
    ],
)
def test_detects_language(
    recognizer,
    audio_file,
    language,
    text,
):
    segments = [
        SimpleNamespace(text=text)
    ]

    info = SimpleNamespace(
        language=language,
        language_probability=0.98,
    )

    recognizer.model.transcribe.return_value = (
        segments,
        info,
    )

    result = recognizer.transcribe(audio_file)

    assert result == text
    assert recognizer.detected_language == language
    assert recognizer.language_probability == 0.98

    call = recognizer.model.transcribe.call_args

    assert call.kwargs["language"] is None
    assert call.kwargs["beam_size"] == 5
    assert call.args[0].dtype == np.float32


def test_empty_transcription_resets_language(
    recognizer,
    audio_file,
):
    recognizer.detected_language = "pt"
    recognizer.language_probability = 0.95

    recognizer.model.transcribe.return_value = (
        [],
        SimpleNamespace(
            language="en",
            language_probability=0.90,
        ),
    )

    assert recognizer.transcribe(audio_file) is None
    assert recognizer.detected_language is None
    assert recognizer.language_probability == 0.0


def test_invalid_audio_format(recognizer, tmp_path):
    audio_file = create_wav(
        tmp_path / "invalid.wav",
        rate=8000,
    )

    assert recognizer.transcribe(audio_file) is None
    assert recognizer.detected_language is None

    recognizer.model.transcribe.assert_not_called()


def test_empty_audio(recognizer, tmp_path):
    audio_file = create_wav(
        tmp_path / "empty.wav",
        frames=0,
    )

    assert recognizer.transcribe(audio_file) is None

    recognizer.model.transcribe.assert_not_called()


def test_transcription_error_resets_language(
    recognizer,
    audio_file,
):
    recognizer.detected_language = "en"
    recognizer.language_probability = 0.99

    recognizer.model.transcribe.side_effect = (
        RuntimeError("Recognition failed")
    )

    assert recognizer.transcribe(audio_file) is None
    assert recognizer.detected_language is None
    assert recognizer.language_probability == 0.0
