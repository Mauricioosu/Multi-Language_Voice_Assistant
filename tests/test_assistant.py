
"""Integration tests for the multilingual assistant."""

import asyncio
from unittest.mock import patch

import pytest

from voice_assistant.assistant import VoiceAssistant
from voice_assistant.config import Settings


@pytest.mark.parametrize(
    "language,question,answer",
    [
        ("pt", "Como você está?", "Estou bem."),
        ("en", "How are you?", "I am fine."),
    ],
)
def test_multilingual_pipeline(
    language,
    question,
    answer,
):
    with (
        patch(
            "voice_assistant.assistant.AudioRecorder"
        ) as recorder,
        patch(
            "voice_assistant.assistant.SpeechRecognizer"
        ) as recognizer,
        patch(
            "voice_assistant.assistant.LocalChatModel"
        ) as chat,
        patch(
            "voice_assistant.assistant.SpeechSynthesizer"
        ) as synthesizer,
        patch(
            "voice_assistant.assistant.asyncio.sleep",
            side_effect=asyncio.CancelledError,
        ),
    ):
        recorder.return_value.record.return_value = True

        recognizer.return_value.transcribe.return_value = (
            question
        )

        recognizer.return_value.detected_language = (
            language
        )

        chat.return_value.SUPPORTED_LANGUAGES = {
            "pt": "Brazilian Portuguese",
            "en": "English",
        }

        chat.return_value.respond.return_value = answer

        assistant = VoiceAssistant(Settings())

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(assistant.run())

        recorder.return_value.record.assert_called_once_with(
            "input.wav"
        )

        recognizer.return_value.transcribe.assert_called_once_with(
            "input.wav"
        )

        chat.return_value.respond.assert_called_once_with(
            question,
            language=language,
        )

        synthesizer.return_value.speak.assert_called_once_with(
            answer,
            language=language,
        )

        synthesizer.return_value.metrics.timings = {
            "tts_synthesis": 0.5,
            "playback": 1.0,
        }


def test_unsupported_language_skips_response():
    with (
        patch(
            "voice_assistant.assistant.AudioRecorder"
        ) as recorder,
        patch(
            "voice_assistant.assistant.SpeechRecognizer"
        ) as recognizer,
        patch(
            "voice_assistant.assistant.LocalChatModel"
        ) as chat,
        patch(
            "voice_assistant.assistant.SpeechSynthesizer"
        ) as synthesizer,
        patch(
            "voice_assistant.assistant.asyncio.sleep",
            side_effect=asyncio.CancelledError,
        ),
    ):
        recorder.return_value.record.return_value = True

        recognizer.return_value.transcribe.return_value = (
            "Hola, necesito ayuda."
        )

        recognizer.return_value.detected_language = "es"

        chat.return_value.SUPPORTED_LANGUAGES = {
            "pt": "Brazilian Portuguese",
            "en": "English",
        }

        assistant = VoiceAssistant(Settings())

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(assistant.run())

        chat.return_value.respond.assert_not_called()
        synthesizer.return_value.speak.assert_not_called()
