
"""Coordinates speech capture, transcription, response and playback."""

import asyncio
import logging

from .audio import AudioRecorder
from .config import Settings
from .llm import LocalChatModel
from .stt import SpeechRecognizer
from .tts import SpeechSynthesizer

logger = logging.getLogger(__name__)


class VoiceAssistant:
    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()

        logger.info("📦 Initializing 100% local assistant...")

        self.recorder = AudioRecorder(settings)
        self.recognizer = SpeechRecognizer(settings)
        self.chat = LocalChatModel(settings)
        self.synthesizer = SpeechSynthesizer(settings)

    async def run(self) -> None:
        logger.info("🚀 Offline assistant started!")

        while True:
            if self.recorder.record("input.wav"):
                text = self.recognizer.transcribe("input.wav")

                if text and len(text) > 2:
                    language = self.recognizer.detected_language

                    if language not in self.chat.SUPPORTED_LANGUAGES:
                        logger.warning(
                            "🌐 Unsupported language: %s",
                            language,
                        )

                        await asyncio.sleep(0.5)
                        continue

                    print(
                        f"👤 User [{language}]: {text}",
                        flush=True,
                    )

                    answer = self.chat.respond(
                        text,
                        language=language,
                    )

                    if answer:
                        print(
                            f"🤖 Assistant [{language}]: {answer}",
                            flush=True,
                        )

                        self.synthesizer.speak(
                            answer,
                            language=language,
                        )

            await asyncio.sleep(0.5)
