
"""Coordinates speech capture, transcription, response and playback."""

import asyncio
import logging

from .audio import AudioRecorder
from .config import Settings
from .llm import LocalChatModel
from .stt import SpeechRecognizer
from .tts import SpeechSynthesizer
from .metrics import PerformanceMetrics

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
            metrics = PerformanceMetrics()

            with metrics.measure("recording"):
                recorded = self.recorder.record("input.wav")

            if recorded:

                with metrics.measure("stt"):
                    text = self.recognizer.transcribe(
                        "input.wav"
                    )

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

                    with metrics.measure("llm"):
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

                        metrics.timings.update(
                            self.synthesizer.metrics.timings
                        )

                        metrics.report(language)

            await asyncio.sleep(0.5)
