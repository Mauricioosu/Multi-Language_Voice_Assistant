
"""Local multilingual speech-to-text using Faster-Whisper."""

import logging
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from .config import Settings

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    def __init__(self, settings: Settings):
        self.rate = settings.rate
        self.detected_language: str | None = None
        self.language_probability: float = 0.0

        self.model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

    def transcribe(self, filename: str | Path) -> str | None:
        self.detected_language = None
        self.language_probability = 0.0

        try:
            with wave.open(str(filename), "rb") as wav_file:
                fmt = (
                    wav_file.getnchannels(),
                    wav_file.getframerate(),
                    wav_file.getsampwidth(),
                )

                if (
                    wav_file.getcomptype() != "NONE"
                    or fmt != (1, self.rate, 2)
                ):
                    raise ValueError(
                        "Esperado WAV PCM mono 16 kHz, 16 bits."
                    )

                raw = wav_file.readframes(
                    wav_file.getnframes()
                )

            if not raw:
                return None

            samples = (
                np.frombuffer(raw, dtype="<i2")
                .astype(np.float32) / 32768.0
            )

            segments, info = self.model.transcribe(
                samples,
                language=None,
                beam_size=5,
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            if not text:
                return None

            self.detected_language = info.language
            self.language_probability = info.language_probability

            logger.info(
                "🌐 Detected language: %s (%.2f%%)",
                self.detected_language,
                self.language_probability * 100,
            )

            logger.info("📝 Transcription: %s", text)

            return text

        except Exception:
            logger.exception("Transcription error")
            return None
