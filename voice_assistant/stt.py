"""Local speech-to-text using Faster-Whisper."""
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
        self.model = WhisperModel(
            settings.whisper_model, device="cpu", compute_type="int8",
            local_files_only=True,
        )

    def transcribe(self, filename: str | Path) -> str | None:
        try:
            with wave.open(str(filename), "rb") as wav_file:
                fmt = (wav_file.getnchannels(), wav_file.getframerate(), wav_file.getsampwidth())
                if wav_file.getcomptype() != "NONE" or fmt != (1, self.rate, 2):
                    raise ValueError("Esperado WAV PCM mono 16 kHz, 16 bits.")
                raw = wav_file.readframes(wav_file.getnframes())
            if not raw:
                return None
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(samples, language="pt", beam_size=5)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            logger.info("📝 Transcrição: %s", text or "(vazia)")
            return text or None
        except Exception:
            logger.exception("Erro na transcrição")
            return None
