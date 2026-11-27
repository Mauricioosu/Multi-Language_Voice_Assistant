
"""Offline multilingual speech synthesis via Piper and aplay."""

import logging
import subprocess

from .config import Settings
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    def __init__(self, settings: Settings):
        self.model_path = settings.piper_model
        self.models = settings.piper_models
        self.metrics = PerformanceMetrics()

    def speak(
        self,
        text: str,
        filename: str = "response.wav",
        *,
        language: str = "pt",
    ) -> None:

        self.metrics = PerformanceMetrics()

        if not text or not text.strip():
            return

        model_path = self.models.get(language)

        if model_path is None:
            logger.warning(
                "Unsupported TTS language: %s",
                language,
            )
            return

        if not model_path.is_file():
            logger.warning(
                "Piper model not found: %s. Text-only response.",
                model_path,
            )
            return

        config_path = model_path.with_suffix(".onnx.json")

        if not config_path.is_file():
            logger.warning(
                "Piper configuration not found: %s. "
                "Text-only response.",
                config_path,
            )
            return

        try:
            logger.info(
                "🔊 Selected voice [%s]: %s",
                language,
                model_path.name,
            )

            with self.metrics.measure("tts_synthesis"):
                subprocess.run(
                    [
                        "piper",
                        "--model",
                        str(model_path),
                        "--output_file",
                        filename,
                    ],
                    input=text,
                    text=True,
                    check=True,
                    timeout=90,
                    stdout=subprocess.DEVNULL,
                )

            with self.metrics.measure("playback"):
                subprocess.run(
                    ["aplay", "-q", filename],
                    check=True,
                    timeout=90,
                )

        except FileNotFoundError as error:
            logger.error(
                "Executable not found: %s. "
                "Install Piper and alsa-utils.",
                error.filename,
            )

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            logger.exception(
                "Speech synthesis or playback failed"
            )
