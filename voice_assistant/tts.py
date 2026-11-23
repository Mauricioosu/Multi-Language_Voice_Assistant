"""Offline speech synthesis and playback via Piper and aplay."""
import logging
import subprocess

from .config import Settings

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    def __init__(self, settings: Settings):
        self.model_path = settings.piper_model

    def speak(self, text: str, filename: str = "response.wav") -> None:
        if not text:
            return
        if not self.model_path.is_file():
            logger.warning("Modelo Piper não encontrado: %s. Resposta apenas em texto.", self.model_path)
            return
        try:
            subprocess.run(
                ["piper", "--model", str(self.model_path), "--output_file", filename],
                input=text, text=True, check=True, timeout=90,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(["aplay", "-q", filename], check=True, timeout=90)
        except FileNotFoundError as error:
            logger.error("Executável ausente: %s. Instale Piper e alsa-utils.", error.filename)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("Falha na síntese/reprodução de voz")
