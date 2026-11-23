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
        logger.info("📦 Inicializando assistente 100%% local...")
        self.recorder = AudioRecorder(settings)
        self.recognizer = SpeechRecognizer(settings)
        self.chat = LocalChatModel(settings)
        self.synthesizer = SpeechSynthesizer(settings)

    async def run(self) -> None:
        logger.info("🚀 Assistente offline iniciado!")
        while True:
            if self.recorder.record("input.wav"):
                text = self.recognizer.transcribe("input.wav")
                if text and len(text) > 2:
                    print(f"👤 Você: {text}", flush=True)
                    answer = self.chat.respond(text)
                    if answer:
                        print(f"🤖 Assistente: {answer}", flush=True)
                        self.synthesizer.speak(answer)
            await asyncio.sleep(0.5)
