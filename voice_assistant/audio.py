"""Microphone capture and voice activity detection."""
import collections
import logging
import wave
from pathlib import Path

import pyaudio
import webrtcvad

from .config import Settings

logger = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vad = webrtcvad.Vad(2)

    def record(self, filename: str | Path = "input.wav") -> bool:
        s = self.settings
        audio = pyaudio.PyAudio()
        stream = None
        frames = []
        pre_speech = collections.deque(maxlen=10)
        silence_frames = 0
        wait_frames = 0
        triggered = False
        try:
            stream = audio.open(
                format=pyaudio.paInt16, channels=1, rate=s.rate,
                input=True, frames_per_buffer=s.chunk,
            )
            logger.info("🎙️ Aguardando fala... (Ctrl+C para encerrar)")
            while True:
                frame = stream.read(s.chunk, exception_on_overflow=False)
                if len(frame) != s.chunk * 2:
                    continue
                is_speech = self.vad.is_speech(frame, s.rate)
                if not triggered:
                    pre_speech.append(frame)
                    wait_frames += 1
                    if is_speech:
                        triggered = True
                        frames.extend(pre_speech)
                        pre_speech.clear()
                        logger.info("⚡ Voz detectada!")
                    elif wait_frames >= s.wait_for_speech_seconds * 1000 // s.chunk_duration_ms:
                        logger.info("⏳ Nenhuma fala detectada.")
                        return False
                else:
                    frames.append(frame)
                    silence_frames = 0 if is_speech else silence_frames + 1
                    if silence_frames >= s.silence_duration_ms // s.chunk_duration_ms:
                        logger.info("✅ Fim da fala.")
                        break
                    if len(frames) >= s.max_record_seconds * 1000 // s.chunk_duration_ms:
                        logger.info("⏱️ Limite de gravação atingido.")
                        break
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
            audio.terminate()

        if len(frames) < 15:
            return False
        with wave.open(str(filename), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(s.rate)
            wav_file.writeframes(b"".join(frames))
        return True
