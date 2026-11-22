
import asyncio
import collections
import json
import logging
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
import wave

import numpy as np
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VoiceAssistantDIO:
    RATE = 16000
    CHUNK_DURATION_MS = 30
    CHUNK = RATE * CHUNK_DURATION_MS // 1000
    SILENCE_DURATION_MS = 1200
    MAX_RECORD_SECONDS = 20
    WAIT_FOR_SPEECH_SECONDS = 15

    def __init__(self):
        logger.info("📦 Inicializando assistente 100%% local...")
        # Evita downloads automáticos: execute a preparação inicial antes.
        self.stt_model = WhisperModel(
            os.getenv("WHISPER_MODEL", "base"),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )
        self.vad = webrtcvad.Vad(2)
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.ollama_url = "http://127.0.0.1:11434/api/chat"
        self.piper_model = Path(os.getenv("PIPER_MODEL", "models/pt_BR-faber-medium.onnx"))
        self.history = [
            {"role": "system", "content": "Você é um assistente de voz útil e conciso. Responda em português brasileiro."}
        ]

    def record_audio(self, filename="input.wav"):
        audio = pyaudio.PyAudio()
        stream = None
        frames = []
        pre_speech = collections.deque(maxlen=10)
        silence_frames = 0
        wait_frames = 0
        silence_limit = self.SILENCE_DURATION_MS // self.CHUNK_DURATION_MS
        wait_limit = self.WAIT_FOR_SPEECH_SECONDS * 1000 // self.CHUNK_DURATION_MS
        record_limit = self.MAX_RECORD_SECONDS * 1000 // self.CHUNK_DURATION_MS
        triggered = False
        try:
            stream = audio.open(
                format=pyaudio.paInt16, channels=1, rate=self.RATE,
                input=True, frames_per_buffer=self.CHUNK,
            )
            logger.info("🎙️ Aguardando fala... (Ctrl+C para encerrar)")
            while True:
                frame = stream.read(self.CHUNK, exception_on_overflow=False)
                if len(frame) != self.CHUNK * 2:
                    continue
                is_speech = self.vad.is_speech(frame, self.RATE)
                if not triggered:
                    pre_speech.append(frame)
                    wait_frames += 1
                    if is_speech:
                        triggered = True
                        frames.extend(pre_speech)
                        pre_speech.clear()
                        logger.info("⚡ Voz detectada!")
                    elif wait_frames >= wait_limit:
                        logger.info("⏳ Nenhuma fala detectada.")
                        return False
                else:
                    frames.append(frame)
                    silence_frames = 0 if is_speech else silence_frames + 1
                    if silence_frames >= silence_limit:
                        logger.info("✅ Fim da fala.")
                        break
                    if len(frames) >= record_limit:
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
            wav_file.setframerate(self.RATE)
            wav_file.writeframes(b"".join(frames))
        return True

    def transcribe_local(self, filename):
        try:
            with wave.open(str(filename), "rb") as wav_file:
                fmt = (wav_file.getnchannels(), wav_file.getframerate(), wav_file.getsampwidth())
                if wav_file.getcomptype() != "NONE" or fmt != (1, self.RATE, 2):
                    raise ValueError("Esperado WAV PCM mono 16 kHz, 16 bits.")
                raw = wav_file.readframes(wav_file.getnframes())
            if not raw:
                return None
            # Array NumPy evita o erro metadata_errors do PyAV.
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            segments, _ = self.stt_model.transcribe(samples, language="pt", beam_size=5)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            logger.info("📝 Transcrição: %s", text or "(vazia)")
            return text or None
        except Exception:
            logger.exception("Erro na transcrição")
            return None

    def get_llm_response(self, text):
        """Consulta somente o Ollama local, via loopback (sem serviços externos)."""
        messages = self.history + [{"role": "user", "content": text}]
        payload = json.dumps({
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 180},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            answer = result["message"]["content"].strip()
            if not answer:
                raise ValueError("Ollama retornou resposta vazia.")
            self.history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ])
            # Limita histórico para não crescer indefinidamente.
            self.history = [self.history[0]] + self.history[1:][-20:]
            return answer
        except urllib.error.URLError:
            logger.exception("Ollama indisponível. Inicie com 'ollama serve' e baixe o modelo.")
        except Exception:
            logger.exception("Erro ao consultar o modelo local")
        return None

    def speak(self, text, filename="response.wav"):
        """Sintetiza com Piper local e reproduz via aplay (ALSA)."""
        if not text:
            return
        if not self.piper_model.is_file():
            logger.warning("Modelo Piper não encontrado: %s. Resposta apenas em texto.", self.piper_model)
            return
        try:
            subprocess.run(
                ["piper", "--model", str(self.piper_model), "--output_file", filename],
                input=text, text=True, check=True, timeout=90,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(["aplay", "-q", filename], check=True, timeout=90)
        except FileNotFoundError as error:
            logger.error("Executável ausente: %s. Instale Piper e alsa-utils.", error.filename)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("Falha na síntese/reprodução de voz")

    async def run(self):
        logger.info("🚀 Assistente offline iniciado!")
        while True:
            if self.record_audio():
                text = self.transcribe_local("input.wav")
                if text and len(text) > 2:
                    print(f"👤 Você: {text}", flush=True)
                    answer = self.get_llm_response(text)
                    if answer:
                        print(f"🤖 Assistente: {answer}", flush=True)
                        self.speak(answer)
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    try:
        assistant = VoiceAssistantDIO()
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        logger.info("Assistente encerrado (Ctrl+C).")
    except Exception:
        logger.exception("Erro inesperado")
