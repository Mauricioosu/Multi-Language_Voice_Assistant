import os
import asyncio
import wave
import pyaudio
import webrtcvad
import collections
import logging
import warnings
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class OfflineVoiceAssistant:
    def __init__(self):
        logger.info("📦 Carregando modelo Whisper local...")
        # Modelo 'base' para CPU. Se travar, mude para 'tiny'
        self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        self.vad = webrtcvad.Vad(3) # Nível 3 para maior estabilidade
        self.history = [{"role": "system", "content": "Você é um assistente offline útil."}]
        
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30
        self.CHUNK = int(self.RATE * self.CHUNK_DURATION_MS / 1000)

    def record_audio(self, filename="input.wav"):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, 
                            input=True, frames_per_buffer=self.CHUNK)
            
            logger.info("🎙️ Aguardando fala...")
            frames = []
            ring_buffer = collections.deque(maxlen=40) 
            triggered = False
            
            # Limpeza de buffer inicial
            for _ in range(0, 15):
                stream.read(self.CHUNK, exception_on_overflow=False)

            while True:
                frame = stream.read(self.CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(frame, self.RATE)

                if not triggered:
                    if is_speech:
                        triggered = True
                        logger.info("⚡ Voz detectada!")
                else:
                    frames.append(frame)
                    ring_buffer.append(is_speech)
                    if ring_buffer.count(False) > 0.9 * ring_buffer.maxlen:
                        logger.info("✅ Fim da fala.")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if len(frames) > 15:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            return True
        return False

    def transcribe_local(self, filename):
        try:
            logger.info("🔍 Processando áudio local...")
            segments, _ = self.stt_model.transcribe(filename, beam_size=5)
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Erro Whisper Local: {e}")
            return None

    def get_llm_response(self, text):
        if not self.client: return "Modo offline (sem LLM)."
        self.history.append({"role": "user", "content": text})
        response = self.client.chat.completions.create(model="gpt-4o", messages=self.history)
        answer = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer

    async def run(self):
        logger.info("🚀 Assistente Offline STT Iniciado.")
        while True:
            try:
                if self.record_audio():
                    text = self.transcribe_local("input.wav")
                    if text and len(text) > 2:
                        print(f"👤 Você: {text}")
                        response = self.get_llm_response(text)
                        print(f"🤖 Assistente: {response}")
                
                logger.info("⏳ Aguardando...")
                await asyncio.sleep(1.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                await asyncio.sleep(2)

if __name__ == "__main__":
    assistant = OfflineVoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        logger.info("Saindo...")