import os
import asyncio
import wave
import pyaudio
import webrtcvad
import collections
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Configuração de Logging Profissional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class VoiceAssistant:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY não encontrada.")
            raise ValueError("Configure a chave no arquivo .env")
        
        self.client = OpenAI(api_key=self.api_key)
        self.vad = webrtcvad.Vad(2)
        self.history = [{"role": "system", "content": "Você é um assistente conciso e útil."}]
        
        # Configurações de Áudio
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30
        self.CHUNK = int(self.RATE * self.CHUNK_DURATION_MS / 1000)

    def record_audio(self, filename="input.wav"):
        """Captura áudio com VAD e garante fechamento de recursos."""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, 
                            input=True, frames_per_buffer=self.CHUNK)
            
            logger.info("🎙️ Aguardando fala...")
            frames = []
            ring_buffer = collections.deque(maxlen=40) 
            triggered = False

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
                    if ring_buffer.count(False) > 0.8 * ring_buffer.maxlen:
                        logger.info("✅ Fim da fala.")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))

    async def get_transcription(self, filename):
        """Encapsula a chamada ao Whisper com tratamento de erro específico."""
        try:
            with open(filename, "rb") as f:
                return self.client.audio.transcriptions.create(model="whisper-1", file=f).text
        except Exception as e:
            logger.error(f"Falha na transcrição: {e}")
            return None

    async def get_chat_response(self, text):
        """Encapsula a lógica do LLM."""
        self.history.append({"role": "user", "content": text})
        try:
            response = self.client.chat.completions.create(model="gpt-4o", messages=self.history)
            answer = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            logger.error(f"Falha no GPT-4o: {e}")
            return "Desculpe, tive um problema ao processar sua resposta."

    async def run(self):
        logger.info("🚀 Assistente Online Iniciado.")
        while True:
            try:
                self.record_audio()
                text = await self.get_transcription("input.wav")
                
                if text and len(text.strip()) > 1:
                    print(f"👤 Você: {text}")
                    response = await self.get_chat_response(text)
                    print(f"🤖 Assistente: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Erro inesperado no loop: {e}")
                await asyncio.sleep(1)