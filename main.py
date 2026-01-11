import os
import asyncio
import wave
import pyaudio
import webrtcvad
import collections
import logging
import warnings
from gtts import gTTS
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

# Silenciar avisos e configurar logs
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class VoiceAssistantDIO:
    def __init__(self):
        logger.info("📦 Inicializando v1.0.0 - Desafio DIO...")
        
        # 1. Motor STT Local (Faster-Whisper)
        self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # 2. Configuração do Cliente OpenAI
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY não encontrada no arquivo .env")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # 3. Componentes de Áudio
        self.vad = webrtcvad.Vad(3)
        self.history = [{"role": "system", "content": "Você é um assistente útil e conciso."}]
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30
        self.CHUNK = int(self.RATE * self.CHUNK_DURATION_MS / 1000)

    def record_audio(self, filename="input.wav"):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, 
                            input=True, frames_per_buffer=self.CHUNK)
            logger.info("🎙️ Aguardando fala...")
            frames, ring_buffer, triggered = [], collections.deque(maxlen=40), False
            
            for _ in range(0, 15): stream.read(self.CHUNK, exception_on_overflow=False)

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
        """Passo 2 do desafio: Whisper Local"""
        try:
            segments, _ = self.stt_model.transcribe(filename, beam_size=5)
            return " ".join([s.text for s in segments]).strip()
        except Exception as e:
            logger.error(f"Erro Whisper: {e}")
            return None

    def get_llm_response(self, text):
        """Passo 3 do desafio: ChatGPT com mensagens de erro customizadas"""
        # Caso 1: API Key ausente no .env
        if not self.api_key:
            return "Modo offline ativo. API KEY não encontrada. Por favor, adicione sua chave no arquivo .env ou use a versão Ollama."
        
        self.history.append({"role": "user", "content": text})
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=self.history,
                timeout=15
            )
            answer = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            error_msg = str(e).lower()
            # Caso 2: Falta de saldo/cota
            if "insufficient_quota" in error_msg or "429" in error_msg:
                return "Modo offline ativo. Seu saldo é insuficiente. Cheque sua carteira na OpenAI ou instale a versão offline com Ollama."
            return f"Houve um erro técnico: {e}"

    def speak(self, text):
        """Passo 4 do desafio: gTTS Output"""
        try:
            logger.info("🔊 Gerando voz...")
            tts = gTTS(text=text, lang='pt')
            filename = "response.mp3"
            tts.save(filename)
            os.system(f"mpg123 -q {filename}")
        except Exception as e:
            logger.error(f"Erro no TTS: {e}")

    async def run(self):
        logger.info("🚀 Assistente v1.0.0 Iniciado!")
        while True:
            try:
                if self.record_audio():
                    text = self.transcribe_local("input.wav")
                    if text and len(text) > 2:
                        print(f"👤 Você: {text}")
                        response = self.get_llm_response(text)
                        print(f"🤖 Assistente: {response}")
                        self.speak(response)
                
                logger.info("⏳ Aguardando próxima interação...")
                await asyncio.sleep(1.5)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    assistant = VoiceAssistantDIO()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        logger.info("Encerrando...")