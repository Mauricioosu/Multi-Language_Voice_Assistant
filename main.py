import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class VoiceAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = [{"role": "system", "content": "Você é um assistente prestativo."}]

    async def run(self):
        print("Assistente pronto! (Pressione Ctrl+C para sair)")
        # Aqui entrará a lógica de captura de áudio que discutiremos a seguir
        pass

if __name__ == "__main__":
    assistant = VoiceAssistant()
    asyncio.run(assistant.run())