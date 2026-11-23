"""Local conversation via Ollama's loopback HTTP API."""
import json
import logging
import urllib.error
import urllib.request

from .config import Settings

logger = logging.getLogger(__name__)


class LocalChatModel:
    def __init__(self, settings: Settings):
        self.model = settings.ollama_model
        self.url = settings.ollama_url
        self.history = [
            {"role": "system", "content": "Você é um assistente de voz útil e conciso. Responda em português brasileiro."}
        ]

    def respond(self, text: str) -> str | None:
        messages = self.history + [{"role": "user", "content": text}]
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 180},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
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
            self.history = [self.history[0]] + self.history[1:][-20:]
            return answer
        except urllib.error.URLError:
            logger.exception("Ollama indisponível. Inicie com 'ollama serve' e baixe o modelo.")
        except Exception:
            logger.exception("Erro ao consultar o modelo local")
        return None
