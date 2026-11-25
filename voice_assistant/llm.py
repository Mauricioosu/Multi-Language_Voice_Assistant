
"""Local multilingual conversation via Ollama's loopback HTTP API."""

import json
import logging
import urllib.error
import urllib.request

from .config import Settings

logger = logging.getLogger(__name__)


class LocalChatModel:
    SUPPORTED_LANGUAGES = {
        "pt": "Brazilian Portuguese",
        "en": "English",
    }

    SYSTEM_PROMPT = (
        "You are a helpful and concise offline voice assistant. "
        "Provide clear and accurate answers suitable for speech synthesis. "
        "You do not have internet access or real-time information. "
        "Do not claim to perform actions outside your capabilities."
    )

    def __init__(self, settings: Settings):
        self.model = settings.ollama_model
        self.url = settings.ollama_url

        self.history = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            }
        ]

    def respond(
        self,
        text: str,
        language: str = "pt",
    ) -> str | None:

        target_language = self.SUPPORTED_LANGUAGES.get(language)

        if target_language is None:
            logger.warning(
                "Unsupported language: %s",
                language,
            )
            return None

        system_message = {
            "role": "system",
            "content": (
                f"{self.SYSTEM_PROMPT} "
                f"Always respond in {target_language}, "
                "regardless of the language used in previous messages."
            ),
        }

        messages = [
            system_message,
            *self.history[1:],
            {"role": "user", "content": text},
        ]

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 180},
        }).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            logger.info(
                "🧠 Generating response in %s",
                target_language,
            )

            with urllib.request.urlopen(
                request, timeout=120
            ) as response:
                result = json.load(response)

            answer = result["message"]["content"].strip()

            if not answer:
                raise ValueError("Ollama returned an empty response.")

            self.history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ])

            self.history = [
                self.history[0]
            ] + self.history[1:][-20:]

            return answer

        except urllib.error.URLError:
            logger.exception(
                "Ollama unavailable. Start it with 'ollama serve'."
            )

        except Exception:
            logger.exception("Local language model error")

        return None
