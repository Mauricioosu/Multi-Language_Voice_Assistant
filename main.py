"""Entrypoint for the offline voice assistant."""
import asyncio
import logging

from voice_assistant.assistant import VoiceAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        asyncio.run(VoiceAssistant().run())
    except KeyboardInterrupt:
        logger.info("Assistente encerrado (Ctrl+C).")
    except Exception:
        logger.exception("Erro inesperado")
        raise


if __name__ == "__main__":
    main()
