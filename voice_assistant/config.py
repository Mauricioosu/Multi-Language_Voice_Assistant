"""Centralized configuration for the offline assistant."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    rate: int = 16000
    chunk_duration_ms: int = 30
    silence_duration_ms: int = 1200
    max_record_seconds: int = 20
    wait_for_speech_seconds: int = 15
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    piper_model: Path = Path(os.getenv("PIPER_MODEL", "models/pt_BR-faber-medium.onnx"))

    @property
    def chunk(self) -> int:
        return self.rate * self.chunk_duration_ms // 1000
