import io
import json
import urllib.error
from unittest.mock import patch

from voice_assistant.config import Settings
from voice_assistant.llm import LocalChatModel


def test_successful_response_updates_history():
    model = LocalChatModel(Settings())
    fake_response = io.BytesIO(json.dumps({"message": {"content": " Brasília. "}}).encode())
    with patch("voice_assistant.llm.urllib.request.urlopen", return_value=fake_response) as request:
        answer = model.respond("Qual é a capital do Brasil?")
    assert answer == "Brasília."
    assert model.history[-2]["role"] == "user"
    assert model.history[-1]["content"] == "Brasília."
    assert request.call_args.kwargs["timeout"] == 120


def test_connection_error_does_not_change_history():
    model = LocalChatModel(Settings())
    with patch("voice_assistant.llm.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        assert model.respond("Olá") is None
    assert len(model.history) == 1


def test_empty_response_does_not_change_history():
    model = LocalChatModel(Settings())
    response = io.BytesIO(b'{"message": {"content": "   "}}')
    with patch("voice_assistant.llm.urllib.request.urlopen", return_value=response):
        assert model.respond("Olá") is None
    assert len(model.history) == 1
