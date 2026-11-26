
"""Unit tests for the local multilingual language model."""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from voice_assistant.config import Settings
from voice_assistant.llm import LocalChatModel


def fake_response(text):
    return io.BytesIO(
        json.dumps(
            {"message": {"content": text}}
        ).encode()
    )


def test_successful_response_updates_history():
    model = LocalChatModel(Settings())

    with patch(
        "voice_assistant.llm.urllib.request.urlopen",
        return_value=fake_response(" Brasília. "),
    ) as request:
        answer = model.respond(
            "Qual é a capital do Brasil?"
        )

    assert answer == "Brasília."
    assert model.history[-2]["role"] == "user"
    assert model.history[-1]["content"] == "Brasília."
    assert request.call_args.kwargs["timeout"] == 120


def test_connection_error_does_not_change_history():
    model = LocalChatModel(Settings())

    with patch(
        "voice_assistant.llm.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        assert model.respond("Olá") is None

    assert len(model.history) == 1


def test_empty_response_does_not_change_history():
    model = LocalChatModel(Settings())

    with patch(
        "voice_assistant.llm.urllib.request.urlopen",
        return_value=fake_response("   "),
    ):
        assert model.respond("Olá") is None

    assert len(model.history) == 1


@pytest.mark.parametrize(
    "language,target_language,question,answer",
    [
        (
            "pt",
            "Brazilian Portuguese",
            "Como você está?",
            "Estou bem.",
        ),
        (
            "en",
            "English",
            "How are you?",
            "I am fine.",
        ),
    ],
)
def test_language_specific_system_prompt(
    language,
    target_language,
    question,
    answer,
):
    model = LocalChatModel(Settings())

    with patch(
        "voice_assistant.llm.urllib.request.urlopen",
        return_value=fake_response(answer),
    ) as request:

        result = model.respond(
            question,
            language=language,
        )

    payload = json.loads(
        request.call_args.args[0].data
    )

    system_message = payload["messages"][0]

    assert result == answer
    assert system_message["role"] == "system"
    assert (
        f"Always respond in {target_language}"
        in system_message["content"]
    )


def test_unsupported_language_does_not_send_request():
    model = LocalChatModel(Settings())

    with patch(
        "voice_assistant.llm.urllib.request.urlopen"
    ) as request:

        result = model.respond(
            "Hola",
            language="es",
        )

    assert result is None
    assert len(model.history) == 1
    request.assert_not_called()


def test_language_switch_preserves_history():
    model = LocalChatModel(Settings())

    responses = [
        fake_response("Olá!"),
        fake_response("Hello!"),
        fake_response("Tudo bem!"),
    ]

    with patch(
        "voice_assistant.llm.urllib.request.urlopen",
        side_effect=responses,
    ) as request:

        model.respond("Oi", language="pt")
        model.respond("Hi", language="en")
        model.respond("Tudo bem?", language="pt")

    expected_languages = [
        "Brazilian Portuguese",
        "English",
        "Brazilian Portuguese",
    ]

    assert request.call_count == 3

    for call, language in zip(
        request.call_args_list,
        expected_languages,
    ):
        payload = json.loads(call.args[0].data)

        assert (
            f"Always respond in {language}"
            in payload["messages"][0]["content"]
        )

    assert len(model.history) == 7
    assert model.history[-1]["content"] == "Tudo bem!"
