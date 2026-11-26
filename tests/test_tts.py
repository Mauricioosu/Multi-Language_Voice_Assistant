
"""Unit tests for multilingual speech synthesis."""

from unittest.mock import patch

import pytest
import subprocess

from voice_assistant.config import Settings
from voice_assistant.tts import SpeechSynthesizer


def create_fake_model(path):
    path.touch()
    path.with_suffix(".onnx.json").touch()
    return path


@pytest.fixture
def voice_models(tmp_path):
    return {
        "pt": create_fake_model(
            tmp_path / "pt_BR-faber-medium.onnx"
        ),
        "en": create_fake_model(
            tmp_path / "en_US-lessac-medium.onnx"
        ),
    }


@pytest.fixture
def synthesizer(voice_models):
    return SpeechSynthesizer(
        Settings(
            piper_model=voice_models["pt"],
            piper_model_en=voice_models["en"],
        )
    )


def test_empty_text_does_not_run_commands(synthesizer):
    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:
        synthesizer.speak("")

    run.assert_not_called()


def test_missing_model_does_not_run_commands(tmp_path):
    synthesizer = SpeechSynthesizer(
        Settings(
            piper_model=tmp_path / "missing.onnx"
        )
    )

    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:
        synthesizer.speak("Olá")

    run.assert_not_called()


def test_synthesis_and_playback_commands(synthesizer):
    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:

        synthesizer.speak(
            "Olá",
            filename="test.wav",
        )

    assert run.call_count == 2

    piper_call = run.call_args_list[0]
    playback_call = run.call_args_list[1]

    assert piper_call.args[0][0] == "piper"
    assert piper_call.kwargs["input"] == "Olá"

    assert playback_call.args[0] == [
        "aplay", "-q", "test.wav"
    ]


@pytest.mark.parametrize(
    "language,text",
    [
        ("pt", "Olá"),
        ("en", "Hello"),
    ],
)
def test_selects_correct_voice(
    synthesizer,
    voice_models,
    language,
    text,
):
    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:

        synthesizer.speak(
            text,
            filename="test.wav",
            language=language,
        )

    assert run.call_count == 2

    command = run.call_args_list[0].args[0]

    assert command == [
        "piper",
        "--model",
        str(voice_models[language]),
        "--output_file",
        "test.wav",
    ]


def test_unsupported_language(synthesizer):
    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:

        synthesizer.speak(
            "Hola",
            language="es",
        )

    run.assert_not_called()


def test_missing_model_configuration(tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.touch()

    synthesizer = SpeechSynthesizer(
        Settings(piper_model=model_path)
    )

    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:

        synthesizer.speak("Olá")

    run.assert_not_called()


def test_switches_voice_between_languages(
    synthesizer,
    voice_models,
):
    with patch(
        "voice_assistant.tts.subprocess.run"
    ) as run:

        synthesizer.speak("Olá", language="pt")
        synthesizer.speak("Hello", language="en")
        synthesizer.speak("Obrigado", language="pt")

    assert run.call_count == 6

    synthesis_calls = run.call_args_list[::2]

    selected_models = [
        call.args[0][2]
        for call in synthesis_calls
    ]

    assert selected_models == [
        str(voice_models["pt"]),
        str(voice_models["en"]),
        str(voice_models["pt"]),
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(
            2, "Executable not found", "piper"
        ),
        subprocess.CalledProcessError(
            1, ["piper"]
        ),
        subprocess.TimeoutExpired(
            ["piper"], 90
        ),
    ],
)
def test_synthesis_failure(
    synthesizer,
    error,
    caplog,
):
    with patch(
        "voice_assistant.tts.subprocess.run",
        side_effect=error,
    ) as run:

        synthesizer.speak("Hello", language="en")

    assert run.call_count == 1
    assert caplog.records
    assert any(
        record.levelname == "ERROR"
        for record in caplog.records
    )


def test_playback_failure(synthesizer, caplog):
    with patch(
        "voice_assistant.tts.subprocess.run",
        side_effect=[
            None,
            subprocess.CalledProcessError(
                1, ["aplay"]
            ),
        ],
    ) as run:

        synthesizer.speak("Hello", language="en")

    assert run.call_count == 2
    assert any(
        record.levelname == "ERROR"
        for record in caplog.records
    )
