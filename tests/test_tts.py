from pathlib import Path
from unittest.mock import patch

from voice_assistant.config import Settings
from voice_assistant.tts import SpeechSynthesizer


def test_empty_text_does_not_run_commands(tmp_path):
    synthesizer = SpeechSynthesizer(Settings(piper_model=tmp_path / "voice.onnx"))
    with patch("voice_assistant.tts.subprocess.run") as run:
        synthesizer.speak("")
    run.assert_not_called()


def test_missing_model_does_not_run_commands(tmp_path):
    synthesizer = SpeechSynthesizer(Settings(piper_model=tmp_path / "missing.onnx"))
    with patch("voice_assistant.tts.subprocess.run") as run:
        synthesizer.speak("Olá")
    run.assert_not_called()


def test_synthesis_and_playback_commands(tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    synthesizer = SpeechSynthesizer(Settings(piper_model=model_path))
    with patch("voice_assistant.tts.subprocess.run") as run:
        synthesizer.speak("Olá", filename="test.wav")
    assert run.call_count == 2
    assert run.call_args_list[0].args[0] == ["piper", "--model", str(model_path), "--output_file", "test.wav"]
    assert run.call_args_list[0].kwargs["input"] == "Olá"
    assert run.call_args_list[1].args[0] == ["aplay", "-q", "test.wav"]
