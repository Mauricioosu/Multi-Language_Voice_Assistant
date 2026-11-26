
"""Unit tests for microphone capture and voice activity detection."""

import wave
from unittest.mock import patch

import pytest

from voice_assistant.audio import AudioRecorder
from voice_assistant.config import Settings


@pytest.fixture
def settings():
    return Settings(
        silence_duration_ms=90,
        max_record_seconds=1,
        wait_for_speech_seconds=1,
    )


@pytest.fixture
def audio_environment(settings):
    with (
        patch(
            "voice_assistant.audio.pyaudio.PyAudio"
        ) as pyaudio_mock,
        patch(
            "voice_assistant.audio.webrtcvad.Vad"
        ) as vad_mock,
    ):
        recorder = AudioRecorder(settings)

        audio = pyaudio_mock.return_value
        stream = audio.open.return_value
        vad = vad_mock.return_value

        frame = b"\x01\x00" * settings.chunk

        stream.read.return_value = frame

        yield recorder, audio, stream, vad, frame


def test_recorder_initialization(settings):
    with patch(
        "voice_assistant.audio.webrtcvad.Vad"
    ) as vad:
        recorder = AudioRecorder(settings)

    assert recorder.settings == settings
    vad.assert_called_once_with(2)


def test_records_speech_until_silence(
    audio_environment,
    settings,
    tmp_path,
):
    recorder, audio, stream, vad, frame = audio_environment

    vad.is_speech.side_effect = (
        [False] * 3
        + [True] * 15
        + [False] * 3
    )

    output = tmp_path / "speech.wav"

    assert recorder.record(output) is True
    assert output.is_file()

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == settings.rate
        assert wav_file.getnframes() == settings.chunk * 21
        assert wav_file.readframes(
            wav_file.getnframes()
        ) == frame * 21

    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    audio.terminate.assert_called_once()


def test_timeout_without_speech(
    audio_environment,
    settings,
    tmp_path,
):
    recorder, audio, stream, vad, _ = audio_environment

    vad.is_speech.return_value = False

    output = tmp_path / "no_speech.wav"

    assert recorder.record(output) is False
    assert not output.exists()

    expected_frames = (
        settings.wait_for_speech_seconds * 1000
        // settings.chunk_duration_ms
    )

    assert stream.read.call_count == expected_frames

    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    audio.terminate.assert_called_once()


def test_maximum_recording_duration(
    audio_environment,
    settings,
    tmp_path,
):
    recorder, audio, stream, vad, _ = audio_environment

    vad.is_speech.return_value = True

    output = tmp_path / "maximum.wav"

    assert recorder.record(output) is True

    expected_frames = (
        settings.max_record_seconds * 1000
        // settings.chunk_duration_ms
    )

    assert stream.read.call_count == expected_frames

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnframes() == (
            expected_frames * settings.chunk
        )

    audio.terminate.assert_called_once()


def test_short_recording_is_discarded(
    audio_environment,
    tmp_path,
):
    recorder, audio, stream, vad, _ = audio_environment

    vad.is_speech.side_effect = [
        True,
        False,
        False,
        False,
    ]

    output = tmp_path / "short.wav"

    assert recorder.record(output) is False
    assert not output.exists()

    stream.close.assert_called_once()
    audio.terminate.assert_called_once()


def test_invalid_audio_frame_is_skipped(
    audio_environment,
    tmp_path,
):
    recorder, audio, stream, vad, frame = audio_environment

    stream.read.side_effect = (
        [b"invalid"]
        + [frame] * 18
    )

    vad.is_speech.side_effect = (
        [True] * 15
        + [False] * 3
    )

    output = tmp_path / "valid.wav"

    assert recorder.record(output) is True
    assert stream.read.call_count == 19
    assert vad.is_speech.call_count == 18


def test_stream_open_failure_releases_audio(
    audio_environment,
    tmp_path,
):
    recorder, audio, stream, vad, _ = audio_environment

    audio.open.side_effect = OSError(
        "Microphone unavailable"
    )

    with pytest.raises(OSError):
        recorder.record(tmp_path / "error.wav")

    audio.terminate.assert_called_once()
    stream.close.assert_not_called()


def test_stream_read_failure_releases_resources(
    audio_environment,
    tmp_path,
):
    recorder, audio, stream, vad, _ = audio_environment

    stream.read.side_effect = OSError(
        "Audio capture failed"
    )

    with pytest.raises(OSError):
        recorder.record(tmp_path / "error.wav")

    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    audio.terminate.assert_called_once()
