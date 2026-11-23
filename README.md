# Offline AI Voice Assistant

**A privacy-focused voice assistant powered by locally running AI models.**

A Python-based voice assistant designed to recognize speech, generate intelligent responses, and synthesize natural-sounding audio without relying on paid cloud APIs.

## Key Features

- **Offline speech recognition:** Faster-Whisper with CPU inference.
- **Voice activity detection:** WebRTC VAD identifies speech segments.
- **Local AI inference:** Ollama with the Qwen 2.5 3B language model.
- **Offline text-to-speech:** Piper TTS with a Brazilian Portuguese voice.
- **Privacy-focused architecture:** Audio and text processing run locally.
- **Linux support:** Developed and tested on Debian.

## System Architecture

```text
Microphone
    |
    v
PyAudio + WebRTC VAD
    |
    v
Faster-Whisper (Speech-to-Text)
    |
    v
Ollama + Qwen 2.5 3B (Local LLM)
    |
    v
Piper TTS (Text-to-Speech)
    |
    v
Speakers
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Speech Recognition | Faster-Whisper |
| Voice Activity Detection | WebRTC VAD |
| Language Model | Qwen 2.5 3B |
| Local Model Runtime | Ollama |
| Speech Synthesis | Piper TTS |
| Audio Processing | PyAudio |
| Operating System | Debian Linux |

## Project Status

The core voice interaction pipeline has been successfully tested on Debian Linux.

**Completed:** Speech capture, voice activity detection, speech transcription, local response generation, and spoken output.

**Planned:** Automated testing, modular architecture, performance benchmarks, and multilingual voice interaction.

## Future Improvements

- [X] Refactor the application into independent modules.
- [X] Implement automated tests using pytest.
- [ ] Add GitHub Actions for continuous integration.
- [ ] Optimize inference latency and audio device management.
- [ ] Implement end-to-end multilingual support.
- [ ] Publish performance benchmarks and a demonstration video.

Developed por Mauricio Rafael de Souza Osuna
