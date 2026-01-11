# Multi-Language Voice Assistant

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Debian%20%7C%20Ubuntu-orange.svg)
![Status](https://img.shields.io/badge/status-v1.0.0--Hybrid-brightgreen.svg)

Este projeto foi desenvolvido como o desafio final do curso da **Digital Innovation One (DIO)**. Trata-se de um assistente de voz inteligente que combina processamento local de alta performance com inteligência em nuvem.

> **Marcos do Projeto:** Esta **v1.0.0** foca na estabilidade do pipeline de áudio e na integração híbrida (STT Local + LLM Cloud).

---

## Fluxo de Funcionamento
[Som ambiente] -> **VAD** (Filtro de voz) -> **Faster-Whisper** (Transcrição Local) -> **GPT-4o** (Inteligência) -> **gTTS** (Sintetizador) -> [Resposta em Áudio]



---

## Diferenciais de Engenharia

- **VAD (Voice Activity Detection):** Implementação de `webrtcvad` nível 3 para garantir que o sistema só processe fala humana, economizando recursos e evitando falsos positivos.
- **Infrerência Local (STT):** Utilização do `faster-whisper` com quantização `int8`, permitindo transcrição rápida mesmo em CPUs domésticas.
- **Tratamento de Exceções Proativo:** Se o saldo da API acabar ou a chave estiver ausente, o assistente utiliza o próprio motor de voz para orientar o usuário sobre como proceder.
- **Estabilidade ALSA/PyAudio:** Gerenciamento rigoroso de buffers e streams para evitar travamentos comuns em sistemas Linux.

---

## Stack Tecnológica

| Camada | Tecnologia | Implementação |
| :--- | :--- | :--- |
| **STT** | Faster-Whisper | Local (CPU) |
| **LLM** | OpenAI GPT-4o | Cloud (API) |
| **TTS** | gTTS | Cloud (API) |
| **VAD** | WebRTCVAD | Local (Real-time) |
| **Audio** | PyAudio / mpg123 | Local (System) |

---

## Instalação e Execução

### 1. Dependências do Sistema
Prepare as bibliotecas de áudio essenciais:
```bash
sudo apt update && sudo apt install portaudio19-dev python3-dev mpg123 -
```
### 2. Preparação do Ambiente Virtual

# Clone e acesse o diretório

```bash
git clone https://github.com/Mauricioosu/Multi-Language_Voice_Assistant.git
cd Multi-Language_Voice_Assistant
```

### 3. Configuração de Variáveis (API Key)
Crie um arquivo .env na raiz do projeto:

```bash
OPENAI_API_KEY=sua_chave_aqui
```

### 4. Execução do Projeto
```bash
python main.py
```

Roadmap de Evolução

- [x] v1.0.0: Versão Híbrida Estável (VAD + Faster-Whisper + gTTS).
 
  [ ] v1.1.0: Integração com Ollama (Llama 3) para LLM 100% Offline.

  [ ] v1.2.0: Migração para Piper TTS (Voz local ultrarrápida).

  [ ] v2.0.0: Suporte Cross-platform nativo (Windows/macOS).

---

Desenvolvido por Mauricio Rafael de Souza Osuna

---
