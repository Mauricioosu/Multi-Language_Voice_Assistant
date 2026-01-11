# Multi-language Voice Assistant 🎙️🤖

Assistente de voz inteligente e multilíngue que utiliza **OpenAI Whisper** para transcrição (STT), **GPT-4o** para processamento de linguagem (LLM) e **OpenAI TTS** para síntese de voz.

## 🚀 Funcionalidades
- **Detecção Automática de Idioma:** Fale em qualquer idioma suportado pelo Whisper.
- **Memória de Contexto:** Mantém o histórico da conversa para respostas coerentes.
- **Saída de Áudio de Alta Qualidade:** Vozes naturais via modelos TTS-1.
- **Arquitetura Assíncrona:** Implementado com `asyncio` para evitar bloqueio de I/O.

## 🛠️ Stack Tecnológica
- **Linguagem:** Python 3.10+
- **APIs:** OpenAI (Whisper-1, GPT-4o, TTS-1)
- **Bibliotecas:** `python-dotenv`, `openai`, `aiohttp`

## 📋 Pré-requisitos
Antes de começar, você precisará de uma chave de API da OpenAI.
1. Obtenha sua chave em: [https://platform.openai.com/](https://platform.openai.com/)
2. Configure o arquivo `.env` (veja abaixo).

## ⚙️ Instalação e Execução

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/Mauricioosu/Multi-Language_Voice_Assistant.git](https://github.com/Mauricioosu/Multi-Language_Voice_Assistant.git)
   cd Multi-Language_Voice_Assistant
