# protoVoice

![protoVoice](https://i.ibb.co/ynptrxys/Screenshot-2026-03-27-at-5-17-20-PM.png)

Sub-200ms real-time voice agent. Speak and get spoken responses faster than human conversational turn-taking.

```
Mic → [Silero VAD] → [Whisper Turbo] → [Qwen 4B] → [Kokoro TTS] → Speaker
        ~1ms             ~55ms            ~150ms        ~50ms
```

**165ms time-to-first-audio. 210ms end-to-end. Zero cold start.**

## Quick Start

```bash
# Docker (single GPU, downloads ~12GB of models on first run)
docker compose up -d

# Or native
pip install -e .
python app.py
```

UI at `http://localhost:7866`. For remote access with mic (HTTPS required), use a reverse proxy or `tailscale funnel 7866`.

## How It Works

1. **Silero VAD** detects when you stop speaking (~1ms, CPU)
2. **Whisper large-v3-turbo** transcribes your speech (~55ms on GPU)
3. **Qwen3.5-4B** streams a response token-by-token (~150ms to first clause)
4. **Sentence chunker** detects boundaries in the token stream
5. **Kokoro 82M** synthesizes each chunk immediately (~50ms per chunk)
6. **Audio plays** before the LLM finishes generating

All models pre-warmed on startup (~5s boot). No cold start penalty.

## Benchmarks

Measured on NVIDIA RTX PRO 6000 Blackwell (96GB):

| Metric | Value |
|--------|-------|
| **Time to first audio (TTFA)** | **165ms avg** (150-180ms) |
| **Total end-to-end** | **210ms avg** (190-230ms) |
| STT (Whisper large-v3-turbo) | 55ms |
| LLM (Qwen3.5-4B, streaming) | 150ms |
| TTS (Kokoro 82M, chunked) | 50ms/chunk |
| Cold start (first turn) | 0ms (pre-warmed) |

165ms TTFA is faster than human conversational turn-taking (~300ms).

## Features

- **Streaming pipeline**: LLM tokens stream through a sentence chunker to TTS — audio plays while the LLM is still generating
- **Interruption**: Start speaking mid-response and it stops, listens, responds to the new input
- **Context memory**: Sliding window of 10 turns with automatic summarization of older context
- **Modes**: Chat, Transcribe, Agent (web search + calculator), Wake Word, and custom skills loaded from `.proto/skills/*.md`
- **Settings sidebar**: Collapsible right-hand drawer for mode, voice, VAD, and LLM settings
- **Voice-safe prompts**: All system prompts (including skills) enforce spoken output — no markdown, emojis, or formatting reaches TTS
- **Self-contained**: Built-in vLLM server for the LLM, or connect to an external one
- **Auth**: Optional login protection via `GRADIO_AUTH`

## Modes

Select a mode from the Settings sidebar.

| Mode | Description |
|------|-------------|
| **Chat** | Default conversational assistant with sliding-window conversation memory |
| **Transcribe** | STT only — speech is transcribed and logged, no LLM or TTS |
| **Agent** | ReAct loop with tools: web search (DuckDuckGo), calculator, and current date/time |
| **Wake Word** | Idles until a trigger phrase is detected, then processes the remainder as Chat |
| **skill:*** | Custom persona — auto-loaded from `.proto/skills/*.md` (see below) |

### Wake Word

Set the trigger phrase in the Settings sidebar (e.g. `Hey Proto`). Everything spoken after the phrase is sent to the LLM; speech without the trigger is ignored.

### Agent

The ReAct loop calls tools as needed before giving a final spoken response. It announces what it's doing ("Let me search for that.") so the conversation doesn't feel silent during tool calls. Maximum 5 iterations per turn.

## Custom Skills

Skills let you create specialized voice personas without touching code. Drop a `.md` file in `.proto/skills/` and it appears as a mode in the UI.

```
.proto/
  skills/
    chef.md          ← becomes "Chef" in the mode dropdown
    spanish_tutor.md ← becomes "Spanish Tutor"
    _README.md       ← ignored (underscore prefix)
```

### Skill file format

```markdown
---
name: Chef
slug: chef
description: A culinary expert for recipes and techniques
voice: af_bella
lang: a
max_tokens: 250
temperature: 0.8
llm_url: null          # override LLM endpoint (default: env LLM_URL)
model: null            # override model name (default: env LLM_SERVED_NAME)
---
You are a warm, knowledgeable chef assistant. Help with recipes, techniques,
and meal planning. Keep responses to 1-3 spoken sentences — no lists or markdown.
```

The body becomes the system prompt. The voice preamble (no markdown, no emojis, spoken sentences) is prepended automatically — you don't need to repeat it.

### Frontmatter fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | filename (title-cased) | Display name in the mode dropdown |
| `slug` | filename stem | URL-safe identifier |
| `description` | `""` | Short description (unused in UI, for your reference) |
| `voice` | `$KOKORO_VOICE` | Kokoro voice name |
| `lang` | `$KOKORO_LANG` | Language code: `a`=American, `b`=British, `j`=Japanese |
| `max_tokens` | `200` | Max LLM output tokens |
| `temperature` | `0.7` | LLM sampling temperature (0.0–1.0) |
| `llm_url` | `$LLM_URL` | Override the LLM endpoint for this skill |
| `model` | `$LLM_SERVED_NAME` | Override the model name for this skill |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7866` | Web UI port |
| `LLM_MODEL` | `Qwen/Qwen3.5-4B` | LLM model (any vLLM-compatible) |
| `WHISPER_MODEL` | `openai/whisper-large-v3-turbo` | STT model |
| `KOKORO_VOICE` | `af_heart` | TTS voice ([54 options](https://huggingface.co/hexgrad/Kokoro-82M)) |
| `KOKORO_LANG` | `a` | Language (`a`=American, `b`=British, `j`=Japanese, etc.) |
| `SYSTEM_PROMPT` | (conversational assistant) | LLM system prompt |
| `GRADIO_AUTH` | (none) | Login auth, format: `user:pass,user2:pass2` |
| `START_VLLM` | `1` | Set `0` to use external LLM |
| `LLM_URL` | `http://localhost:8100/v1` | External LLM endpoint (when `START_VLLM=0`) |
| `HF_HOME` | `/models` | HuggingFace cache directory |
| `NVIDIA_VISIBLE_DEVICES` | `0` | GPU to use |

## GPU Memory Budget (single GPU)

| Component | VRAM |
|-----------|:----:|
| Whisper large-v3-turbo | ~6 GB |
| Qwen3.5-4B (vLLM, 40% util) | ~15 GB |
| Kokoro 82M | ~2 GB |
| **Total** | **~23 GB** |

Fits on any GPU with 24GB+ VRAM. On larger GPUs, increase `--gpu-memory-utilization` for more KV cache (longer conversations, higher concurrency).

## Using an External LLM

To use a larger/faster LLM running elsewhere:

```bash
START_VLLM=0 LLM_URL=http://your-vllm-host:8000/v1 python app.py
```

This skips the built-in vLLM and connects to your existing endpoint. Works with any OpenAI-compatible API.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │          protoVoice              │
                    │                                  │
  Mic (WebRTC) ───►│  Silero VAD                      │
                    │      │                           │
                    │      ▼                           │
                    │  Whisper STT (GPU)               │
                    │      │                           │
                    │      ▼                           │
                    │  Qwen 4B via vLLM ──► streaming  │
                    │      │                  tokens   │
                    │      ▼                           │
                    │  Sentence Chunker                │
                    │      │                           │
                    │      ▼                           │
                    │  Kokoro TTS (GPU) ──► audio      │
                    │      │                chunks     │
  Speaker ◄────────│──────┘                           │
    (WebRTC)       │                                  │
                    └─────────────────────────────────┘
```

## License

MIT
