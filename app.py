#!/usr/bin/env python3
"""
Ava — conversational protoAgent with voice + text chat + A2A

Voice pipeline: Mic → Silero VAD → Whisper STT → LLM → Kokoro TTS → Speaker
Text chat: Gradio chatbot → LLM gateway (Opus 4.6 via protolabs/ava)
A2A: POST /a2a JSON-RPC 2.0 + GET /.well-known/agent.json

Both interfaces use SOUL.md for personality. Voice uses the configured
LLM_URL (local vLLM or gateway). Text chat always uses the gateway.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", os.environ.get("MODEL_DIR", "/models"))

import gradio as gr
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastrtc import ReplyOnPause, Stream
from fastrtc.reply_on_pause import AlgoOptions

from chat.backend import chat as ava_chat
from skills.loader import load_skills
from voice.agent import VoiceAgent, VoiceConfig
from voice.stt import get_stt
from voice.tts import get_kokoro, list_voices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "7866"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8100"))
LLM_URL = os.environ.get("LLM_URL", f"http://localhost:{VLLM_PORT}/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B")
LLM_SERVED_NAME = os.environ.get("LLM_SERVED_NAME", "local")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.environ.get("KOKORO_LANG", "a")
START_VLLM = os.environ.get("START_VLLM", "1") == "1"

# Load SOUL.md for voice system prompt
_SOUL_PATH = Path(__file__).parent / "config" / "SOUL.md"
_SOUL_TEXT = _SOUL_PATH.read_text().strip() if _SOUL_PATH.exists() else "You are Ava, a helpful assistant."

VOICE_PREAMBLE = (
    "You are speaking directly to the user through a voice interface. "
    "Your response will be read aloud by a text-to-speech engine, so: "
    "never use markdown, bullet points, numbered lists, headers, code blocks, or any formatting. "
    "Never use emojis or special unicode characters. "
    "Speak in casual, natural, conversational sentences. "
    "Keep responses short: 1 to 3 sentences unless more detail is truly necessary. "
    "/no_think"
)

VOICE_SYSTEM_PROMPT = VOICE_PREAMBLE + "\n\n" + _SOUL_TEXT

VOICE_LANG_MAP: dict[str, str] = {
    "af_heart": "a", "af_bella": "a", "af_sarah": "a", "af_nicole": "a", "af_sky": "a",
    "am_adam": "a", "am_michael": "a",
    "bf_emma": "b", "bf_isabella": "b",
    "bm_george": "b", "bm_lewis": "b",
}

# Voice state
_algo_options = AlgoOptions(audio_chunk_duration=0.6, started_talking_threshold=0.5, speech_threshold=0.1)
_config = VoiceConfig(
    mode="chat", voice=KOKORO_VOICE, lang=KOKORO_LANG, temperature=0.7, max_tokens=150,
    system_prompt=VOICE_SYSTEM_PROMPT, llm_url=LLM_URL, model=LLM_SERVED_NAME, api_key=LLM_API_KEY,
    whisper_model=WHISPER_MODEL,
)
agent = VoiceAgent()


def voice_handler(audio: tuple[int, np.ndarray]):
    agent.interrupt()
    for event_type, payload in agent.process(audio, _config):
        if event_type == "audio":
            yield payload


# vLLM lifecycle
_vllm_proc = None

def start_vllm():
    global _vllm_proc
    if not START_VLLM:
        logger.info(f"Using external LLM at {LLM_URL}")
        return
    logger.info(f"Starting vLLM with {LLM_MODEL} on port {VLLM_PORT}...")
    _vllm_proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", LLM_MODEL, "--host", "127.0.0.1", "--port", str(VLLM_PORT),
         "--served-model-name", "local", "--max-model-len", "32768",
         "--gpu-memory-utilization", "0.40", "--enable-prefix-caching",
         "--enable-chunked-prefill",
         "--chat-template-kwargs", json.dumps({"enable_thinking": False})],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for _ in range(120):
        try:
            r = httpx.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2.0)
            if r.status_code == 200:
                logger.info("vLLM ready")
                return
        except Exception:
            pass
        time.sleep(1)
    logger.error("vLLM failed to start within 120s")

def stop_vllm():
    global _vllm_proc
    if _vllm_proc:
        _vllm_proc.terminate()
        _vllm_proc.wait(timeout=10)
        _vllm_proc = None


# Pre-warm
def prewarm():
    logger.info("Pre-warming models...")
    t0 = time.time()
    get_stt(WHISPER_MODEL)
    get_kokoro(KOKORO_LANG)
    try:
        httpx.post(f"{LLM_URL}/chat/completions",
                   json={"model": LLM_SERVED_NAME, "messages": [{"role": "user", "content": "Hi"}],
                         "max_tokens": 1, "temperature": 0},
                   headers={"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {},
                   timeout=30.0)
    except Exception as e:
        logger.warning(f"LLM warmup skipped: {e}")
    logger.info(f"All models ready in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Chat handler (text)
# ---------------------------------------------------------------------------
_SLASH_HELP = "**Commands:** `/clear` — reset, `/help` — this message"

def build_ui(skills):
    _conversation_id = str(uuid.uuid4())

    def handle_chat(message: str, history: list[dict]) -> tuple[str, list[dict]]:
        nonlocal _conversation_id
        message = message.strip()
        if not message:
            return "", history
        if message.startswith("/"):
            cmd = message.split()[0].lower()
            if cmd == "/clear":
                _conversation_id = str(uuid.uuid4())
                return "", []
            elif cmd == "/help":
                return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": _SLASH_HELP}]
            else:
                return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"Unknown: `{cmd}`"}]

        chat_history = [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        response = ava_chat(message=message, history=chat_history)
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

    skills_map = {s.slug: s for s in skills}
    mode_choices = [
        ("Chat", "chat"),
        ("Agent", "agent"),
    ] + [(s.name, f"skill:{s.slug}") for s in skills]

    all_voices = list_voices()
    for v in all_voices:
        if v not in VOICE_LANG_MAP:
            VOICE_LANG_MAP[v] = KOKORO_LANG

    def on_mode_change(mode: str):
        if mode.startswith("skill:"):
            slug = mode[6:]
            skill = skills_map.get(slug)
            if skill:
                _config.system_prompt = skill.system_prompt
                _config.voice = skill.voice
                _config.lang = skill.lang
                _config.max_tokens = skill.max_tokens
                _config.temperature = skill.temperature
                _config.llm_url = skill.llm_url or LLM_URL
                _config.model = skill.model or LLM_SERVED_NAME
                return gr.update(value=skill.voice), gr.update(value=skill.temperature), gr.update(value=skill.max_tokens)
        else:
            _config.system_prompt = VOICE_SYSTEM_PROMPT
            _config.voice = KOKORO_VOICE
            _config.lang = KOKORO_LANG
            _config.max_tokens = 150
            _config.temperature = 0.7
            _config.llm_url = LLM_URL
            _config.model = LLM_SERVED_NAME
        _config.mode = mode
        return gr.update(value=KOKORO_VOICE), gr.update(value=0.7), gr.update(value=150)

    def on_voice_change(voice: str):
        _config.voice = voice
        _config.lang = VOICE_LANG_MAP.get(voice, KOKORO_LANG)

    def on_clear_history():
        agent.clear_history()

    with gr.Blocks(
        title="Ava",
        css="footer {display: none !important} .gradio-container > .flex.flex-wrap {display: none !important}",
        analytics_enabled=False,
    ) as demo:

        # Voice — clean, just the WebRTC stream
        Stream(
            ReplyOnPause(voice_handler, algo_options=_algo_options, output_sample_rate=24000, can_interrupt=True),
            modality="audio", mode="send-receive",
            rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]},
        )

        # Settings sidebar
        with gr.Sidebar(label="Settings", open=False, position="right"):
            gr.Markdown("**Mode**")
            mode_dd = gr.Dropdown(choices=mode_choices, value="chat", show_label=False, interactive=True)

            gr.Markdown("**Voice**")
            voice_dd = gr.Dropdown(choices=all_voices, value=KOKORO_VOICE, label="TTS voice", interactive=True)

            gr.Markdown("**LLM**")
            llm_url_box = gr.Textbox(label="Voice LLM URL", value=LLM_URL, interactive=True, max_lines=1)
            llm_model_box = gr.Textbox(label="Voice model", value=LLM_SERVED_NAME, interactive=True, max_lines=1)
            llm_api_key_box = gr.Textbox(label="API key", value=LLM_API_KEY, type="password", interactive=True, max_lines=1)
            temp_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")
            tokens_slider = gr.Slider(50, 500, value=150, step=25, label="Max tokens")

            gr.Markdown("**Session**")
            clear_history_btn = gr.Button("Clear voice history", size="sm", variant="secondary")

        # Sidebar event wiring
        mode_dd.change(fn=on_mode_change, inputs=[mode_dd], outputs=[voice_dd, temp_slider, tokens_slider])
        voice_dd.change(fn=on_voice_change, inputs=[voice_dd])
        temp_slider.change(fn=lambda v: setattr(_config, "temperature", v), inputs=[temp_slider])
        tokens_slider.change(fn=lambda v: setattr(_config, "max_tokens", int(v)), inputs=[tokens_slider])
        llm_url_box.change(fn=lambda v: setattr(_config, "llm_url", v.strip()), inputs=[llm_url_box])
        llm_model_box.change(fn=lambda v: setattr(_config, "model", v.strip()), inputs=[llm_model_box])
        llm_api_key_box.change(fn=lambda v: setattr(_config, "api_key", v.strip()), inputs=[llm_api_key_box])
        clear_history_btn.click(fn=on_clear_history)

        # Text chat — collapsible, hidden by default
        with gr.Accordion("Chat", open=False):
            chatbot = gr.Chatbot(type="messages", height=400, show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Type a message…", show_label=False, scale=9, max_lines=3)
                send_btn = gr.Button("Send", scale=1, variant="primary")

            send_btn.click(fn=handle_chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
            chat_input.submit(fn=handle_chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])

    return demo


# ---------------------------------------------------------------------------
# FastAPI + A2A
# ---------------------------------------------------------------------------
fastapi_app = FastAPI(title="Ava")

def _build_agent_card(host: str) -> dict:
    return {
        "name": "ava",
        "description": "Conversational protoAgent — voice + text chat, suggests delegations when action is needed.",
        "url": f"http://{host}", "version": "0.2.0",
        "provider": {"organization": "protoLabsAI"},
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/markdown"],
        "skills": [{"id": "chat", "name": "Chat", "description": "Free-form conversation."}],
        "security": [],
    }

@fastapi_app.get("/.well-known/agent.json")
@fastapi_app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    return JSONResponse(content=_build_agent_card(request.headers.get("host", f"ava:{PORT}")),
                        headers={"Cache-Control": "public, max-age=60"})

@fastapi_app.post("/a2a")
async def a2a_handler(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
    rpc_id = body.get("id")
    if body.get("method") != "message/send":
        return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown method: {body.get('method')}"}})
    parts = body.get("params", {}).get("message", {}).get("parts", [])
    user_text = "\n".join(p.get("text", "") for p in parts if (p.get("kind") or p.get("type")) == "text").strip()
    if not user_text:
        return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "No text part"}})
    logger.info(f'A2A: "{user_text[:80]}"')
    response_text = ava_chat(message=user_text)
    ctx = body.get("params", {}).get("contextId", str(uuid.uuid4()))
    return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": {
        "id": str(uuid.uuid4()), "contextId": ctx, "status": {"state": "completed"},
        "artifacts": [{"artifactId": str(uuid.uuid4()), "parts": [{"kind": "text", "text": response_text}]}],
    }})

@fastapi_app.get("/healthz")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_vllm()

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        stop_vllm()
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    prewarm()
    skills = load_skills()
    if skills:
        logger.info(f"Loaded skills: {[s.name for s in skills]}")

    demo = build_ui(skills)
    app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    logger.info(f"Ava starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
