#!/usr/bin/env python3
"""
protoVoice — Sub-200ms real-time voice agent

Self-contained voice agent: STT + LLM + TTS on a single GPU.
All models pre-warmed on startup for zero cold start.

Pipeline:
  Mic → [Silero VAD] → [Whisper Turbo] → [Qwen 4B] → [Kokoro TTS] → Speaker
          ~1ms             ~55ms            ~150ms        ~50ms

Benchmarked: 165ms TTFA, 210ms total on RTX PRO 6000 Blackwell.
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import signal

os.environ.setdefault("HF_HOME", os.environ.get("MODEL_DIR", "/models"))

import gradio as gr
import httpx
import numpy as np
import soxr
import torch
from fastrtc import ReplyOnPause, Stream
from fastrtc.reply_on_pause import AlgoOptions
from transformers import pipeline as hf_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — all overridable via env vars
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = int(os.environ.get("PORT", "7866"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8100"))
LLM_URL = os.environ.get("LLM_URL", f"http://localhost:{VLLM_PORT}/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.environ.get("KOKORO_LANG", "a")

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You are a helpful voice assistant. Keep responses concise — 1-3 sentences max. "
    "Be conversational and natural. Do not use markdown, bullet points, or formatting. "
    "Respond as if speaking out loud."
))

MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "10"))
MAX_HISTORY_TOKENS = int(os.environ.get("MAX_HISTORY_TOKENS", "2000"))
SUMMARY_PROMPT = (
    "Summarize this conversation so far in 2-3 sentences. "
    "Focus on key topics discussed and any important facts mentioned."
)

# Whether to start a built-in vLLM process (set to 0 if using external LLM)
START_VLLM = os.environ.get("START_VLLM", "1") == "1"

_vllm_proc = None


# ---------------------------------------------------------------------------
# Built-in vLLM server
# ---------------------------------------------------------------------------
def start_vllm():
    """Start vLLM as a subprocess for the LLM backend."""
    global _vllm_proc
    if not START_VLLM:
        logger.info(f"Using external LLM at {LLM_URL}")
        return

    logger.info(f"Starting vLLM with {LLM_MODEL} on port {VLLM_PORT}...")
    _vllm_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", LLM_MODEL,
            "--host", "127.0.0.1",
            "--port", str(VLLM_PORT),
            "--served-model-name", "local",
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.40",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--chat-template-kwargs", json.dumps({"enable_thinking": False}),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for vLLM to be ready
    for i in range(120):
        try:
            r = httpx.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2.0)
            if r.status_code == 200:
                logger.info(f"vLLM ready on port {VLLM_PORT}")
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


# ---------------------------------------------------------------------------
# Sentence Chunker
# ---------------------------------------------------------------------------
class SentenceChunker:
    BOUNDARY = re.compile(r'(?<=[.!?;:])\s+|(?<=[.!?])\s*$')

    def __init__(self, min_first=10, min_rest=30, max_chars=200):
        self.buffer = ""
        self.chunk_count = 0
        self.min_first = min_first
        self.min_rest = min_rest
        self.max_chars = max_chars

    @property
    def min_chars(self):
        return self.min_first if self.chunk_count == 0 else self.min_rest

    def feed(self, token: str):
        self.buffer += token
        if len(self.buffer) >= self.max_chars:
            text = self.buffer.strip()
            if text:
                self.chunk_count += 1
                yield text
            self.buffer = ""
            return
        pattern = re.compile(r'(?<=[,.!?;:])\s+') if self.chunk_count == 0 else self.BOUNDARY
        matches = list(pattern.finditer(self.buffer))
        if matches:
            last = matches[-1]
            candidate = self.buffer[:last.end()].strip()
            if len(candidate) >= self.min_chars:
                self.chunk_count += 1
                yield candidate
                self.buffer = self.buffer[last.end():]

    def flush(self):
        if self.buffer.strip():
            self.chunk_count += 1
            yield self.buffer.strip()
            self.buffer = ""


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
_stt_pipe = None


def get_stt():
    global _stt_pipe
    if _stt_pipe is None:
        logger.info(f"Loading {WHISPER_MODEL}...")
        t0 = time.time()
        _stt_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device=DEVICE,
            model_kwargs={"attn_implementation": "sdpa"} if DEVICE == "cuda" else {},
        )
        silence = np.zeros(16000, dtype=np.float32)
        _stt_pipe({"raw": silence, "sampling_rate": 16000})
        logger.info(f"Whisper loaded + warmed in {time.time() - t0:.1f}s")
    return _stt_pipe


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
_kokoro_pipe = None


def get_kokoro():
    global _kokoro_pipe
    if _kokoro_pipe is None:
        logger.info("Loading Kokoro 82M...")
        t0 = time.time()
        from kokoro import KPipeline
        _kokoro_pipe = KPipeline(lang_code=KOKORO_LANG)
        list(_kokoro_pipe("Hello.", voice=KOKORO_VOICE, speed=1))
        logger.info(f"Kokoro loaded + warmed in {time.time() - t0:.1f}s")
    return _kokoro_pipe


def tts_kokoro(text: str) -> tuple[int, np.ndarray]:
    pipe = get_kokoro()
    chunks = list(pipe(text, voice=KOKORO_VOICE, speed=1))
    if not chunks:
        return 24000, np.zeros(2400, dtype=np.int16)
    audio = np.concatenate([c[2] for c in chunks])
    return 24000, (audio * 32767).clip(-32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def stream_llm_tokens(text: str, history: list[dict], cancel: threading.Event):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": text})

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST", f"{LLM_URL}/chat/completions",
                json={
                    "model": "local",
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as response:
                for line in response.iter_lines():
                    if cancel.is_set():
                        return
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
    except Exception as e:
        if not cancel.is_set():
            logger.error(f"LLM error: {e}")
            yield "Sorry, I couldn't process that."


def llm_summarize(history: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )},
    ]
    try:
        r = httpx.post(
            f"{LLM_URL}/chat/completions",
            json={"model": "local", "messages": messages, "max_tokens": 100,
                  "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}},
            timeout=15.0,
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Voice Agent
# ---------------------------------------------------------------------------
class VoiceAgent:
    def __init__(self):
        self.history: list[dict] = []
        self.summary: str = ""
        self.cancel = threading.Event()

    def _get_context(self) -> list[dict]:
        messages = []
        if self.summary:
            messages.append({"role": "system", "content": f"Previous conversation summary: {self.summary}"})
        recent = self.history[-(MAX_HISTORY_TURNS * 2):]
        total = sum(len(m["content"]) for m in recent)
        while total > MAX_HISTORY_TOKENS and len(recent) > 2:
            removed = recent.pop(0)
            total -= len(removed["content"])
            if recent and recent[0]["role"] == "assistant":
                total -= len(recent.pop(0)["content"])
        messages.extend(recent)
        return messages

    def _maybe_summarize(self):
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            old = self.history[:-(MAX_HISTORY_TURNS * 2)]
            if old:
                to_sum = []
                if self.summary:
                    to_sum.append({"role": "system", "content": f"Prior summary: {self.summary}"})
                to_sum.extend(old)
                s = llm_summarize(to_sum)
                if s:
                    self.summary = s
                    logger.info(f"[Context] Summarized: {s[:80]}...")
                self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

    def interrupt(self):
        self.cancel.set()

    def process(self, audio_tuple: tuple[int, np.ndarray]):
        self.cancel.clear()
        sr_in, audio_in = audio_tuple
        t_start = time.time()

        # STT
        t0 = time.time()
        stt = get_stt()
        if audio_in.ndim > 1:
            audio_in = audio_in[:, 0] if audio_in.shape[1] < audio_in.shape[0] else audio_in[0]
        if audio_in.dtype != np.float32:
            audio_in = audio_in.astype(np.float32) / max(np.iinfo(audio_in.dtype).max, 1)
        if sr_in != 16000:
            audio_in = soxr.resample(audio_in.reshape(-1), sr_in, 16000)
        result = stt({"raw": audio_in.flatten(), "sampling_rate": 16000})
        user_text = result["text"].strip()
        stt_time = time.time() - t0

        if not user_text:
            return
        logger.info(f"[STT {stt_time:.2f}s] {user_text}")

        # Stream LLM → chunk → TTS
        chunker = SentenceChunker()
        full_response = ""
        t_llm = time.time()
        ttfa = None
        chunks = 0
        tts_t = 0
        interrupted = False

        for token in stream_llm_tokens(user_text, self._get_context(), self.cancel):
            if self.cancel.is_set():
                interrupted = True
                break
            full_response += token
            for sentence in chunker.feed(token):
                if self.cancel.is_set():
                    interrupted = True
                    break
                t1 = time.time()
                sr, audio = tts_kokoro(sentence)
                tts_t += time.time() - t1
                if ttfa is None:
                    ttfa = time.time() - t_start
                    logger.info(f"[TTFA {ttfa:.2f}s] {sentence!r}")
                chunks += 1
                yield sr, audio
            if interrupted:
                break

        if not interrupted:
            for sentence in chunker.flush():
                if self.cancel.is_set():
                    break
                t1 = time.time()
                sr, audio = tts_kokoro(sentence)
                tts_t += time.time() - t1
                chunks += 1
                yield sr, audio

        if full_response.strip():
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": full_response.strip()})
            self._maybe_summarize()

        total = time.time() - t_start
        status = "INTERRUPTED" if interrupted else "COMPLETE"
        logger.info(
            f"[{status} {total:.2f}s] STT={stt_time:.2f}s LLM={time.time()-t_llm:.2f}s "
            f"TTS={tts_t:.2f}s TTFA={ttfa or 0:.2f}s chunks={chunks} "
            f"history={len(self.history)} summary={'yes' if self.summary else 'no'}"
        )


agent = VoiceAgent()


def voice_handler(audio: tuple[int, np.ndarray]):
    agent.interrupt()
    yield from agent.process(audio)


def prewarm():
    logger.info("Pre-warming all models...")
    t0 = time.time()
    get_stt()
    get_kokoro()
    try:
        httpx.post(
            f"{LLM_URL}/chat/completions",
            json={"model": "local",
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": "Hi"}],
                  "max_tokens": 1, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=30.0,
        )
        logger.info("LLM prefix cache warmed")
    except Exception as e:
        logger.warning(f"LLM warmup skipped: {e}")
    logger.info(f"All models ready in {time.time() - t0:.1f}s")


def main():
    # Start built-in vLLM if configured
    start_vllm()

    # Handle shutdown
    def shutdown(sig, frame):
        logger.info("Shutting down...")
        stop_vllm()
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    prewarm()

    demo = gr.Blocks(title="protoVoice")
    with demo:
        gr.Markdown("# protoVoice\nSpeak — I'll respond.")
        Stream(
            ReplyOnPause(
                voice_handler,
                algo_options=AlgoOptions(
                    audio_chunk_duration=0.6,
                    started_talking_threshold=0.5,
                    speech_threshold=0.1,
                ),
                output_sample_rate=24000,
                can_interrupt=True,
            ),
            modality="audio",
            mode="send-receive",
        )

    auth = os.environ.get("GRADIO_AUTH")
    auth_pairs = None
    if auth:
        auth_pairs = [tuple(p.split(":", 1)) for p in auth.split(",")]
        logger.info(f"Auth enabled for {len(auth_pairs)} user(s)")

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        auth=auth_pairs,
    )


if __name__ == "__main__":
    main()
