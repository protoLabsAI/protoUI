"""Duplex voice agent — small model routes, Opus thinks.

Every utterance goes to the small model first. It either:
  1. Responds directly (simple chat) → speak it, done
  2. Calls deep_research tool → speak its filler text via TTS
     while Opus runs the full ReAct loop with tools in parallel,
     then speak Opus's result when ready
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Generator

import numpy as np

from memory.context import assemble_context
from memory.graphiti import add_episode, get_context_block

from .chunker import SentenceChunker
from .llm import llm_chat
from .stt import transcribe
from .tts import tts_kokoro

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    mode: str = "chat"
    voice: str = "af_heart"
    lang: str = "a"
    temperature: float = 0.7
    max_tokens: int = 150
    wake_word: str = ""
    system_prompt: str = ""
    llm_url: str = "http://localhost:8100/v1"
    model: str = "local"
    api_key: str = ""
    whisper_model: str = "openai/whisper-large-v3-turbo"
    timezone: str = "UTC"


class VoiceAgent:
    def __init__(self):
        self.history: list[dict] = []
        self.conversation_id: str = str(uuid.uuid4())
        self.cancel = threading.Event()

    def interrupt(self):
        self.cancel.set()

    def clear_history(self):
        self.history = []
        self.conversation_id = str(uuid.uuid4())

    def _run_opus(self, message: str):
        """Run Opus ReAct agent. Returns AgentResult or None."""
        try:
            from chat.agent import run
            return run(message)
        except Exception as e:
            logger.error(f"[Duplex] Opus error: {e}")
            return None

    def _tts_text(
        self, text: str, config: VoiceConfig
    ) -> Generator[tuple[str, object], None, None]:
        chunker = SentenceChunker()
        for sentence in chunker.feed(text):
            if self.cancel.is_set():
                return
            sr, audio = tts_kokoro(sentence, config.voice, config.lang)
            yield ("audio", (sr, audio))
        for sentence in chunker.flush():
            if self.cancel.is_set():
                return
            sr, audio = tts_kokoro(sentence, config.voice, config.lang)
            yield ("audio", (sr, audio))

    def process(
        self,
        audio_tuple: tuple[int, np.ndarray],
        config: VoiceConfig,
    ) -> Generator[tuple[str, object], None, None]:
        self.cancel.clear()
        t_start = time.time()

        # --- STT ---
        t0 = time.time()
        try:
            user_text = transcribe(audio_tuple, config.whisper_model)
        except Exception as e:
            logger.error(f"STT error: {e}")
            return
        stt_time = time.time() - t0

        if not user_text:
            return
        logger.info(f"[STT {stt_time:.2f}s] {user_text!r}")

        # --- Wake word ---
        mode = config.mode
        if mode == "wake_word":
            word = config.wake_word.strip().lower()
            if word and word not in user_text.lower():
                return
            if word:
                idx = user_text.lower().find(word)
                user_text = user_text[idx + len(word):].strip(" ,.")
            if not user_text:
                return
            mode = "chat"

        if mode == "transcribe":
            yield ("transcript", user_text)
            return

        # --- Memory enrichment ---
        recalled = get_context_block(user_text)
        recent_turns = [
            {"role": h["role"], "content": h["content"], "channel": "ava-voice"}
            for h in self.history
            if h.get("role") in ("user", "assistant") and h.get("content")
        ]
        enriched_message = assemble_context(
            recalled or None, recent_turns, user_text,
        )

        # --- Small model: respond + decide if research needed ---
        t_llm = time.time()
        sm = llm_chat(
            enriched_message,
            config.system_prompt,
            model=config.model,
            llm_url=config.llm_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        logger.info(
            f"[Small {time.time() - t_llm:.2f}s] "
            f"content={bool(sm.content)} research={sm.research_query!r}"
        )

        full_response = ""
        ttfa = None

        if not sm.research_query:
            # Simple chat — small model handled it, no Opus needed
            if sm.content:
                full_response = sm.content
                for event in self._tts_text(sm.content, config):
                    if self.cancel.is_set():
                        break
                    if ttfa is None:
                        ttfa = time.time() - t_start
                    yield event
        else:
            # Research needed — speak filler while Opus works
            opus_q: queue.Queue = queue.Queue()
            opus_done = threading.Event()

            def opus_worker():
                result = self._run_opus(enriched_message)
                opus_q.put(result)
                opus_done.set()

            opus_thread = threading.Thread(target=opus_worker, daemon=True)
            opus_thread.start()

            # Speak small model's filler (if any)
            if sm.content and not self.cancel.is_set():
                logger.info(f"[Duplex] Filler: {sm.content!r}")
                for event in self._tts_text(sm.content, config):
                    if self.cancel.is_set():
                        break
                    if ttfa is None:
                        ttfa = time.time() - t_start
                    yield event

            # Wait for Opus result
            if not self.cancel.is_set():
                opus_done.wait(timeout=60)
                try:
                    agent_result = opus_q.get_nowait()
                    if agent_result and agent_result.text:
                        full_response = agent_result.text
                        logger.info(
                            f"[Duplex] Opus result (tools={agent_result.used_tools})"
                        )
                        for event in self._tts_text(agent_result.text, config):
                            if self.cancel.is_set():
                                break
                            if ttfa is None:
                                ttfa = time.time() - t_start
                            yield event
                except queue.Empty:
                    full_response = sm.content or ""

            if not full_response:
                full_response = sm.content or ""

        # --- History + episode ---
        if full_response.strip():
            self.history.append({"role": "user", "content": user_text})
            self.history.append(
                {"role": "assistant", "content": full_response.strip()}
            )
            threading.Thread(
                target=add_episode,
                args=(user_text, full_response.strip()),
                kwargs={"platform": "ava-voice"},
                daemon=True,
            ).start()

        logger.info(
            f"[DONE {time.time() - t_start:.2f}s] "
            f"STT={stt_time:.2f}s TTFA={ttfa or 0:.2f}s "
            f"history={len(self.history)}"
        )
