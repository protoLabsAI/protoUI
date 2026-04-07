import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Generator

import numpy as np

from .chunker import SentenceChunker
from .llm import stream_a2a_tokens
from .stt import transcribe
from .tts import tts_kokoro

logger = logging.getLogger(__name__)

# Maps voice mode to an A2A skillHint sent with each request
MODE_SKILL_HINTS = {
    "chat": "chat",
    "agent": "research",
    "skill": "skill",
}


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
    a2a_url: str = field(
        default_factory=lambda: os.environ.get("A2A_URL", "http://automaker-server:3008/a2a")
    )
    a2a_api_key: str = field(
        default_factory=lambda: os.environ.get("A2A_API_KEY", "")
    )


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

    def process(
        self,
        audio_tuple: tuple[int, np.ndarray],
        config: VoiceConfig,
    ) -> Generator[tuple[str, object], None, None]:
        """
        Generator yielding:
          ("audio", (sr, np.ndarray))  — audio chunk to play
          ("transcript", str)          — transcription text (transcribe mode)
        """
        self.cancel.clear()
        t_start = time.time()

        # Always transcribe first
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

        # Resolve effective mode (wake_word gates to chat)
        mode = config.mode
        if mode == "wake_word":
            word = config.wake_word.strip().lower()
            if word and word not in user_text.lower():
                logger.debug(f"[Wake] No trigger in: {user_text!r}")
                return
            if word:
                idx = user_text.lower().find(word)
                user_text = user_text[idx + len(word):].strip(" ,.")
            if not user_text:
                return
            logger.info(f"[Wake] Triggered → {user_text!r}")
            mode = "chat"

        # Transcribe-only mode
        if mode == "transcribe":
            yield ("transcript", user_text)
            return

        # All other modes route through A2A — Ava handles agent smarts
        skill_hint = MODE_SKILL_HINTS.get(mode, "chat")
        chunker = SentenceChunker()
        full_response = ""
        ttfa = None
        interrupted = False

        for token in stream_a2a_tokens(
            user_text,
            self.conversation_id,
            self.cancel,
            skill_hint=skill_hint,
            a2a_url=config.a2a_url,
            api_key=config.a2a_api_key,
        ):
            if self.cancel.is_set():
                interrupted = True
                break
            full_response += token
            for sentence in chunker.feed(token):
                if self.cancel.is_set():
                    interrupted = True
                    break
                sr, audio = tts_kokoro(sentence, config.voice, config.lang)
                if ttfa is None:
                    ttfa = time.time() - t_start
                    logger.info(f"[TTFA {ttfa:.2f}s] {sentence!r}")
                yield ("audio", (sr, audio))
            if interrupted:
                break

        if not interrupted:
            for sentence in chunker.flush():
                if self.cancel.is_set():
                    break
                sr, audio = tts_kokoro(sentence, config.voice, config.lang)
                yield ("audio", (sr, audio))

        if full_response.strip():
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": full_response.strip()})

        status = "INTERRUPTED" if interrupted else "DONE"
        logger.info(
            f"[{status} {time.time() - t_start:.2f}s] "
            f"STT={stt_time:.2f}s TTFA={ttfa or 0:.2f}s "
            f"history={len(self.history)}"
        )
