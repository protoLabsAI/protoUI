"""Ava text chat backend — direct LLM call via OpenAI-compatible API.

No LangGraph, no SDK, no A2A delegation. Just a system prompt (SOUL.md)
and a conversation history sent to the LiteLLM gateway. Simplest thing
that works for a chat-only agent.
"""

import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", os.environ.get("LLM_URL", "http://gateway:4000/v1"))
_GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", ""))
_MODEL = os.environ.get("AVA_CHAT_MODEL", "protolabs/ava")

_SOUL_PATH = Path(__file__).parent.parent / "config" / "SOUL.md"
_soul_cache: str | None = None


def _load_soul() -> str:
    global _soul_cache
    if _soul_cache is None:
        if _SOUL_PATH.exists():
            _soul_cache = _SOUL_PATH.read_text().strip()
            logger.info(f"Loaded SOUL.md ({len(_soul_cache)} chars)")
        else:
            _soul_cache = "You are Ava, a helpful conversational assistant. Be concise."
            logger.warning(f"SOUL.md not found at {_SOUL_PATH}, using fallback prompt")
    return _soul_cache


def chat(
    message: str,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Send a message with conversation history to the LLM gateway.

    Args:
        message: The user's current message.
        history: Prior turns as [{"role": "user"|"assistant", "content": "..."}].
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.
    """
    messages = [{"role": "system", "content": _load_soul()}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    headers = {"Content-Type": "application/json"}
    if _GATEWAY_KEY:
        headers["Authorization"] = f"Bearer {_GATEWAY_KEY}"

    try:
        resp = httpx.post(
            f"{_GATEWAY_URL}/chat/completions",
            headers=headers,
            json={
                "model": _MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Chat backend error: {e}")
        return "Sorry, I hit an error processing that. Try again?"
