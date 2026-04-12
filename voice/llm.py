"""Voice LLM calls — direct OpenAI-compatible chat completions.

Uses the configured LLM_URL (shared vLLM or LiteLLM gateway) instead
of routing through A2A. The voice pipeline needs low-latency responses
so we call the LLM directly with the voice system prompt.
"""

import logging
import os
import threading

import httpx

logger = logging.getLogger(__name__)

LLM_URL = os.environ.get("LLM_URL", "http://vllm:8000/v1")
LLM_MODEL = os.environ.get("LLM_SERVED_NAME", "local")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def llm_chat(
    text: str,
    system_prompt: str,
    model: str = "",
    llm_url: str = "",
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 150,
) -> str:
    """Direct chat completion call to the LLM."""
    url = (llm_url or LLM_URL).rstrip("/")
    mdl = model or LLM_MODEL
    key = api_key or LLM_API_KEY

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        r = httpx.post(
            f"{url}/chat/completions",
            headers=headers,
            json={
                "model": mdl,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Voice LLM error: {e}")
        return "Sorry, I couldn't process that."


def stream_llm_tokens(
    text: str,
    system_prompt: str,
    cancel: threading.Event,
    model: str = "",
    llm_url: str = "",
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 150,
):
    """Call LLM and yield the response for TTS processing."""
    if cancel.is_set():
        return
    response = llm_chat(text, system_prompt, model, llm_url, api_key, temperature, max_tokens)
    if not cancel.is_set() and response:
        yield response


# Backwards compat aliases
def a2a_send(text, conversation_id, skill_hint="chat", a2a_url="", api_key=""):
    return llm_chat(text, "You are a helpful assistant.", a2a_url=a2a_url, api_key=api_key)

def stream_a2a_tokens(text, conversation_id, cancel, skill_hint="chat", a2a_url="", api_key=""):
    yield from stream_llm_tokens(text, "You are a helpful assistant.", cancel, llm_url=a2a_url, api_key=api_key)
