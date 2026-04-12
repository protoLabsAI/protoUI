"""Ava text chat — ReAct agent on Opus with tools.

Routes through chat.agent which handles the function-calling loop.
Voice stays on the small local model (no tools, latency-critical).
"""

import threading

from memory.graphiti import add_episode

from .agent import run


def chat(
    message: str,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    result = run(message, history, max_tokens, temperature)
    if result.text:
        threading.Thread(
            target=add_episode,
            args=(message, result.text),
            kwargs={"platform": "ava-chat"},
            daemon=True,
        ).start()
    return result.text
