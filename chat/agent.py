"""ReAct agent — Opus with function calling via OpenAI-compatible API.

Simple loop: send messages with tool specs → model either responds or
calls tools → execute tools → feed results back → repeat. The model
naturally decides whether to chat or use tools based on the input.
"""

import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from memory.context import assemble_context
from memory.graphiti import add_episode, get_context_block

from .tools import TOOL_SPECS, execute_tool

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

_GATEWAY_URL = os.environ.get(
    "OPENAI_BASE_URL", os.environ.get("LLM_URL", "http://gateway:4000/v1")
)
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
            _soul_cache = "You are Ava, a helpful conversational assistant."
            logger.warning(f"SOUL.md not found at {_SOUL_PATH}, using fallback")
    return _soul_cache


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=_GATEWAY_URL, api_key=_GATEWAY_KEY or "unused")
    return _client


class AgentResult:
    __slots__ = ("text", "used_tools")

    def __init__(self, text: str, used_tools: bool):
        self.text = text
        self.used_tools = used_tools


def run(
    message: str,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> AgentResult:
    """Run the ReAct agent loop. Returns response text + whether tools were used."""
    client = _get_client()

    recalled = get_context_block(message)
    recent_turns = _history_to_turns(history)
    enriched_message = assemble_context(recalled or None, recent_turns, message)

    messages: list[dict] = [{"role": "system", "content": _load_soul()}]
    messages.append({"role": "user", "content": enriched_message})

    used_tools = False

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=TOOL_SPECS,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        if not assistant_msg.tool_calls:
            return AgentResult((assistant_msg.content or "").strip(), used_tools)

        used_tools = True
        messages.append(assistant_msg.model_dump(exclude_none=True))

        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            logger.info(f"[ReAct #{iteration}] {name}({args})")
            result = execute_tool(name, args)
            logger.info(f"[ReAct #{iteration}] → {result[:200]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    logger.warning("ReAct loop hit max iterations")
    return AgentResult(
        "I wasn't able to finish processing that. Could you try rephrasing?",
        used_tools,
    )


def _history_to_turns(history: list[dict[str, str]] | None) -> list[dict]:
    if not history:
        return []
    return [
        {"role": h["role"], "content": h["content"], "channel": "ava-chat"}
        for h in history
        if h.get("role") in ("user", "assistant") and h.get("content")
    ]
