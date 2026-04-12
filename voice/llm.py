"""Voice LLM — small model with tool-based routing to Opus.

The small model responds to every utterance. It has a single
`deep_research` tool — if it calls that, the voice agent spawns
Opus for the full ReAct loop with web search and other tools.
"""

import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

LLM_URL = os.environ.get("LLM_URL", "http://vllm:8000/v1")
LLM_MODEL = os.environ.get("LLM_SERVED_NAME", "local")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

_QWEN_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=deep_research>\s*<parameter=query>\s*(.+?)\s*</parameter>",
    re.DOTALL,
)

RESEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": (
            "Search the web or do deeper analysis. Use when the user asks "
            "about current events, needs a fact you're not confident about, "
            "wants a calculation, or asks something that requires up-to-date "
            "information. Pass the user's question as the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to research",
                },
            },
            "required": ["query"],
        },
    },
}


class SmallModelResponse:
    __slots__ = ("content", "research_query")

    def __init__(self, content: str, research_query: str | None):
        self.content = content
        self.research_query = research_query


def llm_chat(
    text: str,
    system_prompt: str,
    model: str = "",
    llm_url: str = "",
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 150,
) -> SmallModelResponse:
    """Call the small model. Returns spoken content + optional research query."""
    url = (llm_url or LLM_URL).rstrip("/")
    mdl = model or LLM_MODEL
    key = api_key or LLM_API_KEY

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        r = httpx.post(
            f"{url}/chat/completions",
            headers=headers,
            json={
                "model": mdl,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "tools": [RESEARCH_TOOL],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30.0,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning") or "").strip()

        research_query = None

        # Check structured tool_calls field first (if vLLM parser works)
        for tc in msg.get("tool_calls") or []:
            if tc.get("function", {}).get("name") == "deep_research":
                try:
                    args = json.loads(tc["function"]["arguments"])
                    research_query = args.get("query", "")
                except (json.JSONDecodeError, KeyError):
                    pass
                break

        # Fallback: parse Qwen's XML tool call format from content
        if not research_query and content:
            m = _QWEN_TOOL_RE.search(content)
            if m:
                research_query = m.group(1).strip()
                content = content[:m.start()].strip()

        return SmallModelResponse(content, research_query)
    except Exception as e:
        logger.error(f"Voice LLM error: {e}")
        return SmallModelResponse("", None)
