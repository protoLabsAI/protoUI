"""
LangGraph-powered text chat backend for protoVoice.

Wraps the A2A routing in a LangGraph StateGraph so conversation state
is managed as a proper graph node — ready for checkpointing, streaming,
and multi-step expansion without changing the public interface.

Public API
----------
text_chat(user_text, conversation_id, skill_hint) -> str
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import a2a_send

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: list[dict]
    conversation_id: str
    skill_hint: str
    response: str


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _call_a2a(state: ChatState) -> dict:
    """Send the latest user message via A2A and return the assistant response."""
    messages = state["messages"]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        logger.warning("[TextChat] No user message in state; returning empty response")
        return {"response": "", "messages": messages}

    user_text = user_msgs[-1].get("content", "")
    a2a_url = os.environ.get("A2A_URL", "http://automaker-server:3008/a2a")
    api_key = os.environ.get("A2A_API_KEY", "")

    response = a2a_send(
        user_text,
        state["conversation_id"],
        state.get("skill_hint", "chat"),
        a2a_url,
        api_key,
    )
    logger.debug(f"[TextChat] response={response[:120]!r}")

    return {
        "response": response,
        "messages": messages + [{"role": "assistant", "content": response}],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph():
    """Compile the LangGraph text chat graph."""
    builder: StateGraph = StateGraph(ChatState)
    builder.add_node("llm", _call_a2a)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_chat(
    user_text: str,
    conversation_id: str | None = None,
    skill_hint: str = "chat",
) -> str:
    """
    Send a text message and return the assistant response.

    Args:
        user_text:       The user's message.
        conversation_id: Optional conversation ID for multi-turn continuity.
        skill_hint:      A2A skill hint — "chat", "research", "skill:<slug>", etc.

    Returns:
        The assistant's response text, or an empty string on error.
    """
    graph = _get_graph()
    state: ChatState = {
        "messages": [{"role": "user", "content": user_text}],
        "conversation_id": conversation_id or str(uuid.uuid4()),
        "skill_hint": skill_hint,
        "response": "",
    }
    try:
        result = graph.invoke(state)
        return result.get("response", "")
    except Exception as e:
        logger.error(f"[TextChat] graph invocation failed: {e}")
        return ""
