import logging
import os
import threading
import uuid

import httpx

logger = logging.getLogger(__name__)

A2A_DEFAULT_URL = os.environ.get("A2A_URL", "http://automaker-server:3008/a2a")
A2A_DEFAULT_API_KEY = os.environ.get("A2A_API_KEY", "")


def _headers(api_key: str) -> dict:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def a2a_send(
    text: str,
    conversation_id: str,
    skill_hint: str = "chat",
    a2a_url: str = A2A_DEFAULT_URL,
    api_key: str = A2A_DEFAULT_API_KEY,
) -> str:
    """POST to the A2A endpoint and return the response text."""
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
            },
            "contextId": conversation_id,
            "metadata": {"skillHint": skill_hint},
        },
    }
    try:
        r = httpx.post(
            a2a_url,
            headers=_headers(api_key),
            json=payload,
            timeout=60.0,
        )
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            logger.error(f"[A2A] error: {body['error']}")
            return "Sorry, I couldn't process that."
        artifacts = body.get("result", {}).get("artifacts", [])
        parts = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("kind") == "text" or part.get("type") == "text":
                    parts.append(part.get("text", ""))
        return " ".join(parts).strip() or "Sorry, I couldn't process that."
    except Exception as e:
        logger.error(f"[A2A] request error: {e}")
        return "Sorry, I couldn't process that."


def stream_a2a_tokens(
    text: str,
    conversation_id: str,
    cancel: threading.Event,
    skill_hint: str = "chat",
    a2a_url: str = A2A_DEFAULT_URL,
    api_key: str = A2A_DEFAULT_API_KEY,
):
    """Call A2A and yield the response text for TTS processing."""
    if cancel.is_set():
        return
    response = a2a_send(text, conversation_id, skill_hint, a2a_url, api_key)
    if not cancel.is_set() and response:
        yield response
