"""Graphiti client — shared temporal knowledge graph for cross-channel memory.

Same API as protoWorkstacean's lib/memory/graphiti-client.ts. Reads user
facts before each response, writes completed episodes after.
"""

import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

GRAPHITI_URL = os.environ.get("GRAPHITI_URL", "http://graphiti:8000")
DEFAULT_USER_GROUP_ID = os.environ.get("DEFAULT_USER_GROUP_ID", "user_josh")
AGENT_NAME = "ava"


def get_context_block(
    message: str,
    group_id: str = "",
    agent_group_id: str = "",
) -> str:
    """Retrieve relevant facts from Graphiti for the user and agent-scoped groups.

    Returns formatted fact block or empty string.
    """
    gid = group_id or DEFAULT_USER_GROUP_ID
    agid = agent_group_id or f"agent_{AGENT_NAME}__{gid}"

    parts = []
    for g in [gid, agid]:
        block = _fetch_facts(g, message)
        if block:
            parts.append(block)

    return "\n".join(parts)


def add_episode(
    user_message: str,
    agent_message: str,
    group_id: str = "",
    platform: str = "ava-voice",
    channel_id: str = "",
) -> None:
    """Write a completed conversation turn to Graphiti (fire-and-forget)."""
    gid = group_id or DEFAULT_USER_GROUP_ID
    agid = f"agent_{AGENT_NAME}__{gid}"
    now = datetime.now(timezone.utc).isoformat()

    messages = [
        {
            "content": user_message,
            "role_type": "user",
            "role": gid.removeprefix("user_"),
            "timestamp": now,
            "source_description": f"{platform} {channel_id}".strip(),
        },
        {
            "content": agent_message,
            "role_type": "assistant",
            "role": AGENT_NAME,
            "timestamp": now,
            "source_description": f"{platform} {channel_id}".strip(),
        },
    ]

    for g in [gid, agid]:
        try:
            r = httpx.post(
                f"{GRAPHITI_URL}/messages",
                json={"group_id": g, "messages": messages},
                timeout=8.0,
            )
            if r.status_code not in (200, 202):
                logger.warning(f"Graphiti addEpisode ({g}): {r.status_code}")
        except Exception as e:
            logger.debug(f"Graphiti addEpisode ({g}) error: {e}")


def _fetch_facts(group_id: str, message: str) -> str:
    try:
        r = httpx.post(
            f"{GRAPHITI_URL}/get-memory",
            json={
                "group_id": group_id,
                "center_node_uuid": None,
                "messages": [
                    {"content": message, "role_type": "user", "role": ""},
                ],
                "max_facts": 15,
            },
            timeout=8.0,
        )
        r.raise_for_status()
        facts = r.json().get("facts", [])

        now = datetime.now(timezone.utc)
        active = []
        for f in facts:
            if f.get("invalid_at"):
                try:
                    if datetime.fromisoformat(f["invalid_at"]) < now:
                        continue
                except (ValueError, TypeError):
                    pass
            if f.get("expired_at"):
                try:
                    if datetime.fromisoformat(f["expired_at"]) < now:
                        continue
                except (ValueError, TypeError):
                    pass
            active.append(f["fact"])

        if not active:
            return ""

        return f"[User context — {group_id}]\n" + "\n".join(
            f"- {fact}" for fact in active
        )
    except Exception as e:
        logger.debug(f"Graphiti getContextBlock ({group_id}) error: {e}")
        return ""
