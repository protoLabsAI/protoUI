"""Context assembly — XML-framed memory injection.

Mirrors protoWorkstacean's assembleContext(). Produces the same structure
so SOUL.md's context awareness instructions work identically regardless
of whether the user reaches Ava through Discord, the web UI, or A2A.
"""

from datetime import datetime, timezone

MEMORY_PREAMBLE = (
    "The following facts were retrieved from your memory about this user. "
    "Use them as background context if relevant — do NOT repeat them back "
    "to the user or reference them unless the user's message specifically "
    "asks about something they relate to. Focus your response on what the "
    "user is actually saying below."
)


def assemble_context(
    recalled_memory: str | None,
    recent_turns: list[dict],
    current_message: str,
) -> str:
    """Build XML-framed context envelope.

    Args:
        recalled_memory: Graphiti fact block (or None/empty).
        recent_turns: List of {"role": "user"|"assistant", "content": "..."} dicts.
        current_message: The user's current input.

    Returns:
        Framed string ready to use as the user message content.
    """
    parts = []

    if recalled_memory and recalled_memory.strip():
        parts.append(
            f"<recalled_memory>\n{MEMORY_PREAMBLE}\n\n{recalled_memory.strip()}\n</recalled_memory>"
        )

    if recent_turns:
        lines = []
        for turn in recent_turns:
            role = "User" if turn.get("role") == "user" else "Assistant"
            ts = turn.get("timestamp", "")
            channel = turn.get("channel", "ava")
            if ts:
                prefix = f"[{ts}[{channel}]]"
            else:
                prefix = f"[{channel}]"
            lines.append(f"{prefix} {role}: {turn['content']}")
        if lines:
            parts.append(
                f"<recent_conversation>\n" + "\n".join(lines) + "\n</recent_conversation>"
            )

    parts.append(f"<current_message>\n{current_message}\n</current_message>")

    return "\n\n".join(parts)
