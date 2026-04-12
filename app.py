#!/usr/bin/env python3
"""
Ava — standalone conversational protoAgent

Text chat UI (Gradio) + A2A JSON-RPC surface.
Opus 4.6 via LiteLLM gateway. No tools. SOUL.md personality.
Voice runs separately via the protovoice container.
"""

import logging
import os
import uuid

import gradio as gr

from chat.backend import chat as ava_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "7866"))

# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------
_SLASH_HELP = (
    "**Available commands:**\n"
    "- `/clear` — clear chat history\n"
    "- `/help` — show this message"
)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    _conversation_id = str(uuid.uuid4())

    def handle_chat(message: str, history: list[dict]) -> tuple[str, list[dict]]:
        nonlocal _conversation_id
        message = message.strip()
        if not message:
            return "", history

        if message.startswith("/"):
            cmd = message.split()[0].lower()
            if cmd == "/clear":
                _conversation_id = str(uuid.uuid4())
                return "", []
            elif cmd == "/help":
                return "", history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": _SLASH_HELP},
                ]
            else:
                return "", history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"Unknown command `{cmd}`. Type `/help`."},
                ]

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        response = ava_chat(message=message, history=chat_history)
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

    with gr.Blocks(title="Ava") as demo:
        gr.Markdown("## Ava")

        chatbot = gr.Chatbot(label="Ava", type="messages", height=600)
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Type a message…",
                show_label=False,
                scale=9,
                max_lines=3,
                autofocus=True,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        send_btn.click(fn=handle_chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(fn=handle_chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])

    return demo


# ---------------------------------------------------------------------------
# A2A JSON-RPC 2.0 handler
# ---------------------------------------------------------------------------
def _build_agent_card(host: str) -> dict:
    return {
        "name": "ava",
        "description": "Conversational protoAgent — thoughtful chat partner, suggests delegations when action is needed.",
        "url": f"http://{host}",
        "version": "0.2.0",
        "provider": {"organization": "protoLabsAI", "url": "https://github.com/protoLabsAI"},
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/markdown"],
        "skills": [{"id": "chat", "name": "Chat", "description": "Free-form conversation."}],
        "securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
        "security": [],
    }


def _mount_a2a_routes(app):
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.get("/.well-known/agent.json")
    @app.get("/.well-known/agent-card.json")
    async def agent_card(request: Request):
        host = request.headers.get("host", f"ava-agent:{PORT}")
        return JSONResponse(content=_build_agent_card(host), headers={"Cache-Control": "public, max-age=60"})

    @app.post("/a2a")
    async def a2a_handler(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})

        rpc_id = body.get("id")
        if body.get("method") != "message/send":
            return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Method not found: {body.get('method')}"}})

        parts = body.get("params", {}).get("message", {}).get("parts", [])
        user_text = "\n".join(p.get("text", "") for p in parts if (p.get("kind") or p.get("type")) == "text").strip()
        if not user_text:
            return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "No text part"}})

        context_id = body.get("params", {}).get("contextId", str(uuid.uuid4()))
        logger.info(f'A2A: "{user_text[:80]}"')

        response_text = ava_chat(message=user_text)
        return JSONResponse(content={
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "id": str(uuid.uuid4()), "contextId": context_id,
                "status": {"state": "completed"},
                "artifacts": [{"artifactId": str(uuid.uuid4()), "parts": [{"kind": "text", "text": response_text}]}],
            },
        })

    logger.info("A2A routes mounted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    demo = build_ui()
    _mount_a2a_routes(demo.app)
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False, show_error=True)


if __name__ == "__main__":
    main()
