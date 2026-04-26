"""Responses WebSocket transport helpers.

The implementation imports ``websocket-client`` lazily so unit tests and HTTP
fallback still work when the optional package is not installed.
"""

from __future__ import annotations

import json
from typing import Any

from gpt_responses import parse_responses_sse


CODEX_RESPONSES_WS_BETA = "responses_websockets=2026-02-06"


def responses_ws_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/responses"):
        base = f"{base}/responses"
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :]
    if base.startswith("wss://") or base.startswith("ws://"):
        return base
    return "wss://" + base


def ws_headers(headers: dict[str, str]) -> list[str]:
    merged = dict(headers)
    merged["x-codex-beta-features"] = CODEX_RESPONSES_WS_BETA
    return [f"{k}: {v}" for k, v in merged.items()]


def build_ws_create_message(payload: dict[str, Any], *, generate: bool = True) -> str:
    body = dict(payload)
    body["generate"] = bool(generate)
    return json.dumps({"type": "response.create", **body}, ensure_ascii=False)


def stream_responses_ws(url: str, headers: dict[str, str], payload: dict[str, Any], *, connect_timeout: int = 10):
    try:
        import websocket  # type: ignore
    except Exception as exc:
        raise RuntimeError("websocket-client is not installed") from exc
    ws = websocket.create_connection(url, header=ws_headers(headers), timeout=connect_timeout)
    try:
        ws.send(build_ws_create_message(payload, generate=payload.get("generate", True)))

        def _lines():
            while True:
                raw = ws.recv()
                if not raw:
                    break
                try:
                    evt = json.loads(raw)
                except Exception:
                    continue
                if evt.get("type") in ("response.completed", "response.failed", "error"):
                    yield "data: " + json.dumps(evt, ensure_ascii=False)
                    break
                yield "data: " + json.dumps(evt, ensure_ascii=False)

        return (yield from parse_responses_sse(_lines())) or []
    finally:
        try:
            ws.close()
        except Exception:
            pass

