"""Codex-style Responses API helpers for GPT-native sessions."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Iterable


def make_conversation_id(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or {}
    return str(cfg.get("conversation_id") or cfg.get("session_id") or uuid.uuid4())


def installation_id() -> str:
    for key in ("CODEX_INSTALLATION_ID", "GA_CODEX_INSTALLATION_ID"):
        if os.environ.get(key):
            return os.environ[key]
    return f"ga-{uuid.getnode():012x}"


def prepare_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    prepared = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            item = {"type": "function"}
            item.update(tool["function"])
            prepared.append(item)
        else:
            prepared.append(tool)
    return prepared


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    for msg in messages:
        if msg.get("type") in ("function_call", "function_call_output", "message"):
            result.append(msg)
            continue
        role = str(msg.get("role", "user")).lower()
        if role == "tool":
            call_id = msg.get("tool_call_id") or (pending_call_ids.pop(0) if pending_call_ids else f"call_{uuid.uuid4().hex[:8]}")
            result.append({"type": "function_call_output", "call_id": call_id, "output": str(msg.get("content", ""))})
            continue
        if role == "system":
            role = "developer"
        if role not in ("developer", "user", "assistant"):
            role = "user"
        content = msg.get("content", "")
        parts = _content_parts_for_responses(content, role)
        result.append({"role": role, "content": parts})
        pending_call_ids = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            pending_call_ids.append(call_id)
            result.append({
                "type": "function_call",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
            })
    return result


def _content_parts_for_responses(content: Any, role: str) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else [{"type": text_type, "text": ""}]
    if not isinstance(content, list):
        return [{"type": text_type, "text": str(content)}]
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        ptype = item.get("type")
        if ptype == "text" and item.get("text") is not None:
            parts.append({"type": text_type, "text": str(item.get("text", ""))})
        elif ptype == "image_url" and role != "assistant":
            url = (item.get("image_url") or {}).get("url")
            if url:
                parts.append({"type": "input_image", "image_url": url})
        elif ptype == "image" and role != "assistant":
            src = item.get("source") or {}
            if src.get("type") == "base64" and src.get("data"):
                media = src.get("media_type", "image/png")
                parts.append({"type": "input_image", "image_url": f"data:{media};base64,{src['data']}"})
        elif ptype == "tool_result":
            text = item.get("content", "")
            parts.append({"type": text_type, "text": str(text)})
    return parts or [{"type": text_type, "text": ""}]


def build_responses_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    instructions: str,
    tools: list[dict[str, Any]] | None,
    reasoning_effort: str | None,
    verbosity: str | None,
    stream: bool,
    prompt_cache_key: str,
    parallel_tool_calls: bool,
    store: bool = False,
    max_output_tokens: int | None = None,
    previous_response_id: str | None = None,
    generate: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": to_responses_input(messages),
        "tool_choice": "auto",
        "parallel_tool_calls": bool(parallel_tool_calls),
        "store": store,
        "stream": bool(stream),
        "prompt_cache_key": prompt_cache_key,
        "client_metadata": {"x-codex-installation-id": installation_id()},
    }
    prepared_tools = prepare_responses_tools(tools)
    if prepared_tools:
        payload["tools"] = prepared_tools
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
        payload["include"] = ["reasoning.encrypted_content"]
    if verbosity:
        payload["text"] = {"verbosity": verbosity}
    if max_output_tokens:
        payload["max_output_tokens"] = max_output_tokens
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if generate is not None:
        payload["generate"] = bool(generate)
    return payload


def parse_responses_sse(lines: Iterable[str | bytes]):
    """Yield text chunks and return GA content blocks from Responses SSE events."""
    text = ""
    function_calls: dict[int, dict[str, str]] = {}
    current_idx = 0
    reasoning = ""
    completed_usage = None
    for line in lines:
        if not line:
            continue
        line = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        if not line.startswith("data:"):
            continue
        raw = line[5:].lstrip()
        if raw == "[DONE]":
            break
        try:
            evt = json.loads(raw)
        except Exception:
            continue
        etype = evt.get("type", "")
        if etype == "response.output_text.delta":
            delta = evt.get("delta", "")
            if delta:
                text += delta
                yield delta
        elif etype == "response.output_text.done" and not text:
            done_text = evt.get("text", "")
            if done_text:
                text += done_text
                yield done_text
        elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            reasoning += evt.get("delta", "")
        elif etype == "response.output_item.added":
            item = evt.get("item", {})
            if item.get("type") == "function_call":
                current_idx = int(evt.get("output_index", len(function_calls)))
                function_calls[current_idx] = {
                    "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": item.get("name", ""),
                    "args": item.get("arguments", "") or "",
                    "default_key": "",
                }
            elif item.get("type") in ("custom_tool_call", "shell_call", "apply_patch_call"):
                current_idx = int(evt.get("output_index", len(function_calls)))
                name, args, default_key = _tool_name_and_args_from_item(item)
                function_calls[current_idx] = {
                    "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "args": args,
                    "default_key": default_key,
                }
        elif etype == "response.function_call_arguments.delta":
            idx = int(evt.get("output_index", current_idx))
            function_calls.setdefault(idx, {"id": f"call_{idx}", "name": "", "args": ""})
            function_calls[idx]["args"] += evt.get("delta", "")
        elif etype == "response.function_call_arguments.done":
            idx = int(evt.get("output_index", current_idx))
            if idx in function_calls and evt.get("arguments") is not None:
                function_calls[idx]["args"] = evt.get("arguments") or ""
        elif etype == "response.completed":
            completed_usage = (evt.get("response") or {}).get("usage") or {}
        elif etype in ("error", "response.failed"):
            err = evt.get("error") or (evt.get("response") or {}).get("error") or {}
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if msg:
                chunk = f"!!!Error: {msg}"
                text += chunk
                yield chunk
            break
    blocks: list[dict[str, Any]] = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if text:
        blocks.append({"type": "text", "text": text})
    for idx in sorted(function_calls):
        fc = function_calls[idx]
        for suffix, args in enumerate(_parse_tool_args(fc["args"], default_key=fc.get("default_key") or "")):
            call_id = fc["id"] if suffix == 0 else f"{fc['id']}_{suffix}"
            blocks.append({"type": "tool_use", "id": call_id, "name": fc["name"], "input": args})
    if completed_usage:
        blocks.append({"type": "usage", "usage": completed_usage})
    return blocks


def parse_responses_json(data: dict[str, Any]):
    text = ""
    blocks: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    text += part["text"]
                    yield part["text"]
        elif item.get("type") == "function_call":
            for args in _parse_tool_args(item.get("arguments", "")):
                blocks.append({
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name", ""),
                    "input": args,
                })
        elif item.get("type") in ("custom_tool_call", "shell_call", "apply_patch_call"):
            name, raw_args, default_key = _tool_name_and_args_from_item(item)
            for args in _parse_tool_args(raw_args, default_key=default_key):
                blocks.append({
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or "",
                    "name": name,
                    "input": args,
                })
    if text:
        blocks.insert(0, {"type": "text", "text": text})
    if data.get("usage"):
        blocks.append({"type": "usage", "usage": data["usage"]})
    return blocks


def _tool_name_and_args_from_item(item: dict[str, Any]) -> tuple[str, str, str]:
    item_type = item.get("type")
    if item_type == "shell_call":
        action = item.get("action") or {}
        command = action.get("command") if isinstance(action, dict) else item.get("command")
        return "shell_command", json.dumps({"command": command or ""}), ""
    if item_type == "apply_patch_call":
        return "apply_patch", str(item.get("input") or item.get("patch") or ""), "patch"
    name = item.get("name") or item.get("tool_name") or ""
    raw = item.get("input")
    if raw is None:
        raw = item.get("arguments", "")
    default_key = "patch" if name == "apply_patch" else "_raw"
    return str(name), str(raw or ""), default_key


def _parse_tool_args(raw: str, *, default_key: str = "") -> list[dict[str, Any]]:
    if not raw:
        return [{}]
    try:
        parsed = json.loads(raw)
        return [parsed if isinstance(parsed, dict) else {"value": parsed}]
    except Exception:
        pass
    parts = re.split(r"(?<=\})(?=\{)", raw)
    if len(parts) > 1:
        out = []
        for part in parts:
            try:
                parsed = json.loads(part)
                out.append(parsed if isinstance(parsed, dict) else {"value": parsed})
            except Exception:
                return [{default_key or "_raw": raw}]
        return out
    return [{default_key or "_raw": raw}]
