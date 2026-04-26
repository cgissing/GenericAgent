"""Runtime memory adapter for GPT-native GenericAgent sessions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TurnMemoryEntry:
    turn_id: int
    user_intent: str = ""
    assistant_action: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results_summary: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_step: str = ""
    source: str = "runtime"

    def one_line(self, max_len: int = 120) -> str:
        parts = []
        if self.assistant_action:
            parts.append(self.assistant_action)
        if self.tool_calls:
            names = ", ".join(t.get("tool_name", "") for t in self.tool_calls[:3])
            parts.append(f"tools={names}")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        if self.artifacts:
            parts.append(f"artifacts={', '.join(self.artifacts[:2])}")
        text = " | ".join(p for p in parts if p) or "GPT turn completed"
        return text[: max_len - 3] + "..." if len(text) > max_len else text


@dataclass
class MemoryEvent:
    event_type: str
    evidence_refs: list[str] = field(default_factory=list)
    created_by_model: bool = False
    model: str | None = None
    cost_class: str = "deterministic"
    payload: dict[str, Any] = field(default_factory=dict)


def build_turn_memory_entry(response, tool_calls, tool_results, turn: int, next_prompt: str = "") -> TurnMemoryEntry:
    content = getattr(response, "content", "") or ""
    content = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", content).strip()
    action = _compact_text(content) or _tool_action(tool_calls)
    summaries = []
    artifacts: list[str] = []
    errors: list[str] = []
    for result in tool_results or []:
        summary, found_artifacts, found_errors = summarize_tool_result(result.get("content", ""))
        if summary:
            summaries.append(summary)
        artifacts.extend(found_artifacts)
        errors.extend(found_errors)
    return TurnMemoryEntry(
        turn_id=turn,
        assistant_action=action,
        tool_calls=[_clean_tool_call(tc) for tc in (tool_calls or []) if tc.get("tool_name") != "no_tool"],
        tool_results_summary=summaries,
        artifacts=_dedupe(artifacts),
        errors=_dedupe(errors),
        next_step=_compact_text(next_prompt, 160),
        source="runtime",
    )


def summarize_tool_result(content: Any) -> tuple[str, list[str], list[str]]:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    artifacts = _extract_artifacts(content)
    errors = _extract_errors(content)
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            status = data.get("status")
            exit_code = data.get("exit_code")
            msg = data.get("msg") or data.get("stderr") or data.get("stdout") or data.get("content") or ""
            head = f"status={status}" if status else "result"
            if exit_code is not None:
                head += f", exit={exit_code}"
            return f"{head}: {_compact_text(str(msg), 160)}", artifacts, errors
    except Exception:
        pass
    return _compact_text(content, 180), artifacts, errors


def entry_to_history_line(entry: TurnMemoryEntry) -> str:
    return f"[Agent] {entry.one_line()}"


def append_l4_memory_event(memory_dir: str | os.PathLike[str], event: MemoryEvent) -> str:
    root = Path(memory_dir) / "L4_raw_sessions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"memory_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False, default=str) + "\n")
    return str(path)


def _compact_text(text: str, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len // 2] + " ... " + text[-max_len // 2 :]


def _tool_action(tool_calls) -> str:
    calls = [tc for tc in (tool_calls or []) if tc.get("tool_name") != "no_tool"]
    if not calls:
        return "direct response"
    return "called " + ", ".join(c.get("tool_name", "") for c in calls[:4])


def _clean_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    args = {k: v for k, v in (tc.get("args") or {}).items() if not str(k).startswith("_")}
    return {"tool_name": tc.get("tool_name", ""), "args": args, "id": tc.get("id", "")}


def _extract_artifacts(text: str) -> list[str]:
    paths = re.findall(r"([A-Za-z]:\\[^\s\"'<>|]+|/[^\s\"'<>|]+)", text)
    urls = re.findall(r"https?://[^\s\"'<>]+", text)
    return _dedupe(paths + urls)[:8]


def _extract_errors(text: str) -> list[str]:
    errors = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ("error", "exception", "traceback", "failed", "exit code: 1", '"status": "error"')):
            errors.append(_compact_text(line, 180))
    return _dedupe(errors)[:6]


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out

