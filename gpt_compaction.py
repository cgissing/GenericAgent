"""Structured compaction helpers for GPT-native sessions."""

from __future__ import annotations

import json
from typing import Any


def should_compact(estimated_chars: int, context_window: int, threshold: float = 0.85) -> bool:
    # GA historically uses roughly 3 chars/token for budget checks.
    return estimated_chars > int(context_window * 3 * threshold)


def build_structured_compaction(
    history_info: list[str],
    checkpoint: dict[str, Any] | None = None,
    turn_entries: list[Any] | None = None,
) -> str:
    checkpoint = checkpoint or {}
    entries = list(turn_entries or [])[-20:]
    recent = history_info[-40:]
    recent_turns = [_turn_entry_summary(entry) for entry in entries]
    key_artifacts = _dedupe([item for entry in entries for item in getattr(entry, "artifacts", [])])[:20]
    open_blockers = _dedupe([item for entry in entries for item in getattr(entry, "errors", [])])[:10]
    completed = [getattr(entry, "assistant_action", "") for entry in entries if getattr(entry, "assistant_action", "")]
    if not completed:
        completed = recent[-20:]
    next_step = _latest_nonempty(getattr(entry, "next_step", "") for entry in reversed(entries))
    body = {
        "current_goal": checkpoint.get("key_info", ""),
        "related_sop": checkpoint.get("related_sop", ""),
        "completed_or_observed": completed[-20:],
        "recent_turns": recent_turns,
        "key_artifacts": key_artifacts,
        "open_blockers": open_blockers,
        "next_step": next_step or "Continue from the latest checkpoint and use tools to verify state.",
        "compaction_source": "runtime_turn_memory" if entries else "history_anchor",
    }
    return "### Context Compaction Handoff\n" + json.dumps(body, ensure_ascii=False, indent=2)


def compaction_payload_from_handoff(handoff: str) -> dict[str, Any]:
    try:
        return json.loads(handoff.split("\n", 1)[1])
    except Exception:
        return {"raw": handoff}


def _turn_entry_summary(entry: Any) -> dict[str, Any]:
    calls = getattr(entry, "tool_calls", []) or []
    return {
        "turn_id": getattr(entry, "turn_id", None),
        "action": getattr(entry, "assistant_action", ""),
        "tools": [call.get("tool_name", "") for call in calls if call.get("tool_name")],
        "tool_results": list(getattr(entry, "tool_results_summary", []) or [])[:6],
        "artifacts": list(getattr(entry, "artifacts", []) or [])[:8],
        "errors": list(getattr(entry, "errors", []) or [])[:6],
        "next_step": getattr(entry, "next_step", ""),
    }


def _latest_nonempty(values) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out

