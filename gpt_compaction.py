"""Structured compaction helpers for GPT-native sessions."""

from __future__ import annotations

import json
from typing import Any


def should_compact(estimated_chars: int, context_window: int, threshold: float = 0.85) -> bool:
    # GA historically uses roughly 3 chars/token for budget checks.
    return estimated_chars > int(context_window * 3 * threshold)


def build_structured_compaction(history_info: list[str], checkpoint: dict[str, Any] | None = None) -> str:
    checkpoint = checkpoint or {}
    recent = history_info[-40:]
    body = {
        "current_goal": checkpoint.get("key_info", ""),
        "related_sop": checkpoint.get("related_sop", ""),
        "completed_or_observed": recent[-20:],
        "open_blockers": [],
        "next_step": "Continue from the latest checkpoint and use tools to verify state.",
    }
    return "### Context Compaction Handoff\n" + json.dumps(body, ensure_ascii=False, indent=2)

