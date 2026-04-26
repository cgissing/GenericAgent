"""Low-cost model routing for memory maintenance tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_POLICY = {
    "ttl_hours": 24,
    "default_main_model": "gpt-5.5",
    "updated_at": None,
    "tasks": {
        "turn_summary": {"mode": "deterministic", "fallback_model": "gpt-5.4-mini"},
        "checkpoint_compaction": {"model": "gpt-5.4-mini", "reasoning_effort": "low"},
        "session_archive": {"model": "gpt-5.4-mini", "reasoning_effort": "low"},
        "long_term_extract": {"model": "gpt-5.4-mini", "reasoning_effort": "medium", "fallback_model": "gpt-5.5"},
        "sop_consolidation": {"model": "gpt-5.5", "reasoning_effort": "medium", "only_when": "conflict_or_high_value"},
        "model_policy_evaluation": {"mode": "deterministic"},
    },
}


@dataclass
class MemoryModelChoice:
    mode: str = "model"
    model: str | None = None
    reasoning_effort: str | None = None
    fallback_model: str | None = None


class MemoryModelRouter:
    def __init__(self, memory_dir: str | os.PathLike[str], main_model: str = "gpt-5.5"):
        self.memory_dir = Path(memory_dir)
        self.policy_path = self.memory_dir / "model_policy.json"
        self.main_model = main_model
        self.policy = self.load_policy()

    def load_policy(self) -> dict[str, Any]:
        if self.policy_path.exists():
            try:
                data = json.loads(self.policy_path.read_text(encoding="utf-8"))
                return _merge_policy(data)
            except Exception:
                pass
        data = dict(DEFAULT_POLICY)
        data["default_main_model"] = self.main_model
        return _merge_policy(data)

    def save_policy(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        data = dict(self.policy)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.policy_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.policy = data

    def is_stale(self) -> bool:
        updated = self.policy.get("updated_at")
        if not updated:
            return True
        try:
            ts = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        except Exception:
            return True
        ttl = int(self.policy.get("ttl_hours", 24))
        return datetime.now(timezone.utc) - ts > timedelta(hours=ttl)

    def choose(self, task_type: str, *, high_value: bool = False) -> MemoryModelChoice:
        task = (self.policy.get("tasks") or {}).get(task_type) or {}
        if task.get("mode") == "deterministic":
            return MemoryModelChoice(mode="deterministic", fallback_model=task.get("fallback_model"))
        model = task.get("model")
        if model == self.main_model and not high_value and task.get("only_when"):
            return MemoryModelChoice(mode="deterministic", fallback_model=model)
        return MemoryModelChoice(
            mode="model",
            model=model,
            reasoning_effort=task.get("reasoning_effort"),
            fallback_model=task.get("fallback_model"),
        )

    def evaluate_available_models(self, available_models: list[str] | None = None) -> dict[str, Any]:
        """Pick a conservative policy from currently configured model names.

        This is intentionally deterministic and cheap. Live canary evaluation can
        be layered on top by callers, but ordinary turn summaries must not call a
        model just to choose a model.
        """
        available = [m for m in (available_models or []) if m]
        cheap = _first_matching(available, ("mini", "flash", "haiku", "glm", "kimi")) or "gpt-5.4-mini"
        mid = _first_matching(available, ("gpt-5.4", "gpt-5.3", "sonnet", "glm")) or cheap
        main = _first_matching(available, ("gpt-5.5",)) or self.main_model
        self.policy = _merge_policy({
            **DEFAULT_POLICY,
            "default_main_model": main,
            "tasks": {
                **DEFAULT_POLICY["tasks"],
                "turn_summary": {"mode": "deterministic", "fallback_model": cheap},
                "checkpoint_compaction": {"model": cheap, "reasoning_effort": "low"},
                "session_archive": {"model": cheap, "reasoning_effort": "low"},
                "long_term_extract": {"model": mid, "reasoning_effort": "medium", "fallback_model": main},
                "sop_consolidation": {"model": main, "reasoning_effort": "medium", "only_when": "conflict_or_high_value"},
            },
        })
        self.save_policy()
        return self.policy


def _merge_policy(data: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_POLICY))
    for k, v in (data or {}).items():
        if k == "tasks" and isinstance(v, dict):
            merged["tasks"].update(v)
        else:
            merged[k] = v
    return merged


def _first_matching(models: list[str], needles: tuple[str, ...]) -> str | None:
    for model in models:
        low = model.lower()
        if any(n in low for n in needles):
            return model
    return None

