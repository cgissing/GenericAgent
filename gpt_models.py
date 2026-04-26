"""Model profile helpers for GPT/Codex-native GenericAgent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GPTModelProfile:
    name: str
    context_win: int = 128000
    default_reasoning_effort: str = "medium"
    supported_reasoning_efforts: tuple[str, ...] = ("low", "medium", "high")
    default_verbosity: str = "low"
    prefer_websocket: bool = False
    supports_parallel_tool_calls: bool = True
    apply_patch_tool_type: str = "freeform"
    shell_type: str = "shell_command"
    extra: dict[str, Any] = field(default_factory=dict)


GPT_MODEL_PROFILES: dict[str, GPTModelProfile] = {
    "gpt-5.5": GPTModelProfile(
        name="gpt-5.5",
        context_win=272000,
        default_reasoning_effort="medium",
        supported_reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_verbosity="low",
        prefer_websocket=True,
        supports_parallel_tool_calls=True,
        apply_patch_tool_type="freeform",
        shell_type="shell_command",
        extra={
            "input_modalities": ["text", "image"],
            "web_search_tool_type": "text_and_image",
            "supports_image_detail_original": True,
        },
    ),
    "gpt-5.4": GPTModelProfile(
        name="gpt-5.4",
        context_win=272000,
        default_reasoning_effort="medium",
        supported_reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_verbosity="low",
        prefer_websocket=True,
    ),
    "gpt-5.4-mini": GPTModelProfile(
        name="gpt-5.4-mini",
        context_win=272000,
        default_reasoning_effort="low",
        supported_reasoning_efforts=("low", "medium", "high"),
        default_verbosity="low",
        prefer_websocket=False,
    ),
    "gpt-5.3-codex": GPTModelProfile(
        name="gpt-5.3-codex",
        context_win=272000,
        default_reasoning_effort="medium",
        supported_reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_verbosity="low",
        prefer_websocket=True,
    ),
}


def get_gpt_model_profile(model: str | None) -> GPTModelProfile:
    """Return a conservative GPT profile for ``model``."""
    key = (model or "gpt-5.5").strip().lower()
    if key in GPT_MODEL_PROFILES:
        return GPT_MODEL_PROFILES[key]
    for known, profile in GPT_MODEL_PROFILES.items():
        if key.startswith(known):
            return profile
    return GPTModelProfile(name=model or "gpt-5.5")


def normalize_reasoning_effort(effort: str | None, profile: GPTModelProfile) -> str | None:
    if not effort:
        return profile.default_reasoning_effort
    effort = str(effort).strip().lower()
    if effort == "max":
        effort = "xhigh"
    if effort in profile.supported_reasoning_efforts:
        return effort
    return profile.default_reasoning_effort

