"""Codex-compatible authentication for GPT-native sessions."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENAI_API_BASE = "https://api.openai.com/v1"
CHATGPT_CODEX_BASE = "https://chatgpt.com/backend-api/codex"


class GPTAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPTAuth:
    mode: str
    base_url: str
    headers: dict[str, str]
    source: str
    account_id: str | None = None
    fedramp: bool = False

    @property
    def redacted(self) -> str:
        auth = self.headers.get("Authorization", "")
        tail = auth[-6:] if len(auth) >= 6 else ""
        return f"{self.mode}:{self.source}:***{tail}"


def codex_home_from_env() -> Path:
    raw = os.environ.get("CODEX_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".codex"


def codex_auth_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or {}
    raw = cfg.get("codex_auth_path")
    if raw:
        return Path(raw).expanduser()
    return codex_home_from_env() / "auth.json"


def load_codex_auth(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    path = codex_auth_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GPTAuthError(f"Failed to read Codex auth file at {path}: {exc}") from exc


def _b64url_json(segment: str) -> dict[str, Any]:
    try:
        padded = segment + "=" * (-len(segment) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def decode_id_token(id_token: str | None) -> dict[str, Any]:
    if not id_token or id_token.count(".") < 2:
        return {}
    return _b64url_json(id_token.split(".")[1])


def _token_from_cfg_or_env(cfg: dict[str, Any]) -> tuple[str | None, str]:
    for key in ("apikey", "api_key", "OPENAI_API_KEY"):
        val = cfg.get(key)
        if val:
            return str(val), f"config.{key}"
    val = os.environ.get("OPENAI_API_KEY")
    if val:
        return val, "env.OPENAI_API_KEY"
    return None, ""


def _codex_api_key(auth: dict[str, Any]) -> str | None:
    val = auth.get("OPENAI_API_KEY")
    return str(val) if val else None


def _codex_tokens(auth: dict[str, Any]) -> dict[str, Any]:
    tokens = auth.get("tokens") or auth.get("chatgpt_tokens") or {}
    return tokens if isinstance(tokens, dict) else {}


def resolve_gpt_auth(cfg: dict[str, Any] | None = None) -> GPTAuth:
    """Resolve GPT auth using Codex-compatible precedence.

    ``auth='codex'`` reuses Codex login state. If ChatGPT OAuth tokens are
    present it uses the ChatGPT Codex backend; otherwise it falls back to an API
    key from Codex auth, env, or GA config. ``auth='api_key'`` forces API key
    mode and never uses ChatGPT tokens.
    """
    cfg = cfg or {}
    requested = str(cfg.get("auth", cfg.get("auth_mode", "codex"))).strip().lower()
    auth_file = load_codex_auth(cfg)

    if requested in ("codex", "chatgpt", "chatgpt_oauth", "oauth"):
        tokens = _codex_tokens(auth_file)
        access_token = tokens.get("access_token")
        if access_token:
            claims = decode_id_token(tokens.get("id_token"))
            account_id = tokens.get("account_id") or claims.get("chatgpt_account_id")
            fedramp = bool(claims.get("chatgpt_account_is_fedramp"))
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            if account_id:
                headers["ChatGPT-Account-ID"] = str(account_id)
            if fedramp:
                headers["X-OpenAI-Fedramp"] = "true"
            return GPTAuth(
                mode="chatgpt_oauth",
                base_url=str(cfg.get("apibase") or cfg.get("base_url") or CHATGPT_CODEX_BASE).rstrip("/"),
                headers=headers,
                source=str(codex_auth_path(cfg)),
                account_id=str(account_id) if account_id else None,
                fedramp=fedramp,
            )

        api_key = _codex_api_key(auth_file)
        if api_key:
            return _api_key_auth(api_key, str(codex_auth_path(cfg)), cfg)

    api_key, source = _token_from_cfg_or_env(cfg)
    if api_key:
        return _api_key_auth(api_key, source, cfg)

    if requested in ("api", "api_key", "apikey"):
        raise GPTAuthError("OpenAI API key not found. Set OPENAI_API_KEY or configure apikey.")
    raise GPTAuthError(
        "Codex login state not found. Run `codex login` for ChatGPT OAuth, "
        "or set OPENAI_API_KEY / apikey for API-key mode."
    )


def _api_key_auth(api_key: str, source: str, cfg: dict[str, Any]) -> GPTAuth:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    return GPTAuth(
        mode="api_key",
        base_url=str(cfg.get("apibase") or cfg.get("base_url") or OPENAI_API_BASE).rstrip("/"),
        headers=headers,
        source=source,
    )

