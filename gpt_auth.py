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


def codex_config_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or {}
    raw = cfg.get("codex_config_path")
    if raw:
        return Path(raw).expanduser()
    return codex_home_from_env() / "config.toml"


def load_codex_auth(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    path = codex_auth_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GPTAuthError(f"Failed to read Codex auth file at {path}: {exc}") from exc


def load_codex_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    path = codex_config_path(cfg)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        try:
            import tomllib  # type: ignore
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore
        return tomllib.loads(text)
    except ModuleNotFoundError:
        return _parse_minimal_toml(text)
    except Exception as exc:
        raise GPTAuthError(f"Failed to read Codex config file at {path}: {exc}") from exc


def _parse_minimal_toml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current = data
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = data
            for part in line[1:-1].split("."):
                current = current.setdefault(part.strip(), {})
            continue
        if "=" not in line:
            continue
        key, value = [x.strip() for x in line.split("=", 1)]
        current[key] = _parse_minimal_toml_value(value)
    return data


def _parse_minimal_toml_value(value: str) -> Any:
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


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


def _configured_base_url(cfg: dict[str, Any], keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = cfg.get(key)
        if value:
            return str(value).rstrip("/")
    codex_base_url = _codex_config_base_url(cfg)
    if codex_base_url:
        return codex_base_url.rstrip("/")
    return default.rstrip("/")


def _codex_config_base_url(cfg: dict[str, Any]) -> str | None:
    config = load_codex_config(cfg)
    for key in ("base_url", "baseurl", "apibase", "api_base"):
        value = config.get(key)
        if value:
            return str(value)
    provider = str(cfg.get("codex_model_provider") or config.get("model_provider") or "").strip()
    providers = config.get("model_providers") or {}
    if provider and isinstance(providers, dict):
        provider_cfg = providers.get(provider)
        if isinstance(provider_cfg, dict):
            for key in ("base_url", "baseurl", "apibase", "api_base"):
                value = provider_cfg.get(key)
                if value:
                    return str(value)
    return None


def _chatgpt_codex_base_url(cfg: dict[str, Any]) -> str:
    return _configured_base_url(
        cfg,
        ("codex_base_url", "chatgpt_base_url", "apibase", "base_url", "baseurl"),
        CHATGPT_CODEX_BASE,
    )


def _openai_api_base_url(cfg: dict[str, Any]) -> str:
    return _configured_base_url(
        cfg,
        ("codex_base_url", "openai_base_url", "apibase", "base_url", "baseurl"),
        OPENAI_API_BASE,
    )


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
                base_url=_chatgpt_codex_base_url(cfg),
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
        base_url=_openai_api_base_url(cfg),
        headers=headers,
        source=source,
    )
