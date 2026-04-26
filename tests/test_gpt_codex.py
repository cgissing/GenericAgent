"""Mocked tests for GPT/Codex-native GenericAgent support."""

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _jwt(claims):
    def enc(data):
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc(claims)}.sig"


class TestCodexGPTAuth(unittest.TestCase):
    def test_api_key_auth_from_config(self):
        from gpt_auth import resolve_gpt_auth

        auth = resolve_gpt_auth({"auth": "api_key", "apikey": "sk-test", "apibase": "https://api.openai.com/v1"})
        self.assertEqual(auth.mode, "api_key")
        self.assertEqual(auth.base_url, "https://api.openai.com/v1")
        self.assertEqual(auth.headers["Authorization"], "Bearer sk-test")
        self.assertNotIn("sk-test", auth.redacted)

    def test_codex_chatgpt_oauth_headers(self):
        from gpt_auth import resolve_gpt_auth

        with tempfile.TemporaryDirectory() as td:
            Path(td, "auth.json").write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "access-secret",
                            "id_token": _jwt(
                                {
                                    "chatgpt_account_id": "acct_123",
                                    "chatgpt_account_is_fedramp": True,
                                }
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": td}, clear=False):
                auth = resolve_gpt_auth({"auth": "codex"})
        self.assertEqual(auth.mode, "chatgpt_oauth")
        self.assertEqual(auth.base_url, "https://chatgpt.com/backend-api/codex")
        self.assertEqual(auth.headers["ChatGPT-Account-ID"], "acct_123")
        self.assertEqual(auth.headers["X-OpenAI-Fedramp"], "true")
        self.assertNotIn("access-secret", auth.redacted)

    def test_codex_oauth_custom_gateway_base_url_is_preserved(self):
        from gpt_auth import resolve_gpt_auth

        with tempfile.TemporaryDirectory() as td:
            Path(td, "auth.json").write_text(
                json.dumps({"tokens": {"access_token": "access-secret"}}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": td}, clear=False):
                auth = resolve_gpt_auth(
                    {
                        "auth": "codex",
                        "apibase": "https://gateway.example/backend-api/codex/",
                    }
                )
        self.assertEqual(auth.mode, "chatgpt_oauth")
        self.assertEqual(auth.base_url, "https://gateway.example/backend-api/codex")
        self.assertEqual(auth.headers["Authorization"], "Bearer access-secret")

    def test_codex_oauth_accepts_semantic_gateway_base_url_aliases(self):
        from gpt_auth import resolve_gpt_auth

        with tempfile.TemporaryDirectory() as td:
            Path(td, "auth.json").write_text(
                json.dumps({"tokens": {"access_token": "access-secret"}}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": td}, clear=False):
                auth = resolve_gpt_auth(
                    {
                        "auth": "codex",
                        "codex_base_url": "https://gateway.example/backend-api/codex",
                    }
                )
        self.assertEqual(auth.base_url, "https://gateway.example/backend-api/codex")

    def test_api_key_custom_gateway_base_url_is_preserved(self):
        from gpt_auth import resolve_gpt_auth

        auth = resolve_gpt_auth(
            {
                "auth": "api_key",
                "apikey": "sk-test",
                "apibase": "https://gateway.example/openai/v1/",
            }
        )
        self.assertEqual(auth.mode, "api_key")
        self.assertEqual(auth.base_url, "https://gateway.example/openai/v1")
        self.assertEqual(auth.headers["Authorization"], "Bearer sk-test")

    def test_api_key_accepts_baseurl_alias(self):
        from gpt_auth import resolve_gpt_auth

        auth = resolve_gpt_auth(
            {
                "auth": "api_key",
                "apikey": "sk-test",
                "baseurl": "https://gateway.example/openai/v1",
            }
        )
        self.assertEqual(auth.base_url, "https://gateway.example/openai/v1")


class TestResponsesPayloadAndParsing(unittest.TestCase):
    def test_payload_has_codex_responses_fields(self):
        from gpt_responses import build_responses_payload

        payload = build_responses_payload(
            model="gpt-5.5",
            instructions="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "file_read", "parameters": {"type": "object"}}}],
            reasoning_effort="medium",
            verbosity="low",
            stream=True,
            prompt_cache_key="conv-1",
            parallel_tool_calls=True,
            store=False,
        )
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertTrue(payload["parallel_tool_calls"])
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
        self.assertEqual(payload["text"], {"verbosity": "low"})
        self.assertEqual(payload["prompt_cache_key"], "conv-1")
        self.assertIn("x-codex-installation-id", payload["client_metadata"])
        self.assertEqual(payload["tools"][0]["name"], "file_read")

    def test_ws_url_and_create_message(self):
        from gpt_responses_ws import CODEX_RESPONSES_WS_BETA, build_ws_create_message, responses_ws_url, ws_headers

        self.assertEqual(
            responses_ws_url("https://chatgpt.com/backend-api/codex"),
            "wss://chatgpt.com/backend-api/codex/responses",
        )
        msg = json.loads(build_ws_create_message({"model": "gpt-5.5"}, generate=False))
        self.assertEqual(msg["type"], "response.create")
        self.assertFalse(msg["generate"])
        self.assertIn(CODEX_RESPONSES_WS_BETA, "\n".join(ws_headers({"Authorization": "Bearer x"})))

    def test_sse_parser_custom_apply_patch(self):
        from gpt_responses import parse_responses_sse

        event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch",
            },
        }
        gen = parse_responses_sse(["data: " + json.dumps(event), "data: [DONE]"])
        with self.assertRaises(StopIteration) as cm:
            next(gen)
        blocks = cm.exception.value
        self.assertEqual(blocks[0]["name"], "apply_patch")
        self.assertEqual(blocks[0]["input"]["patch"], "*** Begin Patch\n*** End Patch")

    def test_auto_make_responses_url_supports_custom_codex_gateway(self):
        from llmcore import auto_make_responses_url

        self.assertEqual(
            auto_make_responses_url("https://gateway.example/backend-api/codex"),
            "https://gateway.example/backend-api/codex/responses",
        )
        self.assertEqual(
            auto_make_responses_url("https://gateway.example/backend-api/codex/responses"),
            "https://gateway.example/backend-api/codex/responses",
        )
        self.assertEqual(
            auto_make_responses_url("https://gateway.example/openai/v1"),
            "https://gateway.example/openai/v1/responses",
        )


class TestGPTMemoryAndTools(unittest.TestCase):
    def test_memory_router_turn_summary_is_deterministic(self):
        from memory_model_router import MemoryModelRouter

        with tempfile.TemporaryDirectory() as td:
            router = MemoryModelRouter(td, main_model="gpt-5.5")
            choice = router.choose("turn_summary")
        self.assertEqual(choice.mode, "deterministic")
        self.assertEqual(choice.fallback_model, "gpt-5.4-mini")

    def test_turn_memory_entry_from_tool_result(self):
        from llmcore import MockResponse
        from memory_adapter import build_turn_memory_entry, entry_to_history_line

        resp = MockResponse("", "Edited file and ran tests.", [], "")
        entry = build_turn_memory_entry(
            resp,
            [{"tool_name": "shell_command", "args": {"command": "pytest"}, "id": "call_1"}],
            [{"tool_use_id": "call_1", "content": json.dumps({"status": "success", "stdout": "ok", "exit_code": 0})}],
            3,
            "next",
        )
        self.assertEqual(entry.turn_id, 3)
        self.assertEqual(entry.tool_calls[0]["tool_name"], "shell_command")
        self.assertIn("status=success", entry.tool_results_summary[0])
        self.assertTrue(entry_to_history_line(entry).startswith("[Agent]"))

    def test_structured_compaction_uses_turn_evidence_and_checkpoint(self):
        from gpt_compaction import build_structured_compaction
        from memory_adapter import TurnMemoryEntry

        entry = TurnMemoryEntry(
            turn_id=7,
            assistant_action="Patched Codex gateway routing.",
            tool_calls=[{"tool_name": "shell_command", "args": {"command": "pytest"}, "id": "call_1"}],
            tool_results_summary=["status=success, exit=0: 15 tests passed"],
            artifacts=["C:\\repo\\gpt_auth.py", "https://gateway.example/backend-api/codex"],
            errors=["HTTP 502 from custom gateway during retry"],
            next_step="Run full unittest discover.",
        )
        handoff = build_structured_compaction(
            ["[Agent] older context"],
            {"key_info": "Preserve custom Codex gateway base URL.", "related_sop": "memory_management_sop.md"},
            [entry],
        )
        payload = json.loads(handoff.split("\n", 1)[1])
        self.assertEqual(payload["current_goal"], "Preserve custom Codex gateway base URL.")
        self.assertEqual(payload["related_sop"], "memory_management_sop.md")
        self.assertIn("C:\\repo\\gpt_auth.py", payload["key_artifacts"])
        self.assertIn("HTTP 502", payload["open_blockers"][0])
        self.assertEqual(payload["next_step"], "Run full unittest discover.")
        self.assertEqual(payload["recent_turns"][0]["turn_id"], 7)
        self.assertEqual(payload["recent_turns"][0]["tools"], ["shell_command"])
        self.assertIn("15 tests passed", payload["recent_turns"][0]["tool_results"][0])

    def test_apply_patch_rejects_path_escape_and_updates_file(self):
        from ga import apply_codex_patch_text

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "a.txt"
            target.write_text("alpha\nbeta\n", encoding="utf-8")
            result = apply_codex_patch_text(
                "*** Begin Patch\n*** Update File: a.txt\n@@\n alpha\n-beta\n+gamma\n*** End Patch",
                root,
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")
            bad = apply_codex_patch_text(
                "*** Begin Patch\n*** Add File: ../bad.txt\n+bad\n*** End Patch",
                root,
            )
            self.assertEqual(bad["status"], "error")
            self.assertIn("escapes", bad["msg"])

    def test_gpt_turn_end_does_not_inject_summary_warning(self):
        from ga import GenericAgentHandler
        from llmcore import MockResponse

        class Backend:
            is_gpt_native = True
            model = "gpt-5.5"
            context_win = 272000

        class Client:
            backend = Backend()

        class Parent:
            llmclient = Client()
            task_dir = None
            verbose = False

        handler = GenericAgentHandler(Parent(), ["[USER]: do work"], tempfile.gettempdir())
        resp = MockResponse("", "Done", [], "")
        with patch("ga.append_l4_memory_event", return_value="memory_events.jsonl"):
            next_prompt = handler.turn_end_callback(resp, [{"tool_name": "no_tool", "args": {}}], [], 1, "", {})
        self.assertNotIn("<summary>", next_prompt)
        self.assertNotIn("[DANGER] 你遗漏了", next_prompt)
        self.assertTrue(handler.history_info[-1].startswith("[Agent]"))


    def test_gpt_compaction_writes_l4_compaction_event_when_threshold_hit(self):
        from ga import GenericAgentHandler
        from llmcore import MockResponse

        class Backend:
            is_gpt_native = True
            model = "gpt-5.5"
            context_win = 1

        class Client:
            backend = Backend()

        class Parent:
            llmclient = Client()
            task_dir = None
            verbose = False

        handler = GenericAgentHandler(Parent(), ["[USER]: do work"], tempfile.gettempdir())
        handler.working = {"key_info": "Finish compact v2", "related_sop": "memory_management_sop.md"}
        events = []

        def capture_event(_memory_dir, event):
            events.append(event)
            return "memory_events.jsonl"

        resp = MockResponse("", "Patched compact handoff.", [], "")
        tool_results = [
            {
                "tool_use_id": "call_1",
                "content": json.dumps(
                    {
                        "status": "error",
                        "stderr": "HTTP 502 from https://gateway.example/backend-api/codex",
                        "exit_code": 1,
                    }
                ),
            }
        ]
        with patch("ga.append_l4_memory_event", side_effect=capture_event):
            next_prompt = handler.turn_end_callback(
                resp,
                [{"tool_name": "shell_command", "args": {"command": "pytest"}, "id": "call_1"}],
                tool_results,
                2,
                "Retry with fallback",
                {},
            )
        self.assertIn("### Context Compaction Handoff", next_prompt)
        self.assertEqual([event.event_type for event in events], ["turn_summary", "compaction"])
        self.assertFalse(events[1].created_by_model)
        self.assertEqual(events[1].cost_class, "deterministic")
        self.assertIn("HTTP 502", json.dumps(events[1].payload, ensure_ascii=False))


class TestNativeToolClientGPTProtocol(unittest.TestCase):
    def test_no_summary_protocol_skips_thinking_prompt(self):
        from llmcore import NativeToolClient

        class Backend:
            no_summary_protocol = True
            name = "dummy"
            system = ""

        client = NativeToolClient(Backend())
        client.set_system("base system")
        self.assertEqual(client.backend.system, "base system")
        self.assertNotIn("<summary>", client.backend.system)


if __name__ == "__main__":
    unittest.main()
