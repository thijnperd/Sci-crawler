"""MCP protocol behaviour: JSON-RPC framing, tool schemas, error handling."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from paperhound.config import Settings
from paperhound.mcp_server import PROTOCOL_VERSIONS, TOOLS, handle_message


SETTINGS = Settings(cache=False, mirrors=["https://sci-hub.se"], mailto="", unpaywall_email="", openalex_key="", s2_key="")

REPO_ROOT = Path(__file__).resolve().parents[1]


def call(method, params=None, message_id=1):
    return handle_message(
        {"jsonrpc": "2.0", "id": message_id, "method": method, "params": params or {}},
        SETTINGS,
    )


class InitializeTests(unittest.TestCase):
    def test_protocol_version_is_echoed(self):
        for version in PROTOCOL_VERSIONS:
            response = call("initialize", {"protocolVersion": version, "capabilities": {}})
            self.assertEqual(response["result"]["protocolVersion"], version)
            self.assertEqual(response["result"]["serverInfo"]["name"], "paperhound")
            self.assertIn("tools", response["result"]["capabilities"])

    def test_unknown_protocol_falls_back(self):
        response = call("initialize", {"protocolVersion": "1999-01-01"})
        self.assertIn(response["result"]["protocolVersion"], PROTOCOL_VERSIONS)

    def test_instructions_explain_the_workflow(self):
        response = call("initialize", {"protocolVersion": PROTOCOL_VERSIONS[0]})
        self.assertIn("screens", response["result"]["instructions"])

    def test_notification_gets_no_reply(self):
        self.assertIsNone(handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}, SETTINGS))

    def test_ping(self):
        self.assertEqual(call("ping"), {"jsonrpc": "2.0", "id": 1, "result": {}})


class ToolListingTests(unittest.TestCase):
    def test_all_tools_are_advertised_with_schemas(self):
        response = call("tools/list")
        tools = {tool["name"]: tool for tool in response["result"]["tools"]}
        self.assertEqual(set(tools), set(TOOLS))
        for name, tool in tools.items():
            self.assertIn("inputSchema", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")
        self.assertIn("query", tools["search_literature"]["inputSchema"]["required"])
        self.assertIn("papers", tools["screen_papers"]["inputSchema"]["required"])
        self.assertIn("explain_scope", tools)

    def test_tools_are_marked_read_only_for_hosts(self):
        tools = {tool["name"]: tool for tool in call("tools/list")["result"]["tools"]}
        for name, tool in tools.items():
            if name == "download_paper":
                # It writes one PDF the caller asked for; still not destructive.
                self.assertFalse(tool["annotations"]["readOnlyHint"], name)
            else:
                self.assertTrue(tool["annotations"]["readOnlyHint"], name)
            self.assertFalse(tool["annotations"]["destructiveHint"], name)
            self.assertTrue(tool["title"], name)

    def test_new_agent_tools_are_advertised(self):
        tools = {tool["name"]: tool for tool in call("tools/list")["result"]["tools"]}
        self.assertIn("download_paper", tools)
        self.assertIn("export_citations", tools)
        self.assertIn("papers", tools["export_citations"]["inputSchema"]["required"])
        self.assertIn("require_in", tools["search_literature"]["inputSchema"]["properties"])
        self.assertIn("expand", tools["search_literature"]["inputSchema"]["properties"])


class ToolCallTests(unittest.TestCase):
    def test_missing_query_is_a_tool_error_not_a_crash(self):
        response = call("tools/call", {"name": "search_literature", "arguments": {}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("query", response["result"]["content"][0]["text"])

    def test_unknown_tool_is_a_protocol_error(self):
        response = call("tools/call", {"name": "nope", "arguments": {}})
        self.assertEqual(response["error"]["code"], -32602)

    def test_bad_arguments_type(self):
        response = call("tools/call", {"name": "search_literature", "arguments": "hello"})
        self.assertEqual(response["error"]["code"], -32602)

    def test_unknown_method(self):
        response = call("nonexistent/method")
        self.assertEqual(response["error"]["code"], -32601)

    def test_invalid_jsonrpc_envelope(self):
        response = handle_message({"id": 1, "method": "ping"}, SETTINGS)
        self.assertEqual(response["error"]["code"], -32600)

    def test_screen_papers_works_offline(self):
        response = call(
            "tools/call",
            {
                "name": "screen_papers",
                "arguments": {
                    "intent": "muscle strength in older adults",
                    "papers": [
                        {
                            "title": "Grip strength in older adults",
                            "abstract": "Muscle strength was measured in older adults.",
                        },
                        {"title": "Muscle fibre typing", "abstract": "Muscle fibres differ."},
                    ],
                    "with_access": False,
                },
            },
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["counts"]["in_scope"], 1)
        self.assertEqual(payload["counts"]["out_of_scope"], 1)
        self.assertEqual(payload["results"][0]["scope"]["bucket"], "in_scope")

    def test_get_paper_access_requires_an_identifier(self):
        response = call("tools/call", {"name": "get_paper_access", "arguments": {}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("doi", response["result"]["content"][0]["text"])

    def test_explain_scope_is_offline_and_shows_the_negation_rule(self):
        response = call(
            "tools/call",
            {
                "name": "explain_scope",
                "arguments": {
                    "intent": "muscle strength in older adults not about rodents",
                    "exclude": ["case reports"],
                },
            },
        )
        self.assertFalse(response["result"]["isError"])
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["head"], "strength")
        self.assertIn("rodents", payload["exclude"])
        self.assertIn("case reports", payload["exclude"])
        self.assertIn("mice", payload["exclude_terms"]["rodents"])
        self.assertEqual(payload["schema_version"], 1)

    def test_structured_content_matches_the_text_block(self):
        response = call(
            "tools/call",
            {
                "name": "screen_papers",
                "arguments": {
                    "intent": "muscle strength in older adults",
                    "papers": [{"title": "Grip strength in older adults", "abstract": "Muscle strength in older adults."}],
                },
            },
        )
        result = response["result"]
        self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])


class NewToolTests(unittest.TestCase):
    def test_download_requires_an_identifier(self):
        response = call("tools/call", {"name": "download_paper", "arguments": {}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("doi", response["result"]["content"][0]["text"])

    def test_export_ris_works_offline(self):
        response = call(
            "tools/call",
            {
                "name": "export_citations",
                "arguments": {
                    "format": "ris",
                    "papers": [
                        {
                            "title": "Grip strength in older adults",
                            "doi": "10.1/a",
                            "year": 2021,
                            "authors": ["Smith J"],
                        }
                    ],
                },
            },
        )
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["count"], 1)
        self.assertTrue(payload["text"].startswith("TY  - JOUR"))
        self.assertIn("10.1/a", payload["text"])

    def test_export_rejects_unknown_format(self):
        response = call(
            "tools/call",
            {
                "name": "export_citations",
                "arguments": {"format": "endnote", "papers": [{"title": "x"}]},
            },
        )
        self.assertTrue(response["result"]["isError"])
        self.assertIn("bibtex", response["result"]["content"][0]["text"])

    def test_screen_accepts_require_in(self):
        response = call(
            "tools/call",
            {
                "name": "screen_papers",
                "arguments": {
                    "intent": "muscle strength in older adults",
                    "require": ["strength"],
                    "require_in": "title",
                    "papers": [
                        {"title": "Grip in older adults", "abstract": "Muscle strength was measured."}
                    ],
                },
            },
        )
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["counts"]["out_of_scope"], 1)


class StdioTransportTests(unittest.TestCase):
    def test_server_speaks_line_delimited_jsonrpc(self):
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSIONS[0]}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            "this is not json",
        ]
        stdin = "\n".join(m if isinstance(m, str) else json.dumps(m) for m in messages) + "\n"
        completed = subprocess.run(
            [sys.executable, "-m", "paperhound", "mcp"],
            input=stdin,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        replies = [json.loads(line) for line in lines]
        # initialize, tools/list and one parse-error reply; the notification is silent.
        self.assertEqual(len(replies), 3)
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "paperhound")
        self.assertGreaterEqual(len(replies[1]["result"]["tools"]), 5)
        self.assertEqual(replies[2]["error"]["code"], -32700)
        # stdout must carry nothing but JSON-RPC; the banner belongs on stderr.
        for line in lines:
            json.loads(line)
        self.assertIn("MCP server ready", completed.stderr)


class CliSmokeTests(unittest.TestCase):
    def test_help_and_version(self):
        for args in (["--help"], ["--version"]):
            completed = subprocess.run(
                [sys.executable, "-m", "paperhound", *args],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_check_accepts_a_file_after_the_intent(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        try:
            handle.write("Grip strength and physical function in older adults\n")
            handle.close()
            completed = subprocess.run(
                [sys.executable, "-m", "paperhound", "check", "muscle strength in older adults", handle.name],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=60,
            )
        finally:
            os.unlink(handle.name)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("in scope", completed.stdout)

    def test_check_can_export_bibtex(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        try:
            handle.write("Grip strength in older adults\n")
            handle.close()
            completed = subprocess.run(
                [
                    sys.executable, "-m", "paperhound", "check",
                    "muscle strength in older adults", handle.name, "--format", "bibtex",
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=60,
            )
        finally:
            os.unlink(handle.name)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("@article{", completed.stdout)

    def test_bare_intent_is_treated_as_find(self):
        completed = subprocess.run(
            [sys.executable, "-m", "paperhound", "find", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        self.assertIn("--min-scope", completed.stdout)


if __name__ == "__main__":
    unittest.main()
