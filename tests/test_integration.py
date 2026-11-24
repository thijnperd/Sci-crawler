"""Integration tests exercising real subprocess CLI pipes, MCP framing, and binary caching."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from papersplease import net

REPO_ROOT = Path(__file__).resolve().parent.parent


class SubprocessPipeTests(unittest.TestCase):
    def test_cli_check_pipe_handles_utf8_under_cp1252_env(self):
        stdin = "Resistance training for muscle hypertrophy by Müller\u2010Schmidt\n"
        with tempfile.TemporaryDirectory(prefix="pp_cli_pipe_") as tmp:
            env = {
                **os.environ,
                "PYTHONIOENCODING": "cp1252",
                "PYTHONUTF8": "0",
                "PAPERSPLEASE_CACHE_DIR": tmp,
            }
            proc = subprocess.run(
                [sys.executable, "-m", "papersplease", "check", "hypertrophy", "-"],
                input=stdin.encode("utf-8"),
                capture_output=True,
                cwd=str(REPO_ROOT),
                env=env,
                timeout=60,
                check=False,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        stdout = proc.stdout.decode("utf-8")
        self.assertIn("Müller", stdout)
        self.assertIn("IN SCOPE", stdout)

    def test_cli_explain_pipe_succeeds_under_cp1252_env(self):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "cp1252"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "papersplease",
                "explain",
                "optimal training for hypertrophy in younger adults since 2015",
            ],
            capture_output=True,
            cwd=str(REPO_ROOT),
            env={**env, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"},
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        self.assertIn("hypertrophy", proc.stdout.decode("utf-8"))

    def test_cli_rejects_invalid_numeric_flags(self):
        for bad_args in (
            ["find", "hypertrophy", "--min-scope", "nan"],
            ["find", "hypertrophy", "--min-scope", "1.5"],
            ["find", "hypertrophy", "--limit", "-3"],
            ["find", "hypertrophy", "--limit", "0"],
            ["find", "hypertrophy", "--deep-limit", "0"],
            ["find", "hypertrophy", "--min-citations", "-1"],
        ):
            proc = subprocess.run(
                [sys.executable, "-m", "papersplease", *bad_args],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=30,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, bad_args)


class SubprocessMcpEdgeTests(unittest.TestCase):
    def test_mcp_survives_malformed_params_and_unicode_over_cp1252_pipe(self):
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "screen_papers",
                    "arguments": {
                        "intent": "hypertrophy",
                        "papers": [
                            {
                                "title": "Muscle hypertrophy review by Müller\u2010Schmidt",
                                "abstract": "Skeletal muscle hypertrophy increased.",
                            }
                        ],
                    },
                },
            },
            {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": [1]},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": "bad"},
            [],
            {"jsonrpc": "2.0", "id": 5, "method": "ping"},
        ]
        stdin = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages) + "\n"
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "cp1252"
        env["PYTHONUTF8"] = "0"
        proc = subprocess.run(
            [sys.executable, "-m", "papersplease", "mcp"],
            input=stdin.encode("utf-8"),
            capture_output=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        replies = [
            json.loads(line)
            for line in proc.stdout.decode("utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(replies), 6)
        self.assertIn("Müller", replies[1]["result"]["content"][0]["text"])
        self.assertEqual(replies[2]["error"]["code"], -32602)
        self.assertEqual(replies[3]["error"]["code"], -32602)
        self.assertEqual(replies[4]["error"]["code"], -32600)
        self.assertEqual(replies[5], {"jsonrpc": "2.0", "id": 5, "result": {}})


class HttpCacheBinaryTests(unittest.TestCase):
    def test_http_cache_round_trips_binary_pdf_payload_losslessly(self):
        payload = b"%PDF-1.4\n" + bytes(range(256)) * 4
        self.assertEqual(len(payload), 1033)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            url = "https://example.org/sample.pdf"
            net._cache_write(cache_dir, url, payload)
            loaded = net._cache_read(cache_dir, url, ttl=3600)
            self.assertEqual(loaded, payload)
            self.assertEqual(len(loaded), 1033)


if __name__ == "__main__":
    unittest.main()
