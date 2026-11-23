"""Drive the PapersPlease MCP server the way an agent host does.

One JSON-RPC message per line over stdio, then a readable transcript. Run it
from the repository root:

    python3 docs/examples/mcp_session.py

It is deliberately dependency-free: any MCP client (Claude Desktop, Cursor,
Codex, a hand-rolled host) speaks the same protocol.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MESSAGES = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "explain_scope",
            "arguments": {"intent": "muscle strength in older adults not about rodents"},
        },
    },
]


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "papersplease", "mcp"],
        cwd=ROOT,
        input="\n".join(json.dumps(message) for message in MESSAGES) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode

    for line in completed.stdout.splitlines():
        reply = json.loads(line)
        result = reply.get("result") or {}
        if "serverInfo" in result:
            info = result["serverInfo"]
            print(f'initialize   {info["name"]} {info["version"]} | protocol {result["protocolVersion"]}')
        elif "tools" in result:
            print(f'tools/list   {len(result["tools"])} tools')
            for tool in result["tools"]:
                flags = "read-only" if tool["annotations"]["readOnlyHint"] else "writes"
                print(f'  - {tool["name"]:18} {tool["title"]:20} {flags}')
        elif "structuredContent" in result:
            data = result["structuredContent"]
            print(f'explain_scope head={data["head"]!r} exclude={data["exclude"]}')
            for facet in data["facets"]:
                print(f'  facet {facet["name"]:10} {", ".join(facet["terms"][:5])}')
            print(f'query {data["query"]}')
        else:
            print(json.dumps(reply)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
