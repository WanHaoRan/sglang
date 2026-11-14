"""Print the text of given transcript lines (0-based index), read-only.

usage: show.py LO HI [MAXCHARS] [--tools]
Prints user text and assistant text blocks; tool calls only with --tools.
"""
import json
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 6000
tools = "--tools" in sys.argv

SKIP = {
    "api-request-blob", "api-request", "api-request-shape", "mode", "permission-mode",
    "session-owner", "atis-latch", "ai-title", "last-prompt", "file-history-snapshot",
    "file-history-delta",
}

with open(SRC, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            o = json.loads(line)
        except Exception as e:  # noqa: BLE001
            print(i, "UNPARSEABLE", e)
            continue
        t = o.get("type")
        if t in SKIP:
            continue
        ts = o.get("timestamp", "")
        msg = o.get("message") or {}
        content = msg.get("content")
        out = []
        if isinstance(content, str):
            out.append(("text", content))
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    out.append(("text", b.get("text", "")))
                elif bt == "tool_use" and tools:
                    out.append(("tool_use", json.dumps(b.get("input", {}), ensure_ascii=False)))
                elif bt == "tool_result" and tools:
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    out.append(("tool_result", str(c)))
        for kind, text in out:
            if not text.strip():
                continue
            print("=== line %d %s %s %s (len %d) ===" % (i, t, ts, kind, len(text)))
            print(text[:mx])
            print()
