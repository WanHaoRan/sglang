"""Print assistant text (and optionally tool calls) of a transcript between two line indices.

Usage: python3 -I -B dump_range.py <transcript.jsonl> <first-idx0> <last-idx0> [maxlen] [with_tools]
Read-only: opens the transcript for reading and prints to stdout.
"""
import json
import sys

path = sys.argv[1]
first = int(sys.argv[2])
last = int(sys.argv[3])
maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
with_tools = len(sys.argv) > 5 and sys.argv[5] == "1"

with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if idx < first:
            continue
        if idx > last:
            break
        try:
            rec = json.loads(line)
        except Exception:
            continue
        kind = rec.get("type")
        if kind not in ("user", "assistant"):
            continue
        if rec.get("isSidechain"):
            continue
        ts = rec.get("timestamp", "")
        content = (rec.get("message") or {}).get("content")
        if isinstance(content, str):
            print("=== idx0=%d ts=%s %s/str" % (idx, ts, kind))
            print(content[:maxlen])
            print()
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            pk = part.get("type")
            if pk == "text":
                text = part.get("text", "")
                if text.strip():
                    print("=== idx0=%d ts=%s %s/text" % (idx, ts, kind))
                    print(text[:maxlen])
                    print()
            elif pk == "tool_use" and with_tools:
                print("=== idx0=%d ts=%s %s/tool_use %s" % (idx, ts, kind, part.get("name")))
                print(json.dumps(part.get("input"), ensure_ascii=False)[:maxlen])
                print()
            elif pk == "tool_result" and with_tools:
                c = part.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                print("=== idx0=%d ts=%s %s/tool_result" % (idx, ts, kind))
                print((c or "")[:maxlen])
                print()
