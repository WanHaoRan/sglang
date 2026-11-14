"""Print AskUserQuestion calls and their answers from a transcript, from a start date on.

Usage: python3 -I -B askq.py <transcript.jsonl> <start-iso-prefix>
Read-only: opens the transcript for reading and prints to stdout.
"""
import json
import sys

path = sys.argv[1]
start = sys.argv[2]
ids = {}

with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if "AskUserQuestion" not in line and not ids:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ts = rec.get("timestamp", "")
        if ts < start:
            continue
        if rec.get("isSidechain"):
            continue
        content = (rec.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use" and part.get("name") == "AskUserQuestion":
                ids[part.get("id")] = idx
                print("=== idx0=%d ts=%s QUESTION" % (idx, ts))
                print(json.dumps(part.get("input"), ensure_ascii=False)[:1800])
                print()
            elif part.get("type") == "tool_result" and part.get("tool_use_id") in ids:
                c = part.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                print("=== idx0=%d ts=%s ANSWER (to idx0=%d)" % (idx, ts, ids[part.get("tool_use_id")]))
                print((c or "")[:1500])
                print()
