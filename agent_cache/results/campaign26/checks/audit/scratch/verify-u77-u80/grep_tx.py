"""Search a session transcript for a regular expression and print the matching records.

Usage: python3 -I -B grep_tx.py <transcript.jsonl> <regex> [start-iso-prefix] [context-chars] [kinds]
kinds: comma list of text,tool_use,tool_result,user (default text,user)
Read-only: opens the transcript for reading and prints to stdout.
"""
import json
import re
import sys

path = sys.argv[1]
pat = re.compile(sys.argv[2])
start = sys.argv[3] if len(sys.argv) > 3 else ""
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 300
kinds = set((sys.argv[5] if len(sys.argv) > 5 else "text,user").split(","))


def parts_of(rec):
    msg = rec.get("message") or {}
    content = msg.get("content")
    role = rec.get("type")
    if isinstance(content, str):
        yield ("user" if role == "user" else "text"), content
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            yield ("user" if role == "user" else "text"), part.get("text", "")
        elif kind == "tool_use":
            yield "tool_use", json.dumps(part.get("input"), ensure_ascii=False)
        elif kind == "tool_result":
            c = part.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            yield "tool_result", c if isinstance(c, str) else ""


with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if not pat.search(line):
            # cheap pre-filter on the raw line (escaped JSON may hide some matches)
            pass
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ts = rec.get("timestamp", "")
        if start and ts < start:
            continue
        if rec.get("type") not in ("user", "assistant"):
            continue
        for kind, text in parts_of(rec):
            if kind not in kinds or not text:
                continue
            for m in pat.finditer(text):
                a = max(0, m.start() - ctx)
                b = min(len(text), m.end() + ctx)
                side = "SIDE " if rec.get("isSidechain") else ""
                print("=== idx0=%d ts=%s %s%s/%s" % (idx, ts, side, rec.get("type"), kind))
                print(text[a:b].replace("\n", "\n    "))
                print()
                break
