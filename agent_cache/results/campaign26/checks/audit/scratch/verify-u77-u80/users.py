"""Print the human-typed user messages of a session transcript from a start date on.

Usage: python3 -I -B users.py <transcript.jsonl> <start-iso-prefix> [maxlen]
Read-only: opens the transcript for reading and prints to stdout.
"""
import json
import sys

path = sys.argv[1]
start = sys.argv[2]
maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 1500


def text_of(content):
    if isinstance(content, str):
        return content, False
    out = []
    tool = False
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text":
                out.append(part.get("text", ""))
            elif kind == "tool_result":
                tool = True
    return "\n".join(out), tool


with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if '"type":"user"' not in line and '"type": "user"' not in line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") != "user":
            continue
        ts = rec.get("timestamp", "")
        if ts < start:
            continue
        msg = rec.get("message") or {}
        text, tool = text_of(msg.get("content"))
        if tool and not text.strip():
            continue
        if rec.get("isMeta"):
            tag = "META"
        elif rec.get("isSidechain"):
            tag = "SIDE"
        else:
            tag = "USER"
        text = text.strip()
        if not text:
            continue
        print("=== idx0=%d ts=%s %s len=%d" % (idx, ts, tag, len(text)))
        print(text[:maxlen])
        print()
