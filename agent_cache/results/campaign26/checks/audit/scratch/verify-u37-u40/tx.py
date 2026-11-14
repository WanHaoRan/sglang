#!/usr/bin/env python3
"""Read-only helper for the session transcript (jsonl).

usage:
  tx.py line N [N ...]          print the text of the given 0-based line indices
  tx.py users FROM [TO]         list real user messages (not tool results) between indices
  tx.py grep PATTERN [FROM] [TO] [--role user|assistant] [--max N]   regex search in text blocks
"""
import json
import re
import sys

PATH = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"


def texts(obj):
    """Yield (kind, text) for the message blocks of one transcript record."""
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return
    content = msg.get("content")
    if isinstance(content, str):
        yield ("text", content)
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            yield ("text", block.get("text", ""))
        elif t == "tool_use":
            yield ("tool_use", json.dumps(block.get("input", {}))[:4000])
        elif t == "tool_result":
            c = block.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            yield ("tool_result", str(c))
        elif t == "thinking":
            yield ("thinking", block.get("thinking", ""))


def is_real_user(obj):
    if obj.get("type") != "user":
        return False
    if obj.get("isMeta"):
        return False
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "text" for b in content)
    return False


def main():
    mode = sys.argv[1]
    args = sys.argv[2:]
    if mode == "line":
        want = {int(a) for a in args if not a.startswith("--")}
        limit = 6000
        for a in args:
            if a.startswith("--limit="):
                limit = int(a.split("=", 1)[1])
        with open(PATH, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i in want:
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        print(f"[{i}] <unparsable: {e}>")
                        continue
                    print(f"===== [{i}] type={obj.get('type')} ts={obj.get('timestamp')} sidechain={obj.get('isSidechain')}")
                    for kind, text in texts(obj):
                        print(f"--- {kind}")
                        print(text[:limit])
                if i > max(want):
                    break
    elif mode == "users":
        lo = int(args[0])
        hi = int(args[1]) if len(args) > 1 else 10**9
        limit = 1500
        with open(PATH, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i < lo:
                    continue
                if i > hi:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not is_real_user(obj) or obj.get("isSidechain"):
                    continue
                parts = [t for k, t in texts(obj) if k == "text"]
                if not parts:
                    continue
                print(f"===== [{i}] ts={obj.get('timestamp')}")
                print("\n".join(parts)[:limit])
    elif mode == "grep":
        pat = re.compile(args[0], re.I)
        rest = [a for a in args[1:] if not a.startswith("--")]
        lo = int(rest[0]) if len(rest) > 0 else 0
        hi = int(rest[1]) if len(rest) > 1 else 10**9
        role = None
        mx = 60
        kinds = {"text"}
        for a in args:
            if a.startswith("--role="):
                role = a.split("=", 1)[1]
            if a.startswith("--max="):
                mx = int(a.split("=", 1)[1])
            if a.startswith("--kinds="):
                kinds = set(a.split("=", 1)[1].split(","))
        n = 0
        with open(PATH, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i < lo:
                    continue
                if i > hi:
                    break
                if not pat.search(line):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("isSidechain"):
                    continue
                if role and obj.get("type") != role:
                    continue
                for kind, text in texts(obj):
                    if kind not in kinds:
                        continue
                    for m in pat.finditer(text):
                        s = max(0, m.start() - 300)
                        e = min(len(text), m.end() + 500)
                        print(f"===== [{i}] type={obj.get('type')} kind={kind} ts={obj.get('timestamp')}")
                        print(text[s:e].replace("\n", "\n    "))
                        n += 1
                        break
                if n >= mx:
                    break


if __name__ == "__main__":
    main()
