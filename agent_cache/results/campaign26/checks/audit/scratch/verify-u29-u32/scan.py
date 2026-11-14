"""Read-only scan of a session transcript (.jsonl). Usage:
  python3 -I -B scan.py <transcript.jsonl> users              -> every user-typed message (index, timestamp, text head)
  python3 -I -B scan.py <transcript.jsonl> grep <regex> [who] -> assistant text / user text matching (who: assistant|user|any|tool)
Prints line index (0-based), timestamp, role, and a window of text around each match."""
import json, re, sys

path, mode = sys.argv[1], sys.argv[2]


def texts(obj):
    """Yield (kind, text) for the message of one transcript line."""
    msg = obj.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        yield "text", content
        return
    for part in content or []:
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t == "text":
            yield "text", part.get("text", "")
        elif t == "tool_use":
            yield "tool_use", json.dumps(part.get("input", {}), ensure_ascii=False)
        elif t == "tool_result":
            c = part.get("content")
            if isinstance(c, str):
                yield "tool_result", c
            else:
                for p in c or []:
                    if isinstance(p, dict) and p.get("type") == "text":
                        yield "tool_result", p.get("text", "")


if mode == "users":
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if '"type":"user"' not in line and '"type": "user"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "user" or obj.get("isSidechain"):
                continue
            for kind, text in texts(obj):
                if kind != "text":
                    continue
                s = text.strip()
                if not s:
                    continue
                print(f"[{i}] {obj.get('timestamp')} meta={obj.get('isMeta')} :: {s[:int(sys.argv[3]) if len(sys.argv) > 3 else 400]!r}")
elif mode == "grep":
    rx = re.compile(sys.argv[3])
    who = sys.argv[4] if len(sys.argv) > 4 else "assistant"
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if not rx.search(line):
                # the raw line is JSON-escaped; simple patterns survive escaping
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            role = obj.get("type")
            if obj.get("isSidechain"):
                continue
            for kind, text in texts(obj):
                if who == "assistant" and not (role == "assistant" and kind == "text"):
                    continue
                if who == "user" and not (role == "user" and kind == "text"):
                    continue
                if who == "tool" and kind not in ("tool_use", "tool_result"):
                    continue
                if who == "tool_use" and kind != "tool_use":
                    continue
                for m in rx.finditer(text):
                    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
                    print(f"[{i}] {obj.get('timestamp')} {role}/{kind} :: ...{text[a:b]!r}...")
                    print()
