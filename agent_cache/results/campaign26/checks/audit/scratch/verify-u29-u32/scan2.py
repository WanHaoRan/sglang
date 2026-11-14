"""Read-only: print matches of a regex in assistant thinking/text/tool_use and user text of a transcript, from a start index.
python3 -I -B scan2.py <transcript> <regex> [start_index] [kinds csv: thinking,text,tool_use,tool_result] [before] [after] [max_per_line]"""
import json, sys, re
path, rx = sys.argv[1], re.compile(sys.argv[2])
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
kinds = set((sys.argv[4] if len(sys.argv) > 4 else "thinking,text,tool_use").split(","))
before = int(sys.argv[5]) if len(sys.argv) > 5 else 300
after = int(sys.argv[6]) if len(sys.argv) > 6 else 400
per = int(sys.argv[7]) if len(sys.argv) > 7 else 1
seen = set()
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if i < start or not rx.search(line):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("isSidechain") or obj.get("type") not in ("assistant", "user"):
            continue
        msg = obj.get("message") or {}
        content = msg.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for p in parts:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t not in kinds:
                continue
            if t == "thinking":
                text = p.get("thinking", "")
            elif t == "text":
                text = p.get("text", "")
            elif t == "tool_use":
                text = json.dumps(p.get("input", {}), ensure_ascii=False)
            elif t == "tool_result":
                c = p.get("content")
                text = c if isinstance(c, str) else "\n".join(q.get("text", "") for q in (c or []) if isinstance(q, dict))
            else:
                continue
            n = 0
            for m in rx.finditer(text):
                a, b = max(0, m.start() - before), min(len(text), m.end() + after)
                key = text[a:b]
                if key in seen:
                    continue
                seen.add(key)
                print(f"[{i}] {obj.get('timestamp')} {obj.get('type')}/{t} :: ...{key!r}...")
                print()
                n += 1
                if n >= per:
                    break
