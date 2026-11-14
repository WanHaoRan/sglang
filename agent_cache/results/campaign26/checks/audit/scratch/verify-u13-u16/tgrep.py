"""Grep a session transcript's user and assistant TEXT (not tool output) for a regex.
usage: tgrep.py file.jsonl regex [min_line] [context_chars] [roles=user,assistant]"""
import json, re, sys
path, rx = sys.argv[1], re.compile(sys.argv[2], re.I)
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 300
roles = set((sys.argv[5] if len(sys.argv) > 5 else "user,assistant").split(","))
with open(path, errors="replace") as f:
    for i, line in enumerate(f, 1):
        if i < lo or not rx.search(line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in roles:
            continue
        msg = d.get("message") or {}
        content = msg.get("content")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text", ""))
        for t in texts:
            for m in rx.finditer(t):
                a, b = max(0, m.start() - ctx), min(len(t), m.end() + ctx)
                print(f"--- line {i} {d.get('timestamp','')} {d.get('type')} sidechain={d.get('isSidechain')}")
                print(t[a:b].replace("\n", " | "))
