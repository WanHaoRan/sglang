"""Search the session transcript for substrings (case-insensitive). Usage:
  python3 -I -B tgrep.py [--from IDX] [--role user|assistant|any] [--ctx N] needle [needle ...]
Prints: line index, timestamp, type, where the hit is (text / tool_use / tool_result), and a snippet around each needle."""
import json, sys

PATH = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
args = sys.argv[1:]
start, role, ctx, maxhits = 0, "any", 160, 60
while args and args[0].startswith("--"):
    k = args.pop(0)
    v = args.pop(0)
    if k == "--from":
        start = int(v)
    elif k == "--role":
        role = v
    elif k == "--ctx":
        ctx = int(v)
    elif k == "--max":
        maxhits = int(v)
needles = [a.lower() for a in args]
hits = 0
with open(PATH, errors="replace") as f:
    for i, line in enumerate(f):
        if i < start:
            continue
        low = line.lower()
        if not any(n in low for n in needles):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        typ = rec.get("type")
        if role != "any" and typ != role:
            continue
        ts = rec.get("timestamp", "")
        msg = rec.get("message") or {}
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            parts.append(("text", content))
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    parts.append(("text", c.get("text") or ""))
                elif c.get("type") == "tool_use":
                    parts.append(("tool_use", json.dumps(c.get("input"))))
                elif c.get("type") == "tool_result":
                    cc = c.get("content")
                    parts.append(("tool_result", cc if isinstance(cc, str) else json.dumps(cc)))
        for where, text in parts:
            tl = text.lower()
            for n in needles:
                p = tl.find(n)
                if p >= 0:
                    hits += 1
                    snippet = text[max(0, p - ctx):p + len(n) + ctx].replace("\n", " ")
                    print(f"{i}\t{ts}\t{typ}\t{where}\t{snippet}")
                    break
        if hits >= maxhits:
            print("... more hits not shown")
            break
print("hits", hits)
