"""Read-only helper: search the session transcript (JSONL) for patterns and print matching lines' index, timestamp, role and a snippet.
usage: tx.py grep <regex> [maxhits] [ctxchars]   |   tx.py show <idx> [maxchars]   |   tx.py range <lo> <hi> [maxchars]"""
import json, re, sys
P = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"

def parts(o):
    """Yield (kind, text) for every text-bearing part of a transcript record."""
    m = o.get("message") or {}
    c = m.get("content")
    if isinstance(c, str):
        yield ("text", c)
    elif isinstance(c, list):
        for p in c:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "text":
                yield ("text", p.get("text") or "")
            elif t == "tool_use":
                yield ("tool_use:" + str(p.get("name")), json.dumps(p.get("input"), ensure_ascii=False))
            elif t == "tool_result":
                cc = p.get("content")
                if isinstance(cc, list):
                    cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                yield ("tool_result", str(cc))
            elif t == "thinking":
                yield ("thinking", p.get("thinking") or "")

mode = sys.argv[1]
if mode == "grep":
    rx = re.compile(sys.argv[2])
    maxhits = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    hits = 0
    with open(P, errors="replace") as f:
        for i, line in enumerate(f):
            if not rx.search(line):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            for kind, text in parts(o):
                for m in rx.finditer(text):
                    s = max(0, m.start() - ctx); e = min(len(text), m.end() + ctx)
                    print(f"[{i}] {o.get('timestamp')} {o.get('type')} {kind}: ...{text[s:e]!r}...")
                    hits += 1
                    break
            if hits >= maxhits:
                break
elif mode in ("show", "range"):
    lo = int(sys.argv[2]); hi = int(sys.argv[3]) if mode == "range" else lo
    maxc = int(sys.argv[4 if mode == "range" else 3]) if len(sys.argv) > (4 if mode == "range" else 3) else 3000
    with open(P, errors="replace") as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i > hi:
                break
            try:
                o = json.loads(line)
            except Exception as ex:
                print(f"[{i}] unparsable: {ex}"); continue
            print(f"==== [{i}] {o.get('timestamp')} type={o.get('type')} sidechain={o.get('isSidechain')}")
            for kind, text in parts(o):
                print(f"--- {kind} ({len(text)} chars)")
                print(text[:maxc])
