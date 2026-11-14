import json, sys, re
path = sys.argv[1]
start = int(sys.argv[2])
pat = re.compile(sys.argv[3])
out = sys.argv[4]
ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 400
n = 0
with open(path, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
    for i, line in enumerate(f):
        if i < start:
            continue
        if '"type":"assistant"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain"):
            continue
        c = d.get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text", "")
                for m in pat.finditer(t):
                    s = max(0, m.start() - ctx)
                    e = min(len(t), m.end() + ctx)
                    o.write("=== line %d ts %s\n...%s...\n\n" % (i, d.get("timestamp"), t[s:e]))
                    n += 1
print("hits", n)
