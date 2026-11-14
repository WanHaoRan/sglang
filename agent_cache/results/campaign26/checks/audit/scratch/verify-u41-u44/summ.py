import json, sys, re
path = sys.argv[1]
idx = int(sys.argv[2])
pat = sys.argv[3]
ctx = int(sys.argv[4])
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < idx: continue
        if i > idx: break
        d = json.loads(line)
        c = d.get("message", {}).get("content")
        t = c if isinstance(c, str) else "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
        open("summary_%d.txt" % idx, "w").write(t)
        for m in re.finditer(pat, t, re.I):
            s = max(0, m.start() - ctx); e = min(len(t), m.end() + ctx)
            print("--- at %d" % m.start()); print(t[s:e]); print()
