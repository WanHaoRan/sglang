import sys, json
path = sys.argv[1]
lo = int(sys.argv[2]); hi = int(sys.argv[3])
out = sys.argv[4]
minlen = int(sys.argv[5]) if len(sys.argv) > 5 else 200
with open(path, encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        if '"type":"assistant"' not in line and '"type": "assistant"' not in line:
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
                if len(t) >= minlen:
                    o.write("##### line %d ts %s\n%s\n\n" % (i, d.get("timestamp", ""), t))
print("ok ->", out)
