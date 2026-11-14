import json, sys
path = sys.argv[1]
out = sys.argv[2]
since = sys.argv[3] if len(sys.argv) > 3 else ""
n = 0
with open(path, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
    for i, line in enumerate(f):
        if '"type":"user"' not in line and '"type": "user"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user":
            continue
        ts = d.get("timestamp", "")
        if since and ts < since:
            continue
        if d.get("isSidechain"):
            continue
        msg = d.get("message", {})
        c = msg.get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        if not texts:
            continue
        t = "\n".join(texts)
        o.write("=== line %d ts %s meta=%s\n%s\n" % (i, ts, d.get("isMeta"), t[:6000]))
        n += 1
print("user text messages:", n)
