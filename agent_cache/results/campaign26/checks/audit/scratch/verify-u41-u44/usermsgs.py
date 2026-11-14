import json, sys
path = sys.argv[1]
since = sys.argv[2] if len(sys.argv) > 2 else "2026-11-11"
out = []
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line[:400] and '"type": "user"' not in line[:400]:
            # cheap prefilter; still parse if contains user type anywhere early
            if '"type":"user"' not in line:
                continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user":
            continue
        ts = d.get("timestamp", "")
        if ts < since:
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
        out.append((i, ts, d.get("isMeta"), t))
for i, ts, meta, t in out:
    print("=== idx0=%d ts=%s meta=%s len=%d" % (i, ts, meta, len(t)))
    print(t[:1500])
    print()
