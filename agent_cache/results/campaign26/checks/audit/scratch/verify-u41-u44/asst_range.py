import json, sys
path = sys.argv[1]
lo, hi = int(sys.argv[2]), int(sys.argv[3])
maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue
        typ = d.get("type")
        if typ not in ("assistant", "user"):
            continue
        if d.get("isSidechain"):
            continue
        msg = d.get("message", {}) or {}
        c = msg.get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        t = "\n".join(x for x in texts if x.strip())
        if not t.strip():
            continue
        if t.startswith("<task-notification") or t.startswith("[Image"):
            continue
        print("=== idx0=%d ts=%s type=%s" % (i, d.get("timestamp", ""), typ))
        print(t[:maxlen])
        print()
