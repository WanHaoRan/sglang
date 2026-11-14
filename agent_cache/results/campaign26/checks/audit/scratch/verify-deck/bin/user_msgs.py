import sys, json
path = sys.argv[1]
since = sys.argv[2] if len(sys.argv) > 2 else "2026-11-10"
out = sys.argv[3]
n = 0
with open(path, encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
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
            has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
            if has_tool_result:
                continue
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        t = "\n".join(texts).strip()
        if not t:
            continue
        n += 1
        o.write("##### line %d ts %s meta=%s\n%s\n\n" % (i, ts, d.get("isMeta"), t[:6000]))
print(n, "user text messages since", since, "->", out)
