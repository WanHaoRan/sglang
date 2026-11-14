import json, sys, re
path = sys.argv[1]
since = sys.argv[2]
pat = re.compile(sys.argv[3], re.I)
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 220
types = set((sys.argv[5] if len(sys.argv) > 5 else "assistant,user").split(","))
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if not pat.search(line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        typ = d.get("type")
        if typ not in types:
            continue
        ts = d.get("timestamp", "")
        if ts < since:
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
        t = "\n".join(texts)
        if t.startswith("<artifact-content-authored-by-others/>") or "This session is being continued from a previous conversation" in t[:600]:
            tag = "SUMMARY"
        else:
            tag = ""
        for m in pat.finditer(t):
            s = max(0, m.start() - ctx); e = min(len(t), m.end() + ctx)
            print("=== idx0=%d ts=%s type=%s %s" % (i, ts, typ, tag))
            print(t[s:e].replace("\n", " \\n "))
            print()
