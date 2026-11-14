import json, sys, re
path = sys.argv[1]
pat = re.compile(sys.argv[2], re.I)
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line:
            continue
        if not pat.search(line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user" or d.get("isSidechain") or d.get("isMeta"):
            continue
        c = d.get("message", {}).get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        t = "\n".join(texts)
        if not t or t.startswith("<task-notification") or "This session is being continued from a previous conversation" in t[:700] or t.startswith("<local-command") or t.startswith("<command-"):
            continue
        if len(t) > 3000:
            continue
        if pat.search(t):
            print("=== idx0=%d ts=%s" % (i, d.get("timestamp", "")))
            print(t[:900]); print()
