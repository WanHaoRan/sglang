import sys, json
path = sys.argv[1]
ids = {}
with open(path, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if "AskUserQuestion" not in line and not ids:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        c = d.get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and b.get("name") == "AskUserQuestion":
                ids[b.get("id")] = i
                print("ASK line %d ts %s\n%s\n" % (i, d.get("timestamp"), json.dumps(b.get("input"), ensure_ascii=False)[:3000]))
            if b.get("type") == "tool_result" and b.get("tool_use_id") in ids:
                cc = b.get("content")
                s = cc if isinstance(cc, str) else json.dumps(cc, ensure_ascii=False)
                print("ANSWER line %d ts %s\n%s\n" % (i, d.get("timestamp"), s[:2000]))
