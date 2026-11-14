import sys, json
path = sys.argv[1]
names = set(sys.argv[2].split(","))
since = sys.argv[3] if len(sys.argv) > 3 else ""
with open(path, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if '"tool_use"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        ts = d.get("timestamp", "")
        if ts < since:
            continue
        c = d.get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in names:
                inp = b.get("input", {})
                s = json.dumps(inp, ensure_ascii=False)
                print("line %d ts %s side=%s %s %s" % (i, ts, d.get("isSidechain"), b.get("name"), s[:700]))
