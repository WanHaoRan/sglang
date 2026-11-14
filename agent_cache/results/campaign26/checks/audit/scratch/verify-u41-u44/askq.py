import json, sys
path = sys.argv[1]; since = sys.argv[2]
ids = {}
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if "AskUserQuestion" not in line and not any(k in line for k in ids):
            continue
        try: d = json.loads(line)
        except Exception: continue
        ts = d.get("timestamp", "")
        if ts < since or d.get("isSidechain"): continue
        c = (d.get("message") or {}).get("content")
        if not isinstance(c, list): continue
        for b in c:
            if not isinstance(b, dict): continue
            if b.get("type") == "tool_use" and b.get("name") == "AskUserQuestion":
                ids[b.get("id")] = i
                print("=== ASK idx0=%d ts=%s" % (i, ts)); print(json.dumps(b.get("input"))[:1500]); print()
            if b.get("type") == "tool_result" and b.get("tool_use_id") in ids:
                cc = b.get("content")
                if isinstance(cc, list): cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                print("=== ANSWER idx0=%d ts=%s" % (i, ts)); print(str(cc)[:1200]); print()
