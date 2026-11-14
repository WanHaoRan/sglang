import json, sys
path = sys.argv[1]
want = set(int(x) for x in sys.argv[2].split(","))
maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
mx = max(want)
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i > mx:
            break
        if i not in want:
            continue
        try:
            d = json.loads(line)
        except Exception as e:
            print("=== idx0=%d PARSE ERROR %s" % (i, e)); continue
        ts = d.get("timestamp", "")
        typ = d.get("type")
        msg = d.get("message", {}) or {}
        c = msg.get("content")
        parts = []
        if isinstance(c, str):
            parts.append(("text", c))
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    parts.append(("text", b.get("text", "")))
                elif b.get("type") == "tool_use":
                    parts.append(("tool_use", json.dumps(b.get("input", {}))[:1500]))
                elif b.get("type") == "tool_result":
                    cc = b.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    parts.append(("tool_result", str(cc)[:1500]))
                elif b.get("type") == "thinking":
                    parts.append(("thinking", (b.get("thinking", "") or "")[:300]))
        print("=== idx0=%d ts=%s type=%s sidechain=%s" % (i, ts, typ, d.get("isSidechain")))
        for k, t in parts:
            print("[%s]" % k)
            print(t[:maxlen])
        print()
