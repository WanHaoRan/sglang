import json, sys, re
path = sys.argv[1]
start = int(sys.argv[2])
pat = re.compile(sys.argv[3])
out = sys.argv[4]
ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 300
types = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else {"user", "assistant"}
kinds = set(sys.argv[7].split(",")) if len(sys.argv) > 7 else {"text"}
n = 0
with open(path, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
    for i, line in enumerate(f):
        if i < start:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in types or d.get("isSidechain"):
            continue
        c = d.get("message", {}).get("content")
        blocks = []
        if isinstance(c, str):
            blocks.append(("text", c))
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                k = b.get("type")
                if k == "text":
                    blocks.append(("text", b.get("text", "")))
                elif k == "thinking":
                    blocks.append(("thinking", b.get("thinking", "")))
                elif k == "tool_use":
                    blocks.append(("tool_use", json.dumps(b.get("input"))))
                elif k == "tool_result":
                    cc = b.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    blocks.append(("tool_result", str(cc)))
        for k, t in blocks:
            if k not in kinds:
                continue
            for m in pat.finditer(t):
                s = max(0, m.start() - ctx)
                e = min(len(t), m.end() + ctx)
                o.write("=== line %d %s/%s ts %s\n...%s...\n\n" % (i, d.get("type"), k, d.get("timestamp"), t[s:e]))
                n += 1
print("hits", n)
