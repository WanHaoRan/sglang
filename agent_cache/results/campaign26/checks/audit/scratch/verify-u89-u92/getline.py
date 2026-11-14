import json, sys
path = sys.argv[1]
want = set(int(x) for x in sys.argv[2].split(","))
out = sys.argv[3]
mx = max(want)
with open(path, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
    for i, line in enumerate(f):
        if i > mx:
            break
        if i not in want:
            continue
        d = json.loads(line)
        msg = d.get("message", {})
        c = msg.get("content")
        o.write("=== line %d type %s ts %s\n" % (i, d.get("type"), d.get("timestamp")))
        if isinstance(c, str):
            o.write(c + "\n")
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    o.write(b.get("text", "") + "\n")
                elif b.get("type") == "tool_use":
                    o.write("[tool_use %s] %s\n" % (b.get("name"), json.dumps(b.get("input"))[:3000]))
                elif b.get("type") == "tool_result":
                    cc = b.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    o.write("[tool_result] %s\n" % (str(cc)[:3000]))
