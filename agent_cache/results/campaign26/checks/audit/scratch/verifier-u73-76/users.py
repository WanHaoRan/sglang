import json, sys
path = sys.argv[1]
out = open(sys.argv[2], "w")
n = 0
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line[:4000] and '"type": "user"' not in line[:4000]:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user":
            continue
        if d.get("isSidechain"):
            continue
        m = d.get("message", {})
        c = m.get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        if not texts:
            continue
        t = "\n".join(texts)
        out.write(f"=== line {i} ts {d.get('timestamp')} meta={d.get('isMeta')} ===\n{t[:6000]}\n\n")
        n += 1
print(n)
