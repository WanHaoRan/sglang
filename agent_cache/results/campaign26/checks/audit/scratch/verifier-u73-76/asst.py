import json, sys
path = sys.argv[1]
out = open(sys.argv[2], "w")
start = int(sys.argv[3])
n = 0
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < start:
            continue
        if '"type":"assistant"' not in line[:6000]:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain"):
            continue
        c = d.get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip():
                out.write(f"=== line {i} ts {d.get('timestamp')} ===\n{b['text']}\n\n")
                n += 1
print(n)
