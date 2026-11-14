import json, sys
path = sys.argv[1]
want = set(int(x) for x in sys.argv[2].split(","))
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i in want:
            d = json.loads(line)
            m = d.get("message", {})
            c = m.get("content")
            if isinstance(c, str):
                txt = c
            else:
                txt = "\n".join((b.get("text") or json.dumps(b)[:3000]) if isinstance(b, dict) else str(b) for b in c)
            print(f"=== line {i} ts {d.get('timestamp')} type {d.get('type')} ===")
            print(txt)
            print()
        if i > max(want):
            break
