import json, sys, re
path, outp, start, pat = sys.argv[1], sys.argv[2], int(sys.argv[3]), re.compile(sys.argv[4], re.I)
out = open(outp, "w")
n = 0
ids = {}
with open(path, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < start:
            continue
        if not pat.search(line):
            # still need tool results for matched ids
            if not ids or '"tool_result"' not in line:
                continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("isSidechain"):
            continue
        c = d.get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                inp = b.get("input", {})
                s = json.dumps(inp)
                if pat.search(s):
                    ids[b.get("id")] = i
                    cmd = inp.get("command") or inp.get("prompt") or s
                    out.write(f"=== line {i} ts {d.get('timestamp')} tool {b.get('name')} ===\n{str(cmd)[:2500]}\n\n")
                    n += 1
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in ids:
                cc = b.get("content")
                if isinstance(cc, list):
                    cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                out.write(f"--- result for line {ids.pop(b.get('tool_use_id'))} (at line {i}) ---\n{str(cc)[-1800:]}\n\n")
print(n)
