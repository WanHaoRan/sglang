import json, sys, glob, os
for path in sys.argv[1:]:
    ids = {}
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if "pytest" not in line and '"tool_result"' not in line:
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
                if b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmd = b.get("input", {}).get("command", "")
                    if "-m pytest" in cmd:
                        ids[b["id"]] = i
                        print(f"=== {os.path.basename(path)} line {i} ts {d.get('timestamp')}\n{cmd[:900]}\n")
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in ids:
                    cc = b.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    print(f"--- result (line {ids.pop(b['tool_use_id'])}):\n{str(cc)[-700:]}\n")
