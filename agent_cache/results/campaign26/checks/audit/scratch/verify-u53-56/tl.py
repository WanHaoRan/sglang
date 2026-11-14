"""Print chosen lines of a session transcript (0-based index), text parts only."""
import json
import sys

path = sys.argv[1]
want = set()
for arg in sys.argv[2:]:
    if "-" in arg:
        a, b = arg.split("-")
        want.update(range(int(a), int(b) + 1))
    else:
        want.add(int(arg))
limit = 6000
top = max(want)
with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if idx > top:
            break
        if idx not in want:
            continue
        try:
            rec = json.loads(line)
        except Exception as exc:  # a broken line is reported, not fatal
            print(f"[{idx}] unparsable: {exc}")
            continue
        kind = rec.get("type")
        stamp = rec.get("timestamp")
        msg = rec.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            parts.append(("text", content))
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append(("text", part.get("text", "")))
                elif ptype == "tool_use":
                    parts.append(("tool_use", json.dumps(part.get("input"))[:1500]))
                elif ptype == "tool_result":
                    body = part.get("content")
                    if isinstance(body, list):
                        body = " ".join(
                            p.get("text", "") for p in body if isinstance(p, dict)
                        )
                    parts.append(("tool_result", str(body)[:1500]))
                elif ptype == "thinking":
                    parts.append(("thinking", part.get("thinking", "")[:800]))
        print(f"===== [{idx}] type={kind} role={role} ts={stamp} sidechain={rec.get('isSidechain')}")
        for ptype, text in parts:
            print(f"--- {ptype}")
            print(text[:limit])
