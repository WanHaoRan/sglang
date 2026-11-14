"""Read-only: print the assistant's text messages (what the user saw) in a time window."""
import json, sys
path, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if '"assistant"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain"):
            continue
        ts = d.get("timestamp", "")
        if not (lo <= ts <= hi):
            continue
        for c in (d.get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "text" and c.get("text", "").strip():
                print(f"[{i}] {ts} ASSISTANT TEXT:\n{c['text']}\n")
