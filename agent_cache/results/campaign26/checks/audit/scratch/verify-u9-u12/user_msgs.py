"""Read-only: print the user's own typed messages (not tool results, not summaries) in a time window."""
import json, sys
path, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line and '"type": "user"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user" or d.get("isSidechain"):
            continue
        ts = d.get("timestamp", "")
        if not (lo <= ts <= hi):
            continue
        c = (d.get("message") or {}).get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for x in c:
                if isinstance(x, dict) and x.get("type") == "text":
                    texts.append(x.get("text", ""))
        for t in texts:
            if t.startswith("<") or "This session is being continued" in t[:200] or len(t) > 3000:
                print(f"[{i}] {ts} (system/summary/long text, {len(t)} chars): {t[:120]!r}")
            else:
                print(f"[{i}] {ts} USER: {t}")
