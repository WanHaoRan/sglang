"""Print user-typed prompts of a session transcript since a date (read only)."""
import json
import sys

path = sys.argv[1]
since = sys.argv[2] if len(sys.argv) > 2 else "2026-11-05"

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line[:4000] and '"type": "user"' not in line[:4000]:
            # cheap prefilter; type key may sit later in the line, so fall through on doubt
            if '"type":"user"' not in line and '"type": "user"' not in line:
                continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user":
            continue
        if d.get("isSidechain") or d.get("isMeta"):
            continue
        ts = d.get("timestamp", "")
        if ts < since:
            continue
        msg = d.get("message", {})
        c = msg.get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        for t in texts:
            t = t.strip()
            if not t:
                continue
            if t.startswith("<system-reminder>") or t.startswith("<task-notification>"):
                continue
            if t.startswith("<local-command") or t.startswith("<command-name>"):
                continue
            print("=== line %d @ %s (len %d) ===" % (i, ts, len(t)))
            print(t[:3000])
            print()
