#!/usr/bin/env python3
"""Read-only: list real user prompts (non-meta, non-tool-result). No prefilter."""
import json, sys
src, out = sys.argv[1], sys.argv[2]
min_ts = sys.argv[3] if len(sys.argv) > 3 else ""
SKIP = ("<system-reminder>", "<task-notification>", "<local-command", "<command-")
n = 0
with open(src, "r", errors="replace") as f, open(out, "w") as fo:
    for idx, line in enumerate(f):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "user" or o.get("isMeta"):
            continue
        ts = o.get("timestamp") or ""
        if ts < min_ts:
            continue
        c = (o.get("message") or {}).get("content")
        texts = []
        if isinstance(c, str):
            texts = [c]
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text") or "")
        for t in texts:
            t = t.strip()
            if not t or t.startswith(SKIP):
                continue
            n += 1
            fo.write("=== L%d %s side=%s len=%d\n%s\n\n" % (idx, ts, o.get("isSidechain"), len(t), t[:6000]))
print("prompts", n)
