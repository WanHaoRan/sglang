"""Scan assistant and human text of a transcript (from a 0-based index) for patterns; print snippets."""
import json
import re
import sys

path, start = sys.argv[1], int(sys.argv[2])
pats = [re.compile(p, re.I) for p in sys.argv[3:]]
with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if idx < start:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("isSidechain") or rec.get("type") not in ("assistant", "user"):
            continue
        content = (rec.get("message") or {}).get("content")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
        for text in texts:
            if rec.get("type") == "user" and len(text) > 5000:
                continue  # context summaries, not the user's words
            for pat in pats:
                for m in pat.finditer(text):
                    lo, hi = max(0, m.start() - 220), min(len(text), m.end() + 320)
                    snippet = text[lo:hi].replace("\n", " ")
                    print(f"[{idx}] {rec.get('type')} {rec.get('timestamp')} /{pat.pattern}/: ...{snippet}...")
                    print()
                    break
