"""List real user prompts (not tool results) of a transcript. usage: uprompts.py file.jsonl [min_line] [maxchars]"""
import json, sys
path = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 700
with open(path, errors="replace") as f:
    for i, line in enumerate(f, 1):
        if i < lo or '"type":"user"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user" or d.get("isSidechain"):
            continue
        c = (d.get("message") or {}).get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
                continue
            texts += [x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"]
        t = "\n".join(texts).strip()
        if not t:
            continue
        print(f"=== line {i} @ {d.get('timestamp','')} (len {len(t)}) meta={d.get('isMeta')} compact={d.get('isCompactSummary')}")
        print(t[:mx])
