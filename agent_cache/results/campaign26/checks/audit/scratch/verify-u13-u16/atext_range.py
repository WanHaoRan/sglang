"""Print assistant text blocks (main chain) in a line range. usage: atext_range.py file lo hi [maxchars]"""
import json, sys
path, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
with open(path, errors="replace") as f:
    for i, line in enumerate(f, 1):
        if i < lo:
            continue
        if i > hi:
            break
        if '"type":"assistant"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain"):
            continue
        for c in (d.get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "text" and c.get("text", "").strip():
                print(f"--- [{i}] {d.get('timestamp','')}")
                print(c["text"][:mx])
