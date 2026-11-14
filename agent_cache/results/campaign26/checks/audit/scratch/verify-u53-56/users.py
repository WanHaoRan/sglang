"""List the human-typed messages of a session transcript from a given 0-based index."""
import json
import sys

path = sys.argv[1]
start = int(sys.argv[2])
width = int(sys.argv[3]) if len(sys.argv) > 3 else 700
with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for idx, line in enumerate(fh):
        if idx < start:
            continue
        if '"type":"user"' not in line and '"type": "user"' not in line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") != "user" or rec.get("isSidechain"):
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
        if not texts:
            continue
        text = "\n".join(texts)
        print(f"===== [{idx}] ts={rec.get('timestamp')} meta={rec.get('isMeta')} len={len(text)}")
        print(text[:width])
