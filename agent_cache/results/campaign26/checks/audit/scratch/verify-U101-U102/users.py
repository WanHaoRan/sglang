import json, sys
src = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
out = open("user_msgs.tsv", "w")
n = 0
with open(src, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line[:4000] and '"type": "user"' not in line[:4000]:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "user":
            continue
        ts = d.get("timestamp", "")
        if ts < "2026-11-10":
            continue
        m = d.get("message", {})
        c = m.get("content")
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        if not texts:
            continue
        t = "\n".join(texts)
        flags = []
        if d.get("isMeta"): flags.append("meta")
        if d.get("isSidechain"): flags.append("side")
        if d.get("isCompactSummary"): flags.append("compact")
        out.write(f"{i}\t{ts}\t{','.join(flags)}\t{len(t)}\t{json.dumps(t[:1500], ensure_ascii=False)}\n")
        n += 1
out.close()
print("user text messages since 2026-11-10:", n)
