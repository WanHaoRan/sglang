import json, sys
src = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
out = open("asst_text.jsonl", "w")
tools = open("asst_tools.jsonl", "w")
n = 0; m2 = 0
with open(src, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < 21800:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        ts = d.get("timestamp", "")
        msg = d.get("message", {})
        c = msg.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text", "").strip():
                out.write(json.dumps({"i": i, "ts": ts, "side": bool(d.get("isSidechain")), "text": b["text"]}, ensure_ascii=False) + "\n")
                n += 1
            elif b.get("type") == "tool_use":
                inp = b.get("input", {})
                s = json.dumps(inp, ensure_ascii=False)
                tools.write(json.dumps({"i": i, "ts": ts, "name": b.get("name"), "input": s[:6000], "len": len(s)}, ensure_ascii=False) + "\n")
                m2 += 1
out.close(); tools.close()
print("assistant text blocks:", n, "tool uses:", m2)
