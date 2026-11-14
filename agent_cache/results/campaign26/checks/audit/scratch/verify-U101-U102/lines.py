import json, sys
src = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
with open(src, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo: continue
        if i > hi: break
        try:
            d = json.loads(line)
        except Exception:
            print(i, "BAD"); continue
        t = d.get("type"); ts = d.get("timestamp", "")
        msg = d.get("message", {})
        c = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(c, str):
            print(f"[{i}] {ts} {t} TEXT: {c[:maxc]}")
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict): continue
                bt = b.get("type")
                if bt == "text":
                    print(f"[{i}] {ts} {t} text: {b.get('text','')[:maxc]}")
                elif bt == "tool_use":
                    print(f"[{i}] {ts} {t} tool_use {b.get('name')}: {json.dumps(b.get('input'), ensure_ascii=False)[:maxc]}")
                elif bt == "tool_result":
                    cc = b.get("content")
                    if isinstance(cc, list):
                        s = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict) and x.get("type") == "text")
                    else:
                        s = str(cc)
                    print(f"[{i}] {ts} {t} tool_result: {s[:maxc]}")
                elif bt == "thinking":
                    pass
        else:
            if t not in ("user", "assistant"):
                s = json.dumps(d, ensure_ascii=False)
                print(f"[{i}] {ts} {t} other: {s[:300]}")
