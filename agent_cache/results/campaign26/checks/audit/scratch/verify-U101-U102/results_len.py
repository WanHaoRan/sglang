import json, sys
src = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
with open(src, "r", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo: continue
        if i > hi: break
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") != "user": continue
        c = d.get("message", {}).get("content")
        if not isinstance(c, list): continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                cc = b.get("content")
                s = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict) and x.get("type") == "text") if isinstance(cc, list) else str(cc)
                has_conc = "-- concerns" in s
                n_conc = s.count("\n   * ")
                print(f"[{i}] {d.get('timestamp','')} tool_result len={len(s)} persisted={'persisted-output' in s[:200]} has_concerns_header={has_conc} bullet_lines={n_conc} head={s[:160]!r}")
