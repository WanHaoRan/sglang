"""Read-only: list transcript lines that mention a pattern, with index, timestamp, role and a snippet around each hit."""
import json, re, sys
path, pat = sys.argv[1], re.compile(sys.argv[2])
width = int(sys.argv[3]) if len(sys.argv) > 3 else 300
maxhits = int(sys.argv[4]) if len(sys.argv) > 4 else 3
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if not pat.search(line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            print(i, "unparsed"); continue
        ts = d.get("timestamp", "")
        typ = d.get("type", "")
        msg = d.get("message") or {}
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            parts.append(("text", content))
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    parts.append(("text", c.get("text", "")))
                elif c.get("type") == "tool_use":
                    parts.append(("tool_use:" + str(c.get("name")), json.dumps(c.get("input"), ensure_ascii=False)))
                elif c.get("type") == "tool_result":
                    cc = c.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    parts.append(("tool_result", str(cc)))
                elif c.get("type") == "thinking":
                    parts.append(("thinking", c.get("thinking", "")))
        for kind, text in parts:
            hits = [m.start() for m in pat.finditer(text)]
            for h in hits[:maxhits]:
                s = text[max(0, h - width):h + width].replace("\n", "\\n")
                print(f"[{i}] {ts} {typ} {kind} sidechain={d.get('isSidechain')}: ...{s}...")
                print()
