"""Print the text parts of given transcript lines. Usage: python3 -I -B tshow.py [--max N] IDX [IDX ...]  (IDX may be A-B)"""
import json, sys

PATH = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
args = sys.argv[1:]
maxc = 6000
if args and args[0] == "--max":
    maxc = int(args[1])
    args = args[2:]
want = set()
for a in args:
    if "-" in a:
        lo, hi = a.split("-")
        want.update(range(int(lo), int(hi) + 1))
    else:
        want.add(int(a))
top = max(want)
with open(PATH, errors="replace") as f:
    for i, line in enumerate(f):
        if i > top:
            break
        if i not in want:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            print(i, "unparsable")
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        out = []
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        out.append(c.get("text") or "")
                    elif c.get("type") == "tool_use":
                        out.append("[tool_use " + str(c.get("name")) + "] " + json.dumps(c.get("input"))[:maxc // 3])
                    elif c.get("type") == "tool_result":
                        cc = c.get("content")
                        out.append("[tool_result] " + (cc if isinstance(cc, str) else json.dumps(cc))[:maxc // 3])
        text = "\n".join(out)
        if not text.strip():
            continue
        print(f"===== {i} {rec.get('timestamp', '')} {rec.get('type')}")
        print(text[:maxc])
