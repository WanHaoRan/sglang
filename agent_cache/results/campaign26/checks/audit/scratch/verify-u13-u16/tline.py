"""Print the text blocks of given line numbers of a transcript. usage: tline.py file.jsonl n1 [n2 ...] [--max=N]"""
import json, sys
path = sys.argv[1]
mx = 6000
want = set()
for a in sys.argv[2:]:
    if a.startswith("--max="):
        mx = int(a[6:])
    else:
        want.add(int(a))
with open(path, errors="replace") as f:
    for i, line in enumerate(f, 1):
        if i not in want:
            continue
        d = json.loads(line)
        msg = d.get("message") or {}
        content = msg.get("content")
        print(f"===== line {i} {d.get('timestamp','')} type={d.get('type')} sidechain={d.get('isSidechain')}")
        if isinstance(content, str):
            print(content[:mx])
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        print(c.get("text", "")[:mx])
                    elif c.get("type") == "tool_use":
                        print("[tool_use]", json.dumps(c.get("input"))[:600])
                    elif c.get("type") == "tool_result":
                        cc = c.get("content")
                        print("[tool_result]", (json.dumps(cc) if not isinstance(cc, str) else cc)[:600])
