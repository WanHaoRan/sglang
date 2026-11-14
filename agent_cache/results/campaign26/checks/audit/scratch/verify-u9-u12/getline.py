"""Read-only: print the tool_use inputs (and optionally tool results) of given transcript line indexes."""
import json, sys
path = sys.argv[1]
want = {int(x) for x in sys.argv[2:]}
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if i not in want:
            continue
        d = json.loads(line)
        print(f"===== [{i}] {d.get('timestamp')} {d.get('type')}")
        content = (d.get("message") or {}).get("content")
        if isinstance(content, str):
            print(content)
        else:
            for c in content or []:
                if c.get("type") == "text":
                    print("--- text\n" + c.get("text", ""))
                elif c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    print("--- tool_use", c.get("name"), c.get("id"))
                    for k, v in inp.items():
                        print(f"[{k}]\n{v}")
                elif c.get("type") == "tool_result":
                    cc = c.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    print("--- tool_result", c.get("tool_use_id"), "\n" + str(cc))
        if max(want) == i:
            break
