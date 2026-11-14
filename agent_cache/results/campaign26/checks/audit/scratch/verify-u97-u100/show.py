#!/usr/bin/env python3
"""Print the text blocks of given transcript line indices (0-based). usage: show.py <jsonl> <idx> [idx...]"""
import sys, json
src = sys.argv[1]
want = set(int(x) for x in sys.argv[2:])
mx = max(want)
with open(src, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i > mx:
            break
        if i not in want:
            continue
        o = json.loads(line)
        typ = o.get("type"); ts = o.get("timestamp", "")
        msg = o.get("message") or {}
        c = msg.get("content")
        if isinstance(c, str):
            print(f"[{i}] {ts} {typ}/str:\n{c}\n-----")
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    print(f"[{i}] {ts} {typ}/text:\n{b.get('text','')}\n-----")
                elif bt == "tool_use":
                    print(f"[{i}] {ts} {typ}/tool_use:{b.get('name')}:\n{json.dumps(b.get('input'), ensure_ascii=False)[:6000]}\n-----")
                elif bt == "tool_result":
                    cc = b.get("content")
                    if isinstance(cc, list):
                        cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    print(f"[{i}] {ts} {typ}/tool_result:\n{str(cc)[:6000]}\n-----")
