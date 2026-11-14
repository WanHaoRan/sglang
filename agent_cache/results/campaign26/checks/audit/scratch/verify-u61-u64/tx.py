#!/usr/bin/env python3
"""Read-only: print user prompts and assistant text blocks of the lead transcript in a line range.
usage: tx.py LO HI [maxchars] [--tools]"""
import json, sys

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
lo, hi = int(sys.argv[1]), int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 100000
tools = '--tools' in sys.argv
with open(SRC, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get('type')
        if t not in ('user', 'assistant'):
            continue
        ts = o.get('timestamp', '')
        c = (o.get('message') or {}).get('content')
        if isinstance(c, str):
            print(f"[{i}] {t} {ts} TEXT:\n{c[:mx]}\n")
            continue
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get('type')
            if bt == 'text':
                print(f"[{i}] {t} {ts} TEXT:\n{b.get('text', '')[:mx]}\n")
            elif tools and bt == 'tool_use':
                print(f"[{i}] {t} {ts} TOOL_USE {b.get('name')}: {json.dumps(b.get('input'), ensure_ascii=False)[:mx]}\n")
            elif tools and bt == 'tool_result':
                cc = b.get('content')
                if isinstance(cc, list):
                    cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                print(f"[{i}] {t} {ts} TOOL_RESULT: {str(cc)[:mx]}\n")
