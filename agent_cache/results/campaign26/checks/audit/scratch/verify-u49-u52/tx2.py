#!/usr/bin/env python3
"""Read-only helper over the lead's transcript.
  tx.py line LO [HI] [MAXCHARS]        print text blocks (user + assistant) of lines LO..HI
  tx.py grep REGEX [LO] [HI] [CTX]     regex over user/assistant text blocks; prints idx, ts, role, snippet
  tx.py grepall REGEX [LO] [HI] [CTX]  as grep, plus thinking blocks and tool_use inputs
"""
import json
import re
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/45edf7fb-7509-4ab1-95e0-795434f50dc8.jsonl"


def blocks(o):
    msg = o.get('message', {}) or {}
    c = msg.get('content')
    if isinstance(c, str):
        return [{'type': 'text', 'text': c}]
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    return []


def lines(lo, hi):
    with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i > hi:
                break
            yield i, line


mode = sys.argv[1]
if mode == 'line':
    lo = int(sys.argv[2])
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else lo
    mx = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    for i, line in lines(lo, hi):
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type')
        if typ not in ('user', 'assistant'):
            continue
        for b in blocks(o):
            if b.get('type') == 'text' and b.get('text', '').strip():
                t = b['text']
                print(f"--- [{i}] {typ} {o.get('timestamp', '')} ---")
                print(t[:mx] + (' ...[cut]' if len(t) > mx else ''))
elif mode in ('grep', 'grepall'):
    pat = re.compile(sys.argv[2], re.I)
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    hi = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    for i, line in lines(lo, hi):
        if not pat.search(line):
            # cheap prefilter on the raw line (escapes aside)
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type')
        if typ not in ('user', 'assistant'):
            continue
        for b in blocks(o):
            bt = b.get('type')
            if bt == 'text':
                t = b.get('text', '')
                tag = 'text'
            elif mode == 'grepall' and bt == 'thinking':
                t = b.get('thinking', '')
                tag = 'thinking'
            elif mode == 'grepall' and bt == 'tool_use':
                t = json.dumps(b.get('input', {}), ensure_ascii=False)
                tag = 'tool_use:' + str(b.get('name'))
            else:
                continue
            for m in pat.finditer(t):
                s = max(0, m.start() - ctx)
                e = min(len(t), m.end() + ctx)
                print(f"[{i}] {typ}/{tag} {o.get('timestamp', '')}: ...{t[s:e]!r}...")
