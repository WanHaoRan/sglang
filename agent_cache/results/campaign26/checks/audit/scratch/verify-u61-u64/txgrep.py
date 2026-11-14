#!/usr/bin/env python3
"""Read-only: grep the lead transcript's user prompts and assistant text blocks (not tool io unless --tools) for a regex.
usage: txgrep.py REGEX [LO] [HI] [ctx] [--tools] [--think]"""
import json, re, sys

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
pat = re.compile(args[0], re.I)
lo = int(args[1]) if len(args) > 1 else 0
hi = int(args[2]) if len(args) > 2 else 10**9
ctx = int(args[3]) if len(args) > 3 else 200
tools = '--tools' in sys.argv
think = '--think' in sys.argv


def emit(i, t, ts, kind, s):
    for m in pat.finditer(s):
        a = max(0, m.start() - ctx)
        b = min(len(s), m.end() + ctx)
        print(f"[{i}] {t} {ts} {kind}: ...{s[a:b]!r}...")


with open(SRC, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        if not pat.search(line):
            continue
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
            emit(i, t, ts, 'TEXT', c)
            continue
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get('type')
            if bt == 'text':
                emit(i, t, ts, 'TEXT', b.get('text', ''))
            elif think and bt == 'thinking':
                emit(i, t, ts, 'THINK', b.get('thinking', ''))
            elif tools and bt == 'tool_use':
                emit(i, t, ts, 'TOOL_USE ' + str(b.get('name')), json.dumps(b.get('input'), ensure_ascii=False))
            elif tools and bt == 'tool_result':
                cc = b.get('content')
                if isinstance(cc, list):
                    cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                emit(i, t, ts, 'TOOL_RESULT', str(cc))
