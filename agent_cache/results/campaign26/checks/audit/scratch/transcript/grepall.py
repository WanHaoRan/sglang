#!/usr/bin/env python3
"""grepall.py PATTERN [lo hi [ctx [kinds]]]
Regex search over the transcript. kinds: comma list of text,thinking,tool_use,tool_result,user (default: text,thinking,user).
Prints line index, timestamp, kind, snippet."""
import json, sys, re

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
pat = re.compile(sys.argv[1], re.I)
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 200
kinds = set((sys.argv[5] if len(sys.argv) > 5 else 'text,thinking,user').split(','))
maxhits = int(sys.argv[6]) if len(sys.argv) > 6 else 60
n = 0
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        if '"type":"assistant"' not in line and '"type":"user"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type')
        if typ not in ('user', 'assistant'):
            continue
        msg = o.get('message', {}) or {}
        c = msg.get('content')
        if isinstance(c, str):
            c = [{'type': 'text', 'text': c}]
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get('type')
            if bt == 'text':
                k = 'user' if typ == 'user' else 'text'
                t = b.get('text', '')
            elif bt == 'thinking':
                k = 'thinking'
                t = b.get('thinking', '')
            elif bt == 'tool_use':
                k = 'tool_use'
                t = json.dumps(b.get('input', {}), ensure_ascii=False)
            elif bt == 'tool_result':
                k = 'tool_result'
                cc = b.get('content')
                if isinstance(cc, list):
                    cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict) and x.get('type') == 'text')
                t = str(cc)
            else:
                continue
            if k not in kinds:
                continue
            for m in pat.finditer(t):
                s = max(0, m.start() - ctx); e = min(len(t), m.end() + ctx)
                print(f"[{i}] {k} {o.get('timestamp','')}: ...{t[s:e]}...".replace('\n', ' | '))
                n += 1
                if n >= maxhits:
                    sys.exit(0)
                break
