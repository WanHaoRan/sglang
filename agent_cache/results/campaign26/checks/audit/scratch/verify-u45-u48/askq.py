#!/usr/bin/env python3
"""Read-only: find AskUserQuestion-style tool uses and their answers since 10 Nov in the main transcript."""
import json
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
ids = {}
names = {}
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if '"timestamp"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get('timestamp', '')
        if ts < '2026-11-10':
            continue
        c = (o.get('message') or {}).get('content')
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get('type') == 'tool_use':
                n = b.get('name', '')
                names[n] = names.get(n, 0) + 1
                if 'ask' in n.lower() or 'question' in n.lower():
                    ids[b.get('id')] = (i, ts)
                    print('ASK', i, ts, json.dumps(b.get('input'))[:1500])
            elif b.get('type') == 'tool_result' and b.get('tool_use_id') in ids:
                cc = b.get('content')
                if isinstance(cc, list):
                    cc = ' '.join(x.get('text', '') for x in cc if isinstance(x, dict))
                print('ANSWER', i, ts, str(cc)[:1500])
print(sorted(names.items(), key=lambda kv: -kv[1])[:30])
