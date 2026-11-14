#!/usr/bin/env python3
"""Read-only: list real user prompts in the other session files."""
import json, sys, glob, os
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
for p in sorted(glob.glob('/home/wanhr/.claude/projects/-home-wanhr-sglang/*.jsonl')):
    if '7c93b1d5' in p:
        continue
    print('#####', os.path.basename(p), os.path.getsize(p))
    n = 0
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('type') != 'user':
                continue
            c = (o.get('message') or {}).get('content')
            texts = []
            if isinstance(c, str):
                texts = [c]
            elif isinstance(c, list):
                texts = [b.get('text', '') for b in c if isinstance(b, dict) and b.get('type') == 'text']
            j = '\n'.join(texts).strip()
            if not j or j.startswith(SKIP) or o.get('isMeta'):
                continue
            if o.get('isCompactSummary') or j.startswith('This session is being continued'):
                print(f'  [{i}] {o.get("timestamp","")} SUMMARY len={len(j)}')
                continue
            n += 1
            if n <= 40:
                print(f'  [{i}] {o.get("timestamp","")} side={o.get("isSidechain")} :: {j[:300]!r}')
    print('  real prompts:', n)
