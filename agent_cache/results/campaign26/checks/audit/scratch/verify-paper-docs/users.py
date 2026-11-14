#!/usr/bin/env python3
"""users.py SRC [MAXLEN] : print real user prompts (not tool results, not sidechains) with line index and timestamp."""
import json, sys

src = sys.argv[1]
mx = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
with open(src, 'r', encoding='utf-8', errors='replace') as f:
    for n, line in enumerate(f):
        if '"type":"user"' not in line and '"type": "user"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('type') != 'user' or o.get('isSidechain'):
            continue
        msg = o.get('message', {}) or {}
        c = msg.get('content')
        texts = []
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get('type') == 'text':
                    texts.append(b.get('text', ''))
        for t in texts:
            if not t.strip():
                continue
            if t.startswith('<system-reminder>') or t.startswith('<task-notification>') or t.startswith('<local-command'):
                continue
            flag = ''
            if o.get('isCompactSummary') or t.startswith('This session is being continued'):
                flag = ' [compact summary]'
                t = t[:200]
            print(f"=== line {n} @ {o.get('timestamp','')} (len {len(t)}){flag} ===")
            print(t[:mx])
            print()
