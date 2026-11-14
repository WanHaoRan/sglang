#!/usr/bin/env python3
"""List real user prompts (not tool results, not system text) of a session transcript: prompts.py FILE [SINCE_ISO]"""
import json, sys
src = sys.argv[1]
since = sys.argv[2] if len(sys.argv) > 2 else ''
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
with open(src, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('type') != 'user' or o.get('isMeta') or o.get('isCompactSummary'):
            continue
        ts = o.get('timestamp', '')
        if ts < since:
            continue
        c = (o.get('message') or {}).get('content')
        texts = [c] if isinstance(c, str) else [b.get('text', '') for b in (c or []) if isinstance(b, dict) and b.get('type') == 'text']
        real = [t for t in texts if t.strip() and not t.lstrip().startswith(SKIP) and not t.startswith('This session is being continued')]
        if real:
            print(f"[{i}] {ts}: " + ' | '.join(t.replace('\n', ' ')[:700] for t in real))
