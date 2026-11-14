#!/usr/bin/env python3
"""List user prompts (not tool results / system reminders) of another session transcript. Read-only."""
import json, sys
SRC = sys.argv[1]
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
n = 0
first = last = None
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        n += 1
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get('timestamp')
        if ts:
            first = first or ts
            last = ts
        if o.get('type') != 'user' or o.get('isMeta'):
            continue
        c = (o.get('message') or {}).get('content')
        texts = []
        if isinstance(c, str):
            texts = [c]
        elif isinstance(c, list):
            texts = [b.get('text', '') for b in c if isinstance(b, dict) and b.get('type') == 'text']
        real = [t for t in texts if t.strip() and not t.lstrip().startswith(SKIP)]
        if real:
            t = '\n'.join(real)
            if t.startswith('This session is being continued'):
                print('[%d] %s SUMMARY len %d' % (i, ts, len(t)))
            else:
                print('[%d] %s PROMPT: %s' % (i, ts, t[:500].replace('\n', ' ')))
print('lines', n, 'first', first, 'last', last)
