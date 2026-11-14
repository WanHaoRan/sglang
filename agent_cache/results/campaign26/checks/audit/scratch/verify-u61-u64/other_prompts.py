#!/usr/bin/env python3
"""Read-only: list real user prompts (and optionally assistant text) of another session transcript.
usage: other_prompts.py FILE [--assistant] [maxchars]"""
import json, sys

src = sys.argv[1]
show_asst = '--assistant' in sys.argv
nums = [a for a in sys.argv[2:] if a.isdigit()]
mx = int(nums[0]) if nums else 3000
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
with open(src, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get('type')
        if t not in ('user', 'assistant'):
            continue
        ts = o.get('timestamp', '')
        c = (o.get('message') or {}).get('content')
        texts = []
        if isinstance(c, str):
            texts = [c]
        elif isinstance(c, list):
            texts = [b.get('text', '') for b in c if isinstance(b, dict) and b.get('type') == 'text']
        if t == 'user':
            if o.get('isMeta') or o.get('isCompactSummary'):
                continue
            real = [x for x in texts if x.strip() and not x.lstrip().startswith(SKIP)]
            if real:
                joined = '\n'.join(real)
                if joined.startswith('This session is being continued'):
                    print(f"=== [{i}] SUMMARY {ts} (len {len(joined)})\n")
                    continue
                print(f"=== [{i}] USER {ts}\n{joined[:mx]}\n")
        elif show_asst:
            for x in texts:
                if x.strip():
                    print(f"--- [{i}] ASSISTANT {ts}\n{x[:mx]}\n")
