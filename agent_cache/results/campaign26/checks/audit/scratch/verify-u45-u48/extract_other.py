#!/usr/bin/env python3
"""Read-only: dump user prompts and assistant text of the ICML discussion session."""
import json, os
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/45edf7fb-7509-4ab1-95e0-795434f50dc8.jsonl'
OUT = os.path.dirname(os.path.abspath(__file__))
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
w = open(os.path.join(OUT, 'icml_session.txt'), 'w')
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type'); ts = o.get('timestamp', '')
        c = (o.get('message') or {}).get('content')
        if typ == 'user':
            texts = [c] if isinstance(c, str) else [b.get('text', '') for b in (c or []) if isinstance(b, dict) and b.get('type') == 'text']
            j = '\n'.join(texts).strip()
            if not j or j.startswith(SKIP) or o.get('isMeta'):
                continue
            kind = 'SUMMARY' if (o.get('isCompactSummary') or j.startswith('This session is being continued')) else 'USER'
            w.write(f"=== line {i} @ {ts} {kind} len={len(j)} ===\n{j if kind=='USER' else j[:300]}\n\n")
        elif typ == 'assistant' and isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get('type') == 'text' and b.get('text', '').strip():
                    w.write(f"=== line {i} @ {ts} ASSISTANT len={len(b['text'])} ===\n{b['text']}\n\n")
w.close()
