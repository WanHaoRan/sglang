#!/usr/bin/env python3
"""Read-only: dump user prompts and assistant text messages since a date from the main transcript."""
import json, sys, os
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
OUT = os.path.dirname(os.path.abspath(__file__))
SINCE = '2026-11-10T00:00:00'
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
up = open(os.path.join(OUT, 'user_prompts.txt'), 'w')
at = open(os.path.join(OUT, 'assistant_text.txt'), 'w')
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if '"timestamp"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get('timestamp', '')
        if ts < SINCE:
            continue
        typ = o.get('type')
        msg = o.get('message', {}) or {}
        c = msg.get('content')
        if typ == 'user':
            texts = []
            has_tr = False
            if isinstance(c, str):
                texts = [c]
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        if b.get('type') == 'text':
                            texts.append(b.get('text', ''))
                        elif b.get('type') == 'tool_result':
                            has_tr = True
            joined = '\n'.join(texts)
            if not joined.strip():
                continue
            kind = 'PROMPT'
            if o.get('isCompactSummary') or joined.startswith('This session is being continued'):
                kind = 'SUMMARY'
            elif o.get('isMeta'):
                kind = 'META'
            elif joined.lstrip().startswith(SKIP):
                kind = 'SYS'
            side = 'SIDE' if o.get('isSidechain') else 'MAIN'
            up.write(f"=== line {i} @ {ts} kind={kind} {side} len={len(joined)} ===\n")
            up.write((joined if kind == 'PROMPT' else joined[:600]) + "\n\n")
        elif typ == 'assistant':
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get('type') == 'text' and b.get('text', '').strip():
                        side = 'SIDE' if o.get('isSidechain') else 'MAIN'
                        at.write(f"=== line {i} @ {ts} {side} len={len(b['text'])} ===\n")
                        at.write(b['text'] + "\n\n")
up.close(); at.close()
