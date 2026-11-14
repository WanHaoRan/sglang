#!/usr/bin/env python3
"""Build a light index of the transcript: per line -> type, timestamp, kind, short text.
Writes index.tsv, user_prompts.txt, summaries_idx.txt in the scratch folder. Read-only on the transcript."""
import json, sys, os

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
OUT = os.path.dirname(os.path.abspath(__file__))

SKIP_PREFIX = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')

def texts_of(content):
    """Return (list of text strings, has_tool_result)"""
    if isinstance(content, str):
        return [content], False
    out = []
    has_tr = False
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get('type')
            if t == 'text':
                out.append(b.get('text', ''))
            elif t == 'tool_result':
                has_tr = True
    return out, has_tr

n = 0
prompts = []
summaries = []
types = {}
with open(SRC, 'r', encoding='utf-8', errors='replace') as f, \
        open(os.path.join(OUT, 'index.tsv'), 'w') as idx:
    for i, line in enumerate(f):
        n += 1
        try:
            o = json.loads(line)
        except Exception as e:
            idx.write(f"{i}\tBADJSON\t\t\t{len(line)}\n")
            continue
        typ = o.get('type')
        ts = o.get('timestamp', '')
        types[typ] = types.get(typ, 0) + 1
        kind = ''
        short = ''
        if typ == 'user':
            msg = o.get('message', {}) or {}
            texts, has_tr = texts_of(msg.get('content'))
            is_meta = bool(o.get('isMeta'))
            is_compact = bool(o.get('isCompactSummary'))
            joined = '\n'.join(texts)
            if joined.startswith('This session is being continued from a previous conversation') or is_compact:
                kind = 'SUMMARY'
                summaries.append((i, ts, len(joined)))
            elif has_tr and not texts:
                kind = 'tool_result'
            elif is_meta:
                kind = 'meta'
            else:
                real = [t for t in texts if t.strip() and not t.lstrip().startswith(SKIP_PREFIX)]
                if real:
                    kind = 'PROMPT'
                    prompts.append((i, ts, '\n'.join(real)))
                else:
                    kind = 'sys' if texts else ('tool_result' if has_tr else 'empty')
            short = joined[:160].replace('\n', ' ').replace('\t', ' ')
        elif typ == 'assistant':
            msg = o.get('message', {}) or {}
            c = msg.get('content')
            kinds = []
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        bt = b.get('type')
                        if bt == 'text':
                            kinds.append('text')
                            if not short:
                                short = b.get('text', '')[:160].replace('\n', ' ').replace('\t', ' ')
                        elif bt == 'tool_use':
                            kinds.append('tool:' + str(b.get('name')))
                        elif bt == 'thinking':
                            kinds.append('think')
            kind = ','.join(kinds)
        else:
            kind = str(o.get('subtype', ''))
        idx.write(f"{i}\t{typ}\t{ts}\t{kind}\t{len(line)}\t{short}\n")

with open(os.path.join(OUT, 'user_prompts.txt'), 'w') as f:
    for i, ts, t in prompts:
        f.write(f"=== line {i} @ {ts} (len {len(t)}) ===\n{t}\n\n")
with open(os.path.join(OUT, 'summaries_idx.txt'), 'w') as f:
    for i, ts, l in summaries:
        f.write(f"{i}\t{ts}\t{l}\n")
print('lines', n)
print('types', types)
print('prompts', len(prompts))
print('summaries', len(summaries))
