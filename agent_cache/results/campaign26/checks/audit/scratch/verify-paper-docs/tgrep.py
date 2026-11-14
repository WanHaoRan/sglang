#!/usr/bin/env python3
"""tgrep.py SRC PATTERN [LO] [HI] [--kinds user,atext,tool,think] [--ctx N]
Regex search over a Claude Code transcript (.jsonl). Read-only.
Prints: line index, timestamp, kind, and a snippet around each match.
"""
import json, re, sys

src = sys.argv[1]
pat = re.compile(sys.argv[2], re.I)
args = sys.argv[3:]
kinds = {'user', 'atext'}
ctx = 220
pos = []
i = 0
while i < len(args):
    a = args[i]
    if a == '--kinds':
        kinds = set(args[i + 1].split(',')); i += 2; continue
    if a == '--ctx':
        ctx = int(args[i + 1]); i += 2; continue
    pos.append(a); i += 1
lo = int(pos[0]) if len(pos) > 0 else 0
hi = int(pos[1]) if len(pos) > 1 else 10 ** 12


def blocks(o):
    msg = o.get('message', {}) or {}
    c = msg.get('content')
    if isinstance(c, str):
        return [{'type': 'text', 'text': c}]
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    return []


with open(src, 'r', encoding='utf-8', errors='replace') as f:
    for n, line in enumerate(f):
        if n < lo:
            continue
        if n > hi:
            break
        if not pat.search(line):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type')
        ts = o.get('timestamp', '')
        side = o.get('isSidechain')
        for b in blocks(o):
            bt = b.get('type')
            if bt == 'text':
                kind = 'user' if typ == 'user' else 'atext'
                text = b.get('text', '')
            elif bt == 'thinking':
                kind = 'think'; text = b.get('thinking', '')
            elif bt == 'tool_use':
                kind = 'tool'; text = json.dumps(b.get('input', {}), ensure_ascii=False)
            elif bt == 'tool_result':
                kind = 'result'
                c = b.get('content')
                if isinstance(c, list):
                    c = '\n'.join(x.get('text', '') for x in c if isinstance(x, dict) and x.get('type') == 'text')
                text = str(c)
            else:
                continue
            if kind not in kinds:
                continue
            for m in pat.finditer(text):
                s = max(0, m.start() - ctx); e = min(len(text), m.end() + ctx)
                snip = text[s:e].replace('\n', ' / ')
                print(f"[{n}] {ts} {kind}{' side' if side else ''}: ...{snip}...")
