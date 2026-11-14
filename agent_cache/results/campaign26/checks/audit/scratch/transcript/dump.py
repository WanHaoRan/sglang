#!/usr/bin/env python3
"""dump.py MODE args...
  line  i [j]           : print full text of user/assistant text blocks for lines i..j (inclusive)
  summ  i               : write summary at line i to summ_<i>.txt
  atext i j [maxchars]  : print assistant text blocks (not thinking, not tool) between lines i..j, with index + ts
  tools i j             : print tool_use short lines between i..j
  grep  pattern [i j]   : regex search over assistant text + user prompt text + tool inputs; print line idx, ts, kind, snippet
"""
import json, sys, os, re

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
OUT = os.path.dirname(os.path.abspath(__file__))

def iter_lines(lo, hi):
    with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i > hi:
                break
            # cheap prefilter
            yield i, line

def blocks(o):
    msg = o.get('message', {}) or {}
    c = msg.get('content')
    if isinstance(c, str):
        return [{'type': 'text', 'text': c}]
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    return []

mode = sys.argv[1]
if mode == 'line':
    lo = int(sys.argv[2]); hi = int(sys.argv[3]) if len(sys.argv) > 3 else lo
    for i, line in iter_lines(lo, hi):
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type')
        if typ not in ('user', 'assistant'):
            continue
        for b in blocks(o):
            bt = b.get('type')
            if bt == 'text':
                print(f"--- [{i}] {typ} {o.get('timestamp','')} text ---")
                print(b.get('text', ''))
            elif bt == 'thinking':
                print(f"--- [{i}] {typ} {o.get('timestamp','')} thinking ---")
                print(b.get('thinking', ''))
            elif bt == 'tool_use':
                print(f"--- [{i}] {typ} {o.get('timestamp','')} tool_use {b.get('name')} ---")
                print(json.dumps(b.get('input', {}), ensure_ascii=False)[:6000])
            elif bt == 'tool_result':
                print(f"--- [{i}] {typ} {o.get('timestamp','')} tool_result ---")
                c = b.get('content')
                if isinstance(c, list):
                    c = '\n'.join(x.get('text', '') for x in c if isinstance(x, dict) and x.get('type') == 'text')
                print(str(c)[:6000])
elif mode == 'summ':
    i0 = int(sys.argv[2])
    for i, line in iter_lines(i0, i0):
        o = json.loads(line)
        txt = '\n'.join(b.get('text', '') for b in blocks(o) if b.get('type') == 'text')
        p = os.path.join(OUT, f'summ_{i0}.txt')
        with open(p, 'w') as f:
            f.write(txt)
        print(p, len(txt), o.get('timestamp'))
elif mode == 'atext':
    lo = int(sys.argv[2]); hi = int(sys.argv[3])
    mx = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
    for i, line in iter_lines(lo, hi):
        if '"type":"assistant"' not in line and '"type": "assistant"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('type') != 'assistant':
            continue
        for b in blocks(o):
            if b.get('type') == 'text' and b.get('text', '').strip():
                t = b.get('text', '')
                print(f"--- [{i}] {o.get('timestamp','')} ---")
                print(t[:mx] + (' ...[cut]' if len(t) > mx else ''))
elif mode == 'tools':
    lo = int(sys.argv[2]); hi = int(sys.argv[3])
    for i, line in iter_lines(lo, hi):
        if '"tool_use"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('type') != 'assistant':
            continue
        for b in blocks(o):
            if b.get('type') == 'tool_use':
                inp = b.get('input', {})
                s = inp.get('description') or inp.get('file_path') or inp.get('command') or json.dumps(inp, ensure_ascii=False)
                print(f"[{i}] {o.get('timestamp','')} {b.get('name')}: {str(s)[:300]}")
elif mode == 'grep':
    pat = re.compile(sys.argv[2], re.I)
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    hi = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    for i, line in iter_lines(lo, hi):
        if '"type":"assistant"' not in line and '"type":"user"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type')
        if typ not in ('user', 'assistant'):
            continue
        for b in blocks(o):
            bt = b.get('type')
            if bt == 'text':
                t = b.get('text', '')
            elif bt == 'thinking':
                continue
            elif bt == 'tool_use':
                continue
            else:
                continue
            for m in pat.finditer(t):
                s = max(0, m.start() - ctx); e = min(len(t), m.end() + ctx)
                print(f"[{i}] {typ} {o.get('timestamp','')}: ...{t[s:e]!r}...")
