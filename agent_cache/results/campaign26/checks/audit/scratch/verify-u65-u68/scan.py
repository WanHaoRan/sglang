#!/usr/bin/env python3
"""Scan the transcript (read-only) from line START for:
  - WebSearch / WebFetch tool uses (name + query/url)
  - any block (assistant text, tool input, tool result, user text) matching the regex in argv[2]
usage: scan.py START REGEX [ctx_chars]"""
import json, re, sys

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
start = int(sys.argv[1])
rx = re.compile(sys.argv[2])
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 140


def show(i, ts, tag, s):
    for m in rx.finditer(s):
        a = max(0, m.start() - ctx); b = min(len(s), m.end() + ctx)
        print('[%d] %s %s: ...%s...' % (i, ts, tag, s[a:b].replace('\n', ' ')))
        break  # one hit per block is enough


n = 0
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        n = i
        if i < start:
            continue
        if not (rx.search(line) or 'WebSearch' in line or 'WebFetch' in line):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get('type'); ts = o.get('timestamp', '')
        msg = o.get('message') or {}
        c = msg.get('content')
        if isinstance(c, str):
            show(i, ts, typ.upper() + '(str)', c)
            continue
        if not isinstance(c, list):
            continue
        for blk in c:
            if not isinstance(blk, dict):
                continue
            bt = blk.get('type')
            if bt == 'text':
                show(i, ts, typ.upper() + ' TEXT', blk.get('text', ''))
            elif bt == 'tool_use':
                nm = blk.get('name')
                inp = json.dumps(blk.get('input'), ensure_ascii=False)
                if nm in ('WebSearch', 'WebFetch'):
                    print('[%d] %s TOOL_USE %s: %s' % (i, ts, nm, inp[:400]))
                else:
                    show(i, ts, 'TOOL_USE ' + str(nm), inp)
            elif bt == 'tool_result':
                cc = blk.get('content')
                if isinstance(cc, list):
                    cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                show(i, ts, 'TOOL_RESULT', cc or '')
print('last line index', n)
