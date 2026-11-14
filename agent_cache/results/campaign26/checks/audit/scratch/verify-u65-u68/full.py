#!/usr/bin/env python3
"""Print the full tool_result / text of given transcript lines (0-based). Read-only."""
import json, sys
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
want = set(int(x) for x in sys.argv[1:])
mxl = max(want)
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i > mxl:
            break
        if i not in want:
            continue
        o = json.loads(line)
        c = (o.get('message') or {}).get('content')
        print('=== [%d] %s %s' % (i, o.get('type'), o.get('timestamp', '')))
        if isinstance(c, str):
            print(c)
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get('type')
                if bt == 'text':
                    print(blk.get('text', ''))
                elif bt == 'tool_use':
                    print('TOOL_USE', blk.get('name'), json.dumps(blk.get('input'), ensure_ascii=False))
                elif bt == 'tool_result':
                    cc = blk.get('content')
                    if isinstance(cc, list):
                        cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                    print('TOOL_RESULT', cc)
