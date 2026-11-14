#!/usr/bin/env python3
"""Print a readable dump of transcript lines [a, b] (0-based, inclusive). Read-only.
usage: rng.py a b [maxchars_per_block] [--tools] [--results]"""
import json, sys

SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
a = int(sys.argv[1]); b = int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 3000
show_tools = '--tools' in sys.argv
show_results = '--results' in sys.argv


def clip(s, n=mx):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + ' ...[+%d chars]' % (len(s) - n)


with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < a:
            continue
        if i > b:
            break
        try:
            o = json.loads(line)
        except Exception:
            print('[%d] BADJSON len %d' % (i, len(line)))
            continue
        typ = o.get('type'); ts = o.get('timestamp', '')
        msg = o.get('message') or {}
        c = msg.get('content')
        if typ == 'assistant' and isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get('type')
                if bt == 'text':
                    print('[%d] %s ASSISTANT TEXT:\n%s\n' % (i, ts, clip(blk.get('text', ''))))
                elif bt == 'tool_use' and show_tools:
                    print('[%d] %s TOOL_USE %s: %s\n' % (i, ts, blk.get('name'), clip(blk.get('input'), min(mx, 1500))))
        elif typ == 'user':
            if isinstance(c, str):
                print('[%d] %s USER(str): %s\n' % (i, ts, clip(c)))
            elif isinstance(c, list):
                for blk in c:
                    if not isinstance(blk, dict):
                        continue
                    bt = blk.get('type')
                    if bt == 'text':
                        print('[%d] %s USER TEXT: %s\n' % (i, ts, clip(blk.get('text', ''))))
                    elif bt == 'tool_result' and show_results:
                        cc = blk.get('content')
                        if isinstance(cc, list):
                            cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                        print('[%d] %s TOOL_RESULT: %s\n' % (i, ts, clip(cc or '', min(mx, 2500))))
        elif typ not in ('assistant', 'user'):
            pass
