import json, sys
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
lo, hi = int(sys.argv[1]), int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 700
with open(SRC, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < lo: continue
        if i > hi: break
        o = json.loads(line)
        t = o.get('type')
        if t in ('api-request-blob', 'api-request', 'api-request-shape', 'mode', 'permission-mode', 'session-owner', 'atis-latch', 'ai-title', 'last-prompt', 'file-history-snapshot', 'file-history-delta'):
            if t in ('mode', 'permission-mode', 'last-prompt'):
                print(i, t, json.dumps(o)[:300])
            continue
        s = json.dumps(o, ensure_ascii=False)
        print(i, t, o.get('timestamp', ''), s[:mx])
        print()
