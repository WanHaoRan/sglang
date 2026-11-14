import json, sys, re
p = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
pat = re.compile(sys.argv[1])
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 160
kinds = set(sys.argv[5].split(',')) if len(sys.argv) > 5 else {'text', 'tool_use', 'user_str', 'tool_result', 'thinking'}
def emit(i, ts, kind, s):
    for m in pat.finditer(s):
        a = max(0, m.start() - ctx); b = min(len(s), m.end() + ctx)
        print(f"{i}\t{ts}\t{kind}\t...{s[a:b]!r}...")
with open(p, 'rb') as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get('type')
        if t not in ('user', 'assistant'):
            continue
        ts = d.get('timestamp', '')
        msg = d.get('message') or {}
        content = msg.get('content')
        if isinstance(content, str):
            if 'user_str' in kinds:
                emit(i, ts, t + ':str', content)
        elif isinstance(content, list):
            for c in content:
                k = c.get('type')
                if k == 'text' and 'text' in kinds:
                    emit(i, ts, t + ':text', c.get('text', ''))
                elif k == 'thinking' and 'thinking' in kinds:
                    emit(i, ts, t + ':thinking', c.get('thinking', ''))
                elif k == 'tool_use' and 'tool_use' in kinds:
                    emit(i, ts, t + ':tool_use', json.dumps(c.get('input', {}), ensure_ascii=False))
                elif k == 'tool_result' and 'tool_result' in kinds:
                    cc = c.get('content')
                    if isinstance(cc, list):
                        s = ''.join(x.get('text', '') for x in cc if isinstance(x, dict))
                    else:
                        s = str(cc)
                    emit(i, ts, t + ':tool_result', s)
