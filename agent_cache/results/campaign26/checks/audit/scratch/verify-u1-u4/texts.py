# print only user-typed text and assistant text (no tool calls/results) in an index range
import json, sys
path = sys.argv[1]
start = int(sys.argv[2]); end = int(sys.argv[3])
maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < start: continue
        if i > end: break
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get('type'); ts = d.get('timestamp', '')
        if d.get('isSidechain'): continue
        msg = d.get('message') or {}
        role = msg.get('role')
        content = msg.get('content')
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get('type') == 'text':
                    texts.append(c.get('text', ''))
        for s in texts:
            if not s.strip(): continue
            s2 = s.replace('\n', '\\n')
            if len(s2) > maxlen: s2 = s2[:maxlen] + ' ...[%d chars]' % len(s2)
            print('%d %s %s/%s: %s' % (i, ts, t, role, s2))
            print()
