import json, sys
path = sys.argv[1]
want = set(int(x) for x in sys.argv[2].split(','))
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i in want:
            try:
                d = json.loads(line)
            except Exception as e:
                print(i, 'PARSE ERROR', e); continue
            ts = d.get('timestamp'); typ = d.get('type')
            msg = d.get('message') or {}
            content = msg.get('content')
            out = []
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list):
                for c in content:
                    if c.get('type') == 'text':
                        out.append(c.get('text', ''))
                    elif c.get('type') == 'tool_use':
                        out.append('[tool_use %s] %s' % (c.get('name'), json.dumps(c.get('input'))[:400]))
                    elif c.get('type') == 'tool_result':
                        cc = c.get('content')
                        out.append('[tool_result] %s' % (json.dumps(cc)[:400]))
            print('=== line', i, ts, typ)
            print('\n'.join(out)[:maxc])
        if i > max(want):
            break
