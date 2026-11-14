import json, sys
src = sys.argv[1]
out = open(sys.argv[2], 'w')
n = 0
with open(src, 'r', errors='replace') as f:
    for i, line in enumerate(f):
        try:
            d = json.loads(line)
        except Exception as e:
            out.write(json.dumps({'i': i, 'err': str(e)[:80]}) + '\n'); continue
        t = d.get('type'); ts = d.get('timestamp')
        rec = {'i': i, 'ts': ts, 'type': t}
        if d.get('isSidechain'): rec['side'] = 1
        m = d.get('message')
        parts = []
        if isinstance(m, dict):
            rec['role'] = m.get('role')
            c = m.get('content')
            if isinstance(c, str):
                parts.append(('text', c))
            elif isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict): continue
                    bt = b.get('type')
                    if bt == 'text': parts.append(('text', b.get('text', '')))
                    elif bt == 'tool_use':
                        inp = b.get('input', {})
                        parts.append(('tool_use', b.get('name', '') + ' ' + json.dumps(inp)[:6000]))
                    elif bt == 'tool_result':
                        cc = b.get('content')
                        if isinstance(cc, list):
                            s = ' '.join(x.get('text', '') for x in cc if isinstance(x, dict))
                        else:
                            s = str(cc)
                        parts.append(('tool_result', s[:3000]))
                    elif bt == 'thinking':
                        pass
        else:
            for k in ('content', 'summary', 'text'):
                if isinstance(d.get(k), str):
                    parts.append((k, d[k][:4000]))
        rec['parts'] = parts
        out.write(json.dumps(rec) + '\n'); n += 1
print('lines', n)
