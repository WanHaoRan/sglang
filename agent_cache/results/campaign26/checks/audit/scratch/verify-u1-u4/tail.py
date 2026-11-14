import json, sys
path = sys.argv[1]
start = int(sys.argv[2]); end = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 400
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < start: continue
        if i > end: break
        try:
            d = json.loads(line)
        except Exception as e:
            print(i, 'BADJSON', len(line)); continue
        t = d.get('type'); ts = d.get('timestamp', '')
        side = d.get('isSidechain')
        msg = d.get('message') or {}
        role = msg.get('role')
        content = msg.get('content')
        out = []
        if isinstance(content, str):
            out.append(('text', content))
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict): continue
                ct = c.get('type')
                if ct == 'text':
                    out.append(('text', c.get('text', '')))
                elif ct == 'tool_use':
                    inp = c.get('input') or {}
                    s = json.dumps(inp, ensure_ascii=False)
                    out.append(('tool_use:' + str(c.get('name')), s))
                elif ct == 'tool_result':
                    cc = c.get('content')
                    if isinstance(cc, list):
                        s = ' '.join(x.get('text', '') for x in cc if isinstance(x, dict))
                    else:
                        s = str(cc)
                    out.append(('tool_result', s))
                elif ct == 'thinking':
                    out.append(('thinking', '(omitted %d chars)' % len(c.get('thinking', ''))))
        else:
            # other types
            keys = [k for k in d.keys() if k not in ('message',)]
            out.append(('other', json.dumps({k: d[k] for k in keys if k in ('subtype', 'content', 'summary', 'operation', 'level')}, ensure_ascii=False)))
        for kind, s in out:
            s = s.replace('\n', '\\n')
            if len(s) > maxlen: s = s[:maxlen] + ' ...[%d chars]' % len(s)
            print('%d %s %s/%s side=%s %s: %s' % (i, ts, t, role, side, kind, s))
