import sys, json, re
# usage: ts.py <jsonl> <regex> [kinds=text,usertext] [start] [end] [ctx]
path = sys.argv[1]; rx = re.compile(sys.argv[2], re.I | re.S)
kinds = set((sys.argv[3] if len(sys.argv) > 3 else 'text').split(','))
a = int(sys.argv[4]) if len(sys.argv) > 4 else 1
b = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
ctx = int(sys.argv[6]) if len(sys.argv) > 6 else 250
with open(path, 'r', errors='replace') as f:
    for i, line in enumerate(f, 1):
        if i < a: continue
        if i > b: break
        if not rx.search(line): 
            # cheap prefilter on raw line (escaped json) - may miss matches spanning escapes; fall through only if raw matches
            continue
        try: d = json.loads(line)
        except Exception: continue
        if d.get('isSidechain'): continue
        ts = d.get('timestamp',''); typ = d.get('type','')
        msg = d.get('message') or {}
        c = msg.get('content') if isinstance(msg, dict) else None
        parts = []
        if isinstance(c, str): parts.append((typ+'/text', c))
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict): continue
                t = p.get('type')
                if t == 'text': parts.append((typ+'/text', p.get('text','')))
                elif t == 'thinking': parts.append((typ+'/thinking', p.get('thinking','')))
                elif t == 'tool_use': parts.append((typ+'/tool_use', json.dumps(p.get('input'))))
                elif t == 'tool_result':
                    cc = p.get('content')
                    if isinstance(cc, list): cc = ' '.join(x.get('text','') for x in cc if isinstance(x, dict))
                    parts.append((typ+'/tool_result', str(cc)))
        for kind, t in parts:
            k = kind.split('/')[1]
            if k not in kinds and 'all' not in kinds: continue
            for m in rx.finditer(t):
                s = max(0, m.start()-ctx); e = min(len(t), m.end()+ctx)
                print(f'[{i}] {ts} {kind} :: ...{t[s:e]}...'.replace('\n', ' \\n '))
                print()
                break
