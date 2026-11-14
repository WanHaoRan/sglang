import sys, json
# usage: tl.py <jsonl> <start_line_1based> <end_line_1based> [maxchars]
path = sys.argv[1]; a = int(sys.argv[2]); b = int(sys.argv[3])
maxc = int(sys.argv[4]) if len(sys.argv) > 4 else 600
def text_of(msg):
    out = []
    c = msg.get('content') if isinstance(msg, dict) else None
    if isinstance(c, str):
        out.append(('text', c))
    elif isinstance(c, list):
        for p in c:
            if not isinstance(p, dict): continue
            t = p.get('type')
            if t == 'text': out.append(('text', p.get('text','')))
            elif t == 'thinking': out.append(('thinking', p.get('thinking','')))
            elif t == 'tool_use': out.append(('tool_use', json.dumps(p.get('input'))[:maxc]))
            elif t == 'tool_result':
                cc = p.get('content')
                if isinstance(cc, list):
                    cc = ' '.join(x.get('text','') for x in cc if isinstance(x, dict))
                out.append(('tool_result', str(cc)))
    return out
with open(path, 'r', errors='replace') as f:
    for i, line in enumerate(f, 1):
        if i < a: continue
        if i > b: break
        try:
            d = json.loads(line)
        except Exception as e:
            print(i, 'BAD JSON'); continue
        ts = d.get('timestamp',''); typ = d.get('type','')
        side = d.get('isSidechain')
        msg = d.get('message') or {}
        parts = text_of(msg)
        if not parts:
            extra = d.get('content') or d.get('summary') or ''
            print(f'[{i}] {ts} {typ} side={side} :: {str(extra)[:maxc]}')
            continue
        for kind, t in parts:
            print(f'[{i}] {ts} {typ}/{kind} side={side} :: {t[:maxc]}')
