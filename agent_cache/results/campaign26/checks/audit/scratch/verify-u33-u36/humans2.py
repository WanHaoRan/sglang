import json, sys
p = sys.argv[1]
maxc = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
with open(p, 'rb') as f:
    for i, line in enumerate(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get('type')
        if t not in ('user', 'assistant'):
            continue
        msg = d.get('message') or {}
        content = msg.get('content')
        text = ''
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            if any(c.get('type') == 'tool_result' for c in content):
                continue
            for c in content:
                if c.get('type') == 'text':
                    text += c.get('text', '')
        if not text.strip():
            continue
        o = d.get('origin')
        kind = o.get('kind') if isinstance(o, dict) else None
        if t == 'user' and (d.get('isMeta') or d.get('isCompactSummary')):
            continue
        print(f"=== {i} {d.get('timestamp','')} {t} origin={kind} len={len(text)}")
        print(text[:maxc])
        print()
