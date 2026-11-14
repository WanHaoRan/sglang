# find assistant text messages (main chain) since a timestamp containing a regex; print snippets around matches
import json, sys, re
path = sys.argv[1]; since = sys.argv[2]; pat = re.compile(sys.argv[3], re.I); width = int(sys.argv[4]) if len(sys.argv) > 4 else 260
seen = set()
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if '"role":"assistant"' not in line and '"role": "assistant"' not in line: continue
        try: d = json.loads(line)
        except Exception: continue
        if d.get('type') != 'assistant' or d.get('isSidechain'): continue
        ts = d.get('timestamp', '')
        if ts < since: continue
        for c in ((d.get('message') or {}).get('content') or []):
            if isinstance(c, dict) and c.get('type') == 'text':
                t = c.get('text', '')
                key = (ts[:16], t[:80])
                if key in seen: continue
                for m in pat.finditer(t):
                    seen.add(key)
                    a = max(0, m.start() - width); b = min(len(t), m.end() + width)
                    print('%d %s: ...%s...' % (i, ts, t[a:b].replace('\n', '\\n')))
                    print()
