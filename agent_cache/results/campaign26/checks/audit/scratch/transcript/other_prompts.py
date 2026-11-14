import json, sys
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
for src in sys.argv[1:]:
    print('#####', src.split('/')[-1])
    with open(src, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if '"type":"user"' not in line: continue
            try: o = json.loads(line)
            except Exception: continue
            if o.get('type') != 'user' or o.get('isMeta'): continue
            c = (o.get('message') or {}).get('content')
            if isinstance(c, str): texts = [c]
            elif isinstance(c, list): texts = [b.get('text', '') for b in c if isinstance(b, dict) and b.get('type') == 'text']
            else: texts = []
            real = [t for t in texts if t.strip() and not t.lstrip().startswith(SKIP)]
            if not real: continue
            t = '\n'.join(real)
            if t.startswith('This session is being continued'): 
                print(f'[{i}] {o.get("timestamp")} <compaction summary, {len(t)} chars>'); continue
            print(f'[{i}] {o.get("timestamp")} {t[:700]!r}')
