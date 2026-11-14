import json, sys
path = sys.argv[1]; start = int(sys.argv[2])
n = 0
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        n = i
        if i < start: continue
        try: d = json.loads(line)
        except Exception: continue
        ts = d.get('timestamp'); typ = d.get('type')
        msg = d.get('message') or {}
        content = msg.get('content')
        if isinstance(content, str):
            print(i, ts, typ, 'STR:', content[:300].replace('\n', ' | '))
        elif isinstance(content, list):
            for c in content:
                t = c.get('type')
                if t == 'text':
                    print(i, ts, typ, 'TEXT:', c.get('text', '')[:500].replace('\n', ' | '))
                elif t == 'tool_use':
                    inp = c.get('input') or {}
                    desc = inp.get('description') or inp.get('file_path') or inp.get('command') or ''
                    print(i, ts, typ, 'TOOL:', c.get('name'), str(desc)[:160].replace('\n', ' | '))
print('last line index', n)
