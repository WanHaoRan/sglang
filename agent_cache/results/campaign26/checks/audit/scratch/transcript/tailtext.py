import json
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
n = 0
with open(SRC, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        n = i
        if i < 30746: continue
        if '"type":"assistant"' not in line and '"type":"user"' not in line: continue
        try: o = json.loads(line)
        except Exception: continue
        t = o.get('type')
        c = (o.get('message') or {}).get('content')
        if t == 'assistant' and isinstance(c, list):
            for b in c:
                if b.get('type') == 'text' and b.get('text', '').strip():
                    print(f"[{i}] {o.get('timestamp')} ASSISTANT TEXT: {b['text'][:600]!r}")
                elif b.get('type') == 'tool_use':
                    inp = b.get('input', {})
                    print(f"[{i}] {o.get('timestamp')} tool {b.get('name')}: {str(inp.get('description') or inp.get('file_path') or '')[:140]}")
        elif t == 'user':
            if isinstance(c, str) and not c.lstrip().startswith('<'):
                print(f"[{i}] {o.get('timestamp')} USER: {c[:300]!r}")
print('last line index', n)
