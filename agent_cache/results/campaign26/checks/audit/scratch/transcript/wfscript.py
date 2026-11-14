import json, sys
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
want = set(int(x) for x in sys.argv[1:])
with open(SRC, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i in want:
            o = json.loads(line)
            for b in o['message']['content']:
                if b.get('type') == 'tool_use':
                    inp = b.get('input', {})
                    s = inp.get('script') or inp.get('prompt') or json.dumps(inp)
                    open(f'tool_{i}.txt', 'w').write(s)
                    print(i, b.get('name'), len(s))
        if i > max(want):
            break
