import json
SRC = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
n = 0
with open(SRC, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i < 21800: continue
        if '"queue-operation"' not in line: continue
        o = json.loads(line)
        if o.get('type') != 'queue-operation': continue
        c = o.get('content')
        if c is None: continue
        if isinstance(c, str) and c.lstrip().startswith('<task-notification>'): continue
        n += 1
        print(i, o.get('timestamp'), o.get('operation'), repr(str(c)[:400]))
print('non-notification queue ops:', n)
