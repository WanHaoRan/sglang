import json, sys
p = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
out = open(sys.argv[1], 'w')
n = 0
with open(p, 'rb') as f:
    for i, line in enumerate(f):
        n = i
        try:
            d = json.loads(line)
        except Exception as e:
            out.write(f"{i}\tBAD\n")
            continue
        t = d.get('type')
        ts = d.get('timestamp', '')
        side = d.get('isSidechain')
        msg = d.get('message') or {}
        role = msg.get('role', '')
        content = msg.get('content')
        kinds = []
        text = ''
        if isinstance(content, str):
            kinds.append('str')
            text = content
        elif isinstance(content, list):
            for c in content:
                k = c.get('type')
                kinds.append(k)
                if k == 'text':
                    text += c.get('text', '')
                elif k == 'tool_use':
                    text += ' [tool_use ' + c.get('name', '') + '] '
        elif content is None:
            # other types
            text = str(d.get('content', ''))[:200] if d.get('content') else ''
        text1 = text.replace('\n', ' \\n ')[:300]
        out.write(f"{i}\t{ts}\t{t}\t{role}\t{side}\t{','.join(kinds)}\t{len(text)}\t{text1}\n")
print('lines', n + 1)
