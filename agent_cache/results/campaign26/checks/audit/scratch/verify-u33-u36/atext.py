import json, sys
p = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
lo = int(sys.argv[1]); hi = int(sys.argv[2]); minlen = int(sys.argv[3]) if len(sys.argv) > 3 else 0
with open(p, 'rb') as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('type') != 'assistant':
            continue
        msg = d.get('message') or {}
        content = msg.get('content')
        text = ''
        if isinstance(content, list):
            for c in content:
                if c.get('type') == 'text':
                    text += c.get('text', '')
        if len(text) < minlen or not text.strip():
            continue
        print(f"=== {i} {d.get('timestamp','')} len={len(text)}")
        print(text)
        print()
