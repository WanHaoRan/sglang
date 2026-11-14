import json, sys
p = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
lo = int(sys.argv[1]); hi = int(sys.argv[2]); maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
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
        t = d.get('type')
        if t not in ('user', 'assistant'):
            continue
        ts = d.get('timestamp', '')
        msg = d.get('message') or {}
        content = msg.get('content')
        if isinstance(content, str):
            print(f"=== {i} {ts} {t}:str\n{content[:maxc]}\n")
        elif isinstance(content, list):
            for c in content:
                k = c.get('type')
                if k == 'text':
                    print(f"=== {i} {ts} {t}:text\n{c.get('text','')[:maxc]}\n")
                elif k == 'thinking':
                    th = c.get('thinking', '')
                    if th:
                        print(f"=== {i} {ts} {t}:thinking\n{th[:maxc]}\n")
                elif k == 'tool_use':
                    inp = c.get('input', {})
                    s = inp.get('command') or inp.get('file_path') or json.dumps(inp, ensure_ascii=False)
                    extra = ''
                    if 'old_string' in inp:
                        extra = f"\n  OLD: {inp['old_string'][:400]}\n  NEW: {inp['new_string'][:600]}"
                    if 'content' in inp and 'file_path' in inp:
                        extra = f"\n  CONTENT: {inp['content'][:maxc]}"
                    print(f"=== {i} {ts} tool_use {c.get('name')}\n{str(s)[:maxc]}{extra}\n")
                elif k == 'tool_result':
                    cc = c.get('content')
                    if isinstance(cc, list):
                        s = ''.join(x.get('text', '') for x in cc if isinstance(x, dict))
                    else:
                        s = str(cc)
                    print(f"=== {i} {ts} tool_result\n{s[:maxc]}\n")
