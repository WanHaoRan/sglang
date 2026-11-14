import json, sys
p = '/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl'
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
with open(p, 'rb') as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('type') != 'user':
            continue
        msg = d.get('message') or {}
        content = msg.get('content')
        text = ''
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            has_tr = any(c.get('type') == 'tool_result' for c in content)
            if has_tr:
                continue
            for c in content:
                if c.get('type') == 'text':
                    text += c.get('text', '')
        if not text.strip():
            continue
        flags = []
        for k in ('isMeta', 'isSidechain', 'isCompactSummary', 'isVisibleInTranscriptOnly', 'userType', 'origin', 'sourceToolUseID'):
            if d.get(k) not in (None, False):
                flags.append(f"{k}={str(d.get(k))[:40]}")
        print(f"=== {i} {d.get('timestamp','')} {' '.join(flags)} len={len(text)}")
        print(text[:1500])
        print()
