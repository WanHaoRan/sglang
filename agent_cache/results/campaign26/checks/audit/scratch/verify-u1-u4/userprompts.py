# list user-typed prompts (not tool results, not system/meta) since a given timestamp
import json, sys
path = sys.argv[1]; since = sys.argv[2]
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if '"type":"user"' not in line[:4000] and '"type": "user"' not in line[:4000]:
            continue
        try: d = json.loads(line)
        except Exception: continue
        if d.get('type') != 'user' or d.get('isSidechain'): continue
        ts = d.get('timestamp', '')
        if ts < since: continue
        if d.get('isMeta'): continue
        c = (d.get('message') or {}).get('content')
        texts = []
        if isinstance(c, str): texts = [c]
        elif isinstance(c, list):
            if any(isinstance(x, dict) and x.get('type') == 'tool_result' for x in c): continue
            texts = [x.get('text', '') for x in c if isinstance(x, dict) and x.get('type') == 'text']
        for t in texts:
            s = t.strip()
            if not s or s.startswith('<task-notification') or s.startswith('<local-command') or s.startswith('<command-') or s.startswith('<system-reminder') or s.startswith('# Workflow authoring') or s.startswith('[Image') or s.startswith('This session is being continued'):
                continue
            print('%d %s: %s' % (i, ts, s.replace('\n', '\\n')[:700]))
            print()
