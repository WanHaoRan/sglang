import json, sys, re
# usage: q.py show <i0> <i1> [maxchars]   |  q.py grep <regex> [kinds] [ts_from] [ts_to] [maxchars]
mode = sys.argv[1]
recs = [json.loads(l) for l in open('idx.jsonl')]
def show(r, mx):
    for k, s in r.get('parts', []):
        s = s.replace('\n', ' \\n ')
        print('  [%s] %s' % (k, s[:mx]))
if mode == 'show':
    a, b = int(sys.argv[2]), int(sys.argv[3]); mx = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    for r in recs[a:b+1]:
        print('#%d %s %s %s%s' % (r['i'], r.get('ts'), r.get('type'), r.get('role'), ' SIDE' if r.get('side') else ''))
        show(r, mx)
elif mode == 'grep':
    rx = re.compile(sys.argv[2], re.I)
    kinds = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != '-' else None
    t0 = sys.argv[4] if len(sys.argv) > 4 else ''
    t1 = sys.argv[5] if len(sys.argv) > 5 else '9'
    mx = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    for r in recs:
        ts = r.get('ts') or ''
        if ts and not (t0 <= ts <= t1): continue
        for k, s in r.get('parts', []):
            if kinds and k not in kinds: continue
            if kinds is None and k == 'tool_result': continue
            for m in rx.finditer(s):
                a = max(0, m.start() - mx // 2); b = min(len(s), m.end() + mx // 2)
                print('#%d %s %s/%s [%s] ...%s...' % (r['i'], ts, r.get('type'), r.get('role'), k, s[a:b].replace('\n', ' \\n ')))
                break
elif mode == 'users':
    t0 = sys.argv[2] if len(sys.argv) > 2 else ''
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    for r in recs:
        ts = r.get('ts') or ''
        if ts < t0: continue
        if r.get('type') == 'user' and r.get('role') == 'user' and not r.get('side'):
            for k, s in r.get('parts', []):
                if k == 'text' and not s.startswith('<system-reminder>'):
                    print('#%d %s %s' % (r['i'], ts, s[:mx].replace('\n', ' \\n ')))
