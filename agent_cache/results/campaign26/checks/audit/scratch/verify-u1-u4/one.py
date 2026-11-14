# print full tool_use inputs and tool_result texts for given line indexes
import json, sys
path = sys.argv[1]
want = set(int(x) for x in sys.argv[2].split(','))
maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
mx = max(want)
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i > mx: break
        if i not in want: continue
        d = json.loads(line)
        msg = d.get('message') or {}
        content = msg.get('content')
        print('=====', i, d.get('timestamp'), d.get('type'))
        if isinstance(content, str):
            print(content[:maxlen])
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict): continue
                ct = c.get('type')
                if ct == 'text': print('[text]', c.get('text', '')[:maxlen])
                elif ct == 'tool_use':
                    inp = c.get('input') or {}
                    print('[tool_use %s]' % c.get('name'))
                    for k, v in inp.items():
                        if isinstance(v, str): print('  %s: %s' % (k, v[:maxlen]))
                        else: print('  %s: %s' % (k, json.dumps(v, ensure_ascii=False)[:maxlen]))
                elif ct == 'tool_result':
                    cc = c.get('content')
                    if isinstance(cc, list):
                        s = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                    else: s = str(cc)
                    print('[tool_result]', s[:maxlen])
