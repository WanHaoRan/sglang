import sys, json, re
# usage: sect.py <jsonl> <line> <regex-start> <nchars>
path=sys.argv[1]; n=int(sys.argv[2]); rx=re.compile(sys.argv[3], re.I|re.S); k=int(sys.argv[4])
with open(path,'r',errors='replace') as f:
    for i,line in enumerate(f,1):
        if i<n: continue
        if i>n: break
        d=json.loads(line); msg=d.get('message') or {}; c=msg.get('content')
        if isinstance(c,list): c='\n'.join(p.get('text','') for p in c if isinstance(p,dict) and p.get('type')=='text')
        c = c or ''
        print('len', len(c), d.get('timestamp'), d.get('type'))
        for m in rx.finditer(c):
            print('@', m.start()); print(c[m.start():m.start()+k]); print('.....')
