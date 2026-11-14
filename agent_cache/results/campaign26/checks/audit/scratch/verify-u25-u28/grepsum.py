import sys, json, re
path=sys.argv[1]; lines=[int(x) for x in sys.argv[2].split(',')]; rx=re.compile(sys.argv[3], re.I)
want=set(lines)
with open(path,'r',errors='replace') as f:
    for i,line in enumerate(f,1):
        if i not in want: continue
        d=json.loads(line); c=(d.get('message') or {}).get('content')
        if isinstance(c,list): c='\n'.join(p.get('text','') for p in c if isinstance(p,dict) and p.get('type')=='text')
        c=c or ''
        hits=[m.start() for m in rx.finditer(c)]
        print(f'[{i}] {d.get("timestamp")} len={len(c)} hits={len(hits)}')
        for h in hits[:6]:
            print('   ...'+c[max(0,h-200):h+260].replace('\n',' \\n ')+'...')
