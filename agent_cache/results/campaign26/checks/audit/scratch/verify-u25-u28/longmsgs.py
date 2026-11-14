import sys, json
path=sys.argv[1]; a=int(sys.argv[2]); b=int(sys.argv[3]); minlen=int(sys.argv[4]); head=int(sys.argv[5])
with open(path,'r',errors='replace') as f:
    for i,line in enumerate(f,1):
        if i<a: continue
        if i>b: break
        if '"assistant"' not in line: continue
        try: d=json.loads(line)
        except Exception: continue
        if d.get('type')!='assistant' or d.get('isSidechain'): continue
        c=(d.get('message') or {}).get('content')
        if not isinstance(c,list): continue
        for p in c:
            if isinstance(p,dict) and p.get('type')=='text':
                t=p.get('text','')
                if len(t)>=minlen:
                    print(f'[{i}] {d.get("timestamp")} len={len(t)} :: {t[:head]}'.replace('\n',' \\n ')); print()
