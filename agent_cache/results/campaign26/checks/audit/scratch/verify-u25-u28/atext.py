import sys, json, re
# assistant text only (what the user was told); usage: atext.py <jsonl> <regex> <start> <end> <ctx>
path=sys.argv[1]; rx=re.compile(sys.argv[2], re.I|re.S); a=int(sys.argv[3]); b=int(sys.argv[4]); ctx=int(sys.argv[5])
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
                for m in rx.finditer(t):
                    s=max(0,m.start()-ctx); e=min(len(t),m.end()+ctx)
                    print(f'[{i}] {d.get("timestamp")} :: ...{t[s:e]}...'.replace('\n',' \\n ')); print()
