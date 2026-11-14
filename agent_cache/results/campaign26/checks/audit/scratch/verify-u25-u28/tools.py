import sys, json, collections
path=sys.argv[1]; name=sys.argv[2]; maxc=int(sys.argv[3])
cnt=collections.Counter()
with open(path,'r',errors='replace') as f:
    for i,line in enumerate(f,1):
        if '"tool_use"' not in line: continue
        try: d=json.loads(line)
        except Exception: continue
        if d.get('isSidechain'): continue
        c=(d.get('message') or {}).get('content')
        if not isinstance(c,list): continue
        for p in c:
            if isinstance(p,dict) and p.get('type')=='tool_use':
                cnt[p.get('name')]+=1
                if name.lower() in str(p.get('name','')).lower():
                    print(f'[{i}] {d.get("timestamp")} {p.get("name")} :: {json.dumps(p.get("input"))[:maxc]}')
print(dict(cnt))
