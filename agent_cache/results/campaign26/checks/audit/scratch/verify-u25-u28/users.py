import sys, json
path=sys.argv[1]; a=int(sys.argv[2]); maxc=int(sys.argv[3]) if len(sys.argv)>3 else 500
with open(path,'r',errors='replace') as f:
    for i,line in enumerate(f,1):
        if i<a: continue
        if '"type":"user"' not in line[:400] and '"type": "user"' not in line[:400]:
            # type may be later in the line; fall back to json parse on small lines
            if len(line) > 200000: 
                pass
        try: d=json.loads(line)
        except Exception: continue
        if d.get('type')!='user' or d.get('isSidechain'): continue
        msg=d.get('message') or {}; c=msg.get('content')
        texts=[]
        if isinstance(c,str): texts=[c]
        elif isinstance(c,list):
            if any(isinstance(p,dict) and p.get('type')=='tool_result' for p in c): continue
            texts=[p.get('text','') for p in c if isinstance(p,dict) and p.get('type')=='text']
        t=' '.join(texts).strip()
        if not t: continue
        head=t[:60]
        tag=''
        if t.startswith('This session is being continued'): tag='[COMPACTION SUMMARY] '; t=t[:120]
        elif t.startswith('<task-notification') : tag='[TASK-NOTIF] '; t=t[:200]
        elif t.startswith('<system-reminder'): tag='[SYS] '; t=t[:160]
        elif t.startswith('<local-command') or t.startswith('<command-'): tag='[CMD] '; t=t[:300]
        print(f'[{i}] {d.get("timestamp")} {tag}{t[:maxc]}'.replace('\n',' \\n '))
