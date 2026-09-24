import glob, json, os
from arrival_loss import RUNS, R
for scale in ["x70","x10","x1"]:
    for arm in ["hbm_host","three_tier_to","three_tier_wc","hbm_lru"]:
        cl = glob.glob(os.path.join(R, RUNS[scale][arm], "client_*", "client.jsonl"))[0]
        n=k=0
        for line in open(cl):
            r=json.loads(line)
            if r.get("kind")!="turn" or r["turn"]<1: continue
            n+=1; k+= (r.get("cached_tokens") or 0) <= 7872
        print(f"{scale} {arm}: returning turns {n}, admitted with <= shared prompt cached: {k} ({100*k/n:.1f}%)")
