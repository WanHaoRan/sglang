import re, sys, collections
for run in sys.argv[1:]:
    lines = [l for l in open(f"/home/wanhr/sglang/agent_cache/results/{run}/server.log", errors="ignore") if "HICACHE_EVT" in l]
    c = collections.Counter(); tok = collections.Counter()
    for i, l in enumerate(lines):
        if "evict_host" not in l: continue
        t = int(re.search(r"tokens=(\d+)", l).group(1))
        nxt = None
        for j in range(i+1, min(len(lines), i+6)):
            if "evict_host" in lines[j]: continue
            nxt = re.search(r"HICACHE_EVT (\w+)", lines[j]).group(1); break
        c[nxt] += 1; tok[nxt] += t
    print(run, "evict_host events by next event:", dict(c), "tokens:", dict(tok))
