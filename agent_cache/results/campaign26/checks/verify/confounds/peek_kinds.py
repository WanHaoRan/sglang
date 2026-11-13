import json, sys, glob, os
R = "/home/wanhr/sglang/agent_cache/results"
run = sys.argv[1]
d = sorted(glob.glob(os.path.join(R, run, "client_*")))[-1]
kinds = {}
keys = {}
n = 0
with open(os.path.join(d, "client.jsonl")) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception as e:
            print("BAD LINE", n, line[:100])
            continue
        n += 1
        k = o.get("kind")
        kinds[k] = kinds.get(k, 0) + 1
        ks = keys.setdefault(k, {})
        for kk in o:
            ks[kk] = ks.get(kk, 0) + 1
        if kinds[k] <= 1 and k != "turn":
            print(k, json.dumps(o)[:1500])
print(kinds)
for k in keys:
    print(k, keys[k])
