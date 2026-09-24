import json, glob, sys, re
P = 64
f = lambda x: x // P * P
for run in sys.argv[1:]:
    base = f"/home/wanhr/sglang/agent_cache/results/{run}"
    cj = glob.glob(base + "/client_*/client.jsonl")[0]
    ep = ed = n = 0
    for line in open(cj):
        r = json.loads(line)
        if r["kind"] != "turn" or r.get("ttft") is None: continue
        n += 1
        pt, ct, c = r["prompt_tokens"], r["completion_tokens"], r["cached_tokens"]
        ep += f(pt) - f(c); ed += f(pt + ct) - f(pt)
    d2h = 0
    for line in open(base + "/server.log", errors="ignore"):
        if "HICACHE_EVT d2h_submit" in line:
            d2h += int(re.search(r"tokens=(\d+)", line).group(1))
    print(f"{run}: turns={n} pred_prefill_end={ep} pred_finish={ed} pred_total={ep+ed} observed_d2h={d2h} ratio={d2h/(ep+ed):.4f} prefill_end_share={ep/(ep+ed):.3f}")
