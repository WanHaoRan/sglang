import json, re, sys, glob, statistics as st
from datetime import datetime
run = sys.argv[1]
base = f"/home/wanhr/sglang/agent_cache/results/{run}"
cj = glob.glob(base + "/client_*/client.jsonl")[0]
turns = []; t_start = None
for line in open(cj):
    r = json.loads(line)
    if r["kind"] == "run": t_start = r["t_start"]
    elif r["kind"] == "turn" and r.get("ttft") is not None: turns.append(r)
ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] (.*)$")
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
pf = {}; d2h = []; h2s = []
for line in open(base + "/server.log", errors="ignore"):
    m = ts_re.match(line)
    if not m: continue
    body = m.group(2)
    if "HICACHE_EVT prefetch_start" in body:
        mm = re.search(r"rid=\S+-c(\d+)-t(\d+) ", body)
        if mm: pf.setdefault((int(mm.group(1)), int(mm.group(2))), ts(m.group(1)))
    elif "HICACHE_EVT d2h_submit" in body:
        d2h.append((ts(m.group(1)), int(re.search(r"tokens=(\d+)", body).group(1))))
    elif "HICACHE_EVT h2s_submit" in body:
        h2s.append((ts(m.group(1)), int(re.search(r"tokens=(\d+)", body).group(1))))
# clock offset: server_ts - (t_start + t_send)
offs = [pf[(t["conv"], t["turn"])] - (t_start + t["t_send"]) for t in turns if (t["conv"], t["turn"]) in pf]
off = st.median(offs)
print(f"run={run} turns={len(turns)} matched_prefetch={len(offs)} offset_median={off:.3f}s p5={sorted(offs)[len(offs)//20]:.3f} p95={sorted(offs)[len(offs)*19//20]:.3f}")
# intervals in server clock
iv = []
for t in turns:
    s = t_start + t["t_send"] + off
    iv.append((s, s + t["ttft"], s + t["latency"], t))
iv.sort(key=lambda x: x[0])
P = 64
def floor(x): return x // P * P
isolated = []
for i, (s, p, e, t) in enumerate(iv):
    ok = True
    for j in range(max(0, i - 200), min(len(iv), i + 200)):
        if j == i: continue
        s2, p2, e2, _ = iv[j]
        if s2 < e + 1.0 and e2 > s - 1.0: ok = False; break
    if ok: isolated.append((s, p, e, t))
print("isolated turns:", len(isolated))
import bisect
dts = [x[0] for x in d2h]
agg = dict(pre=0, at_prefill=0, mid=0, at_finish=0, post=0)
exp_prompt = exp_decode = 0
per = []
for s, p, e, t in isolated:
    lo = bisect.bisect_left(dts, s - 0.5); hi = bisect.bisect_right(dts, e + 1.0)
    pt, ct = t["prompt_tokens"], t["completion_tokens"]
    ep = floor(pt) - floor(t["cached_tokens"]); ed = floor(pt + ct - 1) - floor(pt)
    exp_prompt += ep; exp_decode += ed
    near_p = near_e = other = 0
    for k in range(lo, hi):
        tt, tok = d2h[k]
        if abs(tt - p) <= 0.15 or (tt <= p and tt >= s): near_p += tok  # between send and first token
        elif abs(tt - e) <= 0.15 or tt > e: near_e += tok
        else: other += tok
    per.append((pt, t["cached_tokens"], ct, ep, ed, near_p, near_e, other, p - s, e - p))
    agg["at_prefill"] += near_p; agg["at_finish"] += near_e; agg["mid"] += other
print("expected new-prompt pages tokens (floor(prompt)-floor(cached)) =", exp_prompt, " expected decode-page tokens =", exp_decode)
print("observed d2h tokens: send..first-token(+0.15s) =", agg["at_prefill"], " finish(+/-0.15s or after) =", agg["at_finish"], " between =", agg["mid"])
print("sample (prompt, cached, completion, exp_prompt_wr, exp_dec_wr, obs_at_prefill, obs_at_finish, obs_between, ttft_srv, decode_s):")
for x in per[:15]: print("  ", tuple(round(v, 2) if isinstance(v, float) else v for v in x))
# exact match rate per turn
m1 = sum(1 for x in per if x[5] == x[3]); m2 = sum(1 for x in per if x[6] == x[4])
print(f"turns with obs_at_prefill == expected prompt write: {m1}/{len(per)}; obs_at_finish == expected decode write: {m2}/{len(per)}")
