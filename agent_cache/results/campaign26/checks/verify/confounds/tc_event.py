"""Test (c), detail: the congestion event at positions 8750-9249 of the order run, which carries most of ec_r13's lead.
Per arm on the shared turns there: mean TTFT, turns over 10 s, queue wait, live context at the send; and how much the same
event moved between two runs of the same arm (hint-free, JIT p50).
  nice -n 19 python3 -I -B tc_event.py"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
names = C.LEAD5 + ["jit_p50", "jit_p50_ctl", "hintfree", "hintfree_ctl"]
D = {a: C.load(a) for a in names}
B = {a: C.by_key(C.kept(D[a])) for a in names}
LF = json.load(open(os.path.join(C.CACHE, "load_state_fix.json")))
keys = C.shared_keys(B, C.LEAD5, rank)
for lo, hi in ((8750, 9250), (8750, 9000), (9000, 9250), (5750, 6750)):
    ks = [k for k in keys if lo <= rank.get(k, -1) < hi]
    print(f"positions {lo}-{hi - 1}: {len(ks)} shared turns of LEAD5")
    for a in C.LEAD5:
        v = [B[a][k]["ttft"] for k in ks]
        q = [D[a]["req"].get(f"{k[0]}-{k[1]}", [0])[0] / 1000 for k in ks]
        ctx = [LF[a][f"{k[0]}-{k[1]}"][1] for k in ks]
        ts = [B[a][k]["t_send"] / 60 for k in ks]
        print(f"  {a:11s} mean {C.mean(v):6.3f} median {C.pct(v, .5):6.3f} p90 {C.pct(v, .9):6.2f} max {max(v):5.1f} over 10 s {sum(1 for x in v if x > 10):3d} | queue {C.mean(q):6.3f} | live context {C.mean(ctx) / 1000:7.1f}K max {max(ctx) / 1000:7.1f}K | "
              f"minutes {min(ts):5.1f}-{max(ts):5.1f} | recomputed {C.mean(C.rec(B[a][k]) for k in ks):6.0f}")
print("\nthe same event between two runs of the same arm (their own shared turns at positions 8750-9249, and 5750-6749):")
for a, b in (("hintfree", "hintfree_ctl"), ("jit_p50", "jit_p50_ctl")):
    kk = C.shared_keys(B, [a, b], rank)
    for lo, hi in ((8750, 9250), (5750, 6750)):
        ks = [k for k in kk if lo <= rank.get(k, -1) < hi]
        ma, mb = C.mean(B[a][k]["ttft"] for k in ks), C.mean(B[b][k]["ttft"] for k in ks)
        print(f"  {a} {ma:6.3f}  {b} {mb:6.3f}  on {len(ks)} turns at {lo}-{hi - 1}: apart {abs(ma - mb):.3f} s = {100 * abs(ma - mb) / min(ma, mb):.1f} % of the lower; turns over 10 s {sum(1 for k in ks if B[a][k]['ttft'] > 10)} against {sum(1 for k in ks if B[b][k]['ttft'] > 10)}")
print("\nLEAD5 at positions 8750-9249: spread between arms " + "; ".join(
    f"{b}/{a} {100 * (C.mean(B[b][k]['ttft'] for k in keys if 8750 <= rank.get(k, -1) < 9250) / C.mean(B[a][k]['ttft'] for k in keys if 8750 <= rank.get(k, -1) < 9250) - 1):+.0f} %"
    for a, b in (("ec_r13", "jit_exact"), ("ec_r13", "ec_r1"), ("ec_r13", "nofc"), ("ec_r13", "nofc_noend"), ("nofc_noend", "nofc"))))
