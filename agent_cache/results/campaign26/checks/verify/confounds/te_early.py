"""Test (e), detail: the first 8 minutes of each run, where every arm serves the same requests from the same cache state.
Decode ms per step by running requests from the Decode lines, the number of lines, and the paired per-turn decode time.
  nice -n 19 python3 -I -B te_early.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ARMS = ["baseline", "hintfree", "jit_exact", "jit_p50", "base_rep", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_p50_d_cut"]
D = {a: C.load(a) for a in ARMS}


def med(v):
    v = sorted(v)
    n = len(v)
    return float("nan") if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


print("first 8 minutes: Decode lines with no queue, median ms per step (n) at 1-3, 4-7, 8-11, 12-15 running requests | mean #token K on the 8-15 lines | prefill tokens computed in the window (K) | turns finished")
for a in ARMS:
    t0 = D[a]["t0"]
    cells = []
    for lo, hi in ((1, 3), (4, 7), (8, 11), (12, 15)):
        st = [1000.0 * r / thr for t, r, tok, u, thr, q in D[a]["dec"] if 0 <= t - t0 < 480 and lo <= r <= hi and q == 0 and thr > 0]
        cells.append(f"{med(st):6.2f} ({len(st):3d})")
    tok = [x[2] for x in D[a]["dec"] if 0 <= x[0] - t0 < 480 and 8 <= x[1] <= 15]
    pre = sum(x[2] for x in D[a]["pre"] if 0 <= x[0] - t0 < 480)
    done = sum(1 for t in D[a]["turns"] if t["ttft"] is not None and t["t_send"] + t["latency"] < 480)
    print(f"  {a:14s} {time.strftime('%m-%d %H:%M', time.gmtime(t0))} | " + " ".join(cells) + f" | {C.mean(tok) / 1000:6.1f}K | {pre / 1000:7.1f}K | {done}")

B8 = {a: {(t["conv"], t["turn"]): t for t in D[a]["turns"] if t["ttft"] is not None and t["t_send"] + (t["latency"] or 0) < 480} for a in ARMS}
k8 = [k for k in set.intersection(*(set(v) for v in B8.values())) if len(set((B8[a][k]["pt"], B8[a][k]["ct"]) for a in ARMS)) == 1 and B8[ARMS[0]][k]["ct"] >= 20]
print(f"\nthe same {len(k8)} turns (identical token counts, 20 or more completion tokens) finished within 480 s in all twelve runs: decode ms per completion token median, mean; mean latency; total decode seconds; ratio of total decode seconds to the twelve-run median")
tot = {a: sum(B8[a][k]["latency"] - B8[a][k]["ttft"] for k in k8) for a in ARMS}
ref = med(list(tot.values()))
for a in ARMS:
    per = [1000 * (B8[a][k]["latency"] - B8[a][k]["ttft"]) / B8[a][k]["ct"] for k in k8]
    print(f"  {a:14s} {med(per):6.2f} {C.mean(per):6.2f} | {C.mean(B8[a][k]['latency'] for k in k8):6.3f} | {tot[a]:8.1f} | {tot[a] / ref:6.4f} ({100 * (tot[a] / ref - 1):+5.2f} %)")
print("  send-time drift on these turns (how far each run's clock has slipped by then): mean t_send (s) " + " ".join(f"{a} {C.mean(B8[a][k]['t_send'] for k in k8):.1f}" for a in ARMS))
