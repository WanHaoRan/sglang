"""Test (f4): R4, JIT p50 on the new build (cut at minute 80) against the old build.
1. The paired difference by 10-minute window of the cut run's clock. 2. Hardware and mechanisms in the same minutes
(0-74.7). 3. How much of the first load episode the cut run's kept window covers. 4. Errors in the cut run's server log.
  nice -n 19 python3 -I -B tf_r4.py"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NAMES = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"]
D = {a: C.load(a) for a in NAMES}
B = {a: C.by_key(C.kept(D[a])) for a in NAMES}
keys = C.shared_keys(B, C.CUT7, rank)
cut_end = max(t["t_send"] for t in B["jit_p50_d_cut"].values())
print(f"CUT7 shared returning turns {len(keys)}; the cut run's kept window ends at {cut_end / 60:.1f} min; its last send at {max(t['t_send'] for t in D['jit_p50_d_cut']['turns']) / 60:.1f} min")


def med(v):
    v = sorted(v)
    n = len(v)
    return float("nan") if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


print("\n1. by 10-minute window of the cut run's send time: turns | mean TTFT jit_p50 (c23), jit_p50_ctl, jit_p50_d_cut | d_cut - ctl | c23 - ctl | nofc_noend, ec_r13, jit_exact")
for m in range(0, 80, 10):
    ks = [k for k in keys if m * 60 <= B["jit_p50_d_cut"][k]["t_send"] < (m + 10) * 60]
    if not ks:
        continue
    f = lambda a: C.mean(B[a][k]["ttft"] for k in ks)
    print(f"  {m:2d}-{m + 10:2d} min {len(ks):5d} | {f('jit_p50'):6.3f} {f('jit_p50_ctl'):6.3f} {f('jit_p50_d_cut'):6.3f} | {f('jit_p50_d_cut') - f('jit_p50_ctl'):+6.3f} | {f('jit_p50') - f('jit_p50_ctl'):+6.3f} | "
          f"{f('nofc_noend'):6.3f} {f('ec_r13'):6.3f} {f('jit_exact'):6.3f}")
print("  cumulative from the start: window end | d_cut - ctl | c23 - ctl | smallest lead of nofc_noend/ec_r13/jit_exact over any of the three JIT p50 runs")
for m in (30, 40, 50, 60, 65, 70, 74.7):
    ks = [k for k in keys if B["jit_p50_d_cut"][k]["t_send"] < m * 60]
    f = lambda a: C.mean(B[a][k]["ttft"] for k in ks)
    lead = min(f(j) - f(n) for j in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut") for n in ("nofc_noend", "ec_r13", "jit_exact"))
    print(f"    to {m:5.1f} min {len(ks):5d} turns | {f('jit_p50_d_cut') - f('jit_p50_ctl'):+6.3f} | {f('jit_p50') - f('jit_p50_ctl'):+6.3f} | {lead:+6.3f}")

print("\n2. the same minutes (0 to 74.7 of each run's own clock): decode ms per step by running requests (Decode lines with no queue), first-turn prefill ms per token, SSD ms per page (reads of 500 pages or more), "
      "staged reads, launches refused room, reading turns among own kept returning turns, s2h pages, evicted tokens GPU/host (M)")
for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut"):
    t0 = D[a]["t0"]
    cells = []
    for lo, hi in ((8, 15), (16, 23), (24, 31)):
        st = [1000.0 * r / thr for t, r, tok, u, thr, q in D[a]["dec"] if 0 <= t - t0 < cut_end and lo <= r <= hi and q == 0 and thr > 0]
        cells.append(f"{lo}-{hi} {med(st):6.2f} ({len(st):3d})")
    fresh = []
    for t in D[a]["turns"]:
        if t["turn"] == 0 and t["ttft"] is not None and t["t_send"] < cut_end:
            r = D[a]["req"].get(f"{t['conv']}-0")
            unc = (t["pt"] or 0) - (t["cached"] or 0)
            if r and r[0] < 50 and unc > 2000:
                fresh.append((t["ttft"] - r[0] / 1000) / unc * 1000)
    io = [x for x in D[a]["s2h"] if 0 <= x[0] - t0 < cut_end]
    big = [x[5] / x[3] for x in io if x[3] >= 500]
    allp = sum(x[3] for x in io)
    allms = sum(x[5] for x in io)
    own = [k for k, t in B[a].items() if k[1] > 0 and t["t_send"] < cut_end]
    rd = set((x[1], x[2]) for x in io if x[3] > 0)
    print(f"  {a:14s} decode " + " ".join(cells) + f" | prefill {med(fresh):.4f} ({len(fresh)}) | SSD {med(big):5.2f} ms/page ({len(big)}), all reads {allms / allp:5.2f} ms/page over {allp} pages | "
          f"staged {sum(1 for x in D[a]['stage'] if 0 <= x[0] - t0 < cut_end)} refused {sum(1 for x in D[a]['disp'] if 0 <= x[0] - t0 < cut_end and x[5] == 0)} | "
          f"reading turns {sum(1 for k in own if k in rd)}/{len(own)} | evicted {sum(x[1] for x in D[a]['evict_device'] if 0 <= x[0] - t0 < cut_end) / 1e6:.1f}/{sum(x[1] for x in D[a]['evict_host'] if 0 <= x[0] - t0 < cut_end) / 1e6:.1f}")
print("  first 8 minutes, the same turns in the three runs: decode ms per completion token (median) and mean latency")
B8 = {a: {(t["conv"], t["turn"]): t for t in D[a]["turns"] if t["ttft"] is not None and t["t_send"] + (t["latency"] or 0) < 480} for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut")}
k8 = [k for k in set.intersection(*(set(v) for v in B8.values())) if len(set((B8[a][k]["pt"], B8[a][k]["ct"]) for a in B8)) == 1 and B8["jit_p50"][k]["ct"] >= 20]
for a in B8:
    print(f"    {a:14s} {med([1000 * (B8[a][k]['latency'] - B8[a][k]['ttft']) / B8[a][k]['ct'] for k in k8]):6.2f} ms/token, latency {C.mean(B8[a][k]['latency'] for k in k8):6.3f} s ({len(k8)} turns)")

print("\n3. how much of the first load episode (positions 5500-6999) the cut run's kept window covers")
six = [a for a in C.CUT7 if a != "jit_p50_d_cut"]
k6 = C.shared_keys(B, six, rank)
e6 = [k for k in k6 if 5500 <= rank.get(k, -1) < 7000]
inc = [k for k in e6 if k in B["jit_p50_d_cut"]]
out = [k for k in e6 if k not in B["jit_p50_d_cut"]]
print(f"  episode turns shared by the six complete runs: {len(e6)}; of them kept in the cut run {len(inc)} ({100 * len(inc) / len(e6):.1f} %), not reached {len(out)}")
for a in six:
    print(f"    {a:14s} mean TTFT on the covered episode turns {C.mean(B[a][k]['ttft'] for k in inc):6.3f}; on the uncovered {C.mean(B[a][k]['ttft'] for k in out):6.3f}; whole episode {C.mean(B[a][k]['ttft'] for k in e6):6.3f}")
print(f"  positions of the uncovered episode turns: {min(rank[k] for k in out)}-{max(rank[k] for k in out)}, median {med([rank[k] for k in out]):.0f}; minute they were sent in jit_p50_ctl: "
      f"{min(B['jit_p50_ctl'][k]['t_send'] for k in out) / 60:.1f}-{max(B['jit_p50_ctl'][k]['t_send'] for k in out) / 60:.1f}")
print(f"  on all {len(k6)} turns the six share (to minute 115): " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in k6):.3f}" for a in six))
print("  minute at which each JIT p50 run sent the turn at positions 5500, 6250, 6800 (median over +-50 positions): " + "; ".join(
    f"{a} " + " ".join(f"{med([B[a][k]['t_send'] for k in B[a] if abs(rank.get(k, -10 ** 9) - p) <= 50]) / 60:.1f}" for p in (5500, 6250, 6800)) for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut")))

print("\n4. server log of each JIT p50 run: lines with Traceback / ERROR / Exception / CUDA error / OOM, and the last stamped line")
pat = re.compile(r"Traceback|\bERROR\b|Exception|CUDA error|out of memory|Killed", re.I)
for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut"):
    n = 0
    last = ""
    first_hit = ""
    with open(os.path.join(C.RES, C.RUNS[a], "server.log"), errors="replace") as f:
        for line in f:
            if pat.search(line) and "server_args=" not in line:
                n += 1
                if not first_hit:
                    first_hit = line.strip()[:160]
            if line.startswith("[20"):
                last = line[:25]
    print(f"  {a:14s} {n} lines; first: {first_hit or '-'} | last stamp {last}")
