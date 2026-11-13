"""Test (f3): is the TTFT order carried through to the whole response?
Per arm on the shared returning turns: client TTFT and latency, the server's queue wait and forward time (ReqTimeStats),
decode time, throughput (turns kept in 115 min), and the GPU's mean running and queued requests from the Decode lines.
A turn admitted sooner joins a larger decode batch, so a lower queue wait can come back as a longer forward time.
  nice -n 19 python3 -I -B tf_latency.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
ARMS = [a for a in C.RUNS if a != "order"]
D = {a: C.load(a) for a in ARMS}
B = {a: C.by_key(C.kept(D[a])) for a in ARMS}


def req(a, k, i):
    r = D[a]["req"].get(f"{k[0]}-{k[1]}")
    return r[i] / 1000 if r else 0.0


def table(names, title, ref):
    keys = C.shared_keys(B, names, rank)
    ep = [k for k in keys if C.in_episode(rank, k)]
    print(f"\n{title}: {len(keys)} shared returning turns ({len(ep)} in the episodes)")
    print("  arm            ttft  = pre-entry + queue + rest | latency | forward | decode (latency - ttft) | ms per completion token (median) | episodes: ttft, latency | "
          "paired against " + ref + ": ttft, decode, latency")
    m = {}
    for a in names:
        b = B[a]
        ttft = C.mean(b[k]["ttft"] for k in keys)
        lat = C.mean(b[k]["latency"] for k in keys)
        q = C.mean(req(a, k, 0) for k in keys)
        fwd = C.mean(req(a, k, 1) for k in keys)
        pre = C.mean((D[a]["req"][f"{k[0]}-{k[1]}"][2] - (D[a]["t0"] + b[k]["t_send"])) if f"{k[0]}-{k[1]}" in D[a]["req"] else 0.0 for k in keys)
        dec = lat - ttft
        per = C.pct([1000 * (b[k]["latency"] - b[k]["ttft"]) / b[k]["ct"] for k in keys if b[k]["ct"] >= 20], .5)
        m[a] = (ttft, lat, dec)
        r = B[ref]
        print(f"  {a:14s} {ttft:6.3f} = {pre:5.3f} + {q:6.3f} + {ttft - pre - q:5.3f} | {lat:7.3f} | {fwd:7.3f} | {dec:7.3f} | {per:6.2f} | "
              f"{C.mean(b[k]['ttft'] for k in ep):6.3f} {C.mean(b[k]['latency'] for k in ep):7.3f} | "
              f"{C.mean(b[k]['ttft'] - r[k]['ttft'] for k in keys):+6.3f} {C.mean((b[k]['latency'] - b[k]['ttft']) - (r[k]['latency'] - r[k]['ttft']) for k in keys):+6.3f} {C.mean(b[k]['latency'] - r[k]['latency'] for k in keys):+6.3f}")
    return keys, m


k5, m5 = table(C.LEAD5, "LEAD5", "jit_exact")
print("  spread across the five: ttft %.3f s, latency %.3f s (%.1f %% of the mean latency); order by ttft: %s; by latency: %s" % (
    max(v[0] for v in m5.values()) - min(v[0] for v in m5.values()), max(v[1] for v in m5.values()) - min(v[1] for v in m5.values()),
    100 * (max(v[1] for v in m5.values()) - min(v[1] for v in m5.values())) / C.mean(v[1] for v in m5.values()),
    " < ".join(sorted(m5, key=lambda a: m5[a][0])), " < ".join(sorted(m5, key=lambda a: m5[a][1]))))
print("  the four arms of 2026-11-11 only (one GPU unit): latency spread %.3f s; order by latency: %s" % (
    max(m5[a][1] for a in C.LEAD5[1:]) - min(m5[a][1] for a in C.LEAD5[1:]), " < ".join(sorted(C.LEAD5[1:], key=lambda a: m5[a][1]))))
k7, m7 = table(C.SEVEN, "SEVEN", "jit_exact")
k11, m11 = table(C.ALL11, "ALL11", "jit_exact")
for base in ("baseline", "base_rep"):
    print(f"  share of the way from {base} to jit_exact covered by nofc_noend: ttft {100 * (m11[base][0] - m11['nofc_noend'][0]) / (m11[base][0] - m11['jit_exact'][0]):.1f} %, "
          f"latency {100 * (m11[base][1] - m11['nofc_noend'][1]) / (m11[base][1] - m11['jit_exact'][1]):.1f} % (latency {m11[base][1]:.2f} -> {m11['nofc_noend'][1]:.2f} -> {m11['jit_exact'][1]:.2f})")
kc, mc = table(C.CUT7, "CUT7", "jit_p50_ctl")

print("\nthroughput and GPU occupancy over each run's kept window: kept turns, kept returning turns, minutes, turns per minute | completion tokens per second | "
      "Decode lines: mean running, mean queued, share with a queue | prefill tokens computed (M), cached (M)")
for a in ARMS:
    kt = C.kept(D[a])
    end = kt[-1]["t_send"]
    t0 = D[a]["t0"]
    dec = [x for x in D[a]["dec"] if 0 <= x[0] - t0 < end]
    pre = [x for x in D[a]["pre"] if 0 <= x[0] - t0 < end]
    print(f"  {a:14s} {len(kt):6d} {sum(1 for t in kt if t['turn'] > 0):6d} {end / 60:6.1f} {len(kt) / (end / 60):6.1f} | {sum(t['ct'] or 0 for t in kt) / end:7.1f} | "
          f"{C.mean(x[1] for x in dec):5.2f} {C.mean(x[5] for x in dec):5.2f} {100 * C.mean(1 if x[5] > 0 else 0 for x in dec):5.1f} % | {sum(x[2] for x in pre) / 1e6:6.2f} {sum(x[3] for x in pre) / 1e6:7.1f}")

print("\nsame wall-clock window instead of the same turns: every kept returning turn each arm sent in minutes 0-100 of its own run: turns, mean TTFT, mean latency")
for a in C.SEVEN:
    v = [t for t in C.kept(D[a]) if t["turn"] > 0 and t["t_send"] < 6000]
    print(f"  {a:14s} {len(v):6d} {C.mean(t['ttft'] for t in v):6.3f} {C.mean(t['latency'] for t in v):7.3f}")
