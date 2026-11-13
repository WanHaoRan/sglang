"""Reproduce the stated numbers of readings R1-R4 from the raw logs (through cache/), with my own code.
  nice -n 19 python3 -I -B t0_reproduce.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
D = {a: C.load(a) for a in C.RUNS if a != "order"}
B = {a: C.by_key(C.kept(D[a])) for a in D}
READ = {a: {} for a in D}
for a in D:
    for t, c, n, pages, tok, ms in D[a]["s2h"]:
        READ[a][(c, n)] = READ[a].get((c, n), 0) + pages

print("order run: turns with ttft", len(rank), "max position", max(rank.values()))
print("\nper run: all turns, kept (margin 300 s), kept returning, last kept t_send (min), t0 (UTC epoch), errors, null ttft")
for a in D:
    k = C.kept(D[a])
    print(f"  {a:14s} {len(D[a]['turns']):6d} {len(k):6d} {sum(1 for t in k if t['turn'] > 0):6d} {k[-1]['t_send'] / 60:7.1f} {D[a]['t0']:.0f} "
          f"{sum(1 for t in D[a]['turns'] if t['err'])} {sum(1 for t in D[a]['turns'] if t['ttft'] is None)}")


def table(names, title):
    keys = C.shared_keys(B, names, rank)
    ep = [k for k in keys if C.in_episode(rank, k)]
    ep1 = [k for k in keys if 5500 <= rank.get(k, -1) < 7000]
    ep2 = [k for k in keys if 8750 <= rank.get(k, -1) < 11000]
    calm = [k for k in keys if not C.in_episode(rank, k)]
    absent = [k for k in keys if k not in rank]
    print(f"\n{title}: {len(keys)} shared returning turns; in episodes {len(ep)} (first {len(ep1)}, second {len(ep2)}), elsewhere {len(calm)}, of them absent from the order run {len(absent)}")
    print("  arm            mean  median   p90    p99  >10s | episodes  ep1   ep2  elsewhere  absent-from-order | reading turns (s2h pages>0)  pages  storage>0 (client) | recomputed/turn")
    out = {}
    for a in names:
        b = B[a]
        v = [b[k]["ttft"] for k in keys]
        rd = sum(1 for k in keys if READ[a].get(k, 0) > 0)
        rd_c = sum(1 for k in keys if (b[k]["sto"] or 0) > 0)
        out[a] = C.mean(v)
        print(f"  {a:14s} {C.mean(v):5.3f} {C.pct(v, .5):6.3f} {C.pct(v, .9):6.2f} {C.pct(v, .99):6.1f} {sum(x > 10 for x in v):5d} | "
              f"{C.mean(b[k]['ttft'] for k in ep):6.3f} {C.mean(b[k]['ttft'] for k in ep1):6.3f} {C.mean(b[k]['ttft'] for k in ep2):6.3f} "
              f"{C.mean(b[k]['ttft'] for k in calm):6.3f} {C.mean(b[k]['ttft'] for k in absent):6.3f} | {rd:5d} {sum(READ[a].get(k, 0) for k in keys):8d} {rd_c:5d} | "
              f"{C.mean(C.rec(b[k]) for k in keys):6.0f}")
    return keys, out


k5, m5 = table(C.LEAD5, "R1: LEAD5")
print("  R2: nofc_noend - nofc on LEAD5 shared turns: %+.3f s" % (m5["nofc_noend"] - m5["nofc"]))
k2, m2 = table(["nofc", "nofc_noend"], "R2: the two rivals alone")
print("  R2: nofc_noend - nofc on their own shared turns: %+.3f s" % (m2["nofc_noend"] - m2["nofc"]))
k11, m11 = table(C.ALL11, "R3: ALL11")
for base in ("baseline", "base_rep"):
    for a in ("nofc_noend", "nofc", "ec_r1", "ec_r13", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl"):
        print(f"  share of the way from {base} to jit_exact covered by {a:13s}: {100 * (m11[base] - m11[a]) / (m11[base] - m11['jit_exact']):5.1f} %")
k7, m7 = table(C.SEVEN, "SEVEN")
kc, mc = table(C.CUT7, "R4: CUT7")
print("  R4: jit_p50_d_cut - jit_p50_ctl %+.3f s; jit_p50_d_cut - jit_p50 %+.3f s; jit_p50_ctl - jit_p50 %+.3f s" % (
    mc["jit_p50_d_cut"] - mc["jit_p50_ctl"], mc["jit_p50_d_cut"] - mc["jit_p50"], mc["jit_p50_ctl"] - mc["jit_p50"]))
