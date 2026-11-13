"""Independent recomputation of the headline tables (compare_ec.out, pairs_ec*.out, split_ec*.out) from the cache.
  nice -n 19 python3 -I -B repro_tables.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script's folder from the path
from lib import *

rk = rank()
print(f"order run: {len(rk)} positions (turns with a ttft, turn 0 included); last position {max(rk.values())}")

print("\n[A] per arm, kept turns: turns, up to minute, users started, errors, mean, median, p90, p99, mean queue, recomputed per returning turn, SSD pages (kept turns), kept turns that read")
for a in ALL11 + ["jit_p50_d_cut"]:
    d = arm(a)
    t = d["turns"]
    tt = [r["ttft"] for r in t]
    errs = sum(1 for r in raw(a)["client"]["turns"] if r["error"])
    print(f"  {a:14s} {len(t):6d} {d['end'] / 60:6.1f} {len({r['conv'] for r in t}):4d} {errs:3d} {mean(tt):6.2f} {st.median(tt):6.2f} {pct(tt, .9):6.2f} {pct(tt, .99):6.1f} "
          f"{mean(d['queue'].get((r['conv'], r['turn']), 0) for r in t):6.2f} {mean(rec(r) for r in t if r['turn'] > 0):6.0f} {sum(d['pages'][k] for k in d['by']):9d} {sum(1 for k in d['by'] if d['pages'][k]):6d}"
          f" | own returning {sum(1 for k in d['by'] if k[1] > 0):6d}, share reading {100 * sum(1 for k in d['by'] if k[1] > 0 and d['pages'][k]) / sum(1 for k in d['by'] if k[1] > 0):5.1f} %"
          f" | whole log: s2h pages {sum(d['pages'].values()):8d}, staged {sum(d['staged'].values()):5d}, refused {sum(d['refused'].values()):5d}")


def same(names, label):
    keys = shared(names)
    m = {a: mean(arm(a)["by"][k]["ttft"] for k in keys) for a in names}
    print(f"\n[B] {label}: {len(keys)} returning turns shared; keys absent from the order run {sum(k not in rk for k in keys)}; in episodes {sum(in_episode(k) for k in keys)}")
    print("    arm: mean TTFT = queue + after | recomputed | share of the gain from baseline, base_rep | episodes mean, elsewhere mean | p90 p99 over10 | reading turns, pages")
    ep = [k for k in keys if in_episode(k)]
    calm = [k for k in keys if not in_episode(k)]
    for a in names:
        d = arm(a)
        qq = mean(d["queue"].get(k, 0) for k in keys)
        v = [d["by"][k]["ttft"] for k in keys]
        gain = " ".join(f"{100 * (m[b] - m[a]) / (m[b] - m['jit_exact']):5.1f} %" for b in ("baseline", "base_rep") if b in m and "jit_exact" in m)
        print(f"    {a:14s} {m[a]:6.3f} = {qq:6.3f} + {m[a] - qq:5.3f} | {mean(rec(d['by'][k]) for k in keys):7.1f} | {gain} | {mean(d['by'][k]['ttft'] for k in ep):6.3f} {mean(d['by'][k]['ttft'] for k in calm):6.3f} | "
              f"{pct(v, .9):6.2f} {pct(v, .99):6.1f} {sum(x > 10 for x in v):5d} | {sum(d['pages'][k] > 0 for k in keys):5d} {sum(d['pages'][k] for k in keys):8d}")
    print("    by stretch: " + "  ".join(f"{lo}-{hi}: {sum(lo <= rk.get(k, -1) < hi for k in keys)}" for lo, hi in STRETCHES))
    for lo, hi in STRETCHES:
        ks = [k for k in keys if lo <= rk.get(k, -1) < hi]
        if ks:
            print(f"    {lo:5d}-{hi:5d} {len(ks):5d} | " + " ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in ks):.2f}" for a in names))
    return keys


def pairs(names, keys, label):
    ep = [k for k in keys if in_episode(k)]
    calm = [k for k in keys if not in_episode(k)]
    for title, ks in (("all", keys), ("episodes", ep), ("elsewhere", calm)):
        print(f"    [{label}] {title}: {len(ks)} turns; row minus column [5 %, 95 %], blocks used")
        for i, a in enumerate(names):
            for b in names[:i]:
                cells = blocks_by_rank({k: arm(a)["by"][k]["ttft"] - arm(b)["by"][k]["ttft"] for k in ks})
                d, lo, hi = boot(cells)
                print(f"      {a:14s} - {b:14s} {d:+7.3f} [{lo:+5.2f},{hi:+5.2f}]  blocks {len(cells)}, largest block {max(c[1] for c in cells.values())} turns")


k11 = same(ALL11, "ALL11")
k5 = same(LEAD5, "LEAD5")
pairs(LEAD5, k5, "LEAD5")
k7 = same(SEVEN, "SEVEN")
k9 = same(CTL9, "CTL9 (pairs_ec_ctl.out)")
pairs(CTL9, k9, "CTL9")
kc = same(CUT7, "CUT7 (pairs_ec_cut.out)")
pairs(CUT7, kc, "CUT7")
k3 = same(["jit_exact", "ec_r13", "nofc"], "trio")
k4n = same(["nofc_noend", "nofc", "ec_r13", "hintfree_ctl"], "noend four (split_ec_noend.out)")
k2 = same(["nofc", "nofc_noend"], "nofc pair (stall_ec_noend.out)")
k5s = same(["jit_exact", "ec_r13", "nofc", "jit_p50_ctl", "hintfree_ctl"], "five (split_ec.out)")

print("\n[C] baseline against its repeat on their shared returning turns")
kb = shared(["baseline", "base_rep"])
dif = [arm("base_rep")["by"][k]["ttft"] - arm("baseline")["by"][k]["ttft"] for k in kb]
print(f"    n {len(kb)} mean {mean(arm('baseline')['by'][k]['ttft'] for k in kb):.2f} repeat {mean(arm('base_rep')['by'][k]['ttft'] for k in kb):.2f} change {mean(dif):+.2f}")
