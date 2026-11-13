"""ec_r13_rep against ec_r13 and the other leading arms by stretch of the order run, on the first set (the turns original
HiCache's runs share with them) and on the leading arms' set; where the repeat's extra time sits; its mechanisms.
  nice -n 19 python3 -I -B r2_repeats2.py"""
import os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, rank, mean, pct, rec, STRETCHES, ALL11

rk = rank()
L = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
for names, label in ((ALL11 + ["ec_r13_rep"], "first set: the eleven runs of compare_ec.out plus ec_r13_rep"), (L, "leading arms plus ec_r13_rep")):
    keys = shared(names)
    show = [a for a in names if a in L + ["baseline", "base_rep"]]
    m = {a: mean(arm(a)["by"][k]["ttft"] for k in keys) for a in names}
    print(f"\n== {label}: {len(keys)} shared returning turns")
    print("   mean TTFT: " + "  ".join(f"{a} {m[a]:.3f}" for a in show))
    if "baseline" in m:
        print("   share of the gain from baseline, base_rep to jit_exact: " + "  ".join(f"{a} {100 * (m['baseline'] - m[a]) / (m['baseline'] - m['jit_exact']):.1f} % {100 * (m['base_rep'] - m[a]) / (m['base_rep'] - m['jit_exact']):.1f} %" for a in L))
    for lo, hi in STRETCHES:
        ks = [k for k in keys if lo <= rk.get(k, -1) < hi]
        if ks:
            print(f"   {lo:5d}-{hi:5d} {len(ks):5d} | " + "  ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in ks):.2f}" for a in L))
    ks = [k for k in keys if k not in rk]
    if ks:
        print(f"   absent      {len(ks):5d} | " + "  ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in ks):.2f}" for a in L))
print("\nown returning kept turns: turns | share reading from SSD | staged reads | refused launches | SSD pages (kept turns) | turns over 10 s | minute at own position 7000, 8750, 11000")
for a in L:
    d = arm(a)
    own = [k for k in d["by"] if k[1] > 0]
    t = d["turns"]
    print(f"   {a:11s} {len(own):6d} | {100 * sum(d['pages'][k] > 0 for k in own) / len(own):4.1f} % | {sum(d['staged'].values()):5d} | {sum(d['refused'].values()):5d} | {sum(d['pages'][k] for k in d['by']):7d} | "
          f"{sum(d['by'][k]['ttft'] > 10 for k in own):4d} | " + " ".join(f"{t[p]['t_send'] / 60:6.1f}" for p in (7000, 8750, 11000) if p < len(t)))
