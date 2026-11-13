"""Spot checks of compare_ec.out's time tables from the cache: the 15-minute rows (turns, mean TTFT, mean queue wait, GPU mean
running and queue from Decode lines, prefill K computed) and the 500-turn position bins (mean TTFT, minute reached).
  nice -n 19 python3 -I -B r2_spot_tables.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, mean

for a in ("baseline", "jit_exact", "ec_r13", "nofc_noend", "hintfree_ctl"):
    d = arm(a)
    s = d["server"]
    print(f"== {a}")
    for m in (45, 60, 90, 105):
        x = [r for r in d["turns"] if m * 60 <= r["t_send"] < (m + 15) * 60]
        dd = [b for b in s["dec"] if 0 <= b[0] - d["t0"] < d["end"] and m * 60 <= b[0] - d["t0"] < (m + 15) * 60]
        pp = sum(p[2] for p in s["pre"] if 0 <= p[0] - d["t0"] < d["end"] and m * 60 <= p[0] - d["t0"] < (m + 15) * 60)
        print(f"   min {m:3d}: {len(x):5d} {mean(r['ttft'] for r in x):6.2f} {mean(d['queue'].get((r['conv'], r['turn']), 0) for r in x):6.2f} | GPU {mean(b[1] for b in dd):4.1f} {mean(b[5] for b in dd):4.1f} {pp / 1000:5.0f}")
    t = d["turns"]
    print("   position bins: " + "  ".join(f"{i}: {mean(r['ttft'] for r in t[i:i + 500]):.2f} at min {t[i]['t_send'] / 60:.1f}" for i in (4000, 6000, 8500, 9500) if i + 500 <= len(t)))
