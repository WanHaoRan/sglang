"""The figure entry's numbers against ../../../../compare_c26_ec/pricing_vs_jit.csv (5-minute bins): mean of 'held' over
minutes 45-120 per arm, and the footnote's means over the returning turns the five arms share, from the cache.
  nice -n 19 python3 -I -B r2_fig_csv.py"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, mean

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "compare_c26_ec", "pricing_vs_jit.csv"))
rows = list(csv.DictReader(open(path)))
arms = list(dict.fromkeys(r["arm"] for r in rows))
print("pricing_vs_jit.csv arms:", arms, "| columns:", list(rows[0]))
for a in arms:
    x = [r for r in rows if r["arm"] == a]
    late = [float(r["held"]) for r in x if 45 <= float(r["minute"]) < 120 and r["held"] not in ("", "nan")]
    print(f"  {a:13s} run {x[0]['run'][:15]} bins {len(x)} minutes {x[0]['minute']}..{x[-1]['minute']} | held, mean over bins from minute 45: {mean(late):9.1f} ({len(late)} bins) | max bin {max(late):9.1f}")
five = ["baseline", "jit_p50_ctl", "nofc_noend", "ec_r13", "jit_exact"]
keys = shared(five)
print(f"\nfootnote: {len(keys)} returning turns shared by {five}: " + "  ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in keys):.3f}" for a in five))
