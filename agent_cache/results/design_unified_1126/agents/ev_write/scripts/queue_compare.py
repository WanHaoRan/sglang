"""The SSD write queue sampled every second over a window: ops pending, age of the oldest, tokens, and the chains of
finished requests under pending writes (bound on host memory the writes hold)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_trees
from model import load
from write_held import Pending, q
trees = load_trees()
a, b = float(sys.argv[1]), float(sys.argv[2])
print(f"window {a:.0f}-{b:.0f} s, one sample a second: pending ops p50/p90/max | oldest age s p50/p90/max | pending K tokens p50/p90/max | chains under pending writes of finished requests K p50/p90/max | samples with that bound over 500 K")
for spec in sys.argv[3:]:
    camp, label = spec.split(":")
    m = load(camp, label, trees)
    pd = Pending(m)
    rows = [pd.summary(float(t)) for t in range(int(a), int(b))]
    f = lambda k, s=1.0: f"{q([r[k] for r in rows], .5) / s:.0f}/{q([r[k] for r in rows], .9) / s:.0f}/{max(r[k] for r in rows) / s:.0f}"
    age = [r["oldest_age"] for r in rows]
    print(f"{camp} {label}: {f('n')} | {q(age, .5):.1f}/{q(age, .9):.1f}/{max(age):.1f} | {f('pending', 1e3)} | {f('bound', 1e3)} | {sum(1 for r in rows if r['bound'] > 500000)} of {len(rows)}")
