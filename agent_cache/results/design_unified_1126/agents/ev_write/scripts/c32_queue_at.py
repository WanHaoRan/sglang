"""The SSD write queue at one instant of a campaign 32 arm: the ops pending, oldest first, and what each one holds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from write_held import Pending
camp, label, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
m = load(camp, label, load_trees())
pd = Pending(m)
pend = sorted(pd.at(t), key=lambda o: o["t_sub"])
print(f"{camp} {label} at {t:.0f} s: {len(pend)} ops pending, {sum(o['tokens'] for o in pend)} tokens")
print("op, submitted (s before), done (s after), tokens, write ms, kind, request finished?, chain end (tokens), session")
for o in pend:
    rid = o["rid"]; r = m.reqs.get(rid) if rid else None
    fin = pd.fin.get(rid, float("inf")) <= t if rid else None
    end = None
    if r is not None:
        end = (r.P + r.out) // PAGE * PAGE if o["part"] == "out" else o["p0"] * PAGE + o["tokens"]
    print(f"  op {o['op']:6d} -{t - o['t_sub']:5.1f} s +{(o['t_done'] or 1e9) - t:5.1f} s {o['tokens']:6d} tok {o['io_ms'] or -1:6.0f} ms {str(o['part']):6s} fin={fin} chain {end} {rid}")
