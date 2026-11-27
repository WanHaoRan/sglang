import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
m = load(sys.argv[1], sys.argv[2], load_trees())
un = sorted(m.unmatched_reqs, key=lambda x: x[2])
print(len(un), "unmatched requests; by 10-minute window:", sorted(collections.Counter(int(x[2] // 600) * 10 for x in un).items()))
left = [o for o in m.ops.values() if o["part"] is None]
print(len(left), "ops left; by 10-minute window:", sorted(collections.Counter(int(o["t_sub"] // 600) * 10 for o in left).items()))
for rid, e, tf in un[40:46]:
    r = m.reqs[rid]; st = m.reqstat.get(rid)
    print(rid, "E", e, "t_first %.3f" % tf, "P", r.P, "cached", r.cached, "dev/host/stor", r.dev, r.host, r.stor, "ideal", r.ideal, st)
