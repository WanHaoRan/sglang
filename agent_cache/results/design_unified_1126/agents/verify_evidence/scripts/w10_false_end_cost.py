"""Turns the engine registered as ended that were not the stream's last (table-hint arms): did their next request
need the SSD more often than other returning requests with a like pause?"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab, chains
from vlib import kv, mean
for arm in ("ec_full", "jit_p50"):
    ev = vlib.load(34, arm); tab = runtab.build(34, arm); ch = chains.load_chains(34, arm)
    reqs = tab["reqs"]
    lastreg = {}
    for t, r in ev["nue_register"]:
        d = kv(r); lastreg[d["rid"]] = d["next_use_ms"]
    false_next = set()
    for rid, v in lastreg.items():
        if v == "ended" and rid in reqs and not ch[rid]["last"]:
            false_next.add("%s-t%d" % (rid.rsplit("-t", 1)[0], reqs[rid]["k"] + 1))
    def bucket(w):
        return "<2 s" if w < 2 else "2-10 s" if w < 10 else "10-60 s" if w < 60 else ">=60 s"
    stats = collections.defaultdict(lambda: [0, 0, 0, 0])
    for rid, q in reqs.items():
        if q["k"] < 1 or q["t_send"] < 1800 or q["t_send"] >= 6600: continue
        key = (bucket(q["wait_s"] or 0.0), rid in false_next)
        s = stats[key]
        s[0] += 1; s[1] += 1 if (q["reads"] or q["sto"] > 0) else 0; s[2] += q["sto"]; s[3] += max(q["ideal"] - q["cached"], 0)
    print(f"c34 {arm}: next requests of falsely ended turns sent in minutes 30-110: {sum(v[0] for k, v in stats.items() if k[1])}")
    for b in ("<2 s", "2-10 s", "10-60 s", ">=60 s"):
        f = stats[(b, True)]; o = stats[(b, False)]
        if f[0]:
            print(f"   pause {b:8s}: after a false end n {f[0]:4d}, read from the SSD {100*f[1]/f[0]:5.1f} %, {f[2]/f[0]/1e3:5.1f} K tokens read per request | others n {o[0]:5d}, {100*o[1]/o[0]:5.1f} %, {o[2]/o[0]/1e3:5.1f} K")
    ft = sum(v[2] for k, v in stats.items() if k[1]); at = sum(v[2] for k, v in stats.items())
    print(f"   tokens read from the SSD by the next requests of false ends: {ft/1e6:.2f} M of {at/1e6:.2f} M in minutes 30-110")
