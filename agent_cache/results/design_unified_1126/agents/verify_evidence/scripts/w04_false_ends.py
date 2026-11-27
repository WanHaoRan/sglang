"""ev_write 1b: commits the engine registered as 'ended' (nue_register next_use_ms=ended at the request's finish)
that were not the last request of their stream in the trace."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab, chains
from vlib import kv
for arm in ("ec_full", "jit_p50", "jit_exact", "ec_exact"):
    ev = vlib.load(34, arm); tab = runtab.build(34, arm); ch = chains.load_chains(34, arm)
    reqs = tab["reqs"]
    lastreg = {}
    for t, r in ev["nue_register"]:
        d = kv(r); lastreg[d["rid"]] = (t, d["next_use_ms"])     # the last line per rid is the commit at its finish
    ended = [rid for rid, (t, v) in lastreg.items() if v == "ended" and rid in reqs]
    false = [rid for rid in ended if not ch[rid]["last"]]
    true = [rid for rid in ended if ch[rid]["last"]]
    wtok = sum((reqs[r]["prompt"] - reqs[r]["cached"]) + reqs[r]["out"] for r in false)
    # how many of the false ends had a hint with no entry (matched level)
    lv = collections.Counter((reqs[r]["hint"] or {}).get("matched") for r in false)
    nxt = sum(1 for r in false if ("%s-t%d" % (r.rsplit("-t", 1)[0], reqs[r]["k"] + 1)) in reqs)
    print(f"c34 {arm}: commits registered ended {len(ended)}; last request of the stream {len(true)}; not the last {len(false)} (their next request was sent in {nxt}); "
          f"tokens those turns computed anew {wtok/1e6:.2f} M; hint level of the false ends {dict(lv)}")
