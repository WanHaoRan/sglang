"""Cleaning orders on my SSD replay (open loop; recorded timeline, lookups and pass times/sizes).
 today          oldest file time first (written, or touched by fix 1 when recomputed)
 b_chain        at a session's end every page of its chains gets file time 0
 b_tail         at a session's end its pages past its first prompt (and its outputs) get file time 0
 refresh        at each request's end every stored page of its prompt is touched   (refresh300: once per 300 s per session)
 touch_on_read  the pages a lookup finds are touched
 b_tail+refresh, b_chain+refresh
 oracle         farthest next use (send time of the next request whose prompt holds the page) first
TTFT at stake by N g: tokens lost in minutes 90-110 x 0.043 ms x 128 / requests sent in those minutes."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay
from s01_validate import losses

def live_deleted(D, res, until=6600.0):
    live = shared = tot = 0
    for t, vs in res["deleted"]:
        if t >= until: continue
        for i in vs:
            tot += 1
            if D.next_use(i, t) < float("inf"):
                live += 1
    return tot, live

if __name__ == "__main__":
    REPAIR = "--norepair" not in sys.argv
    arms = [a for a in sys.argv[1:] if not a.startswith("--")] or ["ec_full"]
    for arm in arms:
        D = ssd_replay.Data(34, arm); ssd_replay.first_sets(D); ssd_replay.uses_index(D)
        base = ssd_replay.replay(D, delete=False)
        p1 = D.passes[0][0]
        n90 = sum(1 for q in D.reqs.values() if 5400 <= q["t_send"] < 6600)
        npost = sum(1 for q in D.reqs.values() if p1 <= q["t_send"] < 6600)
        print(f"=== c34 {arm} (repair after a loss: {REPAIR}): requests first pass..6,600 s {npost}; minutes 90-110 {n90}; sessions that ended in the run {len(D.ended_at)} of {len(set(D.sess.values()))}")
        print("   order                 | first pass..6,600 s: requests losing, K tokens | minutes 90-110: K tokens -> TTFT by N g | pages deleted before 6,600 s, live among them")
        runs = [("today", {}), ("b_chain", {}), ("b_chain", {"seed": 1}), ("b_chain", {"seed": 2}), ("b_tail", {}), ("b_tail", {"seed": 1}),
                ("refresh", {}), ("refresh", {"refresh_period": 300.0}), ("touch_on_read", {}), ("b_tail+refresh", {}),
                ("b_chain+refresh", {}), ("oracle", {})]
        for pol, kw in runs:
            t = time.time()
            res = ssd_replay.replay(D, policy=pol, base=(base if REPAIR else None), **kw)
            nl, tl, rows = losses(D, base, res, p1, 6600)
            n2, t2, _ = losses(D, base, res, 5400, 6600)
            tot, live = live_deleted(D, res)
            nm = pol + ("" if not kw else " " + ",".join(f"{k}={v:g}" for k, v in kw.items()))
            print(f"   {nm:22s}| {nl:4d} {tl/1e3:8.0f} | {t2/1e3:8.0f} -> {t2*0.043e-3*128/n90:5.2f} s | {tot:6d} {live:6d}   ({time.time()-t:.0f} s)", flush=True)
