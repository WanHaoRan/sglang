"""ev_fetch 3a: error of the engine's at-arrival wait estimate (ntp_admit pred_wait_ql_ms = (position+0.5)/rate)
against the turn's GPU turn (admission for a turn with no read of its own; my D for a reader).  e = actual - predicted."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv, pct
for arm in ("ec_full", "jit_exact"):
    ev = vlib.load(34, arm); tab = runtab.build(34, arm)
    reqs, stall, T, NQ, NR = tab["reqs"], tab["stall"], tab["T"], tab["NQ"], tab["NR"]
    rows = []
    for t, r in ev["ntp_admit"]:
        d = kv(r); q = reqs.get(d["rid"])
        if q is None: continue
        pred = float(d["pred_wait_ql_ms"]) / 1e3
        s = stall.get(d["rid"])
        D = s["D"] if s else q["adm"]
        j = bisect.bisect_left(T, q["entry"]) - 1
        rows.append((q["entry"], D - q["entry"] - pred, pred, bool(s), q["adm"] - q["entry"] - pred, (NQ[j], NR[j]) if j >= 0 else (0, 0)))
    print(f"c34 {arm}: planned turns {len(rows)}")
    for lo, hi in [(0, 30), (30, 70), (70, 90), (90, 110)]:
        e = [x[1] for x in rows if lo * 60 <= x[0] < hi * 60]
        en = [x[1] for x in rows if lo * 60 <= x[0] < hi * 60 and not x[3]]
        er = [x[4] for x in rows if lo * 60 <= x[0] < hi * 60 and x[3]]
        share_nr = [1 - x[5][1] / x[5][0] for x in rows if lo * 60 <= x[0] < hi * 60 and x[5][0] > 0]
        print(f"   minutes {lo}-{hi}: GPU turn - prediction: median {pct(e,.5):+.2f} p5 {pct(e,.05):+.2f} p20 {pct(e,.2):+.2f} p80 {pct(e,.8):+.2f} p95 {pct(e,.95):+.2f} s (|e| p90 {pct([abs(x) for x in e],.9):.2f});"
              f" non-readers median {pct(en,.5):+.2f}; readers' admission - prediction median {pct(er,.5) if er else float('nan'):+.2f} s; share of the queue waiting on its own read at arrival, mean {100*sum(share_nr)/max(len(share_nr),1):.0f} %")
