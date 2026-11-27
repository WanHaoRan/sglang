"""Estimates of a turn's GPU turn D at its arrival, scored against D (first pass-over, else the admission):
  all    (position + 0.5) / rate, the engine's (every queued turn ahead counts);
  ready  (ready turns ahead + 0.5) / rate: turns ahead whose own read is under way at that moment are left out;
  half   the mean of the two.
By window of minutes, for planned turns at positions >= 2.

  python d_estimators.py <label> [<label> ...]
"""
import bisect
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import runlib as R  # noqa: E402

WINDOWS = ((0, 30), (30, 70), (70, 90), (90, 110))


def main():
    for label in sys.argv[1:]:
        d = R.load(label)
        Q = R.Queue(d)
        RD = R.Reads(d, Q)
        D = R.gpu_turn_refined(Q, RD, R.usage_series(d))
        t0 = d["run_rec"]["t_start"]
        # per rid: intervals during which a read of its own is under way while it is queued
        busy = {}
        for r in RD.reads:
            if r["arr"] is None or r["adm"] is None:
                continue
            a = max(r["park"], r["arr"])
            b = r["land"] if r["land"] is not None else (r["disp"] if r["disp"] is not None else r["park"])
            if b > a:
                busy.setdefault(r["rid"], []).append((a, b))
        order = sorted(Q.arr, key=lambda rid: Q.arr[rid])
        waiting = []   # heap of (admission, rid)
        rows = []
        arrive = R.first_by_rid(d["evts"].get("ntp_arrive", []))
        for rid in order:
            a = Q.arr[rid]
            while waiting and waiting[0][0] <= a:
                heapq.heappop(waiting)
            pos = len(waiting)
            reading = sum(1 for _, other in waiting if any(x <= a < y for x, y in busy.get(other, ())))
            rate = Q.rate(a)
            heapq.heappush(waiting, (Q.adm[rid], rid))
            if rid not in arrive or rate is None or pos < 2:
                continue
            w_all = (pos + 0.5) / rate[0]
            w_ready = (pos - reading + 0.5) / rate[0]
            rows.append((a - t0, D[rid] - a, w_all, w_ready, pos, reading))
        print(f"==== {label}: planned turns arriving at positions >= 2: {len(rows)} ====")
        print("window | n | turns ahead p50, of which reading p50 (share) | error D - estimate (s): all: p10 p50 p90 |e| p50 | "
              "ready: p10 p50 p90 |e| p50 | half: p10 p50 p90 |e| p50")
        for lo, hi in WINDOWS:
            g = [x for x in rows if lo * 60 <= x[0] < hi * 60]
            if len(g) < 20:
                continue
            out = []
            for est in (lambda x: x[2], lambda x: x[3], lambda x: 0.5 * (x[2] + x[3])):
                e = [x[1] - est(x) for x in g]
                out.append(f"{C.q(e, .1):+6.1f} {C.q(e, .5):+6.1f} {C.q(e, .9):+6.1f} {C.q([abs(v) for v in e], .5):5.1f}")
            share = sum(x[5] for x in g) / max(1, sum(x[4] for x in g))
            print(f"  {lo:3d}-{hi:<3d} | {len(g):5d} | {C.q([x[4] for x in g], .5):3d}, {C.q([x[5] for x in g], .5):3d} ({100 * share:.0f} %) | "
                  + " | ".join(out))


if __name__ == "__main__":
    main()
