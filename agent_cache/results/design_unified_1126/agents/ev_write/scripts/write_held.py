"""What pending SSD writes hold in host memory: the written node and, because host eviction takes leaves only, every
host-only ancestor of it, i.e. the session's chain once the GPU has let it go.

No census exists in campaign 32, so this estimates the write-held tokens from the write ops and the join:
  bound(t) = sum over sessions with a pending write whose request has finished of the chain length up to that node
(the chain of a running request is on the GPU and counts as mirror). It is an upper bound: the GPU may still hold
part of a finished session's chain. Calibrated on campaign 34, whose census logs h_write (tokens only writes hold)
and h_multi.

  python3 -I -B write_held.py calibrate <label> [<label> ...]        # campaign 34: bound against the census
  python3 -I -B write_held.py first <label> [<label> ...]            # campaign 32: around the first refusals
"""
from __future__ import annotations

import bisect
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees  # noqa: E402
from model import load  # noqa: E402


def q(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(p * len(ys)))]


class Pending:
    """Pending write ops over time: query the set at t."""

    def __init__(self, m):
        self.m = m
        self.ops = sorted((o for o in m.ops.values()), key=lambda o: o["t_sub"])
        self.t_sub = [o["t_sub"] for o in self.ops]
        self.fin = {rid: st["t"] for rid, st in m.reqstat.items()}
        self.max_lag = max((o["t_done"] - o["t_sub"] for o in self.ops if o["t_done"] is not None), default=0.0)

    def at(self, t: float) -> list:
        lo = bisect.bisect_left(self.t_sub, t - self.max_lag - 1.0)
        hi = bisect.bisect_right(self.t_sub, t)
        return [o for o in self.ops[lo:hi] if o["t_done"] is None or o["t_done"] > t]

    def summary(self, t: float) -> dict:
        m = self.m
        pend = self.at(t)
        tok = sum(o["tokens"] for o in pend)
        oldest = min((o["t_sub"] for o in pend), default=t)
        chains: dict = {}      # session -> deepest chain end among its pending ops of finished requests
        cls = collections.Counter()
        unjoined = 0
        for o in pend:
            rid = o["rid"]
            if rid is None:
                unjoined += o["tokens"]
                continue
            r = m.reqs[rid]
            if o["part"] == "out":
                end = (r.P + r.out) // PAGE * PAGE
                cls["out"] += o["tokens"]
            else:
                end = o["p0"] * PAGE + o["tokens"]
                cls["last" if r.last else "prompt"] += o["tokens"]
            if self.fin.get(rid, float("inf")) <= t:
                chains[r.inst] = max(chains.get(r.inst, 0), end)
        return {"t": t, "pending": tok, "n": len(pend), "oldest_age": t - oldest, "sessions": len(chains),
                "bound": sum(chains.values()), "cls": dict(cls), "unjoined": unjoined}


def calibrate(label: str, trees) -> None:
    m = load("campaign34", label, trees)
    pd = Pending(m)
    rows = []
    for c in m.census:
        if c["t"] < 1800 or c["t"] > 7200:
            continue
        s = pd.summary(c["t"])
        rows.append((c["h_write"] + c["h_multi"], c["h_write"], c["pending_write"], c["n_writes"], s["bound"],
                     s["pending"], s["n"], s["sessions"]))
    hw = sum(r[1] for r in rows)
    bd = sum(r[4] for r in rows)
    pw = sum(r[2] for r in rows)
    nz = [r for r in rows if r[4] > 0]
    print(f"campaign34 {label}: {len(rows)} census lines from minute 30; pending writes (census) mean {pw / len(rows) / 1e3:.1f} K, "
          f"held only by writes mean {hw / len(rows) / 1e3:.1f} K (p90 {q([r[1] for r in rows], .9) / 1e3:.0f} K, max "
          f"{max(r[1] for r in rows) / 1e3:.0f} K); the bound from the join mean {bd / len(rows) / 1e3:.1f} K (p90 "
          f"{q([r[4] for r in rows], .9) / 1e3:.0f} K, max {max(r[4] for r in rows) / 1e3:.0f} K)")
    print(f"   held / pending tokens: {hw / max(pw, 1):.2f}; held / bound: {hw / max(bd, 1):.2f}; lines with the bound "
          f"under the census's held-by-writes: {sum(1 for r in rows if r[4] < r[1])} of {len(rows)}; join's pending "
          f"tokens against the census's: {sum(r[5] for r in rows) / max(pw, 1):.2f}")
    # by size of the pending set
    for lo, hi in ((1, 10000), (10000, 30000), (30000, 60000), (60000, 10**9)):
        sel = [r for r in rows if lo <= r[2] < hi]
        if sel:
            print(f"   census pending {lo / 1e3:.0f}-{hi / 1e3:.0f} K: {len(sel)} lines, held by writes mean "
                  f"{sum(r[1] for r in sel) / len(sel) / 1e3:.0f} K, bound mean {sum(r[4] for r in sel) / len(sel) / 1e3:.0f} K, "
                  f"held/pending {sum(r[1] for r in sel) / max(sum(r[2] for r in sel), 1):.2f}, sessions "
                  f"{sum(r[7] for r in sel) / len(sel):.1f}")


def first(label: str, trees) -> None:
    m = load("campaign32", label, trees)
    pd = Pending(m)
    refused = sorted(d["t"] for d in m.dispatches if not d["launched"])
    short = sorted(t for t, tok, reqd in m.evict_host if tok < reqd)
    print(f"\ncampaign32 {label}: reads refused or cut for room {len(refused)} (first {refused[0]:.0f} s); evictions that "
          f"freed less than asked {len(short)} (first {short[0]:.0f} s)")
    # bursts: refusals separated by more than 120 s start a new burst
    bursts = []
    for t in refused:
        if not bursts or t - bursts[-1][-1] > 120.0:
            bursts.append([t])
        else:
            bursts[-1].append(t)
    print("   bursts of refusals (start, end, count): " + ", ".join(f"{b[0]:.0f}-{b[-1]:.0f} s ({len(b)})" for b in bursts[:8]))
    disp = sorted(m.dispatches, key=lambda d: d["t"])
    dt = [d["t"] for d in disp]
    parks = sorted(((d["park"]["t"], d["t"], d["pages"], d["rid"]) for d in m.dispatches if d["park"]), key=lambda x: x[0])
    eh = m.evict_host
    eht = [e[0] for e in eh]
    d2t = [x[0] for x in m.d2h]
    rd_iv = [(r["t0"], r["t1"], r["pages"]) for r in m.reads]
    usage = m.decode
    ut = [x[0] for x in usage]
    for b in bursts[:3]:
        t0 = b[0]
        print(f"   around the burst starting {t0:.0f} s (minute {t0 / 60:.1f}): t-t0 | pending write tokens, ops, oldest "
              f"age | sessions whose finished request has a pending write, their chains (bound on what writes hold) | "
              f"pinned, reads in flight | timed reads parked (their resident prefixes) | last 30 s: evicted from host, "
              f"new KV, read launches, short evictions | queue")
        for off in (-600, -300, -120, -60, -30, -10, 0, 10, 30, 60):
            t = t0 + off
            s = pd.summary(t)
            i = bisect.bisect_right(dt, t) - 1
            pinned = disp[i]["pinned_tokens"] if i >= 0 else 0
            inflight = sum(pg for a, e, pg in rd_iv if a <= t < e) * PAGE
            parked = [(pt, dtm, pg, rid) for pt, dtm, pg, rid in parks if pt <= t < dtm]
            anchors = 0
            for _pt, _d, _pg, rid in parked:
                got = m.arrival.get(rid)
                if got:
                    anchors += got[1]
            k0, k1 = bisect.bisect_left(eht, t - 30), bisect.bisect_right(eht, t)
            ev = sum(x[1] for x in eh[k0:k1])
            nshort = sum(1 for x in eh[k0:k1] if x[1] < x[2])
            j0, j1 = bisect.bisect_left(d2t, t - 30), bisect.bisect_right(d2t, t)
            newkv = sum(x[2] for x in m.d2h[j0:j1])
            launches = sum(d["pages"] * PAGE for d in disp[bisect.bisect_left(dt, t - 30):bisect.bisect_right(dt, t)] if d["launched"])
            u = bisect.bisect_right(ut, t) - 1
            queue = usage[u][4] if u >= 0 else -1
            print(f"   {off:5d} s | {s['pending'] / 1e3:5.0f}K {s['n']:4d} ops {s['oldest_age']:5.1f} s | {s['sessions']:3d} "
                  f"sessions {s['bound'] / 1e3:6.0f}K | pinned {pinned / 1e3:4.0f}K in flight {inflight / 1e3:4.0f}K | "
                  f"{len(parked):3d} parked ({anchors / 1e3:5.0f}K) | evicted {ev / 1e3:5.0f}K new {newkv / 1e3:4.0f}K "
                  f"reads {launches / 1e3:5.0f}K short {nshort:2d} | queue {queue}")
        s = pd.summary(t0)
        tot = max(s["pending"], 1)
        print(f"      pending writes at the burst's start by kind: " + ", ".join(
            f"{k} {v / 1e3:.0f}K ({v / tot:.0%})" for k, v in sorted(s["cls"].items()))
              + f"; not joined {s['unjoined'] / 1e3:.0f}K")


def main() -> None:
    trees = load_trees()
    mode, labels = sys.argv[1], sys.argv[2:]
    for label in labels:
        (calibrate if mode == "calibrate" else first)(label, trees)


if __name__ == "__main__":
    main()
