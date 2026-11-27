"""The waiting queue rebuilt per request: who waits, and who of them is ready (no lookup or read of its
own under way, so admission would take it).

build(camp, arm) -> dict:
  rows       joined request records (lib.requests) with extra fields:
               ready_spans  [(s, e)] inside [arr, adm) where the request was ready
               busy_q       seconds inside [arr, adm) with a lookup/read under way
  tot, rdy   step functions (Step) of the queue length and of the ready queue length
  usage      Step of the GPU pool use from the batch lines (value of the last line at or before t)
  running    Step of #running-req from the batch lines
  n_live     Step of users in session (client play records)
"""
import bisect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


class Step:
    """Right-continuous step function from (time, value) pairs sorted by time."""

    def __init__(self, ts, vs):
        self.ts, self.vs = ts, vs

    @classmethod
    def from_deltas(cls, deltas):
        deltas.sort()
        ts, vs, v = [], [], 0
        for t, d in deltas:
            v += d
            if ts and ts[-1] == t:
                vs[-1] = v
            else:
                ts.append(t)
                vs.append(v)
        return cls(ts, vs)

    def at(self, t):
        i = bisect.bisect_right(self.ts, t) - 1
        return self.vs[i] if i >= 0 else 0

    def before(self, t):
        """Value just before t."""
        i = bisect.bisect_left(self.ts, t) - 1
        return self.vs[i] if i >= 0 else 0

    def integral(self, a, b):
        """Integral of the function over [a, b)."""
        if b <= a:
            return 0.0
        i = bisect.bisect_right(self.ts, a) - 1
        t, v = a, (self.vs[i] if i >= 0 else 0)
        total = 0.0
        i += 1
        while i < len(self.ts) and self.ts[i] < b:
            total += v * (self.ts[i] - t)
            t, v = self.ts[i], self.vs[i]
            i += 1
        return total + v * (b - t)

    def mean(self, a, b):
        return self.integral(a, b) / (b - a)

    def time_at(self, a, b, pred):
        """Seconds in [a, b) during which pred(value) holds."""
        if b <= a:
            return 0.0
        i = bisect.bisect_right(self.ts, a) - 1
        t, v = a, (self.vs[i] if i >= 0 else 0)
        total = 0.0
        i += 1
        while i < len(self.ts) and self.ts[i] < b:
            if pred(v):
                total += self.ts[i] - t
            t, v = self.ts[i], self.vs[i]
            i += 1
        if pred(v):
            total += b - t
        return total


def ready_spans(r):
    a, d = r["arr"], r["adm"]
    spans, t = [], a
    busy = 0.0
    for s, e in r["busy"]:
        e = d if e is None else e
        s, e = max(s, a), min(e, d)
        if e <= s:
            continue
        if s > t:
            spans.append((t, s))
        busy += e - max(s, t) if e > t else 0.0
        t = max(t, e)
    if d > t:
        spans.append((t, d))
    return spans, busy


def build(camp, arm):
    rows = lib.requests(camp, arm)
    srv = lib.load_server(camp, arm)
    d_tot, d_rdy = [], []
    for r in rows:
        if r["arr"] is None:
            r["ready_spans"], r["busy_q"] = [], 0.0
            continue
        d_tot.append((r["arr"], 1))
        d_tot.append((r["adm"], -1))
        spans, busy = ready_spans(r)
        r["ready_spans"], r["busy_q"] = spans, busy
        for s, e in spans:
            d_rdy.append((s, 1))
            d_rdy.append((e, -1))
    batch = srv["batch"]
    usage = Step([b[0] for b in batch], [b[4] for b in batch])
    running = Step([b[0] for b in batch], [b[2] for b in batch])
    qlog = Step([b[0] for b in batch], [b[3] for b in batch])
    cl = lib.load_client(camp, arm)
    first = {}
    for r in cl["req"]:
        first[r["play"]] = min(first.get(r["play"], 1e18), r["t_send"])
    ends = {p["play"]: p["end"] for p in cl["play"]}
    starts = {p["play"]: p.get("start", first.get(p["play"])) for p in cl["play"]}
    d_live = []
    for p, t in first.items():
        d_live.append((starts.get(p, t), 1))
        if p in ends:
            d_live.append((ends[p], -1))
    return {
        "rows": rows, "tot": Step.from_deltas(d_tot), "rdy": Step.from_deltas(d_rdy), "usage": usage,
        "running": running, "qlog": qlog, "n_live": Step.from_deltas(d_live), "srv": srv, "client": cl,
    }


if __name__ == "__main__":
    import sys
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        st = build(camp, arm)
        srv = st["srv"]
        # check: rebuilt queue length against the logged #queue-req on the batch lines
        diffs = []
        for b in srv["batch"]:
            if 60 <= b[0] < 7000:
                diffs.append(st["tot"].at(b[0]) - b[3])
        diffs.sort()
        n = len(diffs)
        print(f"{spec}: batch lines {n}; rebuilt queue - logged #queue-req: mean {sum(diffs)/n:+.2f}, "
              f"p05 {diffs[int(.05*n)]:+d}, p50 {diffs[n//2]:+d}, p95 {diffs[int(.95*n)]:+d}; "
              f"exact {sum(1 for x in diffs if x == 0)/n:.0%}, within 1 {sum(1 for x in diffs if abs(x) <= 1)/n:.0%}")
        rows = [r for r in st["rows"] if r["arr"] is not None]
        nb = sum(1 for r in rows if r["busy_q"] > 0.05)
        print(f"   requests {len(rows)}, with a lookup/read under way in the queue > 50 ms: {nb}; "
              f"mean queue {st['tot'].mean(60, 6600):.1f}, mean ready queue {st['rdy'].mean(60, 6600):.1f}")
        for lo, hi in ((0, 1800), (1800, 3600), (3600, 5400), (5400, 6600)):
            print(f"   {lo//60:3d}-{hi//60:3d} min: queue {st['tot'].mean(lo, hi):6.2f} (logged lines mean "
                  f"{lib.mean([b[3] for b in srv['batch'] if lo <= b[0] < hi]):6.2f}), ready {st['rdy'].mean(lo, hi):6.2f}, "
                  f"users in session {st['n_live'].mean(lo, hi):6.1f}")
