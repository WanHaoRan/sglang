"""Per-run tables rebuilt from an extract_run.py pickle: arrivals, admissions, reads, queue positions over time."""
import bisect
import collections
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402


def load(label):
    with open(os.path.join(C.EXTRACTS, f"{label}.pkl"), "rb") as f:
        return pickle.load(f)


def first_by_rid(evts):
    out = {}
    for t, rid, fields in evts:
        if rid is not None and rid not in out:
            out[rid] = (t, fields)
    return out


def all_by_rid(evts):
    out = collections.defaultdict(list)
    for t, rid, fields in evts:
        if rid is not None:
            out[rid].append((t, fields))
    return out


RATE_WINDOW_S = 60.0           # next_turn_estimators.RATE_WINDOW_S
RATE_FALLBACK_WINDOW_S = 300.0
MIN_ADMISSIONS = 5


class Queue:
    """The waiting queue rebuilt from the ReqTimeStats lines (every request: entry time and queue duration).
    Order is arrival order (schedule_policy fcfs, no priorities); admission passes over turns whose read is under way."""

    def __init__(self, d):
        rts = d["rts"]
        self.arr = {rid: v[6] for rid, v in rts.items()}
        self.adm = {rid: v[6] + v[4] for rid, v in rts.items()}
        self.done = {rid: v[0] for rid, v in rts.items()}
        self.fwd = {rid: v[5] for rid, v in rts.items()}
        order = sorted(self.arr, key=lambda r: self.arr[r])
        self.adm_sorted = sorted(self.adm.values())
        # for each rid: admission times of the requests ahead of it when it arrived, ascending
        import heapq
        heap = []
        self.ahead = {}
        for rid in order:
            a = self.arr[rid]
            while heap and heap[0] <= a:
                heapq.heappop(heap)
            self.ahead[rid] = sorted(heap)
            heapq.heappush(heap, self.adm[rid])
        self.arr_sorted = [self.arr[r] for r in order]

    def position(self, rid, t):
        """Requests ahead of rid in the queue at time t (0 = head); None if rid is not queued at t."""
        if rid not in self.arr or t < self.arr[rid] or t >= self.adm[rid]:
            return None
        ahead = self.ahead[rid]
        return len(ahead) - bisect.bisect_right(ahead, t)

    def qlen(self, t):
        """Requests in the queue at t."""
        return bisect.bisect_right(self.arr_sorted, t) - bisect.bisect_right(self.adm_sorted, t)

    def rate(self, t):
        """(admissions per s, window) as AdmissionRate.rate at time t; None without 5 admissions in 300 s."""
        hi = bisect.bisect_right(self.adm_sorted, t)
        recent = hi - bisect.bisect_right(self.adm_sorted, t - RATE_WINDOW_S)
        if recent >= MIN_ADMISSIONS:
            return recent / RATE_WINDOW_S, RATE_WINDOW_S
        total = hi - bisect.bisect_right(self.adm_sorted, t - RATE_FALLBACK_WINDOW_S)
        if total >= MIN_ADMISSIONS:
            return total / RATE_FALLBACK_WINDOW_S, RATE_FALLBACK_WINDOW_S
        return None

    def residual_wait(self, rid, t):
        """(position + 0.5) / rate at t, as next_turn_estimators.residual_wait; None when not queued or no rate."""
        pos = self.position(rid, t)
        r = self.rate(t)
        if pos is None or r is None:
            return None
        return (pos + 0.5) / r[0]


# next_turn_estimators.L3LoadModel seeds and next_turn_prefetch margins
ALLOC_LAG_S, READ_FIXED_S, DRAIN_LAG_S = 0.5, 0.044, 0.36
LOAD_FIXED_S = ALLOC_LAG_S + READ_FIXED_S + DRAIN_LAG_S
SEED_S_PER_PAGE = 0.0025
EWMA_ALPHA = 1 / 8
EWMA_MIN_SAMPLES = 8
HOLD_MARGIN_S = 2.0
PAGE = 64


class Reads:
    """Every storage read of a run (srs_park -> srs_dispatch -> s2h_io -> srs_pin), paired per rid in time order."""

    def __init__(self, d, Q):
        ev = d["evts"]
        self.Q = Q
        parks = all_by_rid(ev.get("srs_park", []))
        disps = all_by_rid(ev.get("srs_dispatch", []))
        ios = all_by_rid(ev.get("s2h_io", []))
        pins = all_by_rid(ev.get("srs_pin", []))
        unpins = all_by_rid(ev.get("srs_unpin", []))
        holds = first_by_rid(ev.get("ntp_hold", []))
        stages = first_by_rid(ev.get("ntp_stage", []))
        plans = first_by_rid(ev.get("ntp_plan", []))
        self.reads = []
        for rid, ps in parks.items():
            for i, (tp, fp) in enumerate(ps):
                t_next = ps[i + 1][0] if i + 1 < len(ps) else math.inf
                r = {"rid": rid, "i": i, "park": tp, "pages": fp["pages"], "deadline_at_park": fp.get("deadline_ms"),
                     "arr": Q.arr.get(rid), "adm": Q.adm.get(rid), "disp": None, "io_end": None, "pin": None}
                dd = [(t, f) for t, f in disps.get(rid, []) if tp <= t < t_next]
                if dd:
                    t, f = dd[0]
                    # pending_write_tokens came with the campaign 34 engine
                    r.update(disp=t, waited=f["waited_ms"] / 1e3, inflight_after=f["inflight_pages"],
                             launched=f["launched"], pend_write=f.get("pending_write_tokens"),
                             pinned_at_disp=f["pinned_tokens"])
                    io = [(t2, f2) for t2, f2 in ios.get(rid, []) if t <= t2 < t_next]
                    if io:
                        r.update(io_end=io[0][0], io_ms=io[0][1]["ms"], io_pages=io[0][1]["pages"])
                    pn = [(t2, f2) for t2, f2 in pins.get(rid, []) if t <= t2 < t_next]
                    if pn:
                        r.update(pin=pn[0][0], pin_tokens=pn[0][1]["tokens"])
                        up = [(t2, f2) for t2, f2 in unpins.get(rid, []) if pn[0][0] <= t2]
                        if up:
                            r.update(unpin=up[0][0], unpin_reason=up[0][1]["reason"])
                h = holds.get(rid)
                if h is not None and tp - 0.05 <= h[0] < (r["disp"] if r["disp"] is not None else t_next) + 0.05:
                    r["hold"] = dict(h[1], t=h[0])
                r["staged"] = r["arr"] is None or tp < r["arr"] - 0.02
                if rid in plans:
                    r["pred_arr"] = plans[rid][0] + plans[rid][1]["pred_arr_ms"] / 1e3
                # landing: the pin (pages usable), else the I/O end
                r["land"] = r["pin"] if r["pin"] is not None else r["io_end"]
                self.reads.append(r)
        self.reads.sort(key=lambda r: r["park"])
        for n, r in enumerate(self.reads):
            r["n"] = n
        # EWMA of read ms per page as the engine publishes it: (time, s per page) after each completed read
        samples = sorted((t, f["ms"], f["pages"]) for t, _, f in ev.get("s2h_io", []) if f["pages"] > 0 and f["ms"] > 0)
        self.ewma_t, self.ewma_v = [], []
        ew, n = None, 0
        for t, ms, pages in samples:
            x = ms / pages
            ew = x if ew is None else ew + EWMA_ALPHA * (x - ew)
            n += 1
            if n >= EWMA_MIN_SAMPLES:
                self.ewma_t.append(t)
                self.ewma_v.append(ew / 1e3)
        # write hold intervals
        self.wh = []
        on = None
        for t, _, f in ev.get("srs_write_hold", []):
            if f["state"] == "on":
                on = t
            elif on is not None:
                self.wh.append((on, t))
                on = None
        if on is not None:
            self.wh.append((on, math.inf))
        self.wh_starts = [a for a, _ in self.wh]

    def s_per_page(self, t):
        i = bisect.bisect_right(self.ewma_t, t)
        return SEED_S_PER_PAGE if i == 0 else self.ewma_v[i - 1]

    def write_held(self, t):
        i = bisect.bisect_right(self.wh_starts, t) - 1
        return i >= 0 and self.wh[i][0] <= t < self.wh[i][1]

    def write_hold_overlap(self, a, b):
        """Seconds of [a, b) during which the write hold was on."""
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in self.wh if s < b and e > a)


def usage_series(d):
    """(times, token usage) from the Prefill and Decode batch lines, ascending."""
    pts = sorted([(p[0], p[4]) for p in d["prefill"]] + [(x[0], x[3]) for x in d["decode"]])
    return [p[0] for p in pts], [p[1] for p in pts]


def usage_at(series, t, window=2.0):
    """Mean logged KV pool usage in [t - window, t + window]; falls back to the nearest line."""
    ts, us = series
    lo, hi = bisect.bisect_left(ts, t - window), bisect.bisect_right(ts, t + window)
    if hi > lo:
        return sum(us[lo:hi]) / (hi - lo)
    i = min(len(ts) - 1, max(0, bisect.bisect_left(ts, t)))
    return us[i]


def hold_replay(d, Q, RD, *, dt=0.1, expected_cost=False):
    """Replay the hold's inputs on a time grid, with the line as it actually was (parked and in-flight reads from the
    log). For every read of a queued turn, from its park to its turn's admission: per step (t, position, wait =
    (position + 0.5) / rate or None at positions 0-1, pages ahead = in flight + parked reads ordered before it,
    read s per page, write hold on). The read's own pages are left out of what is ahead of it.
    Returns {read index n: list of steps}."""
    reads = RD.reads
    # in-flight page changes
    flight = []
    for r in reads:
        if r["disp"] is not None and r.get("launched"):
            ack = r["land"] if r["land"] is not None else r["disp"]
            flight.append((r["disp"], r["pages"]))
            flight.append((ack, -r["pages"]))
    flight.sort()
    fl_t = [e[0] for e in flight]
    fl_cum, s = [], 0
    for e in flight:
        s += e[1]
        fl_cum.append(s)

    def inflight(t):
        i = bisect.bisect_right(fl_t, t) - 1
        return fl_cum[i] if i >= 0 else 0

    t_lo = min(r["park"] for r in reads)
    t_hi = max((r["adm"] or r["park"]) for r in reads) + 1.0
    by_park = sorted(reads, key=lambda r: r["park"])
    nxt = 0
    live = []          # reads tracked: parked (really or virtually) until their turn is admitted
    out = {r["n"]: [] for r in reads}
    steps = int((t_hi - t_lo) / dt) + 1
    for k in range(steps):
        t = t_lo + k * dt
        while nxt < len(by_park) and by_park[nxt]["park"] <= t:
            live.append(by_park[nxt])
            nxt += 1
        if not live:
            continue
        rate = Q.rate(t)
        keep = []
        cur = []
        for r in live:
            arrived = r["arr"] is not None and t >= r["arr"]
            end = r["adm"] if r["adm"] is not None else (r["disp"] if r["disp"] is not None else r["park"])
            if arrived and t >= end:
                continue                      # the turn is admitted: nothing left to time
            if not arrived:
                # a staged read: tracked only while really parked, to order the line
                if r["disp"] is not None and t >= r["disp"]:
                    if r["arr"] is None:
                        continue
                    keep.append(r)
                    continue
                keep.append(r)
                if "pred_arr" in r and rate is not None:
                    qn = Q.qlen(t)
                    w = 0.0 if qn == 0 else (qn + 0.5) / rate[0]
                    deadline = r["pred_arr"] + w
                    if expected_cost:
                        deadline = max(deadline, t + 1.0)
                    key = (1, deadline - t) if deadline > t else (0, r["park"])
                else:
                    key = (0, r["park"])
                cur.append((key, r, None, None, True))
                continue
            keep.append(r)
            pos = Q.position(r["rid"], t)
            if pos is None:
                continue
            really_parked = r["disp"] is None or t < r["disp"]
            if pos <= 1 or rate is None:
                cur.append(((0, r["park"]), r, pos, None, really_parked))
            else:
                w = (pos + 0.5) / rate[0]
                cur.append(((1, w), r, pos, w, really_parked))
        live = keep
        if not cur:
            continue
        cur.sort(key=lambda x: x[0])
        fl = inflight(t)
        p = RD.s_per_page(t)
        wh = RD.write_held(t)
        ahead_real = 0
        for key, r, pos, w, really_parked in cur:
            if pos is not None:
                own = r["pages"] if (r["disp"] is not None and r.get("launched") and r["disp"] <= t
                                     and (r["land"] is None or t < r["land"])) else 0
                out[r["n"]].append((t, pos, w, fl - own + ahead_real, p, wh))
            if really_parked:
                ahead_real += r["pages"]
    return out


def release_time(steps, pages, margin):
    """First grid time the hold lets the read go under this margin: positions 0-1, no rate, or
    wait - load <= margin. None when the steps are empty."""
    for t, pos, w, ahead, p, wh in steps:
        if w is None:
            return t
        if w - (LOAD_FIXED_S + p * (pages + ahead)) <= margin:
            return t
    return None


def gpu_turn(Q):
    """D per request: the first time admission took a later arrival while this one still waited (it was passed
    over, so its GPU turn had come), else its own admission. {rid: D}."""
    order = sorted(Q.arr, key=lambda r: Q.arr[r])
    n = len(order)
    # suffix minimum of admission times over later arrivals
    D = {}
    suf = math.inf
    for i in range(n - 1, -1, -1):
        rid = order[i]
        D[rid] = min(Q.adm[rid], suf)
        suf = min(suf, Q.adm[rid])
    return D


def gpu_turn_refined(Q, RD, usage, *, threshold=0.90):
    """D with the light-load case put right. The pass-over time is exact when the pool is full (the slot a later
    arrival took was this turn's), but with room in the pool a turn that waits on its read is often never passed
    over, or late: its turn came as soon as no ready turn was ahead of it. So for a turn with a read of its own:
    t_free = the first time from its arrival at which every turn still ahead of it was itself waiting on a read;
    D = t_free when the pool then had room (logged usage under `threshold`), else the pass-over time. {rid: D}."""
    D0 = gpu_turn(Q)
    busy = {}
    for r in RD.reads:
        if r["arr"] is None or r["adm"] is None:
            continue
        a = max(r["park"], r["arr"])
        b = r["land"] if r["land"] is not None else (r["disp"] if r["disp"] is not None else r["park"])
        b = min(b, r["adm"])
        if b > a:
            busy.setdefault(r["rid"], []).append((a, b))
    order = sorted(Q.arr, key=lambda rid: Q.arr[rid])
    import heapq
    waiting = []
    D = dict(D0)
    for rid in order:
        a = Q.arr[rid]
        while waiting and waiting[0][0] <= a:
            heapq.heappop(waiting)
        ahead = [other for _, other in waiting]
        heapq.heappush(waiting, (Q.adm[rid], rid))
        if rid not in busy:
            continue
        # ready intervals of the turns ahead, from this arrival on: [a, adm) minus their reading intervals
        ev = []
        for other in ahead:
            lo, hi = a, Q.adm[other]
            cuts = sorted((max(x, lo), min(y, hi)) for x, y in busy.get(other, ()) if y > lo and x < hi)
            t = lo
            for x, y in cuts:
                if x > t:
                    ev.append((t, 1)); ev.append((x, -1))
                t = max(t, y)
            if hi > t:
                ev.append((t, 1)); ev.append((hi, -1))
        if not ev:
            t_free = a
        else:
            ev.sort(key=lambda e: (e[0], e[1]))
            n, t_free = 0, None
            if ev[0][0] > a:
                t_free = a
            else:
                for t, x in ev:
                    n += x
                    if n == 0:
                        t_free = t
                        break
        if t_free is None:
            continue
        t_free = min(max(t_free, a), Q.adm[rid])
        if t_free < D0[rid] and usage_at(usage, t_free) < threshold:
            D[rid] = t_free
    return D


def ready_series(Q, RD):
    """Step function of the number of queued turns with no read of their own under way: (times, counts).
    A read is under way from its park to its landing (or to its dispatch when it never launched)."""
    ev = []
    for rid, a in Q.arr.items():
        ev.append((a, 1))
        ev.append((Q.adm[rid], -1))
    for r in RD.reads:
        if r["arr"] is None or r["adm"] is None:
            continue
        a = max(r["park"], r["arr"])
        b = r["land"] if r["land"] is not None else (r["disp"] if r["disp"] is not None else r["park"])
        b = min(b, r["adm"])
        if b > a:
            ev.append((a, -1))
            ev.append((b, 1))
    ev.sort()
    ts, cs, c = [], [], 0
    for t, x in ev:
        c += x
        if ts and ts[-1] == t:
            cs[-1] = c
        else:
            ts.append(t)
            cs.append(c)
    return ts, cs


def idle_share(series, a, b):
    """Share of [a, b) during which at most `self` ready turns waited, i.e. no other ready turn: the caller passes
    an interval in which its own turn is not ready, so count == 0 means nobody else was."""
    ts, cs = series
    if b <= a:
        return float("nan")
    i = bisect.bisect_right(ts, a) - 1
    t, idle = a, 0.0
    while t < b:
        nxt = ts[i + 1] if i + 1 < len(ts) else math.inf
        end = min(b, nxt)
        if (cs[i] if i >= 0 else 0) <= 0:
            idle += end - t
        t = end
        i += 1
    return idle / (b - a)
