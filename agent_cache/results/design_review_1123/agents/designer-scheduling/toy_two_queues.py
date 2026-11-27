"""Stylized two-queue model (throwaway, CPU only): a GPU admission queue and one SSD read server.

Not a calibrated simulator of the engine. It checks one mechanism: when the GPU is saturated and admission passes
over a turn whose read has not landed (wait_complete), a late read reorders admissions and costs little, while early
landing costs host memory; when the GPU is not saturated, a late read is the turn's own stall.

Model
- N users, closed loop, all present from t = 0. A user sends a request, waits for it to complete, then waits a gap:
  92 % tool gaps (two-piece lognormal through p10/p50/p90 = 0.32/1.02/8.07 s), 8 % human gaps (2.8/64/539 s),
  capped at 1,200 s (FreeInference as replayed in campaign 34).
- A session's history L is fixed (lognormal, mean about 25K tokens, 4K-64K); new tokens and outputs per request are
  lognormal (p50 392 / p90 2,060 and p50 109 / p90 681).
- GPU: S slots (the KV pool binds at about 20 running). Admission scans the waiting queue in arrival order and passes
  over a request whose read is parked or in flight. An admitted request prefills (new + recomputed tokens) at
  0.03 s + 0.0414 ms/token on one prefill server; while a prefill runs, every running decode is paused. Decode takes
  24 ms per output token of decode time.
- Host memory: H tokens for idle, queued and pinned histories (the GPU mirror is outside H). A finished request
  inserts its history (most recently used). Eviction is LRU over idle histories; queued histories that were resident
  at arrival are locked (the engine ranks them last). A read takes L tokens of room at dispatch, evicting LRU idle
  histories; without room it is refused and the request recomputes its history at admission.
- One SSD read server: 0.05 s + L / 18,300 tokens/s of service (3.5 ms per 64-token page), plus 0.3 s after
  service before the pages are usable. Reads wait in an earliest-deadline queue; the server holds at most 2 s of work.
- Policies (post-arrival part; the pre-arrival part is added by 'pre_*'):
  nopin    read at arrival, earliest deadline, landed span not pinned (LRU, stamped at landing): HiCache-like
  pin      read at arrival, earliest deadline, landed span pinned until admission
  hold     as pin, but a parked read is held until (requests ahead + 0.5)/rate <= read estimate + 2 s
  holdr    as hold, counting only requests ahead that are not waiting on a read
  pre_exact  hold + a read started before arrival at (exact arrival + W now - read estimate - 1 s), pinned
  pre_p10    hold + the same timed on the gap entry's p10 (JIT p50 lands reads at the p10)
  Pins are capped at 0.3 H; over the cap a landed span is not pinned.
Usage: python3 toy_two_queues.py  (prints one table)
"""
from __future__ import annotations

import heapq
import math
import random
import statistics as st
import sys
from collections import deque

Z90 = 1.2815515655446004


def two_piece(rng, p10, p50, p90):
    z = rng.gauss(0.0, 1.0)
    sig = math.log(p50 / p10) / Z90 if z < 0 else math.log(p90 / p50) / Z90
    return p50 * math.exp(sig * z)


TOOL = (0.32, 1.02, 8.07)
HUMAN = (2.8, 64.0, 539.0)


from statistics import NormalDist
_N = NormalDist()


def support(entry, points=64, cap=1200.0):
    p10, p50, p90 = entry
    lo, hi = math.log(p50 / p10) / Z90, math.log(p90 / p50) / Z90
    out = []
    for k in range(points):
        z = _N.inv_cdf((k + 0.5) / points)
        out.append(min(cap, p50 * math.exp((lo if z < 0 else hi) * z)))
    return out


def ec_read_start(sup, idle, read_s, wait_s, rent):
    """next_turn_policy.read_start on remaining_gaps(sup, idle): seconds from now, or None (read at arrival)."""
    import bisect, itertools
    first = bisect.bisect_right(sup, idle)
    if len(sup) - first < 4:
        return None
    rem = [g - idle for g in sup[first:]]
    stall_at_arrival = max(0.0, read_s - wait_s)
    if stall_at_arrival == 0.0:
        return None
    pre = list(itertools.accumulate(rem, initial=0.0))
    total, n, lead = pre[-1], len(rem), read_s - wait_s
    best_cost, best = stall_at_arrival, None
    for start in sorted({0.0, *(max(0.0, g - lead) for g in rem)}):
        arrived = bisect.bisect_right(rem, start)
        late = bisect.bisect_right(rem, start + lead)
        stall = arrived * stall_at_arrival + ((late - arrived) * (start + lead) - (pre[late] - pre[arrived]))
        pinned = (total - pre[late]) - (n - late) * (start + lead)
        cost = (stall + rent * pinned) / n
        if cost < best_cost:
            best_cost, best = cost, start
    return best


SUPPORT = {TOOL: support(TOOL), HUMAN: support(HUMAN)}


class Sim:
    def __init__(self, *, n_users, host_tokens, policy, slots=21, seed=0, horizon=3600.0, warmup=600.0,
                 read_tok_s=18300.0):
        self.rng = random.Random(seed)
        self.N, self.H, self.policy = n_users, host_tokens, policy
        self.S, self.horizon, self.warmup = slots, horizon, warmup
        self.read_tok_s = read_tok_s
        self.t = 0.0
        self.ev = []
        self.seq = 0
        # sessions
        self.L = [min(64000, max(4000, int(self.rng.lognormvariate(math.log(21000), 0.6)))) for _ in range(n_users)]
        self.state = ["ssd"] * n_users          # 'host' or 'ssd' (not on the GPU)
        self.stamp = [0.0] * n_users            # LRU stamp of a host history
        self.locked = [False] * n_users         # queued with a resident history, or pinned landed read
        self.pinned = [False] * n_users
        self.host_used = 0
        self.pin_used = 0
        # requests
        self.waiting = []                       # list of request dicts, arrival order
        self.running = 0
        self.prefill_q = deque()
        self.prefill_busy_until = None
        self.prefill_cur = None
        self.decode_clock = 0.0
        self.decode_marks = []                  # heap of (mark, seq, req)
        self.last_adv = 0.0
        # storage
        self.parked = {}                        # user -> read dict
        self.inflight_users = set()             # users with a read on the server
        self.server_free_at = 0.0
        self.inflight_work = 0.0
        self.adm_times = deque()
        # pre-arrival plans: user -> dict(arrive_at, gap_p10, staged)
        self.plans = {}
        # metrics
        self.m = dict(ttft=[], wait=[], recompute=0, reqs=0, reads=0, refused=0, lost=0, pin_tok_s=0.0,
                      starve_s=0.0, read_stall=[], early_land=[])
        self._last_metric_t = 0.0

    # ------------------------------------------------------------------ events
    def push(self, t, kind, payload=None):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, payload))

    def gap(self):
        p = HUMAN if self.rng.random() < 0.08 else TOOL
        return min(1200.0, two_piece(self.rng, *p)), p

    def run(self):
        for u in range(self.N):
            self.push(self.rng.uniform(0, 30.0), "arrive", u)
        t_tick = 0.0
        while True:
            t_gpu = self.next_gpu_time()
            t_ev = self.ev[0][0] if self.ev else math.inf
            t_next = min(t_gpu, t_ev, t_tick)
            if t_next > self.horizon:
                break
            self.advance(t_next)
            if t_next == t_tick:
                self.tick()
                t_tick += 0.25
            elif t_next == t_gpu:
                self.gpu_event()
            else:
                _, _, kind, payload = heapq.heappop(self.ev)
                getattr(self, "on_" + kind)(payload)
            self.try_admit()
        return self.summary()

    # ------------------------------------------------------------------ clocks and metrics
    def advance(self, t):
        dt = t - self.last_adv
        if dt > 0:
            if self.prefill_cur is None:
                self.decode_clock += dt
            if self.t >= self.warmup:
                self.m["pin_tok_s"] += self.pin_used * dt
                # starvation: a free slot, a non-empty queue, and nothing ready
                if self.running < self.S and self.waiting and not any(self.ready(r) for r in self.waiting):
                    self.m["starve_s"] += dt
            self.last_adv = t
        self.t = t

    def next_gpu_time(self):
        if self.prefill_cur is not None:
            return self.prefill_busy_until
        if self.decode_marks:
            return self.t + max(0.0, self.decode_marks[0][0] - self.decode_clock)
        return math.inf

    def gpu_event(self):
        if self.prefill_cur is not None and self.t >= self.prefill_busy_until - 1e-12:
            r = self.prefill_cur
            self.prefill_cur = None
            if r["arrive"] >= self.warmup:
                self.m["ttft"].append(self.t - r["arrive"])
            self.seq += 1
            heapq.heappush(self.decode_marks, (self.decode_clock + r["out"] * 0.024, self.seq, r))
            self.start_prefill()
            return
        if self.decode_marks and self.decode_marks[0][0] <= self.decode_clock + 1e-12:
            _, _, r = heapq.heappop(self.decode_marks)
            self.finish(r)

    def start_prefill(self):
        if self.prefill_cur is None and self.prefill_q:
            r = self.prefill_q.popleft()
            toks = r["new"] + r["recompute"]
            self.prefill_cur = r
            self.prefill_busy_until = self.t + 0.03 + toks * 0.0414e-3

    # ------------------------------------------------------------------ host memory
    def evict_for(self, need):
        if self.H - self.host_used >= need:
            return True
        cands = sorted((self.stamp[u], u) for u in range(self.N)
                       if self.state[u] == "host" and not self.locked[u])
        for _, u in cands:
            self.state[u] = "ssd"
            self.host_used -= self.L[u]
            if self.H - self.host_used >= need:
                return True
        return False

    # ------------------------------------------------------------------ users and requests
    def on_arrive(self, u):
        r = dict(user=u, arrive=self.t, new=int(two_piece(self.rng, 120, 392, 2060)),
                 out=max(1, int(two_piece(self.rng, 30, 109, 681))), recompute=0, reading=False)
        self.waiting.append(r)
        plan = self.plans.pop(u, None)
        if self.state[u] == "host":
            self.locked[u] = True              # resident history of a queued turn: locked (ranked last)
            if self.policy == "nopin" and not self.pinned[u]:
                self.locked[u] = False
                self.stamp[u] = self.t
            return
        if u in self.parked:
            r["reading"] = True                # joins its staged read, now timed by its queue position
            self.parked[u]["req"] = r
            return
        if self.is_inflight(u):
            r["reading"] = True
            return
        self.request_read(r)

    def is_inflight(self, u):
        return u in self.inflight_users

    def request_read(self, r, staged=False, deadline=None):
        u = r["user"] if r is not None else None
        if r is not None:
            r["reading"] = True
        rd = dict(user=u, req=r, parked_at=self.t, staged=staged, deadline=deadline)
        self.parked[u] = rd
        self.dispatch()

    def finish(self, r):
        u = r["user"]
        self.running -= 1
        if r["arrive"] >= self.warmup:
            self.m["reqs"] += 1
            self.m["recompute"] += r["recompute"]
        # history back to host (write-through to host), most recently used
        if self.evict_for(self.L[u]):
            self.state[u] = "host"
            self.host_used += self.L[u]
            self.stamp[u] = self.t
        else:
            self.state[u] = "ssd"
        g, entry = self.gap()
        self.push(self.t + g, "arrive", u)
        if self.policy.startswith("pre_") or self.policy.startswith("uni_") or self.policy == "ec_dist":
            self.plans[u] = dict(arrive_at=self.t + g, end=self.t, entry=entry, staged=False)

    def ready(self, r):
        return not r["reading"]

    def rate(self):
        while self.adm_times and self.adm_times[0] < self.t - 60.0:
            self.adm_times.popleft()
        n = len(self.adm_times)
        return n / 60.0 if n >= 5 else None

    def try_admit(self):
        while self.running < self.S and self.waiting:
            pick = None
            for i, r in enumerate(self.waiting):
                if self.ready(r):
                    pick = i
                    break
            if pick is None:
                return
            r = self.waiting.pop(pick)
            u = r["user"]
            if self.state[u] != "host":
                r["recompute"] = self.L[u]    # refused, or landed and lost before admission
                if self.pinned[u]:
                    self.unpin(u)
            else:
                # history goes to the GPU; its host copy leaves the host-only accounting (mirror)
                self.state[u] = "ssd_mirror"
                self.host_used -= self.L[u]
                if self.pinned[u]:
                    self.unpin(u)
                self.locked[u] = False
            if self.state[u] == "ssd_mirror":
                self.state[u] = "gpu"
            self.running += 1
            self.adm_times.append(self.t)
            if r["arrive"] >= self.warmup:
                self.m["wait"].append(self.t - r["arrive"])
            self.prefill_q.append(r)
            self.start_prefill()

    def unpin(self, u):
        if self.pinned[u]:
            self.pinned[u] = False
            self.pin_used -= self.L[u]

    # ------------------------------------------------------------------ storage
    def deadline_of(self, rd):
        if rd["req"] is None:                  # staged before arrival
            return rd["deadline"]
        pos = self.position(rd["req"])
        if pos is None or pos <= 1:
            return -math.inf
        rate = self.rate()
        return -math.inf if rate is None else self.t + (pos + 0.5) / rate

    def position(self, r):
        for i, x in enumerate(self.waiting):
            if x is r:
                return i
        return None

    def read_est(self, L, ahead_work):
        return 0.05 + L / self.read_tok_s + 0.3 + ahead_work + 0.9   # the engine's lags as seeds, roughly

    def not_before(self, rd, ahead_work):
        if self.policy in ("nopin", "pin") or rd["req"] is None:
            return -math.inf
        r = rd["req"]
        pos = self.position(r)
        if pos is None or pos <= 1:
            return -math.inf
        rate = self.rate()
        if rate is None:
            return -math.inf
        if self.policy == "holdr" or self.policy.endswith("_r"):
            pos = sum(1 for x in self.waiting[:pos] if self.ready(x))
        if self.policy.startswith("pull") or self.policy.startswith("uni_"):
            # pull: start once at most K ready requests are ahead of it; pullr and uni_* size K to cover
            # the admissions during the read: K = ceil(rate x read estimate)
            if self.policy == "pullr":
                k = math.ceil(rate * self.read_est(self.L[rd["user"]], ahead_work))
            elif self.policy.startswith("uni_"):
                k = math.ceil(rate * self.read_est(self.L[rd["user"]], 0.0))
            else:
                k = int(self.policy[4:])
            ready_ahead = sum(1 for x in self.waiting[:pos] if self.ready(x))
            return -math.inf if ready_ahead <= k else math.inf
        wait = (pos + 0.5) / rate
        margin = {"late3": -3.0, "late10": -10.0}.get(self.policy, 2.0)
        return self.t + wait - self.read_est(self.L[rd["user"]], ahead_work) - margin

    def dispatch(self):
        if not self.parked:
            return
        order = sorted(self.parked.values(), key=lambda rd: self.deadline_of(rd))
        ahead = max(0.0, self.server_free_at - self.t)
        for rd in order:
            svc = 0.05 + self.L[rd["user"]] / self.read_tok_s
            nb = self.not_before(rd, ahead)
            ahead += svc
            if self.t < nb:
                continue
            if self.deadline_of(rd) != -math.inf and max(0.0, self.server_free_at - self.t) >= 2.0:
                break                           # the server already holds 2 s of work (the window); 'now' reads skip it
            u = rd["user"]
            del self.parked[u]
            if not self.evict_for(self.L[u]):
                self.m["refused"] += 1
                if rd["req"] is not None:
                    rd["req"]["reading"] = False   # admitted later with its history recomputed
                continue
            self.host_used += self.L[u]        # allocated at dispatch
            self.locked[u] = True
            self.inflight_users.add(u)
            start = max(self.t, self.server_free_at)
            self.server_free_at = start + svc
            self.m["reads"] += 1
            self.push(self.server_free_at + 0.3, "landed", rd)

    def on_landed(self, rd):
        u = rd["user"]
        self.inflight_users.discard(u)
        r = rd["req"]
        if r is None:                          # a staged read: did its turn arrive meanwhile?
            r = next((x for x in self.waiting if x["user"] == u), None)
        self.state[u] = "host"
        self.stamp[u] = self.t
        if self.policy == "nopin":
            self.locked[u] = False
        elif self.pin_used + self.L[u] <= 0.3 * self.H:
            self.pinned[u] = True
            self.pin_used += self.L[u]
            self.locked[u] = True
        else:
            self.locked[u] = False
        if r is not None:
            r["reading"] = False
            if r["arrive"] >= self.warmup:
                self.m["read_stall"].append(self.t - r["arrive"])
        elif rd["staged"] and self.t >= self.warmup:
            self.m["early_land"].append(self.t)
        self.dispatch()

    # ------------------------------------------------------------------ periodic: holds and pre-arrival plans
    def tick(self):
        self.dispatch()
        if not (self.policy.startswith("pre_") or self.policy.startswith("uni_") or self.policy == "ec_dist"):
            return
        rate = self.rate()
        qlen = len(self.waiting)
        w_now = 0.0 if qlen == 0 else ((qlen + 0.5) / rate if rate else 0.0)
        for u, p in list(self.plans.items()):
            if p["staged"] or self.state[u] == "host" or u in self.parked:
                continue
            if self.policy.endswith("_ec") or self.policy == "ec_dist":
                r_est = self.read_est(self.L[u], max(0.0, self.server_free_at - self.t))
                if self.policy.startswith("uni_") and w_now >= r_est:
                    continue
                st_in = ec_read_start(SUPPORT[p["entry"]], self.t - p["end"], r_est, w_now, 5e-5 * self.L[u])
                if st_in is not None and st_in <= 0.0 and self.t < p["arrive_at"]:
                    p["staged"] = True
                    self.parked[u] = dict(user=u, req=None, parked_at=self.t, staged=True,
                                          deadline=self.t + r_est)
                continue
            if self.policy.endswith("exact"):
                at = p["arrive_at"]
            else:                               # JIT p50 lands at the entry's p10
                at = p["end"] + p["entry"][0]
            r_est = self.read_est(self.L[u], max(0.0, self.server_free_at - self.t))
            if self.policy.startswith("uni_") and w_now >= r_est:
                continue                        # the queue will hide the read: read after arrival, pulled
            start = at + w_now - r_est - 1.0
            if self.t >= start and self.t < p["arrive_at"]:
                p["staged"] = True
                rd = dict(user=u, req=None, parked_at=self.t, staged=True, deadline=at + w_now)
                self.parked[u] = rd
        self.dispatch()

    # ------------------------------------------------------------------ results
    def summary(self):
        m = self.m
        span = self.horizon - self.warmup
        return dict(ttft=st.mean(m["ttft"]), wait=st.mean(m["wait"]), x=m["reqs"] / span,
                    recompute=m["recompute"] / max(1, m["reqs"]), reads=m["reads"], refused=m["refused"],
                    pin_k=m["pin_tok_s"] / span / 1000.0, starve=m["starve_s"] / span,
                    stall=(st.mean(m["read_stall"]) if m["read_stall"] else 0.0), reqs=m["reqs"])


def main():
    import os
    if os.environ.get("TOY_SET") == "2":
        configs = [("light, memory-bound", 40, 300_000, 18300.0), ("saturated, c34-like", 128, 1_500_000, 18300.0),
                   ("saturated, memory-tight", 128, 700_000, 18300.0),
                   ("saturated, storage-bound (read 7K tok/s)", 128, 1_500_000, 7000.0)]
        policies = ["hold", "pull2", "pullr", "pre_exact", "pre_p10", "uni_exact", "uni_p10"]
    if os.environ.get("TOY_SET") == "3":
        configs = [("light, memory-bound", 40, 300_000, 18300.0), ("saturated, c34-like", 128, 1_500_000, 18300.0),
                   ("saturated, memory-tight", 128, 700_000, 18300.0),
                   ("saturated, storage-bound (read 7K tok/s)", 128, 1_500_000, 7000.0)]
        policies = ["hold", "pull2", "ec_dist", "uni_ec", "uni_exact", "uni_p10"]
    else:
        configs = [("light, memory-bound", 40, 300_000, 18300.0), ("saturated, c34-like", 128, 1_500_000, 18300.0),
                   ("saturated, memory-tight", 128, 700_000, 18300.0)]
        policies = ["nopin", "pin", "hold", "holdr", "late3", "late10", "pull2", "pull8", "pre_exact", "pre_p10"]
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2]
    for label, n, h, rtok in configs:
        print(f"== {label}: {n} users, host {h/1e3:.0f}K tokens, seeds {seeds}")
        print(f"  {'policy':10s} {'TTFT s':>7s} {'wait s':>7s} {'req/s':>6s} {'recomp tok/req':>14s} "
              f"{'reads':>6s} {'refused':>7s} {'pins K':>7s} {'starved':>7s} {'arrival->land s':>15s}")
        for pol in policies:
            rs = [Sim(n_users=n, host_tokens=h, policy=pol, seed=s, read_tok_s=rtok).run() for s in seeds]
            avg = {k: st.mean(r[k] for r in rs) for k in rs[0]}
            spread = " ".join(f"{r['ttft']:.2f}" for r in rs)
            print(f"  {pol:10s} {avg['ttft']:7.2f} {avg['wait']:7.2f} {avg['x']:6.3f} {avg['recompute']:14.0f} "
                  f"{avg['reads']:6.0f} {avg['refused']:7.0f} {avg['pin_k']:7.1f} {avg['starve']*100:6.2f}% "
                  f"{avg['stall']:15.2f}   seeds: {spread}")


if __name__ == "__main__":
    main()
