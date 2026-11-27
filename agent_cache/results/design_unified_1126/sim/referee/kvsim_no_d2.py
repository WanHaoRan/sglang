"""REFEREE ablation variant of kvsim.py (scratch; see make_patched.py).
kvsim: a discrete-event simulator of three-tier KV caching (GPU pool, host memory, one SSD) under the closed-loop
replay of agent sessions, calibrated on campaigns 34 and 32 (see CALIBRATION.md).

  kvsim.py --arm ec_full --cap 128 --host-gb 160 --duration 7200 --seed 0 --out <folder>

Standard library plus numpy. Parts:
  kvsim_trace.py   the workload: the real trace reduced to pages, the client's schedule, the hint table
  kvsim.py         this file: the client loop, the GPU, host memory, the SSD device, the cleaner, the outputs
  kvsim_policy.py  the policies: hicache, jit_p50, jit_exact, ec_full, ec_exact, unified (DESIGN.md section 9 switches)
  kvsim_compare.py arms x settings x seeds with at most 4 processes, and the paired comparison
  CALIBRATION.md   what was fitted to what, what the simulator reproduces and what it leaves out

Model in one paragraph. Users arrive at RATE per second and wait for one of CAP slots; each plays one recorded tree
(streams of requests with recorded waits). A request joins the server's queue. The GPU admits, in arrival order among
requests with no SSD read under way, the first request whose prompt (beyond what the GPU already holds) plus output
fits the KV pool, and stops at the first that does not fit. Admitted requests prefill in batches of at most 8,192
tokens (per-batch overhead plus a cost per token that grows with the position), during which decoding pauses; running
requests then decode one token per step, a step costing a constant plus terms in the number of running requests and
their context. Every new page is copied to host memory (write-through) and queued for the SSD; a page with a write
pending cannot leave host memory. Host memory is freed leaf first in the policy's order. SSD reads and writes share
one device. A cleaner deletes the oldest-written pages when the volume passes 90 %, down to 80 %.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np

# python -I leaves the script's folder off sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # referee copy: modules are one up
import kvsim_trace as kt  # noqa: E402

PAGE = 64
THRESH = 256                      # the engine's prefetch threshold, tokens
KV_BYTES = 98304                  # per token, Qwen3-30B-A3B bf16

# GPU cost sets (CALIBRATION.md section 3 has the anchors). Seconds: a + b x position per prefilled token, batch_s per
# prefill batch, h2d per token loaded from host memory, a decode step of step0 + step_n x min(running, step_n_cap) +
# step_k x tokens of running context. FA3 is fitted to campaigns 32 and 34; Triton is taken from the records
# (cpu_calib_c44/FINDINGS.md, campaign 28) and was checked on campaign 28 only.
BACKENDS = {
    "fa3": {"a": 1.45e-5, "b": 1.18e-9, "batch_s": 0.015, "h2d": 0.85e-6, "step0": 3.3e-3, "step_n": 0.30e-3,
            "step_n_cap": 8, "step_k": 0.0277e-6},
    "triton": {"a": 0.84e-6, "b": 6.42e-9, "batch_s": 0.062, "h2d": 0.85e-6, "step0": 11.2e-3, "step_n": 0.18e-3,
               "step_n_cap": 1 << 30, "step_k": 0.0357e-6},
}
CHUNK = 8192                      # chunked_prefill_size: prefill tokens per batch


class Config:
    """Settings of one run. Constants marked (fit) are listed in CALIBRATION.md with their anchors."""

    def __init__(self, **kw):
        self.arm = "ec_full"
        self.seed = 0
        self.cap = 128
        self.rate = 0.1
        self.duration = 7200.0
        self.grace = 30.0
        self.host_gb = 160.0
        self.gpu_tokens = 668096
        self.max_running = 64                 # max_running_requests (the KV pool binds first in every calibrated run)
        self.gpu_time_scale = 1.0             # multiplies every GPU time (a sensitivity knob; 1 in every calibrated run)
        # (fit) 2,496,449,740,800 B / 98,304 B = 25.40 M tokens, less 2.8 % (filesystem overhead): the first
        # cleaner pass of campaign 34's ec_full at 5,275 s
        self.ssd_tokens = 24_690_000
        self.cleaner_hi = 0.90
        self.cleaner_lo = 0.80
        self.cleaner_interval = 30.0
        self.max_context = 65536
        self.gap_scale = 1.0
        self.gap_cap = 1200.0
        self.trace = kt.DEFAULT_TRACE
        self.backend = "fa3"
        self.fixes = True                     # the four campaign 34 write-path fixes
        self.mirror = True                    # False: drop the host copy of a GPU-held page once its write is acked
        self.switches = ""                    # unified arm: comma-joined DESIGN.md section 9 switches (F1,...,PR1)
        self.hints = None                     # override the arm's hint mode: none | program | dist | exact
        self.tick = 0.25
        self.cut = None                       # summary cut (default duration - 600 s)
        self.out = None
        self.write_fail_p = 0.0               # chance that a write fails (R3); 0 in every calibrated run
        self.noise = 0.0                      # s: each request is sent up to this much later (uniform), to measure
        self.noise_seed = 0                   # the spread between runs with the same users (0 = none)
        # device (fit): tokens per second
        self.read_alone = 24000.0             # 2.67 ms per 64-token page
        self.write_alone = 18800.0            # 3.4 ms per page
        self.beta = 21400.0                   # 2.1 GB/s for reads and writes together
        self.phi = 0.73                       # read share of beta while both threads run
        self.lookup_lag = 0.02                # lookup issued -> hit known (measured, campaign 34)
        self.land_lag = 0.03                  # I/O end -> pages usable, plus the read's fixed 13 ms (measured)
        self.client_lag = 0.03                # reply complete -> next request reaches the queue, beyond the wait
        self.first_token_s = 0.03
        self.unbacked_room_tokens = 0         # host room no logged lock explains (campaign 32); 0 = not modelled
        self.d2h_s_per_token = 2.8e-6         # GPU -> host copy, inline at GPU eviction when the mirror is off
                                              # (measured: 2,304 tokens in 6.5 ms)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"Config has no field {k}")
            setattr(self, k, v)

    @property
    def host_tokens(self) -> int:
        return int(math.ceil(self.host_gb * 1e9 / KV_BYTES / PAGE)) * PAGE

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["host_tokens"] = self.host_tokens
        return d


class Read:
    """One SSD read (a lookup that hit) for the request `r` that will use it."""

    __slots__ = ("r", "nodes", "tokens", "pages", "anchor", "state", "parked_at", "staged", "deadline", "t_disp",
                 "t_land", "pinned", "seq", "io_start", "rid", "lost_from")

    def __init__(self, r, nodes, tokens, anchor, staged):
        self.r, self.nodes, self.tokens, self.anchor, self.staged = r, nodes, tokens, anchor, staged
        self.pages = tokens // PAGE
        self.state = "lookup"
        self.parked_at = self.t_disp = self.t_land = self.io_start = None
        self.deadline = None
        self.pinned = False
        self.lost_from = None         # [referee D5] index of the first node whose SSD copy was gone when the I/O began


class Sim:
    def __init__(self, cfg: Config, workload: kt.Workload | None = None):
        import kvsim_policy as kp
        self.cfg = cfg
        self.cost = dict(BACKENDS[cfg.backend])
        for key in ("a", "b", "batch_s", "h2d", "step0", "step_n", "step_k"):
            self.cost[key] *= cfg.gpu_time_scale
        self.pol = kp.make_policy(cfg)
        self.wl = workload or kt.Workload(trace=cfg.trace, max_context=cfg.max_context, seed=cfg.seed, rate=cfg.rate,
                                          duration=cfg.duration, gap_scale=cfg.gap_scale, gap_cap=cfg.gap_cap,
                                          hint_no_call=True)
        wl = self.wl
        self.hinter = kt.Hinter(self.pol.hint_mode, hint_no_call=True, deadline=cfg.duration)
        n = wl.n_nodes
        self.tok = [int(x) for x in wl.node_tokens]
        self.par = [int(x) for x in wl.node_parent]
        self.kids = [[] for _ in range(n)]
        for i in range(1, n):
            self.kids[self.par[i]].append(i)
        self.in_gpu = [False] * n
        self.in_host = [False] * n
        self.glock = [0] * n
        self.hlock = [0] * n
        self.nct = [0] * n            # children in the tree (GPU or host)
        self.ncg = [0] * n            # children on the GPU
        self.ssd = [False] * n
        self.ssd_t = [0.0] * n
        self.ssd_dead = [False] * n   # R1: the chain of an ended session, first out of the SSD
        self.dirty = [False] * n      # an SSD write is pending (host-locked)
        self.unwritten = [False] * n  # in memory with no SSD copy and no write pending (skipped, failed, deleted)
        self.stamp = [0.0] * n
        self.ver = [0] * n
        self.ever = [False] * n
        self.pin = [0] * n            # pin count by landed reads (also in hlock)
        self.mup = [False] * n        # inserted in one piece with its parent (one engine node): evicted together
        self.orphan = [False] * n     # host pages under a deleted node: unreachable, held until their lock goes
        self.orphan_tok = [0] * n     # [referee D2] tokens those copies hold (a node can be orphaned, inserted again and orphaned again)
        self.wpend = [0] * n          # [referee D2] SSD writes pending per node
        self.seen = [False] * n       # [referee D7] pages of requests whose reply is complete (the client's infinite cache)
        self.in_gpu[0] = self.in_host[0] = True
        self.glock[0] = self.hlock[0] = 1 << 30
        self.C_G, self.C_H, self.C_S = cfg.gpu_tokens, cfg.host_tokens - cfg.unbacked_room_tokens, cfg.ssd_tokens
        self.gpu_used = 0
        self.gpu_unlocked = 0         # tokens of GPU nodes no running request holds
        self.host_used = 0
        self.host_mirror = 0
        self.pending_write = 0
        self.pinned_tokens = 0
        self.inflight_tokens = 0      # dispatched reads not landed (host pages allocated)
        self.inflight_pages = 0
        self.ssd_used = 0
        self.hcand: set = set()
        self.gcand: set = set()
        self.hheap: list = []
        self.gheap: list = []
        self.heap_built = -1e9
        # events
        self.now = 0.0
        self.ev: list = []
        self.seq = 0
        # client
        self.slots_free = cfg.cap
        self.slot_q: deque = deque()
        self.in_session = 0
        self.in_flight = 0
        # GPU
        self.waitq: list = []
        self.running: list = []       # heap of (finish decode clock, uid, request)
        self.n_run = 0
        self.ctx_run = 0.0
        self.dec_clock = 0.0
        self.dec_t = 0.0
        self.step_s = self.cost["step0"]
        self.batch_busy = False
        self.batch_full = False
        self.chunk_req = None
        self.gpu_ver = 0
        self.prefill_private = 0
        self.extra_gpu_s = 0.0        # GPU time owed (inline copies at GPU eviction with the mirror off)
        self.gpu_prefill_s = 0.0
        self.gpu_batches = 0
        # device
        self.rq: deque = deque()      # dispatched reads, FIFO
        self.r_cur = None
        self.r_rem = 0.0
        self.r_rate = 0.0
        self.r_t0 = 0.0
        self.r_ver = 0
        self.wq = None                # set by the policy (FIFO deque or priority list)
        self.w_cur = None
        self.w_rem = 0.0
        self.w_rate = 0.0
        self.w_t0 = 0.0
        self.w_ver = 0
        self.read_ewma = None
        self.read_samples = 0
        self.phi = cfg.phi
        self.ssd_heap: list = []      # (mtime, node) for the cleaner
        self.cleaner_passes: list = []
        # records
        self.done_reqs: list = []
        self.series: list = []
        self.cnt = dict.fromkeys(("lookups", "lookup_hits", "reads", "read_tokens", "refused", "cut", "backup_refused",
                                  "evict_host_tokens", "evict_gpu_tokens", "writes", "write_tokens", "write_skipped",
                                  "rewrite_tokens", "pins", "pin_skips", "unpins", "staged", "capped", "reevict",
                                  "write_hold_on", "bumps", "recompute_tokens", "cleaner_deleted_tokens",
                                  "lost_only_copy_tokens", "write_skipped_ended", "recompute_by_choice", "room_waits",
                                  "write_failed", "storm_cap", "capped_gate", "reevict_capped", "admit_requery", "demoted_tokens"), 0)
        self.win = dict.fromkeys(("r_busy", "w_busy", "adm", "adm_alone", "refused", "evicted", "gpu_idle_ready",
                                  "lam_max"), 0.0)
        self.win_t = 0.0
        self.adm_times: deque = deque()
        self.noise_rng = random.Random(f"{cfg.seed}:noise:{cfg.noise_seed}")
        self.pol.attach(self)

    # ===== events =====

    def at(self, t: float, fn, arg=None) -> None:
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, fn, arg))

    def run(self) -> "Sim":
        cfg = self.cfg
        for p in self.wl.plays:
            self.at(p.arrival, self._play_arrive, p)
        self.at(cfg.tick, self._tick)
        self.at(10.0, self._sample)
        self.at(cfg.cleaner_interval, self._cleaner)
        end = cfg.duration + cfg.grace
        ev = self.ev
        while ev:
            t, _, fn, arg = heapq.heappop(ev)
            if t > end:
                break
            self.now = t
            if arg is None:
                fn()
            else:
                fn(arg)
        self.now = min(end, self.now)
        return self

    # ===== client =====

    def _play_arrive(self, p) -> None:
        if self.slots_free > 0:
            self.slots_free -= 1
            self._play_start(p)
        else:
            self.slot_q.append(p)

    def _play_start(self, p) -> None:
        if self.now >= self.cfg.duration:
            return
        p.t_start = self.now
        p.n_left = len(p.streams)
        self.in_session += 1
        for st in p.streams:
            for r in st:
                self._init_req(r)
        for st in p.streams:
            self._stream_next(st[0])

    @staticmethod
    def _init_req(r) -> None:
        r.t_done = r.t_send = r.t_admit = r.t_ready = r.t_land = r.t_first = r.t_could = None
        r.reading = None
        r.plan = None
        r.hint = None
        r.attempts = r.lookups = r.refused = r.n_reads = r.tok_read = 0
        r.land_nodes = None
        r.pinned_nodes = None
        r.cum = None
        r.waiters = None
        r.last_match = 0
        r.pred_wait = None
        r.alone = False
        r.pos_arr = 0

    def _stream_next(self, r) -> None:
        """Schedule request r of its stream, once its dependencies are done: at their latest end plus the wait."""
        p = r.play
        ready = p.t_start
        for ds, dk in r.deps:
            d = p.streams[ds][dk]
            if d.t_done is None:
                if d.waiters is None:
                    d.waiters = []
                d.waiters.append(r)
                return
            if d.t_done > ready:
                ready = d.t_done
        t = max(self.now, ready + r.wait + (self.cfg.client_lag if r.deps else 0.0))
        if self.cfg.noise > 0:
            t += self.noise_rng.random() * self.cfg.noise
        self.at(max(self.now, min(t, self.cfg.duration)), self._send, r)   # [referee D1] never before now

    def _send(self, r) -> None:
        if self.now >= self.cfg.duration:
            self._abandon(r)
            return
        r.t_send = self.now
        # [referee D7] the client's ideal_cached: at the send, the leading pages of the prompt whose requests have
        # completed, and in 16-token blocks what it keeps of its base once that has completed
        ideal = 0
        for n in r.path:
            if not self.seen[n]:
                break
            ideal += self.tok[n]
        if r.base is not None and r.play.streams[r.base[0]][r.base[1]].t_done is not None:
            ideal = max(ideal, min(r.keep_tokens, r.prompt))
        r.ideal_at_send = ideal
        r.hint = self.hinter.at_send(r, self.now)
        self.in_flight += 1
        self._arrive(r)

    def _abandon(self, r) -> None:
        p = r.play
        st = p.streams[r.s]
        for x in st[r.k:]:
            if x.t_done is None:
                x.t_done = self.now
                for w in x.waiters or ():
                    self._stream_next(w)
                x.waiters = None
        self._stream_end(p)

    def _client_done(self, r) -> None:
        r.t_done = self.now
        seen = self.seen                            # [referee D7]
        for n in r.path:
            seen[n] = True
        self.in_flight -= 1
        p = r.play
        for w in r.waiters or ():
            self._stream_next(w)
        r.waiters = None
        st = p.streams[r.s]
        if r.k + 1 < len(st):
            self._stream_next(st[r.k + 1])
        else:
            self._stream_end(p)

    def _stream_end(self, p) -> None:
        p.n_left -= 1
        if p.n_left == 0:
            p.t_end = self.now
            self.in_session -= 1
            if self.slot_q:
                self._play_start(self.slot_q.popleft())
                # a play that cannot start (deadline) frees nothing further; the run is over
            else:
                self.slots_free += 1

    # ===== tree primitives =====

    def _hc(self, n: int) -> None:
        """Keep the host-evictable set right for node n: off the GPU, a host copy, no lock, no child in the tree."""
        if self.in_host[n] and not self.in_gpu[n] and self.hlock[n] == 0 and self.nct[n] == 0:
            if n not in self.hcand:
                self.hcand.add(n)
                heapq.heappush(self.hheap, (self.pol.prio(n), n, self.ver[n]))
        else:
            self.hcand.discard(n)

    def _gc(self, n: int) -> None:
        if self.in_gpu[n] and self.glock[n] == 0 and self.ncg[n] == 0:
            if n not in self.gcand:
                self.gcand.add(n)
                heapq.heappush(self.gheap, (self.pol.prio(n), n, self.ver[n]))
        else:
            self.gcand.discard(n)

    def touch(self, n: int) -> None:
        """A changed priority (stamp or registration): re-rank the node if it is a candidate."""
        v = self.ver[n] = self.ver[n] + 1
        if n in self.hcand:
            heapq.heappush(self.hheap, (self.pol.prio(n), n, v))
        if n in self.gcand:
            heapq.heappush(self.gheap, (self.pol.prio(n), n, v))

    def stamp_path(self, nodes, upto: int) -> None:
        now = self.now
        stamp, ver, hcand, gcand = self.stamp, self.ver, self.hcand, self.gcand
        for i in range(upto):
            n = nodes[i]
            if stamp[n] != now:
                stamp[n] = now
                if n in hcand or n in gcand:
                    self.touch(n)
        if upto:
            self._split_below(nodes[upto - 1], None)
        self.pol.on_served(nodes, upto)

    def _split_below(self, x: int, keep) -> None:
        """A prompt ends its match at node x (or a new child `keep` is inserted under it): the engine splits the
        node there, so no other child of x stays glued to it."""
        mup = self.mup
        for c in self.kids[x]:
            if mup[c] and c != keep:
                mup[c] = False

    def _rebuild_heaps(self) -> None:
        """Priorities that move with the clock are re-ranked at most once a second; a heap that is mostly stale
        entries is rebuilt."""
        if self.pol.dynamic and self.now - self.heap_built >= 1.0:
            self.heap_built = self.now
            for n in self.pol.dynamic_candidates():
                self.touch(n)
        if len(self.hheap) > 8 * len(self.hcand) + 4096:
            prio, ver = self.pol.prio, self.ver
            self.hheap = [(prio(n), n, ver[n]) for n in self.hcand]
            heapq.heapify(self.hheap)
        if len(self.gheap) > 8 * len(self.gcand) + 4096:
            prio, ver = self.pol.prio, self.ver
            self.gheap = [(prio(n), n, ver[n]) for n in self.gcand]
            heapq.heapify(self.gheap)

    def _hlock_inc(self, n: int) -> None:
        self.hlock[n] += 1
        self.hcand.discard(n)

    def _hlock_dec(self, n: int) -> None:
        self.hlock[n] -= 1
        if self.hlock[n] == 0:
            if self.orphan[n]:
                self.orphan[n] = False
                self.host_used -= self.tok[n]
                self.cnt["evict_host_tokens"] += self.tok[n]
            else:
                self._hc(n)

    def _glock_inc(self, n: int) -> None:
        if self.glock[n] == 0 and self.in_gpu[n]:
            self.gpu_unlocked -= self.tok[n]
            self.gcand.discard(n)
        self.glock[n] += 1

    def _glock_dec(self, n: int) -> None:
        self.glock[n] -= 1
        if self.glock[n] == 0 and self.in_gpu[n]:
            self.gpu_unlocked += self.tok[n]
            self._gc(n)

    def match(self, path) -> tuple:
        """(nodes on the GPU, nodes in the tree) at the head of a path; both sets are prefix-closed."""
        in_gpu, in_host = self.in_gpu, self.in_host
        lo, hi = 0, len(path)
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            n = path[mid - 1]
            if in_gpu[n] or in_host[n]:
                lo = mid
            else:
                hi = mid - 1
        it = lo
        lo, hi = 0, it
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if in_gpu[path[mid - 1]]:
                lo = mid
            else:
                hi = mid - 1
        return lo, it

    # ===== eviction =====

    def _pop(self, heap: list, cand: set):
        ver = self.ver
        while heap:
            _, n, v = heapq.heappop(heap)
            if n in cand and ver[n] == v:
                return n
        return None

    def evict_gpu(self, need: int) -> bool:
        """Free GPU tokens until `need` are free, lowest priority first, leaves first."""
        if self.C_G - self.gpu_used >= need:
            return True
        self._rebuild_heaps()
        while self.C_G - self.gpu_used < need:
            n = self._pop(self.gheap, self.gcand)
            if n is None:
                return False
            self._gpu_drop(n)
        return True

    def _gpu_drop(self, n: int) -> None:
        t = self.tok[n]
        self.gcand.discard(n)
        self.in_gpu[n] = False
        self.gpu_used -= t
        self.gpu_unlocked -= t
        self.cnt["evict_gpu_tokens"] += t
        p = self.par[n]
        self.ncg[p] -= 1
        glued = self.mup[n]
        if self.in_host[n]:
            self.host_mirror -= t
            self._hc(n)
        else:
            kept = False
            if not self.cfg.mirror and self.pol.demote_on_gpu_evict(n):
                # exclusive tiering: the host copy is made when the GPU gives the page up (an inline copy)
                if self.evict_host(t, asker=None):
                    self.in_host[n] = True
                    self.host_used += t
                    self.extra_gpu_s += t * self.cfg.d2h_s_per_token
                    self.cnt["demoted_tokens"] += t
                    kept = True
                    self._hc(n)
            if not kept:
                self._leave_tree(n)
        self._gc(p)
        if glued and p in self.gcand:
            self._gpu_drop(p)                  # one engine node: its upper part leaves the GPU with it

    def _leave_tree(self, n: int) -> None:
        """Node n has no copy in memory any more. Anything under it is unreachable: unlocked host pages go with
        it, locked ones (a write pending, a pin) are held until their lock is released."""
        if self.nct[n] > 0:
            for c in self.kids[n]:
                if self.in_host[c] and not self.in_gpu[c]:
                    if self.hlock[c] == 0:
                        self._host_drop(c, cascade=True)
                    else:
                        self._orphan(c)
        if not self.ssd[n] and self.ever[n]:
            self.cnt["lost_only_copy_tokens"] += self.tok[n]
        self.unwritten[n] = False
        self.mup[n] = False
        p = self.par[n]
        self.nct[p] -= 1
        self.pol.on_node_gone(n)
        self._hc(p)
        if not self.in_host[p]:
            self._gc(p)

    def _orphan(self, c: int) -> None:
        """A locked host node whose parent left the tree: no prompt can reach it; its tokens stay in host memory
        until the lock goes."""
        if self.nct[c] > 0:
            for k in self.kids[c]:
                if self.in_host[k] and not self.in_gpu[k]:
                    if self.hlock[k] == 0:
                        self._host_drop(k, cascade=True)
                    else:
                        self._orphan(k)
        self.hcand.discard(c)
        self.in_host[c] = False
        self.orphan[c] = True
        self.mup[c] = False
        self.nct[self.par[c]] -= 1
        self.pol.on_node_gone(c)

    def evict_host(self, need: int, asker) -> bool:
        """Free host tokens until `need` are free, in the policy's order; False when the evictable leaves run out.
        The policy may stop early (a class it will not take for this asker)."""
        if self.C_H - self.host_used >= need:
            return True
        self._rebuild_heaps()
        heap, cand, ver = self.hheap, self.hcand, self.ver
        stop = self.pol.evict_stop
        while self.C_H - self.host_used < need:
            n = None
            while heap:
                pr, m, v = heap[0]
                if m in cand and ver[m] == v:
                    n = m
                    break
                heapq.heappop(heap)
            if n is None or (stop is not None and stop(pr, n, asker)):
                return False
            heapq.heappop(heap)
            self.pol.on_evict(n, pr)
            self._host_drop(n)
        return True

    def _host_drop(self, n: int, cascade: bool = False) -> None:
        t = self.tok[n]
        if cascade and self.nct[n] > 0:
            for c in self.kids[n]:
                if self.in_host[c] and not self.in_gpu[c]:
                    if self.hlock[c] == 0:
                        self._host_drop(c, cascade=True)
                    else:
                        self._orphan(c)
        self.hcand.discard(n)
        self.in_host[n] = False
        self.host_used -= t
        self.cnt["evict_host_tokens"] += t
        self.win["evicted"] += t
        glued = self.mup[n]
        self._leave_tree(n)
        if glued and not cascade:
            p = self.par[n]
            if p in self.hcand:
                self._host_drop(p)             # one engine node: evicted whole

    # ===== arrival, lookups and reads =====

    def _arrive(self, r) -> None:
        r.pos_arr = len(self.waitq)
        self.waitq.append(r)
        self.pol.on_arrival(r)
        if r.reading is None:
            r.t_ready = self.now
        self._gpu_kick()

    def cum(self, r):
        c = r.cum
        if c is None:
            tok = self.tok
            c = [0]
            s = 0
            for n in r.path:
                s += tok[n]
                c.append(s)
            r.cum = c
        return c

    def lookup(self, r, chain, start: int, *, staged: bool = False) -> bool:
        """Ask the SSD for the nodes of `chain` from index `start` (the first one not in memory). The anchor (the
        node before it) is host-locked until the read lands or is revoked. The answer comes lookup_lag later.
        Returns False when no lookup is made (an anchor with no host copy)."""
        anchor = chain[start - 1] if start > 0 else 0
        if anchor and not self.in_host[anchor] and self.cfg.mirror:
            return False
        rd = Read(r, None, 0, anchor, staged)
        rd.rid = (chain, start)
        r.reading = rd
        if anchor:
            self._hlock_inc(anchor)
        self.cnt["lookups"] += 1
        self.at(self.now + self.cfg.lookup_lag, self._lookup_done, rd)
        return True

    def _lookup_done(self, rd: Read) -> None:
        if rd.state != "lookup":
            return
        chain, start = rd.rid
        ssd, in_host, in_gpu, tok = self.ssd, self.in_host, self.in_gpu, self.tok
        # the anchor is locked, so `start` is still the first node not in memory unless it came back meanwhile
        while start < len(chain) and (in_host[chain[start]] or in_gpu[chain[start]]):
            start += 1
        j = start
        tokens = 0
        while j < len(chain) and ssd[chain[j]]:
            tokens += tok[chain[j]]
            j += 1
        r = rd.r
        if tokens < THRESH:
            self._revoke(rd)
            self.pol.on_lookup_miss(r, rd)
            self._ready(r)
            return
        if start > 0 and chain[start - 1] != rd.anchor:
            if rd.anchor:
                self._hlock_dec(rd.anchor)
            rd.anchor = chain[start - 1]
            self._hlock_inc(rd.anchor)
        rd.nodes = list(chain[start:j])
        rd.tokens = tokens
        rd.pages = tokens // PAGE
        rd.state = "parked"
        rd.parked_at = self.now
        self.cnt["lookup_hits"] += 1
        self.pol.on_lookup_hit(r, rd)

    def _revoke(self, rd: Read) -> None:
        if rd.anchor:
            self._hlock_dec(rd.anchor)
        rd.state = "revoked"
        if rd.r.reading is rd:
            rd.r.reading = None

    def _ready(self, r) -> None:
        """Request r has no read under way any more: admission may take it."""
        if r.t_send is not None and r.t_admit is None and r.reading is None:
            r.t_ready = self.now
            self._gpu_kick()

    def launch(self, rd: Read) -> bool:
        """Dispatch a parked read: take its host pages (the policy makes room, may cut it, or refuse it)."""
        granted = self.pol.make_room(rd)
        if granted < 0:
            return None
        if granted < rd.tokens:
            keep, s = [], 0
            for n in rd.nodes:
                if s + self.tok[n] > granted:
                    break
                keep.append(n)
                s += self.tok[n]
            if s < THRESH:
                granted = 0
            else:
                rd.nodes, rd.tokens, rd.pages = keep, s, s // PAGE
                self.cnt["cut"] += 1
                self.pol.on_refused(rd.r, rd, cut=True)
        if granted <= 0:
            self.cnt["refused"] += 1
            self.win["refused"] += 1
            r = rd.r
            self._revoke(rd)
            self.pol.on_refused(r, rd, cut=False)
            self._ready(r)
            return False
        self.host_used += rd.tokens
        self.inflight_tokens += rd.tokens
        self.inflight_pages += rd.pages
        rd.state = "inflight"
        rd.t_disp = self.now
        self.rq.append(rd)
        self.cnt["reads"] += 1
        self.cnt["read_tokens"] += rd.tokens
        if self.r_cur is None:
            self._read_next()
        return True

    # ----- the device: one read thread and one write thread on a shared disk -----

    def _retime(self) -> None:
        """Either thread started or stopped: settle the work done at the old rates and set the new ones."""
        now, cfg = self.now, self.cfg
        if self.r_cur is not None:
            self.r_rem -= self.r_rate * (now - self.r_t0)
            self.win["r_busy"] += now - max(self.r_t0, self.win_t)
        if self.w_cur is not None:
            self.w_rem -= self.w_rate * (now - self.w_t0)
            self.win["w_busy"] += now - max(self.w_t0, self.win_t)
        self.r_t0 = self.w_t0 = now
        both = self.r_cur is not None and self.w_cur is not None
        self.r_rate = cfg.beta * self.phi if both else cfg.read_alone
        self.w_rate = cfg.beta * (1.0 - self.phi) if both else cfg.write_alone
        if self.r_cur is not None:
            self.r_ver += 1
            self.at(now + max(0.0, self.r_rem) / self.r_rate, self._read_io_done, self.r_ver)
        if self.w_cur is not None:
            self.w_ver += 1
            self.at(now + max(0.0, self.w_rem) / self.w_rate, self._write_io_done, self.w_ver)

    def _read_next(self) -> None:
        if self.r_cur is None and self.rq:
            self._retime()
            rd = self.rq.popleft()
            self.r_cur, self.r_rem = rd, float(rd.tokens)
            rd.io_start = self.now
            rd.lost_from = next((i for i, n in enumerate(rd.nodes) if not self.ssd[n]), None)   # [referee D5]
            self._retime()

    def _read_io_done(self, ver: int) -> None:
        if ver != self.r_ver or self.r_cur is None:
            return
        rd = self.r_cur
        self._retime()
        self.r_cur = None
        self.r_ver += 1
        ms_per_page = (self.now - rd.io_start) * 1000.0 / max(1, rd.pages)
        self.read_samples += 1
        self.read_ewma = ms_per_page if self.read_ewma is None else self.read_ewma + (ms_per_page - self.read_ewma) / 8
        self.at(self.now + self.cfg.land_lag, self._read_landed, rd)
        if self.rq:
            self._read_next()
        elif self.w_cur is not None:
            self._retime()

    def _read_landed(self, rd: Read) -> None:
        """The read's pages become host-only, clean nodes; the policy may pin them until admission."""
        r = rd.r
        now = self.now
        self.inflight_tokens -= rd.tokens
        self.inflight_pages -= rd.pages
        landed = []
        for i_node, n in enumerate(rd.nodes):
            if rd.lost_from is not None and i_node >= rd.lost_from:
                # [referee D5] the cleaner deleted this page (or one before it) before the I/O began: nothing was read
                self.host_used -= self.tok[n]
                self.cnt["read_deleted_tokens"] = self.cnt.get("read_deleted_tokens", 0) + self.tok[n]
                continue
            if self.in_host[n] or self.in_gpu[n]:
                if not self.in_host[n] and not self.cfg.mirror:
                    self.in_host[n] = True      # GPU-only under exclusive tiering: keep the copy that was read
                    self.host_mirror += self.tok[n]     # [referee D6] it is a host copy of a GPU page now
                    landed.append(n)
                    continue
                self.host_used -= self.tok[n]   # came back another way meanwhile: the duplicate is freed
                continue
            if not (self.in_host[self.par[n]] or self.in_gpu[self.par[n]]):
                self.host_used -= self.tok[n]   # its parent is gone (a cut chain): unreachable
                continue
            self.in_host[n] = True
            self.stamp[n] = now
            p = self.par[n]
            self.nct[p] += 1
            self.hcand.discard(p)
            if landed and landed[-1] == p:
                self.mup[n] = True
            self._split_below(p, n)
            landed.append(n)
        rd.state = "landed"
        rd.t_land = now
        rd.nodes = landed
        r.n_reads += 1
        r.tok_read += sum(self.tok[n] for n in landed)
        r.t_land = now
        if r.land_nodes is None:
            r.land_nodes = set()
        r.land_nodes.update(landed)
        if rd.anchor:
            self._hlock_dec(rd.anchor)
        if r.reading is rd:
            r.reading = None
        self.pol.on_landed(r, rd)
        for n in landed:
            self._hc(n)
        self._ready(r)
        self.pol.dispatch()

    def pin_read(self, r, rd: Read) -> None:
        tokens = 0
        for n in rd.nodes:
            self.pin[n] += 1
            self._hlock_inc(n)
            tokens += self.tok[n]
        self.pinned_tokens += tokens
        rd.pinned = True
        if r.pinned_nodes is None:
            r.pinned_nodes = []
        r.pinned_nodes.append((rd, tokens))
        self.cnt["pins"] += 1

    def unpin(self, r) -> int:
        """Release every pin held for request r; returns the tokens released."""
        total = 0
        if r.pinned_nodes:
            for rd, tokens in r.pinned_nodes:
                for n in rd.nodes:
                    self.pin[n] -= 1
                    self._hlock_dec(n)
                rd.pinned = False
                total += tokens
            self.pinned_tokens -= total
            r.pinned_nodes = None
            self.cnt["unpins"] += 1
        return total

    def pinned_of(self, r) -> int:
        return sum(t for _, t in r.pinned_nodes) if r.pinned_nodes else 0

    # ----- writes -----

    def backup(self, n: int, r) -> bool:
        """Write-through: copy a new GPU node to host memory and queue its SSD write. False = refused (no room)."""
        t = self.tok[n]
        if not self.evict_host(t, asker=None):
            self.cnt["backup_refused"] += 1
            self.unwritten[n] = True
            return False
        self.in_host[n] = True
        self.host_used += t
        self.host_mirror += t
        if self.pol.should_write(n, r):
            self.dirty[n] = True
            self._hlock_inc(n)
            self.pending_write += t
            self.pol.write_push(n, r)
            if self.w_cur is None:
                self._write_next()
        else:
            self.unwritten[n] = not self.ssd[n]
        return True

    def dirty_host_only(self) -> int:
        """Pending-write tokens of pages the GPU no longer holds (the rest of the pending writes sit in the mirror)."""
        in_gpu, tok = self.in_gpu, self.tok
        total = sum(tok[n] for n in self.pol.wqueue if not in_gpu[n])
        if self.w_cur is not None and not in_gpu[self.w_cur]:
            total += tok[self.w_cur]
        return total

    def _write_next(self) -> None:
        if self.w_cur is not None:
            return
        while True:
            n = self.pol.write_pop()
            if n is None:
                return
            if self.ssd[n] and self.pol.skip_stored:
                # fix 1: the page files exist; touched, not written again
                self.cnt["write_skipped"] += 1
                self._write_ack(n, touched=True)
                continue
            if self.ssd[n]:
                self.cnt["rewrite_tokens"] += self.tok[n]
            self._retime()
            self.w_cur, self.w_rem = n, float(self.tok[n])
            self._retime()
            return

    def _write_io_done(self, ver: int) -> None:
        if ver != self.w_ver or self.w_cur is None:
            return
        n = self.w_cur
        self._retime()
        self.w_cur = None
        self.w_ver += 1
        self.cnt["writes"] += 1
        self.cnt["write_tokens"] += self.tok[n]
        failed = self.cfg.write_fail_p > 0 and self.pol.rng.random() < self.cfg.write_fail_p
        self._write_ack(n, failed=failed)
        self._write_next()
        if self.w_cur is None and self.r_cur is not None:
            self._retime()

    def _write_ack(self, n: int, *, touched: bool = False, failed: bool = False) -> None:
        t = self.tok[n]
        if failed:
            self.cnt["write_failed"] += 1
            if self.pol.requeue_failed_write(n):
                self.pol.write_push(n, None)
                return
        else:
            if not self.ssd[n]:
                self.ssd[n] = True
                self.ssd_used += t
            self.ssd_t[n] = self.now
            heapq.heappush(self.ssd_heap, (self.now, n))
        self.dirty[n] = False
        self.unwritten[n] = failed
        self.pending_write -= t
        self._hlock_dec(n)
        if not self.cfg.mirror and self.in_gpu[n] and self.in_host[n] and self.hlock[n] == 0 and not failed:
            self.in_host[n] = False            # exclusive tiering: the GPU holds the page, the SSD has the copy
            self.host_used -= t
            self.host_mirror -= t
        self.pol.on_write_ack(n)

    # ----- the cleaner -----

    def _cleaner(self) -> None:
        cfg = self.cfg
        self.at(self.now + cfg.cleaner_interval, self._cleaner)
        if self.ssd_used < cfg.cleaner_hi * self.C_S:
            return
        target = cfg.cleaner_lo * self.C_S
        deleted = 0
        marginal = None
        order = iter(self.pol.cleaner_order(target))
        while self.ssd_used > target:               # [referee D4] test before taking the next node off the order
            n = next(order, None)
            if n is None:
                break
            if not self.ssd[n]:
                continue
            self.ssd[n] = False
            self.ssd_used -= self.tok[n]
            deleted += self.tok[n]
            marginal = n
            if self.in_host[n] or self.in_gpu[n]:
                if not self.dirty[n]:
                    self.unwritten[n] = True
                self.pol.on_ssd_deleted(n)
        self.cnt["cleaner_deleted_tokens"] += deleted
        self.cleaner_passes.append((round(self.now, 1), deleted))
        self.pol.on_cleaner_pass(marginal)

    def ssd_oldest(self):
        """Stored nodes, oldest write (or touch) first: the engine's cleaner order."""
        heap, ssd, ssd_t = self.ssd_heap, self.ssd, self.ssd_t
        while heap:
            t, n = heapq.heappop(heap)
            if ssd[n] and ssd_t[n] == t:
                yield n

    def ssd_touch(self, n: int) -> None:
        if self.ssd[n] and self.ssd_t[n] != self.now:
            self.ssd_t[n] = self.now
            heapq.heappush(self.ssd_heap, (self.now, n))

    # ===== GPU =====

    def _gpu_kick(self) -> None:
        """Something changed that may let the GPU start a prefill batch."""
        if self.batch_busy:
            return
        self._advance_decode()
        if self._form_batch():
            return
        self._arm_decode()

    def _advance_decode(self) -> None:
        now = self.now
        if self.n_run > 0 and not self.batch_busy:
            self.dec_clock += (now - self.dec_t) / self.step_s
        self.dec_t = now

    def _arm_decode(self) -> None:
        self.gpu_ver += 1
        if self.running:
            left = self.running[0][0] - self.dec_clock
            self.at(self.now + max(0.0, left) * self.step_s, self._decode_event, self.gpu_ver)

    def _set_step(self) -> None:
        c = self.cost
        self.step_s = c["step0"] + c["step_n"] * min(self.n_run, c["step_n_cap"]) + c["step_k"] * self.ctx_run

    def _decode_event(self, ver: int) -> None:
        if ver != self.gpu_ver or self.batch_busy:
            return
        self._advance_decode()
        finished = False
        while self.running and self.running[0][0] <= self.dec_clock + 1e-9:
            _, _, r = heapq.heappop(self.running)
            self._finish(r)
            finished = True
        if finished:
            self.batch_full = False
        self._gpu_kick()

    def _form_batch(self) -> bool:
        """One prefill batch: the request still being chunked first, then queued requests in arrival order while
        the pool has room and the batch has tokens left. Returns True when a batch starts."""
        budget = CHUNK
        items = []
        cost = self.cost
        work = 0.0
        if self.chunk_req is not None:
            r = self.chunk_req
            t = min(r.pf_left, budget)
            work += cost["a"] * t + cost["b"] * (r.pf_pos * t + 0.5 * t * t)
            r.pf_pos += t
            r.pf_left -= t
            budget -= t
            items.append(r)
            if r.pf_left > 0:
                budget = 0
            else:
                self.chunk_req = None
        if budget > 0 and not self.batch_full and self.waitq:
            waitq = self.waitq
            pol = self.pol
            i = 0
            n_ready_seen = 0
            while i < len(waitq) and budget > 0 and self.n_run + len(items) < self.cfg.max_running:
                r = waitq[i]
                if r.reading is not None:
                    if r.t_could is None and self._would_fit(r):
                        r.t_could = self.now            # the GPU could have taken it now, were its read done
                    i += 1
                    continue
                res = self._admit(r, n_ready_seen)
                if res == 1:          # skipped this pass (the policy re-reads)
                    i += 1
                    continue
                if res == 2:          # no room in the pool: FCFS stops here
                    self.batch_full = self.n_run > 0 or bool(items)
                    break
                n_ready_seen += 1
                del waitq[i]
                t = min(r.pf_left, budget)
                work += cost["a"] * t + cost["b"] * (r.pf_pos * t + 0.5 * t * t) + cost["h2d"] * (r.tok_host + r.tok_ssd)
                r.pf_pos += t
                r.pf_left -= t
                budget -= t
                items.append(r)
                if r.pf_left > 0:
                    self.chunk_req = r
                    break
        if not items:
            return False
        dt = cost["batch_s"] + work + self.extra_gpu_s
        self.extra_gpu_s = 0.0
        self.gpu_prefill_s += dt
        self.gpu_batches += 1
        self.batch_busy = True
        self.gpu_ver += 1
        self.at(self.now + dt, self._batch_end, items)
        return True

    def _would_fit(self, r) -> bool:
        """Would the pool admit r now if its history were resident? (the admission test, without the read)"""
        path = r.path
        ig, _ = self.match(path)
        cum = self.cum(r)
        own_unlocked = 0
        tok, glock = self.tok, self.glock
        for i in range(ig):
            if glock[path[i]] == 0:
                own_unlocked += tok[path[i]]
        need = r.prompt - cum[ig] + r.out + PAGE
        return need <= self.C_G - self.gpu_used + self.gpu_unlocked - own_unlocked

    def _admit(self, r, n_ready_before: int) -> int:
        """Try to admit r: 0 = admitted, 1 = skipped (a read was issued), 2 = the pool has no room."""
        path = r.path
        ig, it = self.match(path)
        cum = self.cum(r)
        if self.pol.before_admit(r, cum[it]):
            return 1
        # the admission loop matches the prompt before it tests the pool: the path is stamped even if r must wait
        self.stamp_path(path, it)
        tok, in_gpu, glock = self.tok, self.in_gpu, self.glock
        gpu_tokens = cum[ig]
        own_unlocked = 0
        for i in range(ig):
            n = path[i]
            if glock[n] == 0:
                own_unlocked += tok[n]
        need = r.prompt - gpu_tokens + r.out + PAGE
        if need > self.C_G - self.gpu_used + self.gpu_unlocked - own_unlocked:
            return 2
        now = self.now
        for i in range(ig):
            self._glock_inc(path[i])
        if not self.cfg.mirror:
            # [referee D6] mirror off: a GPU eviction copies pages to host and may evict host pages to make room;
            # the host pages this admission is about to load must not be among them
            for i in range(ig, it):
                self._hlock_inc(path[i])
        fits = self.evict_gpu(need)
        if not self.cfg.mirror:
            for i in range(ig, it):
                self._hlock_dec(path[i])
        if not fits:
            for i in range(ig):
                self._glock_dec(path[i])
            return 2
        land = r.land_nodes
        host_tokens = ssd_tokens = 0
        for i in range(ig, it):
            n = path[i]
            t = tok[n]
            if not in_gpu[n]:
                in_gpu[n] = True
                self.gpu_used += t
                self.ncg[self.par[n]] += 1
                self.gcand.discard(self.par[n])
                if self.in_host[n]:
                    self.hcand.discard(n)
                    if not self.cfg.mirror and self.ssd[n] and not self.dirty[n] and self.hlock[n] == 0:
                        self.in_host[n] = False        # exclusive tiering: the GPU holds it, the SSD has the copy
                        self.host_used -= t
                    else:
                        self.host_mirror += t
                glock[n] += 1
            else:
                self._glock_inc(n)
            if land is not None and n in land:
                ssd_tokens += t
            else:
                host_tokens += t
        resident = cum[it]
        # the admission test counts the whole output; the pool then holds what has been generated, half of it on
        # average over the decode (the engine reserves about a tenth of the running requests' remaining output)
        private = r.prompt - resident + (r.out >> 1)
        self.gpu_used += private
        r.private = private
        r.n_locked = it
        r.tok_gpu, r.tok_host, r.tok_ssd = gpu_tokens, host_tokens, ssd_tokens
        r.resident_at_admit = resident
        r.pf_pos = resident
        r.pf_left = r.prompt - resident
        r.t_admit = now
        r.alone = n_ready_before == 0 and not any(x.reading is None for x in self.waitq if x is not r)
        self.unpin(r)
        self.adm_times.append(now)
        self.win["adm"] += 1
        if r.alone:
            self.win["adm_alone"] += 1
        self.pol.on_admit(r)
        return 0

    def _batch_end(self, items) -> None:
        self.batch_busy = False
        self.dec_t = self.now
        for r in items:
            if r.pf_left > 0:
                continue
            self._prefill_done(r)
        self._set_step()
        self._gpu_kick()

    def _prefill_done(self, r) -> None:
        """The prompt's KV exists: first token, the new nodes enter the tree, are copied to host and queued for the
        SSD; the request joins the decode batch."""
        now = self.now
        r.t_first = now + self.cfg.first_token_s
        path = r.path
        tok, in_gpu, in_host = self.tok, self.in_gpu, self.in_host
        ideal_pages = 0
        ever = self.ever
        for n in path:
            if not ever[n]:
                break
            ideal_pages += tok[n]
        # what an infinite cache would hold, as the client counts it: the pages completed in this run and, in
        # 16-token blocks, what the request keeps of its base once that has completed
        if r.base is not None and r.play.streams[r.base[0]][r.base[1]].t_done is not None:
            ideal_pages = max(ideal_pages, min(r.keep_tokens, r.prompt))
        r.ideal_seen = r.ideal_at_send              # [referee D7]
        prev_new = -1
        for i in range(r.n_locked, len(path)):
            n = path[i]
            t = tok[n]
            if in_gpu[n]:
                self.gpu_used -= t          # another request inserted it meanwhile: the duplicate is freed
                r.private -= t
                self._glock_inc(n)
                continue
            was_in_tree = in_host[n]
            if not was_in_tree:
                if prev_new == i - 1:
                    self.mup[n] = True
                self._split_below(self.par[n], n)
                prev_new = i
            in_gpu[n] = True
            r.private -= t
            self.glock[n] += 1
            self.ncg[self.par[n]] += 1
            self.gcand.discard(self.par[n])
            self.stamp[n] = now
            ever[n] = True
            if was_in_tree:
                self.host_mirror += t
                self.hcand.discard(n)
            else:
                self.nct[self.par[n]] += 1
                self.hcand.discard(self.par[n])
                self.backup(n, r)
        r.n_locked = len(path)
        self.pol.on_served(path, len(path))
        self.n_run += 1
        self.ctx_run += r.prompt + 0.5 * r.out
        heapq.heappush(self.running, (self.dec_clock + max(0, r.out - 1), r.uid, r))

    def _finish(self, r) -> None:
        """The reply is complete: the output's pages enter the tree, the request's locks go, the turn commits."""
        now = self.now
        self.n_run -= 1
        self.ctx_run -= r.prompt + 0.5 * r.out
        self._set_step()
        self.gpu_used -= r.private
        r.private = 0
        tw = r.twig
        if tw is not None and self.C_G - self.gpu_used < self.tok[tw] and not self.evict_gpu(self.tok[tw]):
            # [referee D3] no room and nothing evictable: the reply's pages are not cached (the pool is never exceeded)
            self.cnt["twig_dropped"] = self.cnt.get("twig_dropped", 0) + 1
            tw = None
            r.twig, r.twig_pages = None, 0
        if tw is not None:
            t = self.tok[tw]
            self.in_gpu[tw] = True
            self.gpu_used += t
            self.gpu_unlocked += t
            self.stamp[tw] = now
            self.ever[tw] = True
            p = self.par[tw]
            self.ncg[p] += 1
            self.nct[p] += 1
            self.gcand.discard(p)
            self.hcand.discard(p)
            self._split_below(p, tw)
            self.backup(tw, r)
        for n in r.path:
            self._glock_dec(n)
        if tw is not None:
            self._gc(tw)
        r.t_done_srv = now
        self.pol.on_commit(r)
        self._record(r)
        self._client_done(r)

    # ===== periodic =====

    def _tick(self) -> None:
        self.at(self.now + self.cfg.tick, self._tick)
        self.pol.tick()

    def adm_rate(self, now: float):
        """The engine's admission rate: over the last minute, or the last five when the minute holds under 5."""
        adm = self.adm_times
        while adm and adm[0] <= now - 300.0:
            adm.popleft()
        total = len(adm)
        recent = 0
        for t in reversed(adm):
            if t <= now - 60.0:
                break
            recent += 1
        if recent >= 5:
            return recent / 60.0
        if total >= 5:
            return total / 300.0
        return None

    def _sample(self) -> None:
        now = self.now
        self.at(now + 10.0, self._sample)
        if self.r_cur is not None:
            self.win["r_busy"] += now - max(self.r_t0, self.win_t)
        if self.w_cur is not None:
            self.win["w_busy"] += now - max(self.w_t0, self.win_t)
        span = now - self.win_t
        ready = sum(1 for r in self.waitq if r.reading is None)
        dirty_ho = self.dirty_host_only()
        idle_clean = self.host_used - self.host_mirror - dirty_ho - self.pinned_tokens - self.inflight_tokens
        prices = self.pol.prices(ready)
        w = self.win
        self.series.append({
            "t": round(now, 1), "in_session": self.in_session, "in_flight": self.in_flight,
            "queue": len(self.waitq), "ready": ready, "running": self.n_run,
            "pool_used": round((self.gpu_used - self.gpu_unlocked) / self.C_G, 4),
            "pool_alloc": round(self.gpu_used / self.C_G, 4),
            "gpu_prefill_s": round(self.gpu_prefill_s, 3), "gpu_batches": self.gpu_batches,
            "ctx_run": int(self.ctx_run), "step_ms": round(self.step_s * 1e3, 3),
            "read_busy": round(w["r_busy"] / span, 4), "write_busy": round(w["w_busy"] / span, 4),
            "host_mirror": self.host_mirror, "host_idle": idle_clean, "host_dirty": self.pending_write,
            "host_dirty_only": dirty_ho,
            "host_pinned": self.pinned_tokens, "host_inflight": self.inflight_tokens,
            "host_free": self.C_H - self.host_used, "ssd_used": self.ssd_used,
            "refusals": int(w["refused"]), "cleaner_passes": len(self.cleaner_passes),
            "parked": self.pol.parked_count(), "read_ms_page": None if self.read_ewma is None else round(self.read_ewma, 3),
            "mu": prices["mu"], "c": prices["c"], "kappa": prices["kappa"], "lambda": prices["lambda"],
            "sigma": prices["sigma"], "price_engine": prices.get("engine"),
        })
        for k in w:
            w[k] = 0.0
        self.win_t = now

    def _record(self, r) -> None:
        self.done_reqs.append(r)

    # ===== outputs =====

    def request_rows(self) -> list:
        """One row per completed request, in send order."""
        rows = []
        for r in sorted(self.done_reqs, key=lambda x: x.t_send):
            ideal = min(r.prompt, max(r.ideal_seen, 0))
            hist_cached = r.tok_gpu + r.tok_host + r.tok_ssd
            # stall: from the moment the GPU could have taken the turn (were its history resident) to its read's end
            stall = max(0.0, r.t_ready - r.t_could) if r.t_could is not None and r.t_ready is not None else 0.0
            rows.append({
                "play": r.play.pid, "stream": r.s, "k": r.k, "t_send": round(r.t_send, 4),
                "t_admit": round(r.t_admit, 4), "ttft": round(r.t_first - r.t_send, 4),
                "latency": round(r.t_done_srv - r.t_send, 4), "queue_wait": round(r.t_admit - r.t_send, 4),
                "stall": round(stall, 4),
                "prompt": r.prompt, "out": r.out, "ideal": ideal, "tok_gpu": r.tok_gpu, "tok_host": r.tok_host,
                "tok_ssd": r.tok_ssd, "tok_recomputed": max(0, ideal - hist_cached),
                "tok_new": r.prompt - max(ideal, hist_cached), "reads": r.n_reads, "refused": r.refused,
                "first": int(r.k == 0 and r.s == 0), "gap": r.gap, "wait": round(r.wait, 3),
                "pos": r.pos_arr, "alone": int(r.alone), "harness": r.play.harness,
            })
        return rows

    def write_outputs(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        rows = self.request_rows()
        if rows:
            cols = list(rows[0])
            with open(os.path.join(folder, "requests.csv"), "w") as f:
                f.write(",".join(cols) + "\n")
                for row in rows:
                    f.write(",".join(str(row[c]) for c in cols) + "\n")
        if self.series:
            cols = list(self.series[0])
            with open(os.path.join(folder, "series.csv"), "w") as f:
                f.write(",".join(cols) + "\n")
                for row in self.series:
                    f.write(",".join("" if row[c] is None else str(row[c]) for c in cols) + "\n")
        with open(os.path.join(folder, "summary.json"), "w") as f:
            json.dump(summary(self), f, indent=1)


WINDOWS = ((0, 1800), (1800, 3600), (3600, 5400), (5400, 6600))


def summarize_rows(rows: list, *, cut: float, windows=WINDOWS) -> dict:
    """The statistics of the campaign records over requests sent before `cut` (compare calib_measured.py)."""
    r = [x for x in rows if x["t_send"] < cut]
    if not r:
        return {"requests": 0}
    ttft = np.array([x["ttft"] for x in r])
    t = np.array([x["t_send"] for x in r])
    prompt = np.array([x["prompt"] for x in r], float)
    gpu = np.array([x["tok_gpu"] for x in r], float)
    host = np.array([x["tok_host"] for x in r], float)
    ssd = np.array([x["tok_ssd"] for x in r], float)
    ideal = np.array([x["ideal"] for x in r], float)
    first = np.array([x["first"] for x in r])
    cached = gpu + host + ssd
    out = {
        "requests": len(r), "req_per_min": len(r) / (cut / 60.0), "ttft_mean": float(ttft.mean()),
        "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p90": float(np.percentile(ttft, 90)),
        "ttft_p99": float(np.percentile(ttft, 99)),
        "share_gpu": float(gpu.sum() / prompt.sum()), "share_host": float(host.sum() / prompt.sum()),
        "share_ssd": float(ssd.sum() / prompt.sum()), "share_uncached": float(1 - cached.sum() / prompt.sum()),
        "history_recomputed": float(1 - cached.sum() / max(1.0, ideal.sum())),
        "prompt_mean": float(prompt.mean()), "out_mean": float(np.mean([x["out"] for x in r])),
        "queue_wait_mean": float(np.mean([x["queue_wait"] for x in r])),
        "stall_mean": float(np.mean([x["stall"] for x in r])),
        "windows": [],
    }
    for lo, hi in windows:
        m = (t >= lo) & (t < hi)
        if not m.any():
            continue
        ret = m & (first == 0)
        out["windows"].append({
            "window": f"{lo // 60}-{hi // 60} min", "requests": int(m.sum()),
            "req_per_min": float(m.sum() / ((hi - lo) / 60.0)), "ttft_mean": float(ttft[m].mean()),
            "ttft_p50": float(np.percentile(ttft[m], 50)), "ttft_p90": float(np.percentile(ttft[m], 90)),
            "share_gpu": float(gpu[m].sum() / prompt[m].sum()), "share_host": float(host[m].sum() / prompt[m].sum()),
            "share_ssd": float(ssd[m].sum() / prompt[m].sum()),
            "history_recomputed": float(1 - cached[m].sum() / max(1.0, ideal[m].sum())),
            "recomputed_tokens_M": float((ideal[m] - cached[m]).clip(min=0).sum() / 1e6),
            "returning_reading": float((ssd[ret] > 0).mean()) if ret.any() else None,
        })
    sat = (t >= 5400) & (t < 6540)
    if sat.any():
        out["req_per_min_90_109"] = float(sat.sum() / 19.0)
        out["ttft_mean_90_109"] = float(ttft[sat].mean())
    return out


def summary(sim: Sim, cut: float | None = None) -> dict:
    cfg = sim.cfg
    cut = cut or cfg.cut or max(0.0, cfg.duration - 600.0)
    s = summarize_rows(sim.request_rows(), cut=cut)
    ser = sim.series
    late = [x for x in ser if 3600 <= x["t"] < cut]
    sat = [x for x in ser if 5400 <= x["t"] < cut]

    def med(rows, key):
        vals = [x[key] for x in rows if x[key] is not None]
        return float(np.median(vals)) if vals else None

    plays = [p for p in sim.wl.plays if p.t_start is not None]
    s.update({
        "arm": cfg.arm, "seed": cfg.seed, "cut": cut, "config": {k: v for k, v in cfg.as_dict().items() if k != "out"},
        "users_started": len(plays),
        "users_with_request": len({x["play"] for x in sim.request_rows() if x["t_send"] < cut}),
        "slot_wait_mean": float(np.mean([p.t_start - p.arrival for p in plays])) if plays else None,
        "cap_binds_at_min": next((x["t"] / 60.0 for x in ser if x["in_session"] >= cfg.cap), None),
        "cleaner_passes": sim.cleaner_passes,
        "first_cleaner_pass_s": sim.cleaner_passes[0][0] if sim.cleaner_passes else None,
        "counters": dict(sim.cnt),
        "census_60_110": {k: med(late, k) for k in ("host_mirror", "host_idle", "host_dirty", "host_pinned",
                                                     "host_inflight", "host_free", "queue", "ready", "running",
                                                     "pool_used", "read_busy", "write_busy", "parked")},
        "census_90_110": {k: med(sat, k) for k in ("queue", "ready", "running", "pool_used", "read_busy",
                                                    "write_busy", "host_pinned", "host_dirty", "mu", "c", "kappa",
                                                    "lambda", "in_flight")},
        "pending_write_p90_60_110": float(np.percentile([x["host_dirty"] for x in late], 90)) if late else None,
        "pinned_p90_60_110": float(np.percentile([x["host_pinned"] for x in late], 90)) if late else None,
        "refusals_before_cut": int(sum(x["refusals"] for x in ser if x["t"] <= cut)),
        "hints": dict(sim.hinter.stats), "policy": sim.pol.describe(),
    })
    return s


def print_summary(s: dict) -> None:
    print(f"arm {s['arm']} seed {s['seed']}: requests sent before {s['cut']:.0f} s: {s['requests']} "
          f"({s['req_per_min']:.1f}/min; minutes 90-109 {s.get('req_per_min_90_109', float('nan')):.1f}/min)")
    print(f"  TTFT mean {s['ttft_mean']:.2f} p50 {s['ttft_p50']:.2f} p90 {s['ttft_p90']:.2f} p99 {s['ttft_p99']:.2f} s"
          f" | queue wait {s['queue_wait_mean']:.2f} stall {s['stall_mean']:.2f}")
    print(f"  prompt tokens from GPU / host / SSD {s['share_gpu']:.3f} / {s['share_host']:.3f} / {s['share_ssd']:.3f};"
          f" uncached {s['share_uncached']:.3f}; history recomputed {s['history_recomputed']:.4f}")
    for w in s["windows"]:
        rr = w["returning_reading"]
        print(f"    {w['window']:>12s}: n {w['requests']:5d} ({w['req_per_min']:.1f}/min) TTFT {w['ttft_mean']:.2f} "
              f"p50 {w['ttft_p50']:.2f} p90 {w['ttft_p90']:.2f} | GPU/host/SSD {w['share_gpu']:.3f}/"
              f"{w['share_host']:.3f}/{w['share_ssd']:.3f} recomputed {w['history_recomputed']:.4f} "
              f"({w['recomputed_tokens_M']:.2f} M) returning reading {0.0 if rr is None else rr:.3f}")
    c = s["census_60_110"]
    print(f"  minutes 60-110 medians: running {c['running']}, queued {c['queue']} (ready {c['ready']}), pool "
          f"{c['pool_used']}, read thread busy {c['read_busy']}, write {c['write_busy']}; host mirror "
          f"{c['host_mirror']}, idle {c['host_idle']}, dirty {c['host_dirty']} (p90 {s['pending_write_p90_60_110']}), "
          f"pinned {c['host_pinned']} (p90 {s['pinned_p90_60_110']}), free {c['host_free']}")
    print(f"  users started {s['users_started']}, slot wait {s['slot_wait_mean']:.1f} s, cap binds at minute "
          f"{s['cap_binds_at_min']}; cleaner passes {s['cleaner_passes'][:4]}; refused reads before the cut "
          f"{s['refusals_before_cut']}")
    k = s["counters"]
    print(f"  reads {k['reads']} ({k['read_tokens'] / 1e6:.1f} M tokens), refused {k['refused']}, cut {k['cut']}, "
          f"staged {k['staged']}, pins {k['pins']} (skipped {k['pin_skips']}), capped {k['capped']}, writes "
          f"{k['write_tokens'] / 1e6:.1f} M tokens (skipped {k['write_skipped']} nodes, rewritten "
          f"{k['rewrite_tokens'] / 1e6:.1f} M), write holds {k['write_hold_on']}, backups refused {k['backup_refused']}")


def run(cfg: Config) -> Sim:
    sim = Sim(cfg).run()
    if cfg.out:
        sim.write_outputs(cfg.out)
    return sim


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="kvsim: one simulated replay")
    ap.add_argument("--arm", default="ec_full", help="hicache | jit_p50 | jit_exact | ec_full | ec_exact | unified")
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--rate", type=float, default=0.1)
    ap.add_argument("--host-gb", type=float, default=160.0)
    ap.add_argument("--duration", type=float, default=7200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="fa3", choices=sorted(BACKENDS))
    ap.add_argument("--fixes", default="on", choices=("on", "off"))
    ap.add_argument("--mirror", default="on", choices=("on", "off"))
    ap.add_argument("--switches", default="", help="unified arm: e.g. F1,F2,R1 or 'all'")
    ap.add_argument("--hints", default=None, choices=("none", "program", "dist", "exact"))
    ap.add_argument("--gap-scale", type=float, default=1.0)
    ap.add_argument("--gap-cap", type=float, default=1200.0)
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--gpu-tokens", type=int, default=668096)
    ap.add_argument("--ssd-tokens", type=int, default=24_690_000)
    ap.add_argument("--trace", default=kt.DEFAULT_TRACE)
    ap.add_argument("--cut", type=float, default=None)
    ap.add_argument("--noise", type=float, default=0.0, help="send each request up to this many seconds late (uniform)")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--d2h-s-per-token", type=float, default=2.8e-6,
                    help="mirror off: GPU seconds per token copied to host at GPU eviction (0 = an asynchronous copy)")
    a = ap.parse_args(argv)
    return Config(arm=a.arm, cap=a.cap, rate=a.rate, host_gb=a.host_gb, duration=a.duration, seed=a.seed, out=a.out,
                  backend=a.backend, fixes=a.fixes == "on", mirror=a.mirror == "on", switches=a.switches,
                  hints=a.hints, gap_scale=a.gap_scale, gap_cap=a.gap_cap, max_context=a.max_context,
                  gpu_tokens=a.gpu_tokens, ssd_tokens=a.ssd_tokens, trace=a.trace, cut=a.cut,
                  d2h_s_per_token=a.d2h_s_per_token, noise=a.noise, noise_seed=a.noise_seed)


def main(argv=None) -> int:
    cfg = parse_args(argv)
    t0 = time.process_time()
    sim = run(cfg)
    s = summary(sim)
    print_summary(s)
    print(f"  CPU {time.process_time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
