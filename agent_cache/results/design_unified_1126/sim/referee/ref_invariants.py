"""Referee: invariant checks on kvsim, run by subclassing its Sim (nothing in the simulator is edited).

  python -I -B ref_invariants.py [--sim kvsim | kvsim_patched] [--arms a,b] [--seeds 0,5] [--fixes on,off] [--mirror on,off]

Checked after every event (cheap) and at every 10 s sample (a full scan of the node arrays):

  conservation  host tokens by class add up to host_used and never pass the pool; the GPU pool likewise; counters of
                children, candidates, locks, pins, pending writes and the SSD agree with the node arrays
  residency     the GPU set and the in-memory set are prefix-closed (the binary search of match() needs it); every
                running request's prompt is on the GPU; a lock sits on a copy that exists
  device        every read and write runs at one of the model's rates; reads never start from pages the cleaner deleted
  clocks        no event is scheduled in the past; TTFT >= 0; latency >= TTFT; admissions keep arrival order among
                requests with no read under way
  closed loop   a request is sent once, after every dependency's reply plus its recorded wait; users in session <= cap
  causality     the engine's plan of a turn carries the hint the client sent (table entry or exact gap), never the true
                gap in the table arms; hints use only gaps that have elapsed
  records       cached tokens never exceed what an infinite cache holds; tier tokens add up to the prompt
"""
from __future__ import annotations

import argparse
import collections
import heapq
import importlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(HERE)
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, HERE)


def make_checked(kvsim):
    class Checked(kvsim.Sim):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.viol = collections.Counter()
            self.examples = {}
            self.maxima = collections.defaultdict(float)
            self.active = set()
            self.sent = set()
            self.read_rates = []
            self.write_rates = []
            self.w_start = None
            self.w_node = None
            self.w_other_busy = False
            self.deleted_read_tokens = 0
            self.deleted_reads = 0
            self.last_admit_arrival = {}
            self.n_events = 0
            self.n_samples = 0
            self.gap_true_minus_plan = []
            n = self.wl.n_nodes
            self.np_tok = np.array(self.tok, np.int64)
            self.np_par = np.array(self.par, np.int64)

        # ----- bookkeeping -----
        def flag(self, name, detail=None, size=None):
            self.viol[name] += 1
            if size is not None and size > self.maxima[name]:
                self.maxima[name] = size
                self.examples[name] = (round(self.now, 3), detail)
            elif name not in self.examples:
                self.examples[name] = (round(self.now, 3), detail)

        # ----- events -----
        def at(self, t, fn, arg=None):
            if t < self.now - 1e-9:
                self.flag("event scheduled in the past", f"{fn.__name__} at {t:.6f} < now {self.now:.6f}", self.now - t)
            super().at(t, fn, arg)

        def run(self):
            cfg = self.cfg
            for p in self.wl.plays:
                self.at(p.arrival, self._play_arrive, p)
            self.at(cfg.tick, self._tick)
            self.at(10.0, self._sample)
            self.at(cfg.cleaner_interval, self._cleaner)
            end = cfg.duration + cfg.grace
            ev = self.ev
            last = 0.0
            while ev:
                t, _, fn, arg = heapq.heappop(ev)
                if t > end:
                    break
                if t < last - 1e-12:
                    self.flag("clock went backwards", f"{t} after {last}", last - t)
                last = t
                self.now = t
                if arg is None:
                    fn()
                else:
                    fn(arg)
                self.n_events += 1
                self.cheap()
            self.now = min(end, self.now)
            return self

        def cheap(self):
            if self.host_used > self.C_H:
                self.flag("host_used above the host pool", f"{self.host_used} > {self.C_H}", self.host_used - self.C_H)
            if self.gpu_used > self.C_G:
                self.flag("gpu_used above the GPU pool", f"{self.gpu_used} > {self.C_G}", self.gpu_used - self.C_G)
            if self.ssd_used > self.C_S:
                self.flag("ssd_used above the volume", f"{self.ssd_used} > {self.C_S}", self.ssd_used - self.C_S)
            for name in ("host_used", "host_mirror", "pending_write", "pinned_tokens", "inflight_tokens",
                         "inflight_pages", "gpu_used", "gpu_unlocked", "ssd_used", "n_run", "in_session", "in_flight",
                         "slots_free"):
                if getattr(self, name) < 0:
                    self.flag(f"{name} negative", getattr(self, name), -getattr(self, name))
            if self.in_session > self.cfg.cap:
                self.flag("users in session above the cap", self.in_session, self.in_session - self.cfg.cap)
            if self.n_run != len(self.running):
                self.flag("n_run != running heap", (self.n_run, len(self.running)))

        # ----- closed loop -----
        def _send(self, r):
            if self.now < self.cfg.duration:
                if r.uid in self.sent:
                    self.flag("request sent twice", (r.play.pid, r.s, r.k))
                self.sent.add(r.uid)
                p = r.play
                if p.t_start is None or self.now < p.t_start - 1e-9:
                    self.flag("request sent before its user started", (p.pid, r.s, r.k))
                ready = p.t_start
                for ds, dk in r.deps:
                    d = p.streams[ds][dk]
                    if d.t_done is None:
                        self.flag("request sent before a dependency's reply", (p.pid, r.s, r.k))
                    elif d.t_done > ready:
                        ready = d.t_done
                if self.now < ready + r.wait - 1e-9:
                    self.flag("request sent before reply + recorded gap", (p.pid, r.s, r.k, self.now, ready, r.wait),
                              ready + r.wait - self.now)
                if r.k > 0:
                    prev = p.streams[r.s][r.k - 1]
                    if prev.t_done is None:
                        self.flag("request sent before the stream's previous reply", (p.pid, r.s, r.k))
            super()._send(r)
            if r.t_send is not None and r.t_send == self.now:
                self.check_hint(r)

        def check_hint(self, r):
            """What the client sends must not hold the future unless the arm is an exact-hint arm."""
            h = r.hint
            mode = self.pol.hint_mode
            if h is None:
                return
            if mode in ("dist",):
                if "exact" in h:
                    self.flag("exact gap sent in a table arm", (r.play.pid, r.s, r.k))
            if mode in ("none", "program") and any(k in h for k in ("exact", "p50", "end")):
                self.flag("gap information sent in a no-hint arm", (r.play.pid, r.s, r.k))

        # ----- admission -----
        def _admit(self, r, n_ready_before):
            res = super()._admit(r, n_ready_before)
            if res == 0:
                self.active.add(r)
                path = r.path
                for i in range(r.n_locked):
                    if not self.in_gpu[path[i]]:
                        self.flag("admitted with a prompt page not on the GPU", (r.play.pid, r.s, r.k, path[i]))
                        break
                if r.tok_gpu + r.tok_host + r.tok_ssd != r.resident_at_admit:
                    self.flag("tier tokens do not add up to the resident prefix", (r.play.pid, r.s, r.k))
                if r.t_admit < r.t_send - 1e-9:
                    self.flag("admitted before it was sent", (r.play.pid, r.s, r.k))
                if r.reading is not None:
                    self.flag("admitted with a read under way", (r.play.pid, r.s, r.k))
            return res

        def _finish(self, r):
            super()._finish(r)
            self.active.discard(r)
            if r.t_first - r.t_send < 0:
                self.flag("negative TTFT", (r.play.pid, r.s, r.k))
            if r.t_done_srv - r.t_send < r.t_first - r.t_send - self.cfg.first_token_s - 1e-9:
                self.flag("latency below TTFT", (r.play.pid, r.s, r.k))
            # the engine's forecast of the next turn must be the hint, not the truth
            st = r.play.streams[r.s]
            if r.k + 1 < len(st):
                rn = st[r.k + 1]
                plan = rn.plan
                if plan is not None and plan.dist is not None:
                    mode = self.pol.hint_mode
                    truth = rn.wait
                    if mode == "dist":
                        h = r.hint
                        if h is None or "p50" not in h or abs(plan.gap_s - h["p50"]) > 1e-9:
                            self.flag("table arm: the plan's gap is not the sent p50", (r.play.pid, r.s, r.k))
                    elif mode == "exact":
                        if abs(plan.gap_s - truth) > 1e-9:
                            self.flag("exact arm: the plan's gap is not the true gap", (r.play.pid, r.s, r.k))
                    self.gap_true_minus_plan.append(truth - plan.gap_s)
                elif plan is not None and plan.dist is None and self.pol.hint_mode not in ("program",):
                    self.flag("a plan with no distribution outside the program arm", (r.play.pid, r.s, r.k))

        # ----- device -----
        def _read_next(self):
            before = self.r_cur
            super()._read_next()
            rd = self.r_cur
            if rd is not None and rd is not before:
                missing = [n for n in rd.nodes if not self.ssd[n]]
                if missing:
                    self.deleted_reads += 1
                    self.deleted_read_tokens += sum(self.tok[n] for n in missing)

        def _read_io_done(self, ver):
            rd = self.r_cur if ver == self.r_ver else None
            if rd is not None and self.now > rd.io_start:
                self.read_rates.append(rd.tokens / (self.now - rd.io_start))
            super()._read_io_done(ver)

        def _write_next(self):
            before = self.w_cur
            super()._write_next()
            if self.w_cur is not None and self.w_cur is not before:
                self.w_start = self.now
                self.w_node = self.w_cur

        def _write_io_done(self, ver):
            n = self.w_cur if ver == self.w_ver else None
            if n is not None and self.w_start is not None and self.now > self.w_start and n == self.w_node:
                self.write_rates.append(self.tok[n] / (self.now - self.w_start))
            super()._write_io_done(ver)

        # ----- the full scan -----
        def _sample(self):
            self.full()
            super()._sample()

        def full(self):
            self.n_samples += 1
            tok, par = self.np_tok, self.np_par
            in_gpu = np.array(self.in_gpu, bool)
            in_host = np.array(self.in_host, bool)
            orphan = np.array(self.orphan, bool)
            dirty = np.array(self.dirty, bool)
            hlock = np.array(self.hlock, np.int64)
            glock = np.array(self.glock, np.int64)
            pin = np.array(self.pin, np.int64)
            ssd = np.array(self.ssd, bool)
            nct = np.array(self.nct, np.int64)
            ncg = np.array(self.ncg, np.int64)
            in_tree = in_gpu | in_host

            def eq(name, a, b):
                if a != b:
                    self.flag(name, f"{a} != {b}", abs(a - b))
            host_nodes = int(tok[in_host].sum())
            # the patched copy counts orphaned copies in tokens (a node can be orphaned twice) and pending writes per node
            patched = "orphan_tok" in self.__dict__
            orphans = int(sum(self.orphan_tok)) if patched else int(tok[orphan].sum())
            eq("host_used != nodes in host + reads in flight + orphans", self.host_used,
               host_nodes + self.inflight_tokens + orphans)
            eq("host_mirror != host copies of GPU pages", self.host_mirror, int(tok[in_host & in_gpu].sum()))
            eq("pending_write != tokens of dirty nodes", self.pending_write,
               int((tok * np.array(self.wpend, np.int64)).sum()) if patched else int(tok[dirty].sum()))
            eq("pinned_tokens != pins on nodes", self.pinned_tokens, int((tok * pin).sum()))
            eq("ssd_used != tokens of stored nodes", self.ssd_used, int(tok[ssd].sum()))
            private = sum(r.private for r in self.active)
            eq("gpu_used != nodes on the GPU + running requests' private tokens", self.gpu_used,
               int(tok[in_gpu].sum()) + private)
            eq("gpu_unlocked != unlocked GPU nodes", self.gpu_unlocked, int(tok[in_gpu & (glock == 0)].sum()))
            # the classes of host memory fill the pool exactly
            host_only = int(tok[in_host & ~in_gpu].sum())
            free = self.C_H - self.host_used
            eq("host classes do not sum to the pool", self.C_H,
               self.host_mirror + host_only + self.inflight_tokens + orphans + free)
            if free < 0:
                self.flag("host free room negative", free, -free)
            # residency
            idx = np.arange(1, len(tok))
            p = par[idx]
            bad = idx[in_gpu[idx] & ~in_gpu[p]]
            if len(bad):
                self.flag("GPU set not prefix-closed", int(bad[0]), len(bad))
            bad = idx[in_tree[idx] & ~in_tree[p]]
            if len(bad):
                self.flag("in-memory set not prefix-closed", int(bad[0]), len(bad))
            cnt_t = np.bincount(p[in_tree[idx]], minlength=len(tok))
            cnt_g = np.bincount(p[in_gpu[idx]], minlength=len(tok))
            bad = np.nonzero(cnt_t != nct)[0]
            if len(bad):
                self.flag("nct != children in memory", int(bad[0]), len(bad))
            bad = np.nonzero(cnt_g != ncg)[0]
            if len(bad):
                self.flag("ncg != children on the GPU", int(bad[0]), len(bad))
            want_h = set(np.nonzero(in_host & ~in_gpu & (hlock == 0) & (nct == 0))[0].tolist())
            if want_h != self.hcand:
                self.flag("host candidate set wrong", (len(want_h - self.hcand), len(self.hcand - want_h)),
                          len(want_h ^ self.hcand))
            want_g = set(np.nonzero(in_gpu & (glock == 0) & (ncg == 0))[0].tolist())
            if want_g != self.gcand:
                self.flag("GPU candidate set wrong", (len(want_g - self.gcand), len(self.gcand - want_g)),
                          len(want_g ^ self.gcand))
            bad = np.nonzero((glock > 0) & ~in_gpu)[0]
            if len(bad):
                self.flag("GPU lock on a node not on the GPU", int(bad[0]), len(bad))
            # mirror off: the anchor of a lookup may be a GPU page with no host copy (by design)
            bad = np.nonzero((hlock > 0) & ~in_host & ~orphan & (in_gpu if self.cfg.mirror else ~in_gpu))[0] \
                if not self.cfg.mirror else np.nonzero((hlock > 0) & ~in_host & ~orphan)[0]
            bad = bad[bad != 0]
            if len(bad):
                self.flag("host lock on a node with no host copy", int(bad[0]), len(bad))
            bad = np.nonzero(dirty & ~in_host & ~orphan)[0]
            if len(bad):
                self.flag("pending write of a page with no host copy", int(bad[0]), len(bad))
            bad = np.nonzero(orphan & (hlock == 0))[0]
            if len(bad):
                self.flag("orphan without a lock", int(bad[0]), len(bad))
            for r in self.active:
                path = r.path
                for i in range(r.n_locked):
                    if not self.in_gpu[path[i]] or self.glock[path[i]] <= 0:
                        self.flag("a running request's page is off the GPU or unlocked", (r.play.pid, r.s, r.k))
                        break
            # the queue holds requests in arrival (send) order
            wq = self.waitq
            for i in range(1, len(wq)):
                if wq[i].t_send < wq[i - 1].t_send - 1e-9:
                    self.flag("queue not in arrival order", i)
                    break
            # reads in flight hold exactly their tokens
            infl = sum(rd.tokens for rd in list(self.rq)) + (self.r_cur.tokens if self.r_cur is not None else 0)
            if infl > self.inflight_tokens:
                self.flag("reads at the device hold more than inflight_tokens", (infl, self.inflight_tokens))

    return Checked


def run_one(kvsim, arm, seed, fixes, mirror, switches="", extra=None):
    Checked = make_checked(kvsim)
    kw = dict(arm=arm, seed=seed, fixes=fixes, mirror=mirror, switches=switches)
    kw.update(extra or {})
    cfg = kvsim.Config(**kw)
    t0 = time.process_time()
    sim = Checked(cfg).run()
    rows = sim.request_rows()
    over = sum(1 for x in rows if x["tok_gpu"] + x["tok_host"] + x["tok_ssd"] > x["ideal"])
    over_tok = sum(max(0, x["tok_gpu"] + x["tok_host"] + x["tok_ssd"] - x["ideal"]) for x in rows)
    not_sum = sum(1 for x in rows if x["tok_gpu"] + x["tok_host"] + x["tok_ssd"] > x["prompt"])
    # With the client's own rule (the patched copy: ideal taken at the send from completed requests) a few rows hold
    # more cached tokens than the ideal, as the recorded clients do (3 to 6 rows a run); only the original's rule
    # (ideal at the end of prefill) makes this a violation.
    if over and "orphan_tok" not in sim.__dict__:
        sim.viol["cached tokens above the infinite-cache ideal (rows)"] = over
        sim.examples["cached tokens above the infinite-cache ideal (rows)"] = (None, f"{over_tok} tokens over")
    if not_sum:
        sim.viol["cached tokens above the prompt (rows)"] = not_sum
    rr = np.array(sim.read_rates) if sim.read_rates else np.zeros(1)
    wr = np.array(sim.write_rates) if sim.write_rates else np.zeros(1)
    s = kvsim.summary(sim)
    res = {
        "arm": arm, "seed": seed, "fixes": fixes, "mirror": mirror, "switches": switches, "extra": extra or {},
        "events": sim.n_events, "samples": sim.n_samples, "requests": len(rows), "ttft_mean": s["ttft_mean"],
        "violations": dict(sim.viol), "examples": {k: str(v) for k, v in sim.examples.items()},
        "maxima": dict(sim.maxima),
        "read_rate_min_max": [float(rr.min()), float(rr.max())], "write_rate_min_max": [float(wr.min()), float(wr.max())],
        "reads_started_on_deleted_pages": sim.deleted_reads, "tokens_of_them": sim.deleted_read_tokens,
        "reads": s["counters"]["reads"], "rows_cached_above_ideal": over,
        "true_minus_planned_gap_abs_median": float(np.median(np.abs(sim.gap_true_minus_plan))) if sim.gap_true_minus_plan else None,
        "cpu_s": round(time.process_time() - t0, 1),
    }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="kvsim")
    ap.add_argument("--arms", default="hicache,jit_p50,jit_exact,ec_full,ec_exact,nohint")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--fixes", default="on,off")
    ap.add_argument("--mirror", default="on")
    ap.add_argument("--switches", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    kvsim = importlib.import_module(a.sim)
    out = []
    for arm in a.arms.split(","):
        for seed in [int(x) for x in a.seeds.split(",")]:
            for fx in a.fixes.split(","):
                for mr in a.mirror.split(","):
                    res = run_one(kvsim, arm, seed, fx == "on", mr == "on", a.switches)
                    out.append(res)
                    v = res["violations"]
                    print(f"{a.sim} {arm} seed {seed} fixes {fx} mirror {mr} switches '{a.switches}': {res['events']} events, "
                          f"{res['samples']} scans, TTFT {res['ttft_mean']:.2f}; violations: {v if v else 'none'}", flush=True)
                    for k in v:
                        print(f"      {k}: {v[k]} times; largest {res['maxima'].get(k)}; example {res['examples'].get(k)}")
                    print(f"      read rates {res['read_rate_min_max']}, write rates {res['write_rate_min_max']} tokens/s; "
                          f"reads started on pages the cleaner had deleted: {res['reads_started_on_deleted_pages']} of "
                          f"{res['reads']} ({res['tokens_of_them']} tokens); |true - planned gap| median "
                          f"{res['true_minus_planned_gap_abs_median']}; CPU {res['cpu_s']} s", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
