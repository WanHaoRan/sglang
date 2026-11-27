"""verify_math, part 3: the queue weights (math_fetch R11; math_order_price results 1, 2, 3, 7, 8).

Own simulators: Lindley recursion (open FCFS queue), a single-server queue with pass-over,
a closed loop with one FCFS server, and a pool-bound batch server (prefill serial, decode steps shared).
Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vo_queue.py > vo_queue.out
"""
from __future__ import annotations

import heapq
import math
import sys

import numpy as np

SEED = 4242_1126


# ---------------------------------------------------------------- open queue: mu
def open_mu():
    print("== Open FCFS queue: one more second of work on turn k raises summed waiting by B_k (turns after k in its busy period) ==")
    rng = np.random.default_rng([SEED, 1])
    for label, svc in (("M/D/1", "det"), ("M/M/1", "exp"), ("M/G/1 lognormal(sigma 1)", "ln")):
        for rho in (0.5, 0.8, 0.9):
            n = 600_000
            lam = rho  # mean service 1
            a = np.cumsum(rng.exponential(1.0 / lam, size=n))
            if svc == "det":
                s = np.ones(n)
            elif svc == "exp":
                s = rng.exponential(1.0, size=n)
            else:
                s = rng.lognormal(-0.5, 1.0, size=n)
            start = np.empty(n)
            t = 0.0
            for j in range(n):
                st = a[j] if a[j] > t else t
                start[j] = st
                t = st + s[j]
            comp = start + s
            newbp = np.empty(n, dtype=bool)
            newbp[0] = True
            newbp[1:] = a[1:] >= comp[:-1]
            bp = np.cumsum(newbp) - 1
            last = np.zeros(bp[-1] + 1, dtype=np.int64)
            np.maximum.at(last, bp, np.arange(n))
            Bk = last[bp] - np.arange(n)
            Q = np.searchsorted(a, start, side="left") - (np.arange(n) + 1)
            Q = np.maximum(Q, 0)
            pred = (Q + lam * s) / (1.0 - rho)
            use = slice(1000, n - 5000)
            slope = float(np.sum(pred[use] * Bk[use]) / np.sum(pred[use] ** 2))
            mb, mp, mq = Bk[use].mean(), pred[use].mean(), Q[use].mean()
            if svc == "exp":
                th = f"theory E[B] = rho/(1-rho)^2 = {rho / (1 - rho) ** 2:.2f}, E[Q] = rho^2/(1-rho) = {rho * rho / (1 - rho):.2f}"
            else:
                th = ""
            print(f"   {label}, rho = {rho}: mean B_k = {mb:.2f}, mean of (Q + lam s)/(1 - rho) = {mp:.2f}, regression slope through 0 = {slope:.3f}, mean Q at service start = {mq:.2f}; mu = 1 + B: {1 + mb:.2f} against the draft's 1 + Q = {1 + mq:.2f}. {th}")
            if svc == "det" and rho == 0.8:
                for qv in (0, 1, 2, 4):
                    m = Q[use] == qv
                    print(f"      M/D/1 at a service start with Q = {qv}: mean 1 + B_k = {1 + Bk[use][m].mean():.2f}; (1 + Q)/(1 - rho) = {(1 + qv) / (1 - rho):.2f}; 1 + Q = {1 + qv}")
            # finite differences on 150 turns
            if svc != "det":
                ks = rng.integers(2000, n - 20000, size=150)
                worst = 0.0
                delta = 1e-7
                for k in ks:
                    t2 = comp[k] + delta
                    extra = 0.0
                    j = k + 1
                    while j < n:
                        st = a[j] if a[j] > t2 else t2
                        d = st - start[j]
                        if d <= 0:
                            break
                        extra += d
                        t2 = st + s[j]
                        j += 1
                    worst = max(worst, abs(extra / delta - Bk[k]))
                print(f"      finite difference (delta = 1e-7) of the summed wait against B_k on 150 turns: worst gap {worst:.2e}")


def open_bound():
    print("\n== Open FCFS queue: base + sum_k (1 + B_k) x extra work_k is a lower bound on summed TTFT (convexity) ==")
    rng = np.random.default_rng([SEED, 9])

    def waits(a, s):
        t = 0.0
        tot = 0.0
        for j in range(len(a)):
            st = a[j] if a[j] > t else t
            tot += st - a[j]
            t = st + s[j]
        return tot

    viol = 0
    slack_min = float("inf")
    ratio = []
    for trial in range(300):
        n = 3000
        rho = float(rng.choice([0.5, 0.8, 0.95]))
        a = np.cumsum(rng.exponential(1.0 / rho, size=n))
        s = rng.exponential(1.0, size=n)
        start = np.empty(n)
        t = 0.0
        for j in range(n):
            st = a[j] if a[j] > t else t
            start[j] = st
            t = st + s[j]
        comp = start + s
        newbp = np.empty(n, dtype=bool)
        newbp[0] = True
        newbp[1:] = a[1:] >= comp[:-1]
        bp = np.cumsum(newbp) - 1
        last = np.zeros(bp[-1] + 1, dtype=np.int64)
        np.maximum.at(last, bp, np.arange(n))
        Bk = last[bp] - np.arange(n)
        extra = np.where(rng.random(n) < rng.choice([0.01, 0.1, 0.5]), rng.exponential(rng.choice([0.05, 1.0, 5.0]), size=n), 0.0)
        base = waits(a, s) + 0.0
        pert = waits(a, s + extra) + float(extra.sum())  # each turn's own extra prefill is in its TTFT
        lin = float(np.sum((1 + Bk) * extra))
        slack = (pert - base) - lin
        slack_min = min(slack_min, slack)
        if slack < -1e-6 * max(1.0, abs(lin)):
            viol += 1
        if lin > 0:
            ratio.append((pert - base) / lin)
    print(f"   300 random paths of 3,000 turns, extra work on 1 to 50 % of the turns, up to several service times: bound violated {viol} times; true rise over the linear bound: median {np.median(ratio):.2f}, max {np.max(ratio):.1f}")


# ---------------------------------------------------------------- pass-over: c
def passover_c():
    print("\n== Single server with pass-over: what a stall-second costs (the weight c) ==")
    rng = np.random.default_rng([SEED, 2])
    # L1: exact formula with a queue that never empties
    worst = 0.0
    for trial in range(2000):
        m_q = 40
        s = rng.exponential(1.0, size=m_q)
        S = rng.uniform(0.1, 12.0)
        # everyone arrived at 0 in index order; turn 0 not ready until S
        base_start = np.concatenate([[0.0], np.cumsum(s)[:-1]])
        t = 0.0
        ready0 = S
        done0 = False
        st = np.empty(m_q)
        j = 1
        jumpers = []
        while True:
            if not done0 and t >= ready0:
                st[0] = t
                t += s[0]
                done0 = True
                continue
            if j < m_q:
                st[j] = t
                if not done0:
                    jumpers.append(j)
                t += s[j]
                j += 1
            else:
                if not done0:
                    t = ready0
                    continue
                break
        delta = float(np.sum(st - base_start))
        formula = float(np.sum(s[jumpers]) - len(jumpers) * s[0])
        if j >= m_q and not jumpers:
            continue
        if len(jumpers) < m_q - 1:  # the queue did not run dry during the stall
            worst = max(worst, abs(delta - formula))
    print(f"   L1 (ready work behind for the whole stall): change of the summed wait = sum of the jumpers' work - (number of jumpers) x own work; worst error over 2,000 draws {worst:.2e}")

    def stall_cost(rho, svc, S, samples, rng):
        n = 400_000
        lam = rho
        a = np.cumsum(rng.exponential(1.0 / lam, size=n))
        s = np.ones(n) if svc == "det" else rng.exponential(1.0, size=n)
        start = np.empty(n)
        t = 0.0
        for j in range(n):
            st = a[j] if a[j] > t else t
            start[j] = st
            t = st + s[j]
        tot = 0.0
        tot2 = 0.0
        alone = 0
        idle_tot = 0.0
        ks = rng.integers(1000, n - 50_000, size=samples)
        for i in ks:
            D = start[i]
            ready = D + S
            t = D
            j = i + 1  # next index not yet started (others are served in index order)
            done_i = False
            delta = 0.0
            if not (j < n and a[j] <= D):
                alone += 1
            idle = 0.0
            while True:
                if not done_i and t >= ready:
                    delta += t - start[i]
                    t += s[i]
                    done_i = True
                    continue
                if j < n and a[j] <= t:
                    delta += t - start[j]
                    t += s[j]
                    j += 1
                    continue
                # nobody ready now
                nxt = a[j] if j < n else float("inf")
                if not done_i:
                    t2 = min(ready, nxt)
                    idle += t2 - t
                    t = t2
                    continue
                # i done and nobody waiting: the perturbed system is empty; the base one is too
                break
            # customers from j on start at the same times in both systems
            tot += delta / S
            tot2 += (delta / S) ** 2
            idle_tot += idle / S
        mean = tot / samples
        se = math.sqrt(max(tot2 / samples - mean * mean, 0.0) / samples)
        return mean, se, alone / samples, idle_tot / samples

    for svc, name in (("det", "M/D/1"), ("exp", "M/M/1")):
        for rho in (0.3, 0.6, 0.8):
            mean, se, alone, idle = stall_cost(rho, svc, 0.05, 60_000 if svc == "det" else 150_000, rng)
            lst = math.exp(-rho) if svc == "det" else 1.0 / (1.0 + rho)
            print(f"   {name}, rho = {rho}, stall of 0.05 service times: cost per stall-second {mean:.3f} (se {se:.3f}); theory 1/E[exp(-lam s)] = {1 / lst:.3f}; P(no other turn waiting when due) = {alone:.3f} (theory {(1 - rho) / lst:.3f}); share of the stall with the server idle {idle:.3f}; price of an idle second 1/(1 - rho) = {1 / (1 - rho):.2f}")
    mean, se, alone, idle = stall_cost(0.8, "exp", 5.0, 60_000, rng)
    print(f"   M/M/1, rho = 0.8, long stall (5 service times): cost per stall-second {mean:.3f} (se {se:.3f}); P(no other turn waiting when due) = {alone:.3f}")


# ---------------------------------------------------------------- closed loop, one server
def closed_single():
    print("\n== Closed loop, one FCFS server, saturated: one injected second raises summed response time by N ==")
    rng = np.random.default_rng([SEED, 3])

    def run(N, M, pause, svc, inject_at=None, delta=0.0):
        nxt = [(pause[u][0], u) for u in range(N)]
        heapq.heapify(nxt)
        k = [0] * N
        resp = np.zeros((N, M))
        t = 0.0
        injected = False
        idle = 0.0
        while nxt:
            arr, u = heapq.heappop(nxt)
            if arr > t:
                idle += arr - t
                t = arr
            s = svc[u][k[u]]
            if inject_at is not None and not injected and t + s > inject_at:
                s += delta
                injected = True
            t += s
            resp[u, k[u]] = t - arr
            k[u] += 1
            if k[u] < M:
                heapq.heappush(nxt, (t + pause[u][k[u]], u))
        return resp, idle

    for N in (8, 32, 128):
        M = 400
        pause = rng.exponential(0.6 * N, size=(N, M + 1)) * rng.lognormal(0, 0.5, size=(N, M + 1))  # mean pause < N x mean work: saturated
        svc = rng.lognormal(-0.125, 0.5, size=(N, M + 1))
        base, idle = run(N, M, pause, svc)
        slopes = []
        for t0 in (0.15, 0.3, 0.45, 0.6):
            T0 = t0 * M * N * 1.0
            pert, _ = run(N, M, pause, svc, inject_at=T0, delta=1e-6)
            slopes.append((pert.sum() - base.sum()) / 1e-6)
        fin = []
        for d in (0.5, 2.0):
            vals = []
            for rep in range(40):
                T0 = rng.uniform(0.1, 0.6) * M * N
                pert, _ = run(N, M, pause, svc, inject_at=T0, delta=d)
                vals.append((pert.sum() - base.sum()) / d)
            fin.append((d, float(np.mean(vals)), float(np.median(vals)), float(np.std(vals) / math.sqrt(len(vals)))))
        # sustained: every request gets eps more work; mean response over the middle half of each user's requests
        eps = 0.05
        more, idle2 = run(N, M, pause, svc + eps)
        mid = slice(M // 4, 3 * M // 4)
        print(f"   N = {N}: every request {eps} s longer: mean response time (middle half of each user's requests) {base[:, mid].mean():.3f} -> {more[:, mid].mean():.3f} s, slope {(more[:, mid].mean() - base[:, mid].mean()) / eps:.1f} per second of work (N = {N})")
        print(f"   N = {N}: idle time in the run {idle:.1f} s (start-up and run-out only); slope of summed response time for delta = 1e-6 at four times: {[round(v, 3) for v in slopes]}; finite injections: " + "; ".join(f"{d} s -> mean {m:.1f} (se {se:.1f}), median {md:.1f}" for d, m, md, se in fin))


# ---------------------------------------------------------------- closed loop, pool-bound batch
class Batch:
    """N users; at most B running; prefill serial (decode paused); a decode step with n running takes a + b n."""

    def __init__(self, N, B, seed, mean_pause=20.0, n_out_mean=300, prefill=0.11, a=3.5e-3, b=0.99e-3, M=None):
        self.N, self.B, self.a, self.b, self.prefill = N, B, a, b, prefill
        r = np.random.default_rng(seed)
        self.M = M
        m = M or 4000
        # pauses: 90 % short (lognormal median 1.7 s), 10 % long; scaled to the requested mean
        short = r.lognormal(math.log(1.7), 1.3, size=(N, m))
        long_ = r.lognormal(math.log(60.0), 1.2, size=(N, m))
        raw = np.where(r.random((N, m)) < 0.9, short, long_)
        raw = np.minimum(raw, 1200.0)
        self.pause = raw * (mean_pause / raw.mean())
        self.nout = np.maximum(1, r.geometric(1.0 / n_out_mean, size=(N, m)))
        self.lag_u = r.random((N, m))
        self.lag_e = r.exponential(1.0, size=(N, m))
        self.first = r.uniform(0.0, 5.0, size=N)

    def run(self, T_end, warm=0.0, reader_share=0.0, lag_mean=0.0, extra_prefill=0.0, inject_at=None, delta=0.0, reader_extra=0.0):
        N, B = self.N, self.B
        arrivals = [(self.first[u], u) for u in range(N)]
        heapq.heapify(arrivals)
        k = [0] * N
        waiting = []  # (arrival, user, ready), in arrival order
        running = []  # [user, tokens_left, arrival]
        t = 0.0
        ttft = {}
        resp = {}
        injected = False
        b_dec_at_inject = None
        stats = dict(done=0, sum_ttft=0.0, sum_resp=0.0, n_ttft=0, pausing_time=0.0, running_time=0.0, empty_slot_time=0.0, t0=None, sum_ttft_r=0.0, n_r=0, sum_ttft_o=0.0, n_o=0, sum_pre=0.0)
        last_t = 0.0

        def account(t_new):
            nonlocal last_t
            if t_new > warm:
                lo = max(last_t, warm)
                dt = t_new - lo
                if dt > 0:
                    stats["pausing_time"] += dt * len(arrivals)
                    stats["running_time"] += dt * len(running)
                    if waiting:
                        stats["empty_slot_time"] += dt * (B - len(running))
            last_t = t_new

        while t < T_end:
            while arrivals and arrivals[0][0] <= t:
                arr, u = heapq.heappop(arrivals)
                kk = k[u]
                is_reader = self.lag_u[u, kk] < reader_share
                lag = lag_mean * self.lag_e[u, kk] if is_reader else 0.0
                waiting.append((arr, u, arr + lag, is_reader))
            pick = None
            if len(running) < B:
                for idx, w in enumerate(waiting):
                    if w[2] <= t:
                        pick = idx
                        break
            if pick is not None:
                arr, u, _, is_reader = waiting.pop(pick)
                p = self.prefill + extra_prefill + (reader_extra if is_reader else 0.0)
                if inject_at is not None and not injected and t >= inject_at:
                    b_dec_at_inject = len(running)
                    p += delta
                    injected = True
                running.append([u, int(self.nout[u, k[u]]), arr])
                account(t + p)
                t += p
                ttft[(u, k[u])] = t - arr
                if t > warm:
                    stats["sum_ttft"] += t - arr
                    stats["n_ttft"] += 1
                    stats["sum_pre"] += p
                    if is_reader:
                        stats["sum_ttft_r"] += t - arr
                        stats["n_r"] += 1
                    else:
                        stats["sum_ttft_o"] += t - arr
                        stats["n_o"] += 1
                continue
            if running:
                step = self.a + self.b * len(running)
                account(t + step)
                t += step
                keep = []
                for item in running:
                    item[1] -= 1
                    if item[1] <= 0:
                        u = item[0]
                        resp[(u, k[u])] = t - item[2]
                        if t > warm:
                            stats["done"] += 1
                            stats["sum_resp"] += t - item[2]
                        k[u] += 1
                        if self.M is None or k[u] < self.M:
                            heapq.heappush(arrivals, (t + self.pause[u, k[u] % self.pause.shape[1]], u))
                    else:
                        keep.append(item)
                running = keep
                continue
            # nothing running and nothing ready
            cands = [w[2] for w in waiting] + ([arrivals[0][0]] if arrivals else [])
            if not cands:
                break
            t_new = max(min(cands), t + 1e-9)
            account(t_new)
            t = t_new
        stats["T"] = t - warm
        stats["b_dec"] = b_dec_at_inject
        return stats, ttft, resp


def closed_batch():
    print("\n== Closed loop, pool-bound batch server (N = 128, B = 20, step 3.5 + 0.99 n ms, prefill 0.11 s, 300 output tokens) ==")
    N, B = 128, 20
    sim = Batch(N, B, seed=[SEED, 4])
    T, warm = 9000.0, 1500.0
    g0 = sim.prefill + (sim.a + sim.b * B) * 300 / B

    def report(tag, st):
        X = st["done"] / st["T"]
        Z = st["pausing_time"] / st["T"] / X
        Drun = st["running_time"] / st["T"] / X
        mt = st["sum_ttft"] / st["n_ttft"]
        mr = st["sum_resp"] / st["done"]
        tpre = st["sum_pre"] / st["n_ttft"]
        ident = N / X - Z - Drun + tpre
        full = (N - B) / X - Z + tpre
        empty = st["empty_slot_time"] / st["T"]
        extra = ""
        if st["n_r"]:
            extra = f"; readers {st['sum_ttft_r'] / st['n_r']:.1f} s, others {st['sum_ttft_o'] / st['n_o']:.1f} s"
        print(f"   {tag}: X = {X:.4f}/s (1/X = {1 / X:.4f} s, g0 = {g0:.4f}); mean TTFT {mt:.2f} s; N/X - Z - D_run + t_pre = {ident:.2f}; (N - B)/X - Z + t_pre = {full:.2f}; N/X = {N / X:.2f} = R + Z = {mr + Z:.2f}; running {st['running_time'] / st['T']:.2f}, slots empty while a turn waits {empty:.3f}{extra}")
        return X, mt, Z, mr, tpre

    X0, t0_, Z0, r0, tp0 = report("no reads late", sim.run(T, warm)[0])
    floor = (N - B) * g0 - Z0 + tp0
    print(f"      floor (N - B) g0 - Z + own prefill = {floor:.2f} s (mean TTFT {t0_:.2f} s)")
    for lag in (10.0, 40.0):
        report(f"25 % of turns ready {lag:.0f} s (mean) after arrival", sim.run(T, warm, reader_share=0.25, lag_mean=lag)[0])
    report("25 % ready 150 s after arrival (pool starves)", sim.run(T, warm, reader_share=0.25, lag_mean=150.0)[0])
    for e in (0.05, 0.3):
        Xe, te, Ze, re_, tpe = report(f"{e} s more prefill on every request", sim.run(T, warm, extra_prefill=e)[0])
        print(f"      d(TTFT)/d(work) = {(te - t0_) / e:.1f} (N - B = {N - B}); d(response)/d(work) = {(re_ - r0) / e:.1f} (N = {N}); d(response - TTFT)/d(work) = {((re_ - te) - (r0 - t0_)) / e:.1f} (B = {B})")
    # readers carrying 3 times the prefill, on time or late: same work per request -> same mean
    a1 = report("readers (25 %) with 0.22 s more prefill, on time", sim.run(T, warm, reader_share=0.25, lag_mean=0.0, reader_extra=0.22)[0])
    a2 = report("readers (25 %) with 0.22 s more prefill, 20 s late", sim.run(T, warm, reader_share=0.25, lag_mean=20.0, reader_extra=0.22)[0])
    # single injection of 1e-6 s of prefill: fixed set of requests (the first M per user)
    M = 60
    simM = Batch(N, B, seed=[SEED, 5], M=M)
    base_st, base_ttft, base_resp = simM.run(1e9)
    keys = sorted(base_resp.keys())
    out = []
    for frac in (0.08, 0.16, 0.24, 0.32):
        t_inj = frac * base_st["T"]
        st, tt, rr = simM.run(1e9, inject_at=t_inj, delta=1e-6)
        d_resp = sum(rr[k_] - base_resp[k_] for k_ in keys) / 1e-6
        d_ttft = sum(tt[k_] - base_ttft[k_] for k_ in keys) / 1e-6
        out.append((round(d_resp, 2), round(d_ttft, 2), st["b_dec"]))
    print(f"   one injection of 1e-6 s of prefill ({M} requests per user, all counted): (response slope, TTFT slope, turns decoding at the injection) = {out}; predicted (N, N - turns decoding)")
    # saturation point
    print("   where the pool fills: slots used / B and cost of late reads by users in session")
    for n_users in (40, 56, 64, 72, 96):
        s2 = Batch(n_users, B, seed=[SEED, 6, n_users])
        b0 = s2.run(6000.0, 1000.0)[0]
        b1 = s2.run(6000.0, 1000.0, reader_share=0.25, lag_mean=10.0)[0]
        X = b0["done"] / b0["T"]
        ttft0 = b0["sum_ttft"] / b0["n_ttft"]
        ttft1 = b1["sum_ttft"] / b1["n_ttft"]
        exposed = 0.25 * 10.0
        print(f"      N = {n_users}: slots used {b0['running_time'] / b0['T'] / B:.3f}; mean TTFT {ttft0:.2f} -> {ttft1:.2f} s with 25 % of turns 10 s late: {(ttft1 - ttft0) / exposed:.2f} per second of lag; slots empty while a turn waits {b1['empty_slot_time'] / b1['T']:.3f}")


# ---------------------------------------------------------------- closed-loop counterexample to the bound
def closed_counterexample():
    print("\n== Closed loop: more work can lower summed response time (three users, one FCFS server) ==")

    def run(extra):
        # user A: 5 s (+extra), then 0.9 s after its reply a 10 s request; B: 1 s at 0.1 s, then 1 s 0.001 s after its reply; C: 1 s at 5.95 s
        prog = {"A": [(0.0, 5.0 + extra), (0.9, 10.0)], "B": [(0.1, 1.0), (0.001, 1.0)], "C": [(5.95, 1.0)]}
        pend = [(prog[u][0][0], u, 0) for u in prog]
        heapq.heapify(pend)
        t = 0.0
        tot = 0.0
        log = []
        while pend:
            arr, u, i = heapq.heappop(pend)
            t = max(t, arr) + prog[u][i][1]
            tot += t - arr
            log.append((u, i, round(arr, 3), round(t, 3)))
            if i + 1 < len(prog[u]):
                heapq.heappush(pend, (t + prog[u][i + 1][0], u, i + 1))
        return tot, log

    b, lb = run(0.0)
    p, lp = run(0.2)
    print(f"   summed response time: {b:.3f} s; with 0.2 s more work on A's first request: {p:.3f} s (change {p - b:+.3f} s; 'mu = N = 3' predicts +0.600 s)")
    print(f"   order without: {[(u, i) for u, i, _, _ in lb]}; with: {[(u, i) for u, i, _, _ in lp]}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["open_mu", "open_bound", "passover_c", "closed_single", "closed_batch", "closed_counterexample"]
    for w in which:
        globals()[w]()
