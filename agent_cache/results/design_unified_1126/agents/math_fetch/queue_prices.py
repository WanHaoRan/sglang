"""Check 5 (DESIGN.md section 1.5): the two weights mu and c in toy queues with identical requests.

Q1  open queue (M/D/1, first come first served): summed extra wait per extra second of GPU work
    injected at a service start, against 1 + Q and (1 + Q) / (1 - rho).
Q2  one server that passes over a turn whose read has not landed: summed extra wait per second
    of stall of one turn (the true c), against P(no other ready turn when the turn is due).
Q3  closed loop (N users, k slots, a prefill pass runs alone, decode steps advance all running
    requests together): summed extra TTFT per extra second of prefill, against N.

These are toys, not the engine: they check the arithmetic of the draft's two sentences.
"""

import heapq
import math

import numpy as np


def q1():
    print("== Q1 open M/D/1 queue: cost of one more second of GPU work at a service start ==")
    print("   rho    Q at the injection: mean extra summed wait per second (own second included) | 1 + Q | (1 + Q)/(1 - rho)")
    rng = np.random.default_rng(1)
    tau = 1.0
    delta = 1e-3
    for rho in [0.5, 0.8, 0.9]:
        n = 400_000
        A = np.cumsum(rng.exponential(tau / rho, n))
        start = np.empty(n)
        t = 0.0
        for i in range(n):
            s = A[i] if A[i] > t else t
            start[i] = s
            t = s + tau
        Qb = np.searchsorted(A, start, side="right") - 1 - np.arange(n)
        picks = rng.integers(1000, n - 5000, 20000)
        by = {}
        for i in picks:
            t = start[i] + tau + delta
            j = i + 1
            dsum = 0.0
            while j < n:
                s = A[j] if A[j] > t else t
                d = s - start[j]
                if d <= 1e-13:
                    break
                dsum += d
                t = s + tau
                j += 1
            by.setdefault(int(Qb[i]), []).append(1.0 + dsum / delta)
        cells = []
        for Q in [0, 1, 2, 4, 8]:
            if Q in by and len(by[Q]) >= 30:
                v = np.array(by[Q])
                cells.append(f"Q={Q}: {v.mean():6.2f} +- {v.std() / math.sqrt(len(v)):.2f} | {1 + Q:3d} | {(1 + Q) / (1 - rho):6.2f}")
        allv = np.concatenate([np.array(v) for v in by.values()])
        print(f"   {rho:4.2f}   " + "   ".join(cells))
        print(f"          over all injections: {allv.mean():.2f}; mean of 1 + Q: {1 + Qb[picks].mean():.2f}; mean of (1 + Q)/(1 - rho): {(1 + Qb[picks].mean()) / (1 - rho):.2f}")


def q2():
    print("== Q2 one server with pass-over: summed extra wait per second of one turn's stall ==")
    print("   rho    true c (mean)   P(no other ready turn at its due time)   c given none ready   c given some ready   (1 + rho) for comparison")
    rng = np.random.default_rng(2)
    tau = 1.0
    for S in [0.2, 1.0]:
        print(f"   stall S = {S} service times")
        for rho in [0.3, 0.6, 0.8, 0.9, 0.95]:
            n = 400_000
            A = np.cumsum(rng.exponential(tau / rho, n))
            start = np.empty(n)
            t = 0.0
            for i in range(n):
                s = A[i] if A[i] > t else t
                start[i] = s
                t = s + tau
            Qb = np.searchsorted(A, start, side="right") - 1 - np.arange(n)
            picks = rng.integers(1000, n - 20000, 12000)
            c_all, c0, c1 = [], [], []
            for i in picks:
                ready = start[i] + S          # the tagged turn's read lands S after its due time
                t = start[i]                  # the server is free at the tagged turn's due time
                j = i + 1                     # next other turn, in arrival order
                tagged_done = False
                dsum = 0.0
                while True:
                    if (not tagged_done) and t >= ready:
                        dsum += t - start[i]
                        t += tau
                        tagged_done = True
                        continue
                    if j < n and A[j] <= t:
                        dsum += t - start[j]          # may be negative: an other turn admitted earlier
                        t += tau
                        j += 1
                        continue
                    # nobody ready: the server idles until the next arrival or the landing
                    nxt = A[j] if j < n else float("inf")
                    if not tagged_done:
                        t = min(nxt, ready)
                        continue
                    # tagged done and the server idle before the next arrival: back in step with the baseline
                    break
                c = dsum / S
                c_all.append(c)
                (c0 if Qb[i] == 0 else c1).append(c)
            c_all, c0, c1 = np.array(c_all), np.array(c0), np.array(c1)
            print(f"   {rho:4.2f}   {c_all.mean():6.3f} +- {c_all.std() / math.sqrt(len(c_all)):.3f}   {np.mean(Qb[picks] == 0):6.3f}"
                  f"                                  {c0.mean():6.3f}               {c1.mean() if len(c1) else float('nan'):6.3f}               {1 + rho:5.2f}")


def closed_loop(N, k, p, steps, step_s, Z, T, inject_at=None, delta=0.0, seed=0):
    """Closed loop with identical requests. A prefill pass runs alone for p seconds (decode stands still);
    otherwise every decoding request advances one step per step_s. Admission: first come first served
    while fewer than k requests hold a slot. Returns per-user lists of (arrival, first token, end) and
    the number decoding during the injected pass."""
    rng = np.random.default_rng(seed)
    arrivals = [(float(rng.uniform(0.0, Z)), u) for u in range(N)]
    heapq.heapify(arrivals)
    queue = []                       # (arrival time, user), in arrival order
    decoding = []                    # heap of (decode clock at completion, user, arrival, first token)
    dclock = 0.0                     # decode time elapsed (advances only outside prefill passes)
    t = 0.0
    rec = [[] for _ in range(N)]
    injected = False
    n_dec_at_inject = None
    holding = 0
    while t < T:
        # move arrivals that have happened into the queue
        while arrivals and arrivals[0][0] <= t:
            queue.append(heapq.heappop(arrivals))
        if queue and holding < k:
            a, u = queue.pop(0)
            dur = p
            if (not injected) and inject_at is not None and t >= inject_at:
                dur += delta
                injected = True
                n_dec_at_inject = len(decoding)
            t += dur                              # prefill pass: decode stands still
            holding += 1
            heapq.heappush(decoding, (dclock + steps * step_s, u, a, t))
            continue
        # no prefill possible: decode until the next completion or the next arrival
        t_done = t + (decoding[0][0] - dclock) if decoding else float("inf")
        t_arr = arrivals[0][0] if arrivals else float("inf")
        if t_done <= t_arr or holding >= k:
            if not decoding:
                t = t_arr
                continue
            dclock = decoding[0][0]
            t = t_done
            _, u, a, ft = heapq.heappop(decoding)
            holding -= 1
            rec[u].append((a, ft, t))
            heapq.heappush(arrivals, (t + Z, u))
        else:
            dclock += t_arr - t
            t = t_arr
    return rec, n_dec_at_inject


def q3():
    print("== Q3 closed loop, identical requests: summed extra TTFT per extra second of prefill ==")
    N, k, p, steps, step_s, Z = 128, 20, 0.10, 280, 0.024, 25.0
    T = 1500.0
    base, _ = closed_loop(N, k, p, steps, step_s, Z, T)
    done = sum(len(r) for r in base)
    ttft = np.mean([ft - a for r in base for (a, ft, e) in r if a > 300.0])
    dec = np.mean([e - ft for r in base for (a, ft, e) in r if a > 300.0])
    X = sum(1 for r in base for (a, ft, e) in r if 300.0 < e <= T) / (T - 300.0)
    print(f"   N={N}, k={k} slots, prefill {p} s, decode {steps} steps x {step_s * 1e3:.0f} ms, pause {Z} s:"
          f" throughput {X:.3f}/s, mean TTFT {ttft:.2f} s, mean decode {dec:.2f} s; identity N/X - Z - decode = {N / X - Z - dec:.2f} s")
    delta = 1e-3
    print("   injection at (s)   decoding during the pass   extra summed TTFT / delta   extra summed decode time / delta   sum    N - decoding")
    for t0 in [400.0, 437.3, 512.9, 640.1]:
        pert, ndec = closed_loop(N, k, p, steps, step_s, Z, T, inject_at=t0, delta=delta)
        d_ttft, d_dec = 0.0, 0.0
        for u in range(N):
            m = min(len(base[u]), len(pert[u])) - 1      # drop each user's last request (near the horizon)
            for n in range(m):
                a0, f0, e0 = base[u][n]
                a1, f1, e1 = pert[u][n]
                d_ttft += (f1 - a1) - (f0 - a0)
                d_dec += (e1 - f1) - (e0 - f0)
        print(f"   {t0:8.1f}           {ndec:3d}                        {d_ttft / delta:8.2f}                    {d_dec / delta:8.2f}                        {(d_ttft + d_dec) / delta:7.2f}   {N - ndec}")


if __name__ == "__main__":
    q1()
    q2()
    q3()
