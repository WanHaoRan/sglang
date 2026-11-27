"""How many seconds of response time does one extra second of GPU work cost?

Toy check of the 'mu = 1 + requests queued behind' claim, not a model of the engine.
Closed network: N users, exponential think time Z, one FCFS server with exponential
service of mean s (a stand-in for the KV-bound GPU at saturation). At time t0 the job in
service gets delta seconds more work. Common random numbers per (user, cycle) pair the
two runs job by job; the cost is the summed change in response time over the jobs both
runs completed. Compare with 1 + Q(t0), Q = jobs waiting behind the one in service.

Open variant: Poisson arrivals at the same rate, one job each, same service draws.
"""
import heapq
import random
import statistics as st
import sys


def draw(seed, kind, u, c, mean):
    return random.Random(seed * 1_000_003 + kind * 7_919 + u * 104_729 + c).expovariate(1.0 / mean)


def closed_run(N, Z, s, horizon, t0, delta, seed):
    events = [(draw(seed, 1, u, 0, Z), 0, u) for u in range(N)]
    heapq.heapify(events)
    cycle = [0] * N
    queue, in_service, resp = [], None, {}
    injected, q0 = False, None
    while events:
        t, kind, u = heapq.heappop(events)
        if t > horizon:
            break
        if kind == 0:
            queue.append((t, u))
        else:
            arr, v = in_service
            resp[(v, cycle[v])] = t - arr
            in_service = None
            cycle[v] += 1
            heapq.heappush(events, (t + draw(seed, 1, v, cycle[v], Z), 0, v))
        if in_service is None and queue:
            arr, v = queue.pop(0)
            in_service = (arr, v)
            heapq.heappush(events, (t + draw(seed, 2, v, cycle[v], s), 1, v))
        if not injected and t >= t0 and in_service is not None:
            injected, q0 = True, len(queue)
            if delta:
                for i, (et, k, w) in enumerate(events):
                    if k == 1:
                        events[i] = (et + delta, k, w)
                heapq.heapify(events)
    return resp, q0


def open_run(rate, s, horizon, t0, delta, seed):
    t, arrivals, j = 0.0, [], 0
    while t < horizon:
        t += draw(seed, 3, 0, j, 1.0 / rate)
        arrivals.append(t)
        j += 1
    free, resp, injected, q0 = 0.0, {}, False, None
    starts = []
    for j, a in enumerate(arrivals):
        start = max(a, free)
        svc = draw(seed, 2, 0, j, s)
        if not injected and start + svc >= t0:
            # the job in service at t0 (or the next one) carries the extra work
            injected = True
            q0 = sum(1 for b in arrivals[j + 1:] if b <= t0)
            svc += delta
        free = start + svc
        resp[j] = free - a
    return resp, q0


def paired(base, hit, delta):
    common = base.keys() & hit.keys()
    return sum(hit[k] - base[k] for k in common) / delta, len(base) - len(hit)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    dl = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    print(f"delta = {dl} s of extra work, {reps} injections per row")
    for N, Z, s in ((30, 23.0, 0.46), (45, 23.0, 0.46), (51, 23.0, 0.46), (60, 23.0, 0.46), (80, 23.0, 0.46), (128, 23.0, 0.46), (128, 32.0, 1.02)):
        mus, qs, lost = [], [], []
        for rep in range(reps):
            t0 = 1500.0 + 41.0 * rep
            b, q0 = closed_run(N, Z, s, 5000.0, t0, 0.0, rep)
            h, _ = closed_run(N, Z, s, 5000.0, t0, dl, rep)
            mu, l = paired(b, h, dl)
            mus.append(mu)
            qs.append(q0 + 1)
            lost.append(l)
        print(f"closed N={N} Z={Z} s={s}: d(sum R)/d(work) mean {st.mean(mus):.1f} "
              f"median {st.median(mus):.1f}; 1+Q at t0 mean {st.mean(qs):.1f} median {st.median(qs):.1f}; "
              f"jobs lost in the horizon {st.mean(lost):+.2f}")
    for util in (0.7, 0.9, 0.97):
        s = 0.46
        rate = util / s
        mus, qs = [], []
        for rep in range(reps):
            t0 = 1500.0 + 41.0 * rep
            b, q0 = open_run(rate, s, 5000.0, t0, 0.0, rep)
            h, _ = open_run(rate, s, 5000.0, t0, dl, rep)
            mu, _ = paired(b, h, dl)
            mus.append(mu)
            qs.append(q0 + 1)
        print(f"open util={util}: d(sum R)/d(work) mean {st.mean(mus):.1f} median {st.median(mus):.1f}; "
              f"1+Q at t0 mean {st.mean(qs):.1f} median {st.median(qs):.1f}")


if __name__ == "__main__":
    main()
