#!/usr/bin/env python3
"""Counterexample: in a closed loop the sum of response times is neither convex nor monotone in GPU work.

Three users, one first-come-first-served server, pauses fixed per request.  Adding 0.2 s of GPU work to
one request lowers the summed response time of the same five requests by 8.4 s, because the later return
of its user lets a short request go ahead of a long one.  So "base cost + mu * (extra work)" is not a
lower bound in the closed loop, and a Lagrangian bound built on that linearisation is not one either.
In the open model (fixed arrival times, fixed order) the sum is convex in the work (Lindley), and the
linearisation at any base point is a lower bound.
Output: closed_counterexample.out
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netsim import simulate  # noqa: E402

# (pause before the request, work of the request), per user
PLAN = {
    0: [(0.0, 5.0), (0.9, 10.0)],     # A: a 5 s request, then after 0.9 s a 10 s request
    1: [(0.1, 1.0), (0.001, 1.0)],    # B: two 1 s requests
    2: [(5.95, 1.0)],                 # C: one 1 s request, arriving at 5.95 s
}
NAMES = {0: "A", 1: "B", 2: "C"}


def draw(u, k, rng):
    if k >= len(PLAN[u]):
        return None
    z, s = PLAN[u][k]
    return (z, s, 0.0, 0.0, 0)


def run(extra):
    need = [(u, k) for u in PLAN for k in range(len(PLAN[u]))]
    rec, _ = simulate(3, 1, 0.0, 0.0, draw, 1, 0.0, need=need, extra_prefill=extra)
    return rec


def main():
    out = []
    base = run({})
    pert = run({(0, 0): 0.2})
    out.append("Closed loop, 3 users, one FCFS server: 0.2 s of extra work on A's first request")
    out.append("%-8s %28s %28s" % ("request", "base: arrive start resp", "+0.2 s: arrive start resp"))
    tb = tp = 0.0
    for key in sorted(base):
        b, p = base[key], pert[key]
        tb += b[4] - b[0]
        tp += p[4] - p[0]
        out.append("%s%d       %8.3f %8.3f %8.3f   %8.3f %8.3f %8.3f" % (
            NAMES[key[0]], key[1] + 1, b[0], b[2], b[4] - b[0], p[0], p[2], p[4] - p[0]))
    out.append("sum of response times: %.3f s -> %.3f s  (change %+.3f s for +0.2 s of work;" % (tb, tp, tp - tb))
    out.append("the first-order rule mu = N = 3 predicts +0.6 s)")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "closed_counterexample.out"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
