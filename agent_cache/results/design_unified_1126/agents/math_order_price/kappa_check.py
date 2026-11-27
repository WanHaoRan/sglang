#!/usr/bin/env python3
"""The price of device time kappa (DESIGN.md section 6): a check of its definition.

Device: one machine serving transfers (reads and writes) in earliest-due-date order among released jobs,
without preemption.  Objective term: sum_j w_j (C_j - d_j)+  (w_j = c_j for a read, lambda * L_p for a write).

Sample-path fact: one more device-second of work at time t, in front of the jobs behind, moves the
completion of every later job of the same busy period by one second, so
    kappa(t) = sum of w_j over the jobs that complete after t in the same busy period and complete late.
The draft's estimator counts the transfers waiting at t that are already past their due date (a subset:
it leaves out jobs that will arrive during the busy period and waiting jobs that are not late yet but
will complete late).  The script checks the exact formula by finite differences and measures how much the
estimator understates it.
Output: kappa_check.out
"""
import os
import random
import statistics
import sys

OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def run(jobs, extra=None):
    """jobs: list of (release, size, due, weight).  EDD among released, no preemption.  Returns completions."""
    n = len(jobs)
    extra = extra or {}
    order_rel = sorted(range(n), key=lambda j: jobs[j][0])
    done = [None] * n
    start = [None] * n
    t = 0.0
    i = 0
    avail = []
    served = 0
    while served < n:
        while i < n and jobs[order_rel[i]][0] <= t + 1e-15:
            avail.append(order_rel[i])
            i += 1
        if not avail:
            t = jobs[order_rel[i]][0]
            continue
        j = min(avail, key=lambda q: (jobs[q][2], q))
        avail.remove(j)
        start[j] = t
        t += jobs[j][1] + extra.get(j, 0.0)
        done[j] = t
        served += 1
    return start, done


def cost(jobs, done):
    return sum(w * max(0.0, c - d) for (r, p, d, w), c in zip(jobs, done))


def main():
    rng = random.Random(9)
    P("Price of device time: exact externality against the draft's estimator")
    P("")
    P("%5s %10s %22s %22s %12s" % ("load", "late jobs", "max |slope - formula|", "mean kappa (formula)", "draft / exact"))
    for load in (0.6, 0.8, 0.9, 0.95):
        worst = 0.0
        exact_vals, draft_vals = [], []
        late_share = []
        for rep in range(60):
            n = 400
            t = 0.0
            jobs = []
            for _ in range(n):
                t += rng.expovariate(load / 0.7)
                is_read = rng.random() < 0.6
                size = rng.uniform(0.3, 2.0) if is_read else rng.uniform(0.05, 0.6)
                due = t + size + rng.expovariate(1.0 / 3.0)
                w = rng.uniform(0.5, 1.5) if is_read else rng.uniform(0.05, 2.0)
                jobs.append((t, size, due, w))
            start, done = run(jobs)
            base = cost(jobs, done)
            late_share.append(sum(1 for j in range(n) if done[j] > jobs[j][2]) / n)
            order = sorted(range(n), key=lambda j: start[j])
            pos = {j: i for i, j in enumerate(order)}
            for _ in range(8):
                k = rng.randrange(50, n - 50)
                delta = 1e-7
                _, done2 = run(jobs, {k: delta})
                slope = (cost(jobs, done2) - base) / delta
                # exact: weights of late jobs from k on, in k's busy period
                tot = 0.0
                end = done[k]
                if done[k] > jobs[k][2]:
                    tot += jobs[k][3]
                for j in order[pos[k] + 1:]:
                    if start[j] > end + 1e-12:
                        break
                    if done[j] > jobs[j][2]:
                        tot += jobs[j][3]
                    end = done[j]
                worst = max(worst, abs(slope - tot))
                # draft: transfers waiting when k starts (released, not started) that are already past due
                tk = start[k]
                dr = sum(jobs[j][3] for j in range(n)
                         if j != k and jobs[j][0] <= tk and start[j] > tk and jobs[j][2] < tk)
                exact_vals.append(tot)
                draft_vals.append(dr)
        P("%5.2f %9.0f%% %22.2e %22.2f %12.2f" % (
            load, 100 * statistics.mean(late_share), worst, statistics.mean(exact_vals),
            statistics.mean(draft_vals) / max(statistics.mean(exact_vals), 1e-12)))
    P("")
    P("The exact price is the summed cost rate of every transfer that completes late behind the added work in")
    P("the same busy period; the estimator of section 6 (waiting transfers already past due) is a lower bound.")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kappa_check.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())
