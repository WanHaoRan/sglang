#!/usr/bin/env python3
"""The lateness weight c: what one second of stall of one turn costs the sum of TTFT under pass-over.

Claim checked (DESIGN.md 1.5): "a second of stall costs c = P(no other ready turn when turn i is due)
seconds of summed TTFT: about 1 at light load, about 0 once ready work always waits."

  B1  Exact lemmas on designed open instances (single server, fixed arrival times).
        L1 ready work always waits:   d(sum wait) = sum of the jumpers' work - m * (stalled turn's work)
                                      (0 for equal work; (1 - s_i / s_bar) per stall-second on average)
        L2 nothing else is ready:     d(sum wait) = stall * (1 + later turns in the busy period)
  B2  Open M/M/1 with pass-over at several loads: measured cost per stall-second against the draft's c
      and against (idle time caused) x (1 + later turns in the busy period).
  B3  Closed loop, single server: late reads on 25 % of turns, common random numbers.
        mean TTFT changes by N x (idle time added per request)  [accounting identity]
        so the cost per stall-second is  (price of a GPU-second) x P(GPU idles), not P(GPU idles).
  B4  Pool-bound batch, FA3-like, N = 128: late reads move waiting from other turns to readers and leave the
      mean unchanged while no slot stays empty; unequal work does not change that in the closed loop.
Stdlib only.  Output: exp_c.out
"""
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netsim import simulate, summarize  # noqa: E402
from exp_mu import TAU_A, TAU_B, fa3_draw, mm1n  # noqa: E402

OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def open_draw(arrivals, sizes):
    def draw(u, k, rng):
        if k > 0:
            return None
        return (arrivals[u], sizes[u], 0.0, 0.0, 0)
    return draw


def run_open(arrivals, sizes, ready_abs=None):
    n = len(arrivals)
    rec, st = simulate(n, 1, 0.0, 0.0, open_draw(arrivals, sizes), 1, 0.0,
                       need=[(u, 0) for u in range(n)], ready_abs=ready_abs)
    return rec, st


def sum_wait(rec):
    return sum(r[2] - r[0] for r in rec.values())


def later_in_busy_period(rec, key):
    """Number of jobs that start after `key` in the same busy period (single server, records of a run)."""
    jobs = sorted((r[2], r[4], k) for k, r in rec.items())
    idx = [i for i, j in enumerate(jobs) if j[2] == key][0]
    cnt = 0
    end = jobs[idx][1]
    for j in jobs[idx + 1:]:
        if j[0] > end + 1e-12:
            break
        cnt += 1
        end = j[1]
    return cnt


def b1_lemmas():
    P("B1. Designed open instances (single server, pass-over).")
    rng = random.Random(5)
    m = 60
    arrivals = [1e-9 * i for i in range(m)]
    P("  L1: 60 turns queued at time 0, the first one is not ready for S seconds; the others are ready.")
    for label, sizes in (
            ("equal work 1 s", [1.0] * m),
            ("stalled turn 0.1 s, others 1 s", [0.1] + [1.0] * (m - 1)),
            ("stalled turn 5 s, others 1 s", [5.0] + [1.0] * (m - 1))):
        base, _ = run_open(arrivals, sizes)
        row = []
        for S in (0.5, 3.0, 10.0, 25.5):
            pert, _ = run_open(arrivals, sizes, ready_abs={(0, 0): S})
            d = sum_wait(pert) - sum_wait(base)
            jump = [k for k, r in pert.items() if r[2] < pert[(0, 0)][2]]
            pred = sum(sizes[k[0]] for k in jump) - len(jump) * sizes[0]
            row.append("S=%4.1f: d(sum wait) %+7.2f (formula %+7.2f, %d jumpers)" % (S, d, pred, len(jump)))
        P("    %-32s %s" % (label, "; ".join(row)))
    vals, preds, cs = [], [], []
    for rep in range(2000):
        sizes = [rng.expovariate(1.0) for _ in range(m)]
        base, _ = run_open(arrivals, sizes)
        S = 10.0
        pert, _ = run_open(arrivals, sizes, ready_abs={(0, 0): S})
        d = sum_wait(pert) - sum_wait(base)
        jump = [k for k, r in pert.items() if r[2] < pert[(0, 0)][2]]
        pred = sum(sizes[k[0]] for k in jump) - len(jump) * sizes[0]
        vals.append(d)
        preds.append(pred)
        cs.append(d / S)
    P("    independent exponential work (mean 1 s), S = 10 s, 2,000 draws: max |d - formula| = %.1e;" % (
        max(abs(a - b) for a, b in zip(vals, preds))))
    P("      mean cost per stall-second %+.3f (se %.3f); sd of one draw %.2f  -> zero on average, not turn by turn" % (
        statistics.mean(cs), statistics.pstdev(cs) / math.sqrt(len(cs)), statistics.pstdev(cs)))
    P("  L2: the stalled turn is alone when due; Poisson arrivals behind it (rate 0.7, work 1 s); S = 0.001 s.")
    worst = 0.0
    rows = []
    for rep in range(200):
        n = 400
        t = 0.0
        arrivals = [0.0]
        for _ in range(n - 1):
            t += rng.expovariate(0.7)
            arrivals.append(t + 0.5)
        sizes = [rng.expovariate(1.0) for _ in range(n)]
        base, _ = run_open(arrivals, sizes)
        S = 1e-3
        pert, _ = run_open(arrivals, sizes, ready_abs={(0, 0): S})
        d = (sum_wait(pert) - sum_wait(base)) / S
        b = later_in_busy_period(base, (0, 0))
        worst = max(worst, abs(d - (1 + b)))
        rows.append((d, b))
    P("    200 draws: max |d(sum wait)/S - (1 + later turns in the busy period)| = %.3f; "
      "the slope ranges from %.0f to %.0f (the draft's c is 1 here)" % (
          worst, min(r[0] for r in rows), max(r[0] for r in rows)))
    P("")


def b2_open_stat():
    P("B2. Open M/M/1 with pass-over (work mean 1 s): the 300th of 1,200 turns is stalled S = 2 s after it is due;")
    P("    1,500 draws a load.  Theory for exchangeable work: c -> 1 for long stalls (stationarity) and")
    P("    c = (1 - rho^2) / (1 - rho) = 1 + rho for short ones; it never falls below 1 while rho < 1.")
    P("    c_hat = mean d(sum wait)/S;  c_draft = P(no other turn waiting when it is due)  [theory 1 - rho^2];")
    P("    c_idle = mean (GPU idle time caused)/S;  c_idle/(1-rho) = idle time priced at 1 + rho/(1-rho);")
    P("    c_corr = mean (idle caused) x (1 + later turns in its busy period of the stalled run)/S  (upper estimate)")
    P("    %5s %9s %8s %9s %9s %14s %9s" % ("rho", "c_hat", "(se)", "c_draft", "c_idle", "c_idle/(1-rho)", "c_corr"))
    rng = random.Random(77)
    for rho in (0.2, 0.5, 0.8, 0.9, 0.95):
        ch, cd, ci, cc = [], [], [], []
        for rep in range(1500):
            n = 1200
            t = 0.0
            arrivals = []
            for _ in range(n):
                t += rng.expovariate(rho)
                arrivals.append(t)
            sizes = [rng.expovariate(1.0) for _ in range(n)]
            base, _ = run_open(arrivals, sizes)
            tgt = (300, 0)
            D = base[tgt][2]
            S = 2.0
            pert, _ = run_open(arrivals, sizes, ready_abs={tgt: D + S})
            d = sum_wait(pert) - sum_wait(base)
            ch.append(d / S)
            waiting_at_D = sum(1 for k, r in base.items() if k != tgt and r[0] <= D and r[2] > D)
            cd.append(1.0 if waiting_at_D == 0 else 0.0)
            # idle time in the perturbed run between D and the target's start
            busy = 0.0
            for k, r in pert.items():
                if k == tgt:
                    continue
                lo, hi = max(r[2], D), min(r[4], pert[tgt][2])
                if hi > lo:
                    busy += hi - lo
            idle = (pert[tgt][2] - D) - busy
            ci.append(idle / S)
            cc.append(idle * (1 + later_in_busy_period(pert, tgt)) / S)
        se = statistics.pstdev(ch) / math.sqrt(len(ch))
        P("    %5.2f %9.3f %8.3f %9.3f %9.3f %14.3f %9.3f" % (
            rho, statistics.mean(ch), se, statistics.mean(cd), statistics.mean(ci),
            statistics.mean(ci) / (1 - rho), statistics.mean(cc)))
    P("")


def closed_draw(z_mean, s_mean, reader_frac, lag_mean, reader_scale=1.0):
    def draw(u, k, rng):
        z = rng.expovariate(1.0 / z_mean)
        s = rng.expovariate(1.0 / s_mean)
        x = rng.random()
        lu = rng.random()
        tag, lag = 0, 0.0
        if x < reader_frac:
            tag = 1
            s *= reader_scale
            if lag_mean > 0:
                lag = -lag_mean * math.log(1.0 - lu)
        return (z, s, 0.0, lag, tag)
    return draw


def exposed_stall(rec, t0, t1):
    tot = 0.0
    n = 0
    for r in rec.values():
        if r[4] < 0 or r[0] < t0 or r[0] >= t1:
            continue
        n += 1
        if r[8] >= 0 and r[1] > r[8]:
            tot += r[1] - r[8]
    return tot, n


def b3_closed():
    P("B3. Closed loop, single server (work mean 0.46 s, pause mean 20 s; knee at N* = 44.5).")
    P("    25 % of turns are readers whose read lands Exp(mean 4 s) after arrival; same random numbers without lag.")
    P("    dTTFT = change of the mean TTFT of all turns;  stall = exposed stall per turn (landing - due)+;")
    P("    c_eff = dTTFT / stall;  c_draft = share of the exposed stall time with the GPU idle;")
    P("    identity: dTTFT = N x (idle time added per turn);  mu = N x (1 - absorbed) from the closed form (A3)")
    P("    c_sec6 = the estimator of section 6: share of admissions made with no other ready turn waiting")
    P("    %4s %6s %8s %8s %8s %7s %8s %7s %9s %7s | %9s %9s" % (
        "N", "util", "TTFT0", "dTTFT", "stall", "c_eff", "c_draft", "c_sec6", "N*d(idle)", "mu", "readers", "others"))
    for n_users, t_end in ((16, 300000.0), (32, 300000.0), (40, 300000.0), (44, 300000.0), (48, 300000.0),
                           (56, 200000.0), (64, 200000.0), (96, 150000.0)):
        t0, t1 = 0.1 * t_end, 0.95 * t_end
        rec0, st0 = simulate(n_users, 1, 0.0, 0.0, closed_draw(20.0, 0.46, 0.25, 0.0), 3, t_end, t_stat0=t0)
        rec1, st1 = simulate(n_users, 1, 0.0, 0.0, closed_draw(20.0, 0.46, 0.25, 4.0), 3, t_end, t_stat0=t0)
        a = summarize(rec0, t0, t1, by_tag=True)
        b = summarize(rec1, t0, t1, by_tag=True)
        stall_tot, n1 = exposed_stall(rec1, t0, t1)
        stall = stall_tot / n1
        d = b["ttft"] - a["ttft"]
        # idle per completed request in the statistics window
        n0c = sum(1 for r in rec0.values() if r[4] >= t0)
        n1c = sum(1 for r in rec1.values() if r[4] >= t0)
        d_idle = st1["idle"] / n1c - st0["idle"] / n0c
        h = 1e-5
        mu = (mm1n(n_users, 20.0, 0.46 + h)["R"] - mm1n(n_users, 20.0, 0.46 - h)["R"]) / (2 * h)
        c_draft = st1["starve_time"] / max(stall_tot * (st1["T"] / (t1 - t0)), 1e-9)
        P("    %4d %6.3f %8.3f %8.3f %8.3f %7.3f %8.3f %7.3f %9.3f %7.1f | %9.2f %9.2f" % (
            n_users, st0["busy"] / st0["T"], a["ttft"], d, stall, d / max(stall, 1e-9), c_draft,
            st1["admit_alone"] / max(st1["admits"], 1), n_users * d_idle, mu, b["tags"][1][1], b["tags"][0][1]))
    P("    Same at N = 96 with readers carrying 3 times the GPU work (0.25 of turns):")
    n_users, t_end = 96, 150000.0
    t0, t1 = 0.1 * t_end, 0.95 * t_end
    rec0, st0 = simulate(n_users, 1, 0.0, 0.0, closed_draw(20.0, 0.46, 0.25, 0.0, 3.0), 3, t_end, t_stat0=t0)
    rec1, st1 = simulate(n_users, 1, 0.0, 0.0, closed_draw(20.0, 0.46, 0.25, 4.0, 3.0), 3, t_end, t_stat0=t0)
    a = summarize(rec0, t0, t1, by_tag=True)
    b = summarize(rec1, t0, t1, by_tag=True)
    P("      mean TTFT %.3f -> %.3f s (requests %d -> %d); readers %.2f -> %.2f; others %.2f -> %.2f; GPU idle %.5f -> %.5f" % (
        a["ttft"], b["ttft"], a["n"], b["n"], a["tags"][1][1], b["tags"][1][1], a["tags"][0][1], b["tags"][0][1],
        st0["idle"] / st0["T"], st1["idle"] / st1["T"]))
    P("")


def b4_batch():
    P("B4. Pool-bound batch, FA3-like (N = 128, B = 20, 40,000 s): 25 % of turns are readers whose read lands")
    P("    Exp(mean L) after arrival.  Same random numbers in every row.")
    P("    %-34s %7s %8s %9s %9s %9s %11s %9s" % (
        "case", "X", "TTFT", "readers", "others", "passed", "empty slots", "N/X"))
    t_end = 40000.0
    t0, t1 = 0.25 * t_end, 0.95 * t_end
    base = None
    for label, lag, scale in (("reads on time", 0.0, 1.0), ("lag mean 10 s", 10.0, 1.0),
                              ("lag mean 40 s", 40.0, 1.0), ("lag mean 150 s", 150.0, 1.0),
                              ("lag mean 600 s", 600.0, 1.0),
                              ("on time, readers 3x prefill", 0.0, 3.0),
                              ("lag mean 40 s, readers 3x prefill", 40.0, 3.0)):
        rec, st = simulate(128, 20, TAU_A, TAU_B, fa3_draw(reader_frac=0.25, lag_mean=lag, reader_p_scale=scale),
                           11, t_end, t_stat0=t0)
        a = summarize(rec, t0, t1, by_tag=True)
        X = sum(1 for r in rec.values() if r[4] >= t0) / st["T"]
        P("    %-34s %7.3f %8.2f %9.2f %9.2f %8.1f%% %11.4f %9.2f" % (
            label, X, a["ttft"], a["tags"][1][1], a["tags"][0][1], 100.0 * st["passed_over"] / max(st["admits"], 1),
            st["empty_slot_wait"] / st["T"], 128 / X))
        if base is None:
            base = (X, a["ttft"])
    eps_b = (TAU_A / (TAU_A + TAU_B * 20)) / 20
    P("    One slot empty for a second loses a/(B tau(B)) = %.4f GPU-seconds (the fixed part of a decode step," % eps_b)
    P("    shared by B); at N = 128 that is N x %.4f = %.2f seconds of summed response time." % (eps_b, 128 * eps_b))
    P("")


def b5_batch_sweep():
    P("B5. Pool-bound batch, FA3-like, B = 20: users N from light load to saturation; 25 % readers, lag Exp(mean 10 s).")
    P("    c_eff = dTTFT / exposed stall per turn;  c_empty = slot-seconds left empty while turns wait / exposed")
    P("    stall time, capped at 1 (the share of stall-seconds in which no other ready turn takes the slot);")
    P("    c_sec6 = share of admissions made with no other ready turn waiting.")
    P("    slots used = mean share of the 20 slots in use without lags.")
    P("    %4s %10s %8s %8s %8s %7s %8s %7s | %9s %9s" % (
        "N", "slots used", "TTFT0", "dTTFT", "stall", "c_eff", "c_empty", "c_sec6", "readers", "others"))
    for n_users, t_end in ((30, 60000.0), (50, 60000.0), (60, 60000.0), (64, 60000.0), (68, 60000.0),
                           (72, 60000.0), (80, 60000.0), (96, 40000.0), (128, 40000.0)):
        t0, t1 = 0.25 * t_end, 0.95 * t_end
        rec0, st0 = simulate(n_users, 20, TAU_A, TAU_B, fa3_draw(reader_frac=0.25, lag_mean=0.0), 11, t_end, t_stat0=t0)
        rec1, st1 = simulate(n_users, 20, TAU_A, TAU_B, fa3_draw(reader_frac=0.25, lag_mean=10.0), 11, t_end, t_stat0=t0)
        a = summarize(rec0, t0, t1, by_tag=True)
        b = summarize(rec1, t0, t1, by_tag=True)
        stall_tot, n1 = exposed_stall(rec1, t0, t1)
        stall = stall_tot / n1
        d = b["ttft"] - a["ttft"]
        full = 1.0 - (st0["empty_slot_wait"] + st0["empty_slot_nowait"]) / st0["T"] / 20.0
        c_empty = st1["empty_slot_wait"] / max(stall_tot * (st1["T"] / (t1 - t0)), 1e-9)
        P("    %4d %10.3f %8.3f %8.3f %8.3f %7.3f %8.3f %7.3f | %9.2f %9.2f" % (
            n_users, full, a["ttft"], d, stall, d / max(stall, 1e-9), min(c_empty, 1.0),
            st1["admit_alone"] / max(st1["admits"], 1), b["tags"][1][1], b["tags"][0][1]))
    P("")


def main():
    P("The lateness weight c under pass-over")
    P("")
    b1_lemmas()
    b2_open_stat()
    b3_closed()
    b4_batch()
    b5_batch_sweep()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_c.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
