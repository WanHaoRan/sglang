#!/usr/bin/env python3
"""The weight mu in a closed loop: one more second of GPU work, single server and pool-bound batch.

Checks (DESIGN.md 1.5 and result 8):
  A1  Saturated closed loop, single server, tiny injection: d(sum of response times)/d(work) = N exactly,
      on every sample path (each user's timeline is pushed once).
  A2  Finite injections: the mean is N, single injections are noisy (the service order changes).
  A3  Below saturation: mu = N * d(cycle time per request)/d(work per request) = N * (1 - a), a = the share
      of an extra GPU-second absorbed by idle time.  M/M/1//N in closed form, against 1 + Q, and a check of
      the simulator against the closed form.
  A4  Pool-bound batch (B slots, prefill pauses decode): d(sum of response)/d(work) = N, d(sum of TTFT)/
      d(work) = N - (requests past their first token), residence in the running set = B / X.
Stdlib only.  Output: exp_mu.out
"""
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netsim import simulate, summarize  # noqa: E402

OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def draw_exp(z_mean, s_mean):
    def draw(u, k, rng):
        return (rng.expovariate(1.0 / z_mean), rng.expovariate(1.0 / s_mean), 0.0, 0.0, 0)
    return draw


def sums(rec, k_hi):
    s_resp = s_ttft = 0.0
    n = 0
    for (u, k), r in rec.items():
        if k <= k_hi:
            s_resp += r[4] - r[0]
            s_ttft += r[3] - r[0]
            n += 1
    return s_resp, s_ttft, n


def in_service_at(rec, t):
    """Requests admitted and not done at time t; and which of them have passed their first token."""
    running = []
    for key, r in rec.items():
        if 0 <= r[2] <= t and (r[4] < 0 or r[4] > t):
            running.append((key, r[3] >= 0 and r[3] <= t))
    return running


def prefilling_at(rec, t):
    best = None
    for key, r in rec.items():
        if 0 <= r[2] <= t and r[3] > t:
            # admitted, first token not yet out; the one being prefilled has the earliest admit
            if best is None or r[2] < rec[best][2]:
                best = key
    return best


def a1_exact(n_users, z_mean, s_mean, k_hi, seeds, delta=1e-6):
    draw = draw_exp(z_mean, s_mean)
    slopes = []
    idle_tot = 0.0
    for seed in seeds:
        need = [(u, k) for u in range(n_users) for k in range(k_hi + 1)]
        rec0, st0 = simulate(n_users, 1, 0.0, 0.0, draw, seed, 0.0, need=need)
        t_all = st0["t_final"]
        t_inj = 0.25 * t_all
        cands = in_service_at(rec0, t_inj)
        if not cands:
            continue
        key = cands[0][0]
        rec1, _ = simulate(n_users, 1, 0.0, 0.0, draw, seed, 0.0, need=need, extra_prefill={key: delta})
        r0, _, n0 = sums(rec0, k_hi)
        r1, _, n1 = sums(rec1, k_hi)
        assert n0 == n1 == len(need)
        slopes.append((r1 - r0) / delta)
        idle_tot += st0["idle"] / st0["T"]
    return slopes, idle_tot / max(len(slopes), 1)


def mm1n(n_users, z_mean, s_mean):
    """M/M/1//N (machine repair): utilisation, throughput, mean response, mean number waiting."""
    rho = s_mean / z_mean
    logs = [0.0]
    for n in range(1, n_users + 1):
        logs.append(logs[-1] + math.log(n_users - n + 1) + math.log(rho))
    m = max(logs)
    w = [math.exp(x - m) for x in logs]
    tot = sum(w)
    p = [x / tot for x in w]
    util = 1.0 - p[0]
    x = util / s_mean
    resp = n_users / x - z_mean
    in_sys = sum(n * p[n] for n in range(n_users + 1))
    return dict(U=util, X=x, R=resp, Q=in_sys - util, L=in_sys)


def a3_table(z_mean, s_mean):
    P("   M/M/1//N with pause mean %.1f s and service mean %.3f s (saturation point N* = 1 + Z/s = %.1f)" % (
        z_mean, s_mean, 1 + z_mean / s_mean))
    P("   mu = dR/ds (response seconds per second of work added in proportion to each request's work)")
    P("   %5s %8s %8s %9s %9s %9s %9s %8s" % ("N", "util", "queue Q", "R (s)", "mu", "1 + Q", "mu / N", "absorbed"))
    rows = []
    for n in (8, 16, 24, 32, 40, 44, 48, 56, 64, 80, 96, 128):
        h = 1e-5 * s_mean
        a = mm1n(n, z_mean, s_mean)
        up = mm1n(n, z_mean, s_mean + h)
        dn = mm1n(n, z_mean, s_mean - h)
        mu = (up["R"] - dn["R"]) / (2 * h)
        rows.append((n, a, mu))
        P("   %5d %8.4f %8.2f %9.2f %9.2f %9.2f %9.3f %8.3f" % (
            n, a["U"], a["Q"], a["R"], mu, 1 + a["Q"], mu / n, 1 - mu / n))
    return rows


def a3_sim_check(n_users, z_mean, s_mean, eta, t_end, seed):
    def draw_scaled(scale):
        def draw(u, k, rng):
            return (rng.expovariate(1.0 / z_mean), scale * rng.expovariate(1.0 / s_mean), 0.0, 0.0, 0)
        return draw
    rec0, st0 = simulate(n_users, 1, 0.0, 0.0, draw_scaled(1.0), seed, t_end)
    rec1, _ = simulate(n_users, 1, 0.0, 0.0, draw_scaled(1.0 + eta), seed, t_end)
    a = summarize(rec0, 0.1 * t_end, 0.95 * t_end)
    b = summarize(rec1, 0.1 * t_end, 0.95 * t_end)
    return a, b, (b["resp"] - a["resp"]) / (eta * s_mean), st0


def fa3_draw(extra_p=0.0, reader_frac=0.0, lag_mean=0.0, reader_p_scale=1.0):
    """FA3-like workload (campaign 34 constants): pauses 92 % tool (lognormal, median 1.0 s, mean 6.0 s) and
    8 % human (lognormal, median 64 s, mean 190 s, capped at 1,200 s); prefill mean 0.137 s per request
    (lognormal, sigma 1); outputs mean 280 tokens (lognormal, median 109)."""
    sig_t = math.sqrt(2 * math.log(6.0 / 1.0))
    sig_h = math.sqrt(2 * math.log(190.0 / 64.0))
    sig_o = math.sqrt(2 * math.log(280.0 / 109.0))
    mu_p = math.log(0.137) - 0.5

    def draw(u, k, rng):
        if rng.random() < 0.92:
            z = rng.lognormvariate(0.0, sig_t)
        else:
            z = min(rng.lognormvariate(math.log(64.0), sig_h), 1200.0)
        p = rng.lognormvariate(mu_p, 1.0) + extra_p
        n_out = rng.lognormvariate(math.log(109.0), sig_o)
        x = rng.random()
        lag_u = rng.random()
        tag = 0
        lag = 0.0
        if x < reader_frac:
            tag = 1
            lag = -lag_mean * math.log(1.0 - lag_u) if lag_mean > 0 else 0.0
            p *= reader_p_scale
        return (z, p, n_out, lag, tag)
    return draw


TAU_A = 0.0035
TAU_B = 0.00099   # 0.47 ms per running request + 0.0162 ms per K of context at about 32 K per request


def a4_batch(n_users, n_slots, t_end, seed):
    draw0 = fa3_draw()
    t0 = time.time()
    rec0, st0 = simulate(n_users, n_slots, TAU_A, TAU_B, draw0, seed, t_end, t_stat0=0.25 * t_end)
    a = summarize(rec0, 0.25 * t_end, 0.95 * t_end)
    T = st0["T"]
    B_bar = st0["run_int"] / T
    Q_bar = st0["q_int"] / T
    X = sum(1 for r in rec0.values() if r[4] >= 0.25 * t_end) / T
    # mean pause of the requests completed in the window: recompute from the draws is not stored; use Little
    z_eff = (n_users - B_bar - Q_bar) / X
    P("   base: X = %.3f req/s (g = 1/X = %.3f s), running %.1f, queued %.1f, GPU idle %.4f of the time, "
      "slot-seconds empty while turns wait: %.4f per second" % (
          X, 1 / X, B_bar, Q_bar, st0["idle"] / T, st0["empty_slot_wait"] / T))
    P("         mean TTFT %.2f s, mean response %.2f s, mean queue wait %.2f s, admission to first token %.3f s" % (
        a["ttft"], a["resp"], a["wait"], a["pre"]))
    P("         identity pieces: N/X = %.2f, Zeff = (N - in system)/X = %.2f, B/X = %.2f, queued/X = %.2f" % (
        n_users / X, z_eff, B_bar / X, Q_bar / X))
    P("         (N - B)/X - Zeff + (admission to first token) = %.2f  against mean TTFT %.2f" % (
        (n_users - B_bar) / X - z_eff + a["pre"], a["ttft"]))
    P("         time-average decode batch %.1f; predicted g = prefill + tau(B) * n_out / B = %.3f s" % (
        st0["n_dec_int"] / max(st0["dec_time"], 1e-9),
        a["p"] + (TAU_A + TAU_B * (st0["n_dec_int"] / max(st0["dec_time"], 1e-9))) * 280.0
        / (st0["n_dec_int"] / max(st0["dec_time"], 1e-9))))
    P("         (%.0f s of CPU)" % (time.time() - t0))
    return rec0, st0, a


def a4_rate(n_users, n_slots, t_end, seed, eps):
    rec0, st0 = simulate(n_users, n_slots, TAU_A, TAU_B, fa3_draw(), seed, t_end, t_stat0=0.25 * t_end)
    rec1, st1 = simulate(n_users, n_slots, TAU_A, TAU_B, fa3_draw(extra_p=eps), seed, t_end, t_stat0=0.25 * t_end)
    a = summarize(rec0, 0.25 * t_end, 0.95 * t_end)
    b = summarize(rec1, 0.25 * t_end, 0.95 * t_end)
    B0 = st0["run_int"] / st0["T"]
    B1 = st1["run_int"] / st1["T"]
    X0 = a["n"] / (0.70 * t_end)
    X1 = b["n"] / (0.70 * t_end)
    return a, b, B0, B1, X0, X1


def a4_exact(n_users, n_slots, k_hi, seed, delta=1e-6):
    draw = fa3_draw()
    need = [(u, k) for u in range(n_users) for k in range(k_hi + 1)]
    rec0, st0 = simulate(n_users, n_slots, TAU_A, TAU_B, draw, seed, 0.0, need=need)
    t_inj = 0.3 * st0["t_final"]
    key = prefilling_at(rec0, t_inj)
    if key is None:
        # move to the next prefill start
        later = sorted((r[2], k2) for k2, r in rec0.items() if r[2] > t_inj)
        key = later[0][1]
        t_inj = rec0[key][2] + 1e-9
    running = in_service_at(rec0, rec0[key][2] + 1e-12)
    n_past_first = sum(1 for k2, past in running if past and k2 != key)
    n_wait_pf = sum(1 for k2, past in running if (not past) and k2 != key)
    rec1, _ = simulate(n_users, n_slots, TAU_A, TAU_B, draw, seed, 0.0, need=need, extra_prefill={key: delta})
    r0, f0, _ = sums(rec0, k_hi)
    r1, f1, _ = sums(rec1, k_hi)
    return (r1 - r0) / delta, (f1 - f0) / delta, n_past_first, n_wait_pf, st0


def main():
    P("The weight mu: one more second of GPU work (closed loop)")
    P("")
    P("A1. Single server, saturated closed loop, injection of 1e-6 s into the request in service at 25 % of the run.")
    P("    d(sum of response times)/d(work) over all requests with index <= K of every user; theory: exactly N.")
    for n_users, z_mean, k_hi in ((8, 2.0, 300), (32, 5.0, 300), (128, 20.0, 200)):
        slopes, idle = a1_exact(n_users, z_mean, 1.0, k_hi, seeds=range(1, 9))
        P("    N = %3d, pause mean %4.1f s, service mean 1 s: slopes min %.4f max %.4f (8 paths); "
          "server idle %.5f of the time" % (n_users, z_mean, min(slopes), max(slopes), idle))
    P("")
    P("A2. Same, finite injections (N = 32, pause mean 5 s, service mean 1 s; 300 paths each):")
    n_users, z_mean, k_hi = 32, 5.0, 150
    draw = draw_exp(z_mean, 1.0)
    for delta in (0.01, 0.1, 1.0, 5.0):
        vals = []
        for seed in range(100, 400):
            need = [(u, k) for u in range(n_users) for k in range(k_hi + 1)]
            rec0, st0 = simulate(n_users, 1, 0.0, 0.0, draw, seed, 0.0, need=need)
            cands = in_service_at(rec0, 0.25 * st0["t_final"])
            if not cands:
                continue
            key = cands[0][0]
            rec1, _ = simulate(n_users, 1, 0.0, 0.0, draw, seed, 0.0, need=need, extra_prefill={key: delta})
            vals.append((sums(rec1, k_hi)[0] - sums(rec0, k_hi)[0]) / delta)
        se = statistics.pstdev(vals) / math.sqrt(len(vals))
        P("    delta = %5.2f s: mean %7.2f (se %.2f), median %7.2f, min %9.1f, max %9.1f   [N = %d]" % (
            delta, statistics.mean(vals), se, statistics.median(vals), min(vals), max(vals), n_users))
    P("")
    P("A3. Below saturation (closed form).")
    a3_table(20.0, 0.46)
    P("   Simulator against the closed form (proportional scaling of every service by 1.02, common random numbers):")
    for n_users, t_end in ((32, 400000.0), (44, 400000.0), (64, 200000.0)):
        a, b, mu_sim, st0 = a3_sim_check(n_users, 20.0, 0.46, 0.02, t_end, 7)
        h = 1e-5
        ana = (mm1n(n_users, 20.0, 0.46 + h)["R"] - mm1n(n_users, 20.0, 0.46 - h)["R"]) / (2 * h)
        ana2 = (mm1n(n_users, 20.0, 0.46 * 1.02)["R"] - mm1n(n_users, 20.0, 0.46)["R"]) / (0.02 * 0.46)
        P("     N = %3d: simulated R %.3f s (closed form %.3f); simulated dR/ds %.1f; closed form %.1f "
          "(derivative) and %.1f (same 2 %% step); %d requests" % (
              n_users, a["resp"], mm1n(n_users, 20.0, 0.46)["R"], mu_sim, ana, ana2, a["n"]))
    P("")
    P("A4. Pool-bound batch, FA3-like constants: N = 128 users, B = 20 slots, decode step 3.5 + 0.99 n ms,")
    P("    prefill pauses decode.  40,000 s simulated.")
    a4_batch(128, 20, 40000.0, 11)
    P("    Sustained extra prefill work eps on every request (common random numbers):")
    for eps in (0.05, 0.555):
        a, b, B0, B1, X0, X1 = a4_rate(128, 20, 40000.0, 11, eps)
        dg = 1 / X1 - 1 / X0
        P("      eps = %.3f s: g %.3f -> %.3f (dg = %.3f); mean TTFT %.2f -> %.2f; response %.2f -> %.2f" % (
            eps, 1 / X0, 1 / X1, dg, a["ttft"], b["ttft"], a["resp"], b["resp"]))
        P("                    d(response)/dg = %.1f [N = 128];  d(TTFT)/dg = %.1f [N - B = %.1f];  "
          "d(response - TTFT)/dg = %.1f [B = %.1f]" % (
              (b["resp"] - a["resp"]) / dg, (b["ttft"] - a["ttft"]) / dg, 128 - B0,
              ((b["resp"] - b["ttft"]) - (a["resp"] - a["ttft"])) / dg, B0))
    P("    One injection of 1e-6 s into a prefill (sample-path slopes, 6 paths):")
    for seed in range(21, 27):
        s_resp, s_ttft, n_past, n_wait, st0 = a4_exact(128, 20, 120, seed)
        P("      d(sum response) = %.3f [N = 128];  d(sum TTFT) = %.3f;  running past first token at the "
          "injection: %d  ->  N - that = %d" % (s_resp, s_ttft, n_past, 128 - n_past))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_mu.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
