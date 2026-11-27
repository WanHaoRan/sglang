"""The write as retention one tier down: checks of DESIGN.md section 4.2 and result 7.3.

Per token of a page held in host memory during one idle gap (survival S on a grid, mass S_n at
infinity = the session has ended):
  dirty (no SSD copy): losing the host copy costs kd = mu*gamma at the return (recompute);
  clean (SSD copy):    losing it costs kc at the return (a read);
  write:               costs w = kappa/beta now (device time), turns dirty into clean;
  rent:                lam per second while the host copy is held.

A. Single gap, constant prices. Three-state backward induction against the closed-form claims:
   (i)   writing and then holding is never strictly better than holding and writing later;
   (ii)  the optimum is one of two lives: W = hold to a_w, write, drop (index with kc + w);
         N = hold to a_d, drop dirty (index with kd); W iff min C_W <= min C_N;
   (iii) value of a write (against never writing) = integral over K in [kc, kd] of P(pay | K) - w.
B. Cost of missing the due date: holding a clean-droppable page costs L*(lam - kc*hazard)/s, not lam*L.
C. Life of a page as a function of (lam, mu, kappa) on the tool law with an end probability.
D. A device that is idle or busy (two-state Markov price): write when idle, wait when busy.
E. Many gaps (the page stays dirty across returns until the session ends): lazy against eager,
   share of pages ever written, against the closed form x/(x + e - x e).

Run: python -I -B write_model.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hold_index import (  # noqa: E402
    held_time,
    index_hull,
    surv_lognormal,
    surv_point,
    surv_two_piece_lognormal,
    surv_uniform,
    surv_weibull,
    with_end_mass,
)

RNG = np.random.default_rng(42)
HOLD, WRITE, DROP = 0, 1, 2


# ----------------------------------------------------------------------------- A. single gap DP
def dp_single(surv: np.ndarray, dt: float, lam: float, kd: float, kc: float, w: float):
    n = len(surv) - 1
    vc = np.zeros(n + 1)
    vd = np.zeros(n + 1)
    act_c = np.full(n + 1, DROP)
    act_d = np.full(n + 1, DROP)
    for k in range(n - 1, -1, -1):
        if surv[k] <= 0.0:
            continue
        pi = (surv[k] - surv[n]) / surv[k]
        q = surv[k + 1] / surv[k]
        drop_c, hold_c = kc * pi, lam * dt + q * vc[k + 1]
        vc[k], act_c[k] = (drop_c, DROP) if drop_c <= hold_c else (hold_c, HOLD)
        options = (lam * dt + q * vd[k + 1], w + vc[k], kd * pi)  # HOLD, WRITE, DROP
        best = int(np.argmin(options))
        # prefer DROP over WRITE over HOLD on exact ties, so that ties do not look like "write then hold"
        for cand in (DROP, WRITE, HOLD):
            if options[cand] <= options[best] + 1e-15:
                best = cand
                break
        vd[k], act_d[k] = options[best], best
    return vc, vd, act_c, act_d


def two_lives(surv: np.ndarray, dt: float, lam: float, kd: float, kc: float, w: float):
    held = held_time(surv, dt)
    p_inf = surv[-1]
    c_w = lam * held + (kc + w) * (surv - p_inf) + w * p_inf
    c_n = lam * held + kd * (surv - p_inf)
    return c_w, c_n


def check_single_gap() -> None:
    dt, n = 0.05, 2400
    t = np.linspace(0.0, dt * n, n + 1)
    laws = {
        "tool LN": surv_two_piece_lognormal(t, 0.41, 1.69, 16.8),
        "tool LN + 10% ended": with_end_mass(surv_lognormal(t, 1.69, 1.79), 0.10),
        "weibull k=0.5 + 30% ended": with_end_mass(surv_weibull(t, 5.0, 0.5), 0.30),
        "uniform[15,25] + 20% ended": with_end_mass(surv_uniform(t, 15.0, 25.0), 0.20),
        "point 20 s": surv_point(t, 20.0),
        "random law": np.concatenate(([1.0], np.sort(RNG.random(n))[::-1] * 0.999)),
    }
    cases = v_err = write_then_hold = life_err = 0
    worst = 0.0
    for name, surv in laws.items():
        surv = np.where(surv < 1e-12, 0.0, surv)
        for _ in range(60):
            kc = 10 ** RNG.uniform(-5, -3.5)
            kd = kc * 10 ** RNG.uniform(-0.3, 2.5)  # sometimes kd < kc + w: writing cannot pay
            w = kc * 10 ** RNG.uniform(-2, 1)
            lam = kc * 10 ** RNG.uniform(-3, 0.5)
            vc, vd, act_c, act_d = dp_single(surv, dt, lam, kd, kc, w)
            c_w, c_n = two_lives(surv, dt, lam, kd, kc, w)
            claim = min(float(np.min(c_w)), float(np.min(c_n)))
            cases += 1
            err = abs(vd[0] - claim) / max(claim, 1e-300)
            worst = max(worst, err)
            if err > 1e-9:
                v_err += 1
            # (i) whenever WRITE is chosen, the clean page is dropped at once
            wk = np.where((act_d == WRITE) & (surv > 1e-6))[0]
            if len(wk) and np.any(act_c[wk] == HOLD):
                # a hold after the write is a tie only if holding costs the same as dropping
                k = wk[act_c[wk] == HOLD][0]
                pi = (surv[k] - surv[-1]) / surv[k]
                if vc[k] < kc * pi - 1e-12 * max(1.0, kc):
                    write_then_hold += 1
            # (ii) which life the DP follows from age 0
            k = 0
            while act_d[k] == HOLD and k < n:
                k += 1
            dp_life = "W" if act_d[k] == WRITE else "N"
            thm_life = "W" if np.min(c_w) <= np.min(c_n) else "N"
            if dp_life != thm_life and abs(np.min(c_w) - np.min(c_n)) > 1e-9 * claim:
                life_err += 1
    print(f"  {cases} random cases: value mismatches {v_err} (worst rel. err {worst:.1e}); "
          f"'write, then hold' strictly optimal in {write_then_hold}; wrong life in {life_err}")


def check_value_of_write() -> None:
    """V0(kd) - V0(kc) = integral_{kc}^{kd} P(pay | K) dK  (never-write values of one gap)."""
    dt, n = 0.05, 2400
    t = np.linspace(0.0, dt * n, n + 1)
    surv = with_end_mass(surv_lognormal(t, 1.69, 1.79), 0.10)
    held = held_time(surv, dt)
    lam = 2e-5

    def v0_and_pay(k_restore: float) -> tuple[float, float]:
        c = lam * held + k_restore * (surv - surv[-1])
        s = int(np.argmin(c))
        return float(c[s]), float(surv[s] - surv[-1])

    kc, kd = 7e-5, 4.0e-3
    ks = np.linspace(kc, kd, 200_001)
    pays = np.array([v0_and_pay(k)[1] for k in ks[:: 400]])
    integral = float(np.trapezoid(pays, ks[::400]))
    v_c, p_c = v0_and_pay(kc)
    v_d, p_d = v0_and_pay(kd)
    print(f"  tool LN + 10% ended, lam=2e-5, kc=7e-5, kd=4e-3: V0(kd)-V0(kc) = {v_d - v_c:.6e}; integral of P(pay) = {integral:.6e}")
    print(f"     bounds (kd-kc)*P(pay|kd) = {(kd-kc)*p_d:.6e} <= value <= (kd-kc)*P(pay|kc) = {(kd-kc)*p_c:.6e}")


# ----------------------------------------------------------------------------- B. missing the due date
def due_date_cost() -> None:
    dt, n = 0.01, 60_000
    t = np.linspace(0.0, dt * n, n + 1)
    surv = surv_two_piece_lognormal(t, 0.41, 1.69, 16.8)
    held = held_time(surv, dt)
    kc_over_l = 7e-5
    nu = index_hull(surv, held, kc_over_l)
    for lam in (5e-6, 2e-5):
        k_due = int(np.argmax(nu <= lam))
        c = lam * held + kc_over_l * surv
        print(f"  tool law, kc/L = 7e-5, lam = {lam:.0e}: due date (clean index falls to lam) at idle age {k_due*dt:.2f} s")
        for late in (1.0, 5.0, 20.0, 60.0, 300.0):
            k2 = min(n, k_due + int(round(late / dt)))
            extra = (c[k2] - c[k_due]) / surv[k_due]  # per token, given not back at the due date
            rate_end = lam - nu[k2]  # falling hazard: cost rate per token-second at the late time
            print(
                f"     {late:6.1f} s late: extra cost {extra:.3e} per token = {extra/(lam*late):.3f} of lam*lateness; "
                f"cost rate at that time {rate_end/lam:.3f} of lam"
            )


# ----------------------------------------------------------------------------- C. life of a page
def life_table() -> None:
    dt, n = 0.05, 24_000  # 1200 s
    t = np.linspace(0.0, dt * n, n + 1)
    gamma = 0.043e-3  # s of GPU per token, FA3 under load
    beta = 21_400.0  # tokens/s
    base = surv_two_piece_lognormal(t, 0.41, 1.69, 16.8)
    print("  tool law + P(ended) = 0.05; kc = kappa/beta (a read at device price, no exposed stall); w = kappa/beta; kd = mu*gamma")
    print("  lam        mu    kappa |  clean TTL  |  life of a dirty page (lazy)             | write worth it?  P(return) at the write")
    for lam in (1e-7, 1e-6, 1e-5, 1e-4):
        for mu, kappa in ((1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (20.0, 1.0), (94.0, 1.0), (94.0, 30.0)):
            surv = with_end_mass(base, 0.05)
            kc = kappa / beta
            w = kappa / beta
            kd = mu * gamma
            held = held_time(surv, dt)
            c_c = lam * held + kc * (surv - surv[-1])
            a_c = int(np.argmin(c_c)) * dt
            c_w, c_n = two_lives(surv, dt, lam, kd, kc, w)
            kw, kn = int(np.argmin(c_w)), int(np.argmin(c_n))
            if kd <= kc:
                life = f"hold {kn*dt:7.1f} s, drop dirty (recompute cheaper than a read)"
                worth = "no"
                pi = float("nan")
            elif c_w[kw] <= c_n[kn]:
                life = f"hold {kw*dt:7.1f} s, write, drop"
                worth = "yes"
                pi = (surv[kw] - surv[-1]) / surv[kw]
            else:
                life = f"hold {kn*dt:7.1f} s, drop dirty, never write"
                worth = "no"
                pi = (surv[kn] - surv[-1]) / surv[kn]
            print(f"  {lam:.0e}  {mu:5.0f}  {kappa:5.0f} | {a_c:8.1f} s | {life:40s} | {worth:3s}  {pi:.3f}")


# ----------------------------------------------------------------------------- D. idle/busy device
def idle_busy_device() -> None:
    dt, n = 0.05, 4800
    t = np.linspace(0.0, dt * n, n + 1)
    surv = with_end_mass(surv_two_piece_lognormal(t, 0.41, 1.69, 16.8), 0.05)
    lam, kc, kd = 5e-6, 4.7e-5, 4.0e-3
    w_busy = 4.7e-5
    print(f"  tool law + 5% ended, lam={lam}, kc={kc}, kd={kd}; device price 0 when idle, {w_busy} per token when busy")
    for busy_share, mean_busy_s in ((0.2, 2.0), (0.6, 5.0), (0.9, 20.0)):
        to_idle = dt / mean_busy_s
        to_busy = to_idle * busy_share / (1 - busy_share)
        vc = np.zeros(n + 1)
        vd = np.zeros((n + 1, 2))
        act = np.full((n + 1, 2), DROP)
        for k in range(n - 1, -1, -1):
            pi = (surv[k] - surv[n]) / surv[k]
            q = surv[k + 1] / surv[k]
            vc[k] = min(kc * pi, lam * dt + q * vc[k + 1])
            nxt_idle = (1 - to_busy) * vd[k + 1, 0] + to_busy * vd[k + 1, 1]
            nxt_busy = to_idle * vd[k + 1, 0] + (1 - to_idle) * vd[k + 1, 1]
            for dev, nxt, w in ((0, nxt_idle, 0.0), (1, nxt_busy, w_busy)):
                options = (lam * dt + q * nxt, w + vc[k], kd * pi)
                best = int(np.argmin(options))
                vd[k, dev], act[k, dev] = options[best], best
        first_write_busy = next((k * dt for k in range(n) if act[k, 1] == WRITE), None)
        idle_always_writes = bool(np.all(act[: n - 1, 0][surv[: n - 1] > surv[-1] + 1e-9] != HOLD))
        eager_idle = vc[0]  # born with the device idle: write at once, free
        eager_busy = w_busy + vc[0]
        print(
            f"    busy {busy_share:.0%} of the time, busy spells {mean_busy_s:.0f} s: born idle -> value {vd[0,0]:.4e} "
            f"(eager {eager_idle:.4e}); born busy -> value {vd[0,1]:.4e} (eager {eager_busy:.4e}, saving "
            f"{(eager_busy - vd[0,1])/w_busy:.2f} of a write); in the busy state the write starts at age {first_write_busy}; "
            f"idle state never holds: {idle_always_writes}"
        )


# ----------------------------------------------------------------------------- E. many gaps
class Law:
    def __init__(self, p10: float, p50: float, p90: float, cap_s: float, dt: float):
        n = int(round(cap_s / dt))
        self.t = np.linspace(0.0, cap_s, n + 1)
        self.surv = surv_two_piece_lognormal(self.t, p10, p50, p90)
        self.surv[-1] = 0.0  # the replay caps gaps: whoever is still out returns at the cap
        self.dt = dt


def dp_multi(laws: list[Law], probs: list[float], end_prob: float, lam: float, kd: float, kc: float, w: float):
    """Fixed point over the continuation values after a return (page clean: cc; page dirty: cd)."""
    cc = cd = 0.0
    out = None
    for _ in range(500):
        vals_c, vals_d, acts = [], [], []
        for law in laws:
            surv, dt = law.surv, law.dt
            n = len(surv) - 1
            vc = np.zeros(n + 1)
            vd = np.zeros(n + 1)
            act_d = np.full(n + 1, DROP)
            act_c = np.full(n + 1, DROP)
            # at the cap everyone returns: the last point is a certain return
            for k in range(n - 1, -1, -1):
                pi = 1.0  # no mass at infinity inside a gap; the end of a session is drawn at the return
                q = surv[k + 1] / surv[k]
                r = 1.0 - q
                drop_c = pi * (kc + cc)
                hold_c = lam * dt + r * cc + q * vc[k + 1]
                vc[k], act_c[k] = (drop_c, DROP) if drop_c <= hold_c else (hold_c, HOLD)
                options = (lam * dt + r * cd + q * vd[k + 1], w + vc[k], pi * (kd + cd))
                best = int(np.argmin(options))
                vd[k], act_d[k] = options[best], best
            vals_c.append(vc[0])
            vals_d.append(vd[0])
            acts.append((act_c, act_d))
        new_cc = (1 - end_prob) * sum(p * v for p, v in zip(probs, vals_c))
        new_cd = (1 - end_prob) * sum(p * v for p, v in zip(probs, vals_d))
        done = abs(new_cc - cc) + abs(new_cd - cd) < 1e-13
        cc, cd = new_cc, new_cd
        out = (vals_c, vals_d, acts)
        if done:
            break
    return cc, cd, out


def multi_gap() -> None:
    dt = 0.1
    tool = Law(0.41, 1.69, 16.8, 1200.0, dt)
    human = Law(3.1, 96.0, 1040.0, 1200.0, dt)
    laws, probs = [tool, human], [0.92, 0.08]
    end_prob = 1.0 / 29.2  # 29.2 requests per user (campaign 28 load model)
    gamma, beta = 0.043e-3, 21_400.0
    print("  gaps: tool (92 %) and human (8 %), class known at the turn's end; a turn is the session's last with prob 1/29.2")
    print("  lam       mu  kappa | lazy: write age tool / human (s) | share of pages ever written | closed form | (eager - lazy)/w")
    for lam, mu, kappa in ((2e-6, 94.0, 1.0), (2e-5, 94.0, 1.0), (2e-5, 94.0, 10.0), (2e-4, 94.0, 1.0), (2e-5, 5.0, 1.0)):
        kc = kappa / beta
        w = kappa / beta
        kd = mu * gamma
        cc, cd, (vals_c, vals_d, acts) = dp_multi(laws, probs, end_prob, lam, kd, kc, w)
        ages, xs = [], []
        for law, (act_c, act_d) in zip(laws, acts):
            k = 0
            n = len(law.surv) - 1
            while k < n and act_d[k] == HOLD:
                k += 1
            what = {WRITE: "write", DROP: "drop dirty"}[int(act_d[k])] if k < n else "hold to return"
            ages.append(f"{k*law.dt:7.1f} ({what})")
            xs.append(law.surv[k] if (k < n and act_d[k] == WRITE) else 0.0)
        x = sum(p * v for p, v in zip(probs, xs))
        ever = x / (x + end_prob - x * end_prob) if x > 0 else 0.0
        lazy = sum(p * v for p, v in zip(probs, vals_d))
        eager = w + sum(p * v for p, v in zip(probs, vals_c))
        print(
            f"  {lam:.0e}  {mu:4.0f}  {kappa:4.0f} | {ages[0]:>22s} / {ages[1]:>22s} | x per gap {x:.4f} | "
            f"ever written {ever:.3f} | {(eager - lazy)/w:+.3f}"
        )
    # the share of pages ever due for a plain TTL of d seconds, against the measured 56-67 % read back
    print("  plain TTL d: share of gaps longer than d (x) and share of pages ever due x/(x + e - x e), e = 1/29.2:")
    for d in (10.0, 30.0, 60.0, 120.0, 300.0):
        k = int(round(d / dt))
        x = 0.92 * tool.surv[k] + 0.08 * human.surv[k]
        print(f"     d = {d:5.0f} s: x = {x:.4f}  ever due = {x / (x + end_prob - x * end_prob):.3f}")


def main() -> None:
    print("=" * 100)
    print("A. Single gap, constant prices")
    check_single_gap()
    check_value_of_write()
    print("=" * 100)
    print("B. Cost of missing a write's due date")
    due_date_cost()
    print("=" * 100)
    print("C. Life of a page")
    life_table()
    print("=" * 100)
    print("D. Idle/busy device")
    idle_busy_device()
    print("=" * 100)
    print("E. Many gaps")
    multi_gap()


if __name__ == "__main__":
    main()
