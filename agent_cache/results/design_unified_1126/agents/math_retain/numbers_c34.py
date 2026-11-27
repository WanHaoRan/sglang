"""Numbers behind the Retain / write results, from the measured constants of campaign 34.

Everything here is arithmetic on the brief's constants plus the model of DESIGN.md; nothing is
measured by this script. Prices are per token: seconds of summed TTFT.

Run: python -I -B numbers_c34.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hold_index import held_time, index_hull, surv_two_piece_lognormal  # noqa: E402

BETA = 21_400.0  # tokens/s, one SSD shared by reads and writes
LAGS = 0.9  # s, fixed lags of a read
GAMMA = 0.043e-3  # s of GPU per recomputed token under load (0.037-0.044 ms)
N_USERS = 128
MU_SAT = 0.73 * N_USERS  # measured: 94 TTFT-seconds per GPU-second
HOST_ROOM = 920_000.0  # tokens of host memory beyond the mirror (0.90-0.92 M)
MIRROR = 650_000.0
RATE = 2.2  # returning requests per second at saturation (129-133 a minute)
L_HIST = 24_900.0  # mean returning history, tokens
END_PROB = 1.0 / 29.2


def laws(dt: float, cap: float, which: str):
    n = int(round(cap / dt))
    t = np.linspace(0.0, cap, n + 1)
    if which == "trace":
        tool = surv_two_piece_lognormal(t, 0.41, 1.69, 16.8)
        human = surv_two_piece_lognormal(t, 3.1, 96.0, 1040.0)
    else:  # as replayed in campaign 34 ec_full
        tool = surv_two_piece_lognormal(t, 0.32, 1.02, 8.07)
        human = surv_two_piece_lognormal(t, 2.8, 64.0, 539.0)
    tool[-1] = 0.0
    human[-1] = 0.0
    return t, tool, human


def hazard_from(surv: np.ndarray, dt: float) -> np.ndarray:
    h = np.zeros_like(surv)
    h[:-1] = (surv[:-1] - surv[1:]) / (np.maximum(surv[:-1], 1e-300) * dt)
    return h


def restore_costs() -> None:
    print("1. Restore cost per token (seconds of summed TTFT per token)")
    print(f"   recompute: mu*gamma = {GAMMA:.2e} (mu = 1), {2*GAMMA:.2e} (mu = 2), {MU_SAT*GAMMA:.2e} (mu = {MU_SAT:.0f}, saturation)")
    for tokens in (2_000.0, 8_000.0, 25_000.0, 60_000.0):
        read_idle = (LAGS + tokens / BETA) / tokens  # c = 1, W = 0, device free: the whole read is exposed
        print(
            f"   read, light load (c = 1, W = 0, kappa = 0), L = {tokens:6.0f}: (lags + L/beta)/L = {read_idle:.2e} per token "
            f"= {read_idle/GAMMA:5.2f} x recompute at mu = 1"
        )
    print(f"   read, saturation (c = 0): kappa/beta = kappa x {1/BETA:.2e} per token, no size term")
    print(f"   write: kappa/beta = kappa x {1/BETA:.2e} per token")
    print(f"   so at saturation dirty/clean index ratio = mu*gamma*beta/kappa = {MU_SAT*GAMMA*BETA:.0f}/kappa")


def frontier_price() -> None:
    print("2. Price of host room read off the eviction frontier: lam = (K/L) * hazard(T_C) for a pooled falling-hazard law")
    dt = 0.05
    for which, share_tool, share_human in (("trace", 0.92, 0.08), ("replayed", 0.921, 0.079)):
        t, tool, human = laws(dt, 1200.0, which)
        pooled = share_tool * tool + share_human * human
        held = held_time(pooled, dt)
        nu = index_hull(pooled, held, 1.0)  # sup F/tau: equals the hazard where the hazard falls
        haz = hazard_from(pooled, dt)
        print(f"   gap laws: {which}")
        for tc in (10.0, 30.0, 60.0, 120.0, 300.0):
            k = int(round(tc / dt))
            # average the raw hazard over 2 s to remove grid noise
            w = int(round(1.0 / dt))
            hz = float(np.mean(haz[k - w : k + w]))
            lam_light = GAMMA * nu[k]
            lam_sat_clean = (1.0 / BETA) * nu[k]
            lam_sat_dirty = MU_SAT * GAMMA * nu[k]
            occ = RATE * L_HIST * held[k]
            print(
                f"     eviction age T_C = {tc:5.0f} s: pooled hazard {hz:.4f}/s, sup F/tau {nu[k]:.4f}/s | lam = {lam_light:.1e} (light load, K/L = gamma) "
                f"| {lam_sat_clean:.1e} x kappa (saturation, clean) | {lam_sat_dirty:.1e} (only copy, mu = {MU_SAT:.0f}) "
                f"| idle tokens held at this TTL = {occ/1e6:.2f} M"
            )


def ttl_table() -> None:
    print("3. Time a page is kept (first idle age with index <= lam), per gap class, trace-wide laws")
    dt = 0.05
    t, tool, human = laws(dt, 1200.0, "trace")
    for name, surv in (("tool", tool), ("human", human)):
        held = held_time(surv, dt)
        base = index_hull(surv, held, 1.0)
        for lam in (5e-7, 2e-6, 5e-5):
            row = []
            for label, k_tok in (("K/L=gamma (light)", GAMMA), ("clean, kappa=1", 1.0 / BETA), (f"only copy, mu={MU_SAT:.0f}", MU_SAT * GAMMA)):
                nu = k_tok * base
                idx = np.where(nu <= lam)[0]
                age = idx[0] * dt if len(idx) else float("inf")
                row.append(f"{label}: {age:7.1f} s")
            print(f"   {name:5s} gap, lam = {lam:.0e}: " + " | ".join(row))


def mirror() -> None:
    print("4. The mirror (host copies of pages the GPU still holds)")
    print(f"   mirror {MIRROR/1e6:.2f} M tokens of a {1_627_648/1e6:.2f} M pool; host-only room {HOST_ROOM/1e6:.2f} M -> room without the mirror x{(HOST_ROOM+MIRROR)/HOST_ROOM:.2f}")
    for lam in (5e-7, 2e-6, 5e-5):
        rent = lam * MIRROR
        print(f"   lam = {lam:.0e}: mirror rent {rent:.2f} TTFT-s per second = {rent/RATE:.2f} s per request at {RATE} requests/s")
    d2h = 354_000.0  # tokens/s, one measured copy: 2,304 tokens in 6.5 ms
    print(f"   backup at the turn's end instead of at the end of prefill: copy of {L_HIST:.0f} tokens takes {L_HIST/d2h*1e3:.0f} ms "
          f"(one measured copy: 2,304 tokens in 6.5 ms); at {RATE} requests/s that is {RATE*L_HIST/d2h:.2f} of one copy stream")
    print(f"   if that copy blocked the scheduler it would cost mu*L/rate = {MU_SAT*L_HIST/d2h:.1f} TTFT-s per request at mu = {MU_SAT:.0f}")


def ever_due() -> None:
    print("5. Share of pages whose host copy is ever given up before the session ends, x/(x + e - x e), e = 1/29.2")
    dt = 0.05
    for which in ("trace", "replayed"):
        t, tool, human = laws(dt, 1200.0, which)
        row = []
        for d in (30.0, 60.0, 120.0):
            k = int(round(d / dt))
            x = 0.92 * tool[k] + 0.08 * human[k]
            row.append(f"TTL {d:.0f} s: x = {x:.3f}, ever due {x/(x + END_PROB - x*END_PROB):.3f}")
        print(f"   {which:9s}: " + " | ".join(row))
    print("   measured (campaign 34): 56-67 % of prompt positions are read back from the SSD; an idle prefix stays about a minute in host memory")


def fluid() -> None:
    print("6. Host room and device at saturation (rough fluid balance)")
    demand = RATE * L_HIST
    print(f"   history flow: {RATE} requests/s x {L_HIST:.0f} tokens = {demand/1e3:.1f} K tokens/s; device for reads about 17.4-18.0 K tokens/s")
    print(f"   -> at most {17_700/demand:.2f} of the history flow can come from the SSD (measured: 24-28 % of returning requests read)")
    for wait in (10.0, 20.0, 30.0, 50.0):
        print(f"   queue wait {wait:4.0f} s: queued histories alone hold {demand*wait/1e6:.2f} M tokens (room {HOST_ROOM/1e6:.2f} M)")


def main() -> None:
    restore_costs()
    frontier_price()
    ttl_table()
    mirror()
    ever_due()
    fluid()


if __name__ == "__main__":
    main()
