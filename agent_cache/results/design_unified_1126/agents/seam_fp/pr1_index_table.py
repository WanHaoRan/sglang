"""PR1 check (CPU): the holding index nu(a)/k = sup_{h >= h_min} F(h | a) / tau(h | a) on the engine's own
GapDistribution (continuous survival, not the 64-point support), against
  - the same sup taken over the 64-point support (what a planner on remaining_gaps() would compute), and
  - chance_within(idle, 30 s) / 30 s, the score the return_chance eviction strategy already ranks by.
tau(h | a) = integral_0^h S(a + u) / S(a) du (trapezoid on a geometric grid).
Usage: <venv python> pr1_index_table.py"""
import importlib.util, math, sys
spec = importlib.util.spec_from_file_location("ntp_policy", "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py")
pol = importlib.util.module_from_spec(spec); sys.modules["ntp_policy"] = pol; spec.loader.exec_module(pol)

def index_cont(d, a, h_min, h_max=1200.0, steps=400):
    s0 = d.survival(a)
    if s0 <= 0:
        return 0.0, 0.0
    best, best_h = 0.0, 0.0
    # geometric grid from 1 ms so the integral below h_min is accurate
    u_prev, tau = 0.0, 0.0
    s_prev = 1.0
    ratio = (h_max / 1e-3) ** (1 / steps)
    u = 1e-3
    for _ in range(steps + 1):
        s = d.survival(a + u) / s0
        tau += 0.5 * (s_prev + s) * (u - u_prev)
        if u >= h_min and tau > 0:
            v = (1.0 - s) / tau
            if v > best:
                best, best_h = v, u
        u_prev, s_prev = u, s
        u *= ratio
    return best, best_h

def index_support(d, a):
    rem = pol.remaining_gaps(support=d.support(), idle_s=a)
    if not rem:
        return None
    n, acc, best = len(rem), 0.0, 0.0
    for j, g in enumerate(rem, start=1):
        acc += g
        tau = (acc + (n - j) * g) / n
        if tau > 0:
            best = max(best, (j / n) / tau)
    return best

entries = {"tool 0.41/1.69/16.8": pol.GapDistribution.from_quantiles(p10=0.414, p50=1.692, p90=16.795),
           "user 3.1/96/1040": pol.GapDistribution.from_quantiles(p10=3.084, p50=96.228, p90=1040.492),
           "sharp 18/20/22": pol.GapDistribution.from_quantiles(p10=18.0, p50=20.0, p90=22.0)}
for name, d in entries.items():
    print(f"== {name}: sigma_low {d.sigma_low:.2f}, sigma_high {d.sigma_high:.2f}")
    print("   idle s | hazard-like sup (h >= 0.5 s) at h | sup (h >= 5 s) at h | 64-point support sup | chance_within(30 s)/30")
    for a in (0.0, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0):
        b1, h1 = index_cont(d, a, 0.5); b5, h5 = index_cont(d, a, 5.0)
        sup = index_support(d, a)
        fixed = d.chance_within(idle_s=a, horizon_s=30.0) / 30.0
        print(f"   {a:6.0f} | {b1:9.5f} at {h1:7.1f} s | {b5:9.5f} at {h5:7.1f} s | {'      none' if sup is None else f'{sup:10.5f}'} | {fixed:9.5f}")
