"""Reconstruct host-pool occupancy O(t) and the device-resident host mirror M(t)
from HICACHE_EVT lines.

O(t) = sum d2h_submit tokens + sum prefetch 'loaded' tokens - sum evict_host tokens
M(t) = sum d2h_submit tokens + sum h2d_submit tokens - sum evict_device tokens
       (backed-up nodes that currently also hold a device copy; write_through,
        threshold 1, so every inserted device node is backed up)
U(t) = device tokens in use by running requests (#token of latest Decode batch
       line / token usage*668160 on Prefill lines) -- upper bound on the locked set.
"""
import re, sys, statistics as st
from datetime import datetime

H = 1_627_648
D = 668_160
R = "/home/wanhr/sglang/agent_cache/results/"
runs = sys.argv[1:]

ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
def ts(line):
    m = ts_re.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None

def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]

for run in runs:
    W = L = E = HE = P = 0
    U = 0
    t0 = None
    first_he = None
    samples = []  # (t, O, M, U) after each event
    with open(R + run + "/server.log", errors="replace") as f:
        for line in f:
            if "HICACHE_EVT" not in line and "prefetch success" not in line and "batch," not in line:
                continue
            t = ts(line)
            if t is None:
                continue
            if "Decode batch" in line:
                m = re.search(r"#token: (\d+)", line); U = int(m.group(1))
                if t0 is None: t0 = t
                continue
            if "Prefill batch" in line:
                m = re.search(r"token usage: ([\d.]+)", line); U = float(m.group(1)) * D
                if t0 is None: t0 = t
                continue
            m = re.search(r"HICACHE_EVT (\w+) .*?tokens=(\d+)", line)
            if m:
                ev, n = m.group(1), int(m.group(2))
                if ev == "d2h_submit": W += n
                elif ev == "h2d_submit": L += n
                elif ev == "evict_device": E += n
                elif ev == "evict_host":
                    HE += n
                    if first_he is None: first_he = (t, W + P - HE + n, W + L - E, U)
                else:
                    continue
            elif "prefetch success" in line:
                P += int(re.search(r"loaded=(\d+)", line).group(1))
            else:
                continue
            samples.append((t, W + P - HE, W + L - E, U))
    O = [s[1] for s in samples]; M = [s[2] for s in samples]
    print(f"== {run}")
    print(f"  totals: d2h={W:,} h2d={L:,} evict_device={E:,} evict_host={HE:,} storage_loaded={P:,}")
    print(f"  O max={max(O):,} ({max(O)/H:.3f} of H);  M min={min(M):,} max={max(M):,} ({max(M)/D:.3f} of D)")
    if first_he:
        t, o, mm, u = first_he
        print(f"  first evict_host at +{(t-t0)/60:.1f} min: O_before={o:,} ({o/H:.3f} H), M={mm:,} ({mm/H:.3f} of H), U={u:,.0f}")
        post = [s for s in samples if s[0] >= t]
        Ms = [s[2] for s in post]; Us = [s[3] for s in post]
        idle = [max(0, s[2] - s[3]) for s in post]
        # time-weighted mean of M over post period
        tw = sum((post[i+1][0]-post[i][0]) * post[i][2] for i in range(len(post)-1)) / (post[-1][0]-post[0][0])
        twu = sum((post[i+1][0]-post[i][0]) * post[i][3] for i in range(len(post)-1)) / (post[-1][0]-post[0][0])
        twi = sum((post[i+1][0]-post[i][0]) * max(0, post[i][2]-post[i][3]) for i in range(len(post)-1)) / (post[-1][0]-post[0][0])
        print(f"  after host full ({(post[-1][0]-post[0][0])/60:.0f} min, n={len(post)}):")
        print(f"    M (mirror): time-wtd mean={tw:,.0f} ({tw/H:.3f} of H), p10={q(Ms,.1):,} p50={q(Ms,.5):,} p90={q(Ms,.9):,}")
        print(f"    U (in use by running reqs, <= locked+decode tail): time-wtd mean={twu:,.0f} ({twu/H:.3f} of H), p50={q(Us,.5):,.0f}")
        print(f"    M-U (mirror of idle, unlocked device-resident KV, lower bound): time-wtd mean={twi:,.0f} ({twi/H:.3f} of H)")
    else:
        print("  host never evicted")
