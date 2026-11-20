"""Campaign 28 or 29: does the kind of hint matter after long waits? Pairs two arms on the returning requests both
sent before 6,600 s and splits them by the wait before the request, which is what its hint had to estimate.
On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/hint_effect.py results/campaign28 jit_exact jit_p50 > results/campaign28/checks/hint_effect_jit_exact_jit_p50.out
Per group: requests, users, TTFT mean in each arm, B - A with a 95 % interval over users (2,000 draws, seed 0), the
share of requests that read from SSD and the share of the history computed again in each arm. Stdlib only."""
import collections
import json
import random
import sys

cdir, a_name, b_name = sys.argv[1:4]
GROUPS = (("0-1 s", 0, 1), ("1-10 s", 1, 10), ("10-60 s", 10, 60), ("60-300 s", 60, 300), ("300 s or more", 300, 1e9), ("all", 0, 1e9))


def load(label):
    with open(f"{cdir}/manifest.txt") as f:
        run, programs = next(line.split()[1:3] for line in f if line.split()[0] == label)
    out = {}
    with open(f"results/{run}/{programs}/client.jsonl") as f:
        for line in f:
            if '"kind":"req"' in line:
                r = json.loads(line)
                if r["phase"] == "profile" and r["k"] > 0 and r["t_send"] < 6600 and r.get("ttft") is not None and not r.get("error"):
                    out[(r["play"], r["stream"], r["k"])] = r
    return out


def cached(r):
    d = r.get("cached_details") or {}
    return sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)


def again(rs):
    ideal = sum(r["ideal_cached"] for r in rs)
    return 100 * sum(max(0, r["ideal_cached"] - cached(r)) for r in rs) / ideal if ideal else float("nan")


def ssd(rs):
    return 100 * sum(int((r.get("cached_details") or {}).get("storage") or 0) > 0 for r in rs) / max(1, len(rs))


a, b = load(a_name), load(b_name)
shared = sorted(set(a) & set(b))
rng = random.Random(0)
print(f"{cdir.split('/')[-1]}: A = {a_name}, B = {b_name}; {len(shared)} returning requests both sent before 6,600 s, by the wait before the request")
print(f"{'wait before':14s} | {'requests':>8s} {'users':>5s} | {'A TTFT':>7s} {'B TTFT':>7s} | {'B - A':>7s} {'interval':>17s} | {'from SSD A, B':>14s} | {'computed again A, B':>19s}")
for name, lo, hi in GROUPS:
    keys = [k for k in shared if lo <= a[k]["wait_s"] < hi]
    if not keys:
        continue
    by_play = collections.defaultdict(lambda: [0.0, 0])
    for k in keys:
        by_play[k[0]][0] += b[k]["ttft"] - a[k]["ttft"]
        by_play[k[0]][1] += 1
    plays = list(by_play.values())
    draws = []
    for _ in range(2000):
        pick = rng.choices(plays, k=len(plays))
        draws.append(sum(p[0] for p in pick) / sum(p[1] for p in pick))
    draws.sort()
    ra, rb = [a[k] for k in keys], [b[k] for k in keys]
    ta, tb = sum(r["ttft"] for r in ra) / len(keys), sum(r["ttft"] for r in rb) / len(keys)
    print(f"{name:14s} | {len(keys):8d} {len(plays):5d} | {ta:7.2f} {tb:7.2f} | {tb - ta:+7.2f} [{draws[49]:+7.2f}, {draws[1949]:+7.2f}] | "
          f"{ssd(ra):5.1f}% {ssd(rb):5.1f}% | {again(ra):8.2f}% {again(rb):8.2f}%")
