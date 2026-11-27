"""Referee: the floor under a paired difference. Replicas of the same users (one seed) with up to 5 ms of send
jitter differ only by chaos; two replicas of the SAME arm, paired request by request, give the size of a difference
that means nothing. Then every arm pair over all replica pairs.

  python -I -B ref_noise.py <runs folder> <seed> <noise seeds, comma-joined> [setting prefix, e.g. 'fixes=off,']
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reflib  # noqa: E402

runs, seed = sys.argv[1], int(sys.argv[2])
nseeds = [int(x) for x in sys.argv[3].split(",")]
prefix = sys.argv[4] if len(sys.argv) > 4 else ""
ARMS = ("hicache", "jit_p50", "jit_exact", "ec_full", "ec_exact")


def folder(arm, k):
    return os.path.join(runs, reflib_slug(f"{prefix}noise=0.005,noise_seed={k}"), arm, f"seed{seed}")


def reflib_slug(text):
    return "".join(c if c.isalnum() or c in "._-=" else "_" for c in text) or "default"


req = {(arm, k): reflib.sim_requests(folder(arm, k)) for arm in ARMS for k in nseeds
       if os.path.exists(os.path.join(folder(arm, k), "requests.csv"))}
print(f"seed {seed}, setting '{prefix}noise=0.005', replicas {nseeds}")
print("\n| arm | mean TTFT of the replicas, s | max - min | same arm, replica against replica: paired difference, mean abs / max abs, s |")
print("|---|---|---|---|")
for arm in ARMS:
    ks = [k for k in nseeds if (arm, k) in req]
    means = [float(np.mean(list(req[(arm, k)].values()))) for k in ks]
    d = [reflib.paired(req[(arm, a)], req[(arm, b)], draws=50)["diff"] for a, b in itertools.combinations(ks, 2)]
    print(f"| {arm} | {', '.join(f'{m:.2f}' for m in means)} | {max(means) - min(means):.2f} | {np.mean(np.abs(d)):.2f} / {np.max(np.abs(d)):.2f} |")
print("\n| pair A -> B | paired B - A over all replica pairs (A's replica x B's replica): mean [min, max], s | share of replica pairs with the sign of the mean |")
print("|---|---|---|")
for a, b in (("jit_p50", "ec_full"), ("jit_p50", "jit_exact"), ("ec_full", "jit_exact"), ("ec_full", "ec_exact"),
             ("jit_exact", "ec_exact"), ("hicache", "ec_full")):
    d = [reflib.paired(req[(a, i)], req[(b, j)], draws=50)["diff"] for i in nseeds for j in nseeds
         if (a, i) in req and (b, j) in req]
    d = np.array(d)
    sign = np.sign(d.mean())
    print(f"| {a} -> {b} | {d.mean():+.2f} [{d.min():+.2f}, {d.max():+.2f}] | {100 * (np.sign(d) == sign).mean():.0f} % of {len(d)} |")
