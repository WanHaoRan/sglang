"""Campaign 28 or 29: how long a session sits idle between turns, by harness group and by the cap on a recorded wait.
In the container, from agent_cache/scripts/:
  python3 -B ../results/campaign28/checks/think_times.py ../results/campaign28 [max context] > ../results/campaign28/checks/think_times.out
The idle time per turn against the reply time sets how much session KV waits outside the running batch: with
think time Z and reply time R, the sessions hold about (1 + Z / R) times the KV of the running requests.
Over the users that arrive (the client's own loader and draw). Stdlib and the client modules only."""
import random
import sys
import types

sys.path.insert(0, ".")
import replay_programs  # noqa: E402

cdir = sys.argv[1]
settings = dict(line.strip().split("=", 1) for line in open(f"{cdir}/settings.env") if "=" in line)
args = types.SimpleNamespace(trace=settings["TRACE"], wait_field="wait", split_idle=None, split_compaction=False,
                             max_context=int(sys.argv[2]) if len(sys.argv) > 2 else int(settings["MAX_CONTEXT"]),
                             sample="shuffle", seed=int(settings["SEED"]), trees=0)
header, trees, dropped = replay_programs.load_trees(args)
rng, t, n = random.Random(f"{args.seed}:arrivals"), 0.0, 0
while t < float(settings["DURATION"]) and n < len(trees):   # run_arrivals' own draw
    n += 1
    t += rng.expovariate(float(settings["RATE"]))
users = trees[:n]
B = header.get("block_tokens") or 0
CAPS = (1200.0, 3600.0, 7200.0, float("inf"))


def prompt(r):
    return (r["keep"] + len(r["new"])) * B + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


print(f"{settings['TRACE'].split('/')[-1]}, max context {args.max_context:,}: {len(trees)} sessions kept; the {len(users)} that arrive")
print(f"{'group':22s} | {'users':>5s} {'turns':>6s} {'a user':>6s} | {'prompt K':>8s} {'output':>6s} | {'wait p50 s':>10s} {'p90':>6s} {'p99':>7s} {'max':>8s} | "
      f"{'over 60 s':>9s} {'over 300 s':>10s} {'over 1200 s':>11s} | mean wait s capped at 1200, 3600, 7200, not capped | idle time held by waits over 60 s (cap 1200)")
groups = {"all": [], "eval:swe-bench": [], "the other harnesses": []}
for tree in users:
    name = "eval:swe-bench" if (tree.get("system") or ("?",))[0] == "eval:swe-bench" else "the other harnesses"
    groups["all"].append(tree)
    groups[name].append(tree)
for name, ts in groups.items():
    waits, prompts, outs, turns = [], 0, 0, 0
    for tree in ts:
        for st in tree["streams"]:
            for k, r in enumerate(st["requests"]):
                turns += 1
                prompts += prompt(r)
                outs += r["out"]
                if k:
                    waits.append(float(r["wait"] or 0))
    if not waits:
        continue
    means = " ".join(f"{sum(min(w, c) for w in waits) / turns:7.1f}" for c in CAPS)
    capped = [min(w, 1200.0) for w in waits]
    long_share = sum(w for w in capped if w > 60) / max(1e-9, sum(capped))
    print(f"{name:22s} | {len(ts):5d} {turns:6d} {turns / len(ts):6.1f} | {prompts / turns / 1e3:8.1f} {outs / turns:6.0f} | {q(waits, .5):10.2f} {q(waits, .9):6.1f} {q(waits, .99):7.0f} {max(waits):8.0f} | "
          f"{100 * sum(w > 60 for w in waits) / len(waits):8.1f}% {100 * sum(w > 300 for w in waits) / len(waits):9.1f}% {100 * sum(w > 1200 for w in waits) / len(waits):10.2f}% | "
          f"{means} | {100 * long_share:5.1f}%")
