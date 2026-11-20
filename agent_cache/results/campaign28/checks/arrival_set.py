"""The users a campaign 28 or 29 arm replays: the trees the client keeps and draws, by harness, with their output lengths.
In the container, from agent_cache/scripts/:
  python3 ../results/campaign28/checks/arrival_set.py ../results/campaign28/settings.env > ../results/campaign28/checks/arrival_set.out
Uses the client's own loader and arrival times, so the set is the one every arm sees (same seed, same order): the
users whose arrival time falls before DURATION. Under a binding cap fewer are admitted; that count is each arm's own.
Stdlib and the client modules only."""
import collections
import random
import sys
import types

sys.path.insert(0, ".")
import replay_programs  # noqa: E402

settings = dict(line.strip().split("=", 1) for line in open(sys.argv[1]) if "=" in line)
args = types.SimpleNamespace(trace=settings["TRACE"], wait_field="wait", split_idle=None, split_compaction=False,
                             max_context=int(settings["MAX_CONTEXT"]), sample="shuffle", seed=int(settings["SEED"]), trees=0)
header, trees, dropped = replay_programs.load_trees(args)
rate, duration, gap_cap = float(settings["RATE"]), float(settings["DURATION"]), float(settings["GAP_CAP"])
rng, t, n = random.Random(f"{args.seed}:arrivals"), 0.0, 0
while t < duration and n < len(trees):   # run_arrivals' own draw
    n += 1
    t += rng.expovariate(rate)
users = trees[:n]


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


print(f"{settings['TRACE'].split('/')[-1]}: {len(trees)} trees kept under {args.max_context:,} tokens, {dropped} dropped; {n} arrive in {duration:.0f} s at {rate}/s")
by = collections.defaultdict(lambda: [0, 0, 0, [], []])   # users, requests, output tokens, waits, keyed requests
outs, systems = [], collections.Counter()
for tree in users:
    system = tuple(tree.get("system") or ())
    systems[system] += 1
    row = by[system[0] if system else "?"]
    row[0] += 1
    for st in tree["streams"]:
        rs = st["requests"]
        for k, r in enumerate(rs):
            row[1] += 1
            row[2] += r["out"]
            outs.append(r["out"])
            if k:
                row[3].append(min(float(r["wait"] or 0), gap_cap))
            if r.get("key"):
                row[4].append(1)
total = sum(r[1] for r in by.values())
print(f"agent systems (harness, model) among them: {len(systems)}; users in the largest {systems.most_common(3)}; systems with one user {sum(1 for c in systems.values() if c == 1)}")
print(f"output tokens a request: mean {sum(outs) / len(outs):.0f}, p50 {q(outs, .5)}, p99 {q(outs, .99)}, max {max(outs)}; over 8,000: {sum(o > 8000 for o in outs)}, over 16,000: {sum(o > 16000 for o in outs)}")
print(f"{'harness':18s} {'users':>6s} {'requests':>9s} {'share':>6s} {'out/request':>11s} {'wait p50 s':>10s} {'waits >60 s':>11s} {'keyed':>6s}")
for name, (u, reqs, out, waits, keyed) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"{name[:18]:18s} {u:6d} {reqs:9d} {100 * reqs / total:5.1f}% {out / reqs:11.0f} {q(waits, .5):10.2f} {100 * sum(w > 60 for w in waits) / max(1, len(waits)):10.1f}% "
          f"{100 * len(keyed) / reqs:5.1f}%")
