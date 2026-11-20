"""Campaign 28 or 29: how many requests (turns) a session has, in the trace and in what each arm replayed.
In the container, from agent_cache/scripts/:
  python3 -B ../results/campaign28/checks/turn_counts.py ../results/campaign28 > ../results/campaign28/checks/turn_counts.out
Uses the client's own loader and arrival draw (as arrival_set.py). Three sets of sessions: every session of the
converted trace, those kept under MAX_CONTEXT, and the users whose arrival falls inside DURATION. Then, per arm of
the manifest, the requests sent before 6,600 s: how many a user got through, and how far into its session.
A turn is one request to the model, on any stream of the session (main agent or subagent). Stdlib and the client."""
import collections
import json
import random
import sys
import types

sys.path.insert(0, ".")
import replay_programs  # noqa: E402

cdir = sys.argv[1]
settings = dict(line.strip().split("=", 1) for line in open(f"{cdir}/settings.env") if "=" in line)
CUT = 6600.0
BINS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 10**9))


def load(max_context):
    args = types.SimpleNamespace(trace=settings["TRACE"], wait_field="wait", split_idle=None, split_compaction=False,
                                 max_context=max_context, sample="shuffle", seed=int(settings["SEED"]), trees=0)
    return replay_programs.load_trees(args)[1]


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


def turns(tree):
    return sum(len(st["requests"]) for st in tree["streams"])


def harness(tree):
    return (tree.get("system") or ("?",))[0]


def row(name, counts):
    n, total = len(counts), sum(counts)
    bins = " ".join(f"{100 * sum(lo <= c <= hi for c in counts) / n:5.1f}" for lo, hi in BINS)
    # Requests that sit in sessions of over 50 turns: what a 50-turn cut would touch.
    deep = 100 * sum(c for c in counts if c > 50) / total
    print(f"{name[:26]:26s} | {n:6d} {total:8d} | {total / n:6.1f} {q(counts, .1):4d} {q(counts, .25):4d} {q(counts, .5):4d} {q(counts, .75):4d} "
          f"{q(counts, .9):4d} {q(counts, .99):5d} {max(counts):5d} | {bins} | {deep:5.1f}%")


HEAD = (f"{'':26s} | {'users':>6s} {'requests':>8s} | {'mean':>6s} {'p10':>4s} {'p25':>4s} {'p50':>4s} {'p75':>4s} {'p90':>4s} {'p99':>5s} {'max':>5s} | "
        f"share of sessions with 1, 2-5, 6-10, 11-20, 21-50, 51-100, 101-200, over 200 requests | requests in sessions over 50")
everything, kept = load(0), load(int(settings["MAX_CONTEXT"]))
rate, duration = float(settings["RATE"]), float(settings["DURATION"])
rng, t, n = random.Random(f"{settings['SEED']}:arrivals"), 0.0, 0
while t < duration and n < len(kept):   # run_arrivals' own draw
    n += 1
    t += rng.expovariate(rate)
users = kept[:n]
print(f"{settings['TRACE'].split('/')[-1]}: requests (turns) per session\n{HEAD}")
for name, trees in (("whole trace", everything), (f"kept under {int(settings['MAX_CONTEXT']):,} tokens", kept), (f"arrive in {duration:.0f} s", users)):
    row(name, [turns(x) for x in trees])
    groups = collections.defaultdict(list)
    for x in trees:
        groups["  eval:swe-bench" if harness(x) == "eval:swe-bench" else "  the other harnesses"].append(turns(x))
    for g in sorted(groups):
        row(g, groups[g])
side = [x for x in users if len(x["streams"]) > 1]
print(f"arriving users with a subagent stream: {len(side)} of {len(users)}; their requests off the main stream: "
      f"{sum(len(st['requests']) for x in side for st in x['streams'][1:])} of {sum(turns(x) for x in users)}")
print(f"\narriving users by harness\n{HEAD}")
groups = collections.defaultdict(list)
for x in users:
    groups[harness(x)].append(turns(x))
for g in sorted(groups, key=lambda g: -sum(groups[g])):
    row(g, groups[g])

total_of = {x["tree"]: turns(x) for x in users}
print(f"\nwhat each arm replayed: requests sent before {CUT:.0f} s\n"
      f"{'arm':10s} | {'users':>5s} {'requests':>8s} | {'sent a user: mean':>17s} {'p50':>4s} {'p90':>4s} {'max':>4s} | {'sessions sent whole':>19s} {'cut short':>9s} "
      f"{'share of their turns sent':>25s} | {'turn number of a request: p50':>29s} {'p90':>4s} {'p99':>4s} {'max':>4s} | {'requests at turn 50 or later':>28s}")
with open(f"{cdir}/manifest.txt") as f:
    arms = [line.split()[:3] for line in f if line.strip()]
for label, run, programs in arms:
    sent, tree_of, ks = collections.Counter(), {}, []
    with open(f"../results/{run}/{programs}/client.jsonl") as f:
        for line in f:
            if '"kind":"req"' not in line:
                continue
            r = json.loads(line)
            if r["phase"] == "profile" and r["t_send"] < CUT and r.get("ttft") is not None and not r.get("error"):
                sent[r["play"]] += 1
                tree_of[r["play"]] = r["tree"]
                ks.append(r["k"])
    counts = list(sent.values())
    whole = sum(sent[p] >= total_of[tree_of[p]] for p in sent)
    asked = sum(total_of[tree_of[p]] for p in sent)
    print(f"{label:10s} | {len(counts):5d} {sum(counts):8d} | {sum(counts) / len(counts):17.1f} {q(counts, .5):4d} {q(counts, .9):4d} {max(counts):4d} | "
          f"{whole:19d} {len(counts) - whole:9d} {100 * sum(counts) / asked:24.1f}% | {q(ks, .5):29d} {q(ks, .9):4d} {q(ks, .99):4d} {max(ks):4d} | "
          f"{100 * sum(k >= 50 for k in ks) / len(ks):27.1f}%")
