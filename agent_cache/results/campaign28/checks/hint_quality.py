"""Campaign 28 or 29, an arm with table hints: how close the hinted wait is to the true one, by harness, by the
table level that answered and by the length of the true wait. On the host, from agent_cache/, after compare_arms.py
(which writes checks/tree_systems.json):
  python3 -I -B results/campaign28/checks/hint_quality.py results/campaign28 jit_p50 > results/campaign28/checks/hint_quality_jit_p50.out
Over the requests sent before 6,600 s that name a next request. Stdlib only."""
import collections
import json
import sys

cdir, label = sys.argv[1], sys.argv[2]
with open(f"{cdir}/manifest.txt") as f:
    run, programs = next(line.split()[1:3] for line in f if line.split()[0] == label)
with open(f"{cdir}/checks/tree_systems.json") as f:
    systems = json.load(f)
hints = []
with open(f"results/{run}/{programs}/client.jsonl") as f:
    for line in f:
        if '"kind":"req"' not in line:
            continue
        r = json.loads(line)
        if r["phase"] == "profile" and r["t_send"] < 6600 and r.get("ttft") is not None and (r.get("hint") or {}).get("next_rid"):
            hints.append((systems.get(r["tree"], ["?"])[0], r["hint"]))


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


def level(h):
    m = h.get("matched")
    return "none" if m is None else "all replies" if m == "*" else "tool and word" if "/" in m else "tool"


def bucket(h):
    t = h["true_gap_s"]
    return "0-1 s" if t < 1 else "1-10 s" if t < 10 else "10-60 s" if t < 60 else "60-300 s" if t < 300 else "300 s or more"


def row(name, hs):
    hit = [h for h in hs if h.get("matched") is not None and h.get("p50") is not None]
    inside = sum(h["p10"] <= h["true_gap_s"] <= h["p90"] for h in hit)
    ratio = [h["p50"] / h["true_gap_s"] for h in hit if h["true_gap_s"] > 0]
    err = [h["p50"] - h["true_gap_s"] for h in hit]
    early = [-e for e in err if e < 0]
    late = [e for e in err if e > 0]
    print(f"{name[:20]:20s} | {len(hs):6d} | {100 * len(hit) / max(1, len(hs)):6.1f}% | {100 * inside / max(1, len(hit)):6.1f}% | "
          f"{q(ratio, .5):6.2f} {q(ratio, .1):6.2f} {q(ratio, .9):7.2f} | {q([h['true_gap_s'] for h in hs], .5):8.2f} {q([h['p50'] for h in hit], .5):8.2f} | "
          f"{100 * len(early) / max(1, len(hit)):5.1f}% {sum(early) / max(1, len(early)):8.1f} | {100 * len(late) / max(1, len(hit)):5.1f}% {sum(late) / max(1, len(late)):8.1f}")


HEAD = (f"{'':20s} | {'hints':>6s} | {'entry':>7s} | {'inside':>7s} | {'p50 / true: median, p10, p90':>21s} | {'true p50 s':>8s} {'hint p50':>8s} | "
        f"{'hint short: share, mean s':>14s} | {'hint long: share, mean s':>14s}")
ended = sum(bool(h.get("ended")) for _, h in hints)
hints = [(s, h) for s, h in hints if not h.get("ended")]
print(f"{cdir.split('/')[-1]} {label}: {len(hints)} hints of a wait, and {ended} that say the session ends (left out below). "
      f"entry: the table had a row; inside: true wait within the hint's p10 to p90;\n"
      f"hint short: the hinted p50 is under the true wait (an early read holds room that long); hint long: over it (the read starts late)")
for title, key in (("by harness", lambda x: x[0]), ("by the table level that answered", lambda x: level(x[1])), ("by true wait", lambda x: bucket(x[1]))):
    groups = collections.defaultdict(list)
    for item in hints:
        groups[key(item)].append(item[1])
    print(f"\n{title}\n{HEAD}")
    order = ["0-1 s", "1-10 s", "10-60 s", "60-300 s", "300 s or more"] if title == "by true wait" else sorted(groups, key=lambda g: -len(groups[g]))
    for g in order:
        if groups.get(g):
            row(g, groups[g])
    row("all", [h for _, h in hints])
no_call = [h for _, h in hints if h.get("reply_key") == "_no_call"]
print(f"\nreplies with no tool call: {len(no_call)} hints ({100 * len(no_call) / max(1, len(hints)):.1f} %)")
if no_call:
    print(HEAD)
    row("no tool call", no_call)
    row("a tool call", [h for _, h in hints if h.get("reply_key") != "_no_call"])
