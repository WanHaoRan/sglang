"""Campaign 28 or 29: how long a run that starts empty takes to reach its steady population, and how a session's
context grows turn by turn. In the container, from agent_cache/scripts/:
  python3 -B ../results/campaign28/checks/start_state.py ../results/campaign28 <users/s> [exclude=<harness>] [gap_cap=<s>] [ms_token=<ms>] [hours=<h>] [max_context=<tokens>]
A model, not a run: no queueing (a load under the GPU's capacity). Sessions arrive by the client's own loader and
arrival draw and are replayed on paper: a turn starts after the recorded wait (capped at GAP_CAP) and its reply takes
1 s + 0.158 ms a new prompt token + ms_token (default 26) an output token. Main streams only.
Prints, by time since the start: users in session, the KV their contexts hold, the part idle for over 60 s, and the
part past a host pool of 100 or 70 GB; then the level the run settles at (the last third of the horizon), the
session lengths, and the prompt size by turn number. Stdlib and the client modules only."""
import random
import sys
import types

sys.path.insert(0, ".")
import replay_programs  # noqa: E402

opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
argv = [a for a in sys.argv if "=" not in a]
cdir, rate = argv[1], float(argv[2])
settings = dict(line.strip().split("=", 1) for line in open(f"{cdir}/settings.env") if "=" in line)
args = types.SimpleNamespace(trace=settings["TRACE"], wait_field="wait", split_idle=None, split_compaction=False,
                             max_context=int(opts.get("max_context", settings["MAX_CONTEXT"])), sample="shuffle",
                             seed=int(settings["SEED"]), trees=0)
header, trees, dropped = replay_programs.load_trees(args)
trees = [x for x in trees if (x.get("system") or ("?",))[0] != opts.get("exclude")]
gap_cap = float(opts.get("gap_cap", settings["GAP_CAP"]))
ms_token = float(opts.get("ms_token", 26.0))
hours = float(opts.get("hours", 12.0))
B = header.get("block_tokens") or 0
PREFILL_MS = 0.158              # per new prompt token, campaign 28's measurement
TOKEN_BYTES = 160 * 2**30 / 1627648   # host bytes a token, from the 160 GB pool of 1,627,648 tokens
BIN = 60.0


def prompt(r):
    return (r["keep"] + len(r["new"])) * B + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]


def new_tokens(r, k):
    if k == 0:
        return prompt(r)
    return len(r["new"]) * B + int(r.get("tail") or 0) if "new" in r else r["append"]


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


rng, t, starts = random.Random(f"{args.seed}:arrivals"), 0.0, []
for _ in trees:   # run_arrivals' own draw, with no cap on users in session
    if t >= hours * 3600:
        break
    starts.append(t)
    t += rng.expovariate(rate)
nbins = int(hours * 3600 / BIN)
users, kv, idle = [0.0] * nbins, [0.0] * nbins, [0.0] * nbins


def spread(series, lo, hi, weight):
    """Add weight x (seconds of [lo, hi) inside each bin) to the series."""
    b = int(max(lo, 0.0) // BIN)
    while hi > lo and b < nbins and b * BIN < hi:
        series[b] += weight * (min(hi, (b + 1) * BIN) - max(lo, b * BIN))
        b += 1


spans, firsts, lasts, at_cut = [], [], [], []
by_turn = {}
for tree, start in zip(trees, starts):
    rs = tree["streams"][0]["requests"]
    now, held_time, life = start, 0.0, 0.0
    for k, r in enumerate(rs):
        now += min(float(r["wait"] or 0), gap_cap) if k else 0.0
        reply = 1.0 + new_tokens(r, k) * PREFILL_MS / 1e3 + r["out"] * ms_token / 1e3
        nxt = now + reply + (min(float(rs[k + 1]["wait"] or 0), gap_cap) if k + 1 < len(rs) else 0.0)
        ctx = prompt(r) + r["out"]
        spread(kv, now, nxt, ctx)
        spread(idle, now + reply + 60.0, nxt, ctx)
        held_time += ctx * (nxt - now)
        life += nxt - now
        slot = by_turn.setdefault(min(k, 60) // 5 if k else -1, [0, 0, 0])
        slot[0] += 1
        slot[1] += prompt(r)
        slot[2] += new_tokens(r, k)
        now += reply
    spread(users, start, now, 1.0)
    spans.append(now - start)
    firsts.append(prompt(rs[0]))
    lasts.append(prompt(rs[-1]) + rs[-1]["out"])
    at_cut.append(held_time / life)

print(f"{settings['TRACE'].split('/')[-1]}, max context {args.max_context:,}, options {opts or 'none'}: {len(trees)} sessions; {rate} users/s, waits capped at {gap_cap:.0f} s, "
      f"{ms_token:.0f} ms an output token; {len(starts)} arrive in {hours:.0f} h")
tail = range(2 * nbins // 3, nbins)
steady_users = sum(users[b] for b in tail) / (len(tail) * BIN)
steady_kv = sum(kv[b] for b in tail) / (len(tail) * BIN)
steady_idle = sum(idle[b] for b in tail) / (len(tail) * BIN)
pool = {gb: gb * 2**30 / TOKEN_BYTES for gb in (100, 70)}
print(f"settles at (hours {2 * hours / 3:.0f} to {hours:.0f}): {steady_users:.0f} users in session, {steady_kv / 1e6:.2f} M tokens of session KV ({steady_kv * TOKEN_BYTES / 2**30:.0f} GB), "
      f"{steady_idle / 1e6:.2f} M of it idle for over 60 s; host pools: 100 GB = {pool[100] / 1e6:.2f} M tokens, 70 GB = {pool[70] / 1e6:.2f} M")
print(f"{'minutes since start':>19s} | {'users in session':>16s} | {'session KV M':>12s} {'of the settled':>14s} | {'idle over 60 s M':>16s} {'of the settled':>14s} | KV past a host pool of 100 GB, 70 GB (M tokens)")
for minute in (5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480):
    b = minute - 1   # the minute that ends there
    if b >= nbins:
        break
    u, k, i = users[b] / BIN, kv[b] / BIN, idle[b] / BIN
    print(f"{minute:19d} | {u:16.0f} | {k / 1e6:12.2f} {100 * k / steady_kv:13.0f}% | {i / 1e6:16.2f} {100 * i / max(1.0, steady_idle):13.0f}% | "
          f"{max(0.0, k - pool[100]) / 1e6:10.2f} {max(0.0, k - pool[70]) / 1e6:10.2f}")
for level in (0.5, 0.8, 0.9):
    hit = next((b + 1 for b in range(nbins) if kv[b] / BIN >= level * steady_kv), None)
    hit_idle = next((b + 1 for b in range(nbins) if idle[b] / BIN >= level * steady_idle), None)
    print(f"session KV first at {100 * level:.0f} % of the settled level: minute {hit}; idle KV: minute {hit_idle}")
window = range(30, min(nbins, 120))
print(f"a 2 h run, minutes 30 to 120: session KV averages {sum(kv[b] for b in window) / (len(window) * BIN) / 1e6:.2f} M tokens "
      f"({100 * sum(kv[b] for b in window) / (len(window) * BIN) / steady_kv:.0f} % of the settled level), idle KV {sum(idle[b] for b in window) / (len(window) * BIN) / 1e6:.2f} M "
      f"({100 * sum(idle[b] for b in window) / (len(window) * BIN) / max(1.0, steady_idle):.0f} %)")
print(f"session length on paper (s): mean {sum(spans) / len(spans):.0f}, p50 {q(spans, .5):.0f}, p90 {q(spans, .9):.0f}, p99 {q(spans, .99):.0f}, max {max(spans):.0f}; "
      f"sessions over 1 h: {100 * sum(s > 3600 for s in spans) / len(spans):.1f} % of sessions, {100 * sum(s for s in spans if s > 3600) / sum(spans):.0f} % of the time in session")
ratio = [f / l for f, l in zip(firsts, lasts)]
print(f"first prompt K tokens: p10 {q(firsts, .1) / 1e3:.1f}, p50 {q(firsts, .5) / 1e3:.1f}, p90 {q(firsts, .9) / 1e3:.1f}, mean {sum(firsts) / len(firsts) / 1e3:.1f}; "
      f"context at the session's end: p10 {q(lasts, .1) / 1e3:.1f}, p50 {q(lasts, .5) / 1e3:.1f}, p90 {q(lasts, .9) / 1e3:.1f}, mean {sum(lasts) / len(lasts) / 1e3:.1f}; "
      f"first / end: p10 {q(ratio, .1):.2f}, p50 {q(ratio, .5):.2f}, p90 {q(ratio, .9):.2f}")
print(f"context at a uniform instant of a session (what a mid-session start gives): mean {sum(at_cut) / len(at_cut) / 1e3:.1f} K over sessions, "
      f"{sum(a * s for a, s in zip(at_cut, spans)) / sum(spans) / 1e3:.1f} K weighted by session length (what a running system holds)")
print("turn number | requests | prompt K (mean) | new K (mean)")
for slot in sorted(by_turn):
    n, p, w = by_turn[slot]
    name = "1" if slot < 0 else (f"{5 * slot + 1 if slot else 2}-{5 * slot + 5}" if slot < 12 else "61 or more")
    print(f"{name:>11s} | {n:8d} | {p / n / 1e3:15.1f} | {w / n / 1e3:12.2f}")
