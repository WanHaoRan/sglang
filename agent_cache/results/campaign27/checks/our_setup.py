"""The public traces under the campaigns' own schedule: every recorded session is one arriving user. What the knobs of
campaigns 21-26 (users arriving at ARRIVAL a second, at most CAP in session, TURNS requests a user, gaps scaled and
capped at 1,200 s) would ask of this box, from the traces alone.
In the container, from agent_cache/scripts/:
  python3 ../results/campaign27/checks/our_setup.py > ../results/campaign27/checks/our_setup.out
A user is a whole session (no cut into programs); its root stream is cut after TURNS requests, and a subagent request
counts while it started before that cut. Service time is estimated: TTFT_S plus the recorded output tokens times TPOT_S
(measured here on sessions under 65,536 tokens: 25 to 48 ms a token at light load, 99 ms saturated). In session = busy
on the root stream plus its waits; the cap binds when ARRIVAL times that exceeds CAP. Stdlib only."""
import sys

sys.path.insert(0, ".")
from program_trace import iter_trees, read_header  # noqa: E402

TRACES = (("TraceLab", "../data/converted/tracelab_v0.0.2.programs.jsonl.gz"),
          ("FreeInference, one week", "../data/converted/freeinference_2026-07-12.programs.jsonl.gz"),
          ("AgentX 256K", "../data/converted/agentx_062126_256k.programs.jsonl.gz"))
CONTEXT_CAPS = (65536, 131072)
SCALES = (1, 10)
import os  # noqa: E402

# campaign 26's 44-user setting unless OUR_CAP, OUR_TURNS or OUR_TRACES say otherwise (OUR_TURNS=0: no cut)
ARRIVAL, CAP, TURNS, GAP_CAP_S = 0.1, int(os.environ.get("OUR_CAP", 44)), int(os.environ.get("OUR_TURNS", 50)) or None, 1200.0
if os.environ.get("OUR_TRACES"):
    TRACES = tuple((p.split("=")[0], p.split("=")[1]) for p in os.environ["OUR_TRACES"].split(","))
TPOT_S, TTFT_S = 0.045, 0.3
CAPACITY = 400.0   # output tokens a second this box sustains at these context sizes: 342 in campaign 26, 294 to 460 in the probes


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def users(path, context_cap):
    """Per session kept: (requests, output tokens, busy s on the root, busy s in all, root waits s unscaled, mean root prompt)."""
    header = read_header(path)
    block = header.get("block_tokens") or 0
    size = lambda r: (r["keep"] + len(r["new"])) * block + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]
    out = []
    for tree in iter_trees(path):
        if max(size(r) + r["out"] for st in tree["streams"] for r in st["requests"]) > context_cap:
            continue
        root = tree["streams"][0]["requests"][:TURNS]
        cut = root[-1]["t"] + float(root[-1].get("dur") or 0)
        others = [r for st in tree["streams"][1:] for r in st["requests"] if r["t"] <= cut]
        busy = lambda rs: sum(TTFT_S + r["out"] * TPOT_S for r in rs)
        out.append((len(root) + len(others), sum(r["out"] for r in root + others), busy(root), busy(root) + busy(others),
                    [float(r["wait"] or 0) for r in root[1:]], sum(size(r) for r in root) / len(root)))
    return out


for name, path in TRACES:
    for context_cap in CONTEXT_CAPS:
        us = users(path, context_cap)
        if len(us) < 50:
            print(f"===== {name}, sessions under {context_cap:,} tokens: {len(us)} users, too few")
            continue
        n = len(us)
        print(f"===== {name}, sessions under {context_cap:,} tokens: {n} users; requests a user mean {sum(u[0] for u in us) / n:.1f}, p50 {q([u[0] for u in us], .5)}; "
              f"output tokens a request {sum(u[1] for u in us) / sum(u[0] for u in us):.0f}; root prompt mean {sum(u[5] for u in us) / n / 1000:.0f}K")
        print(f"  {'gaps':>5s} {'in session min, mean':>21s} {'p50':>6s} {'waiting share':>13s} {'waits >60 s':>11s} | arriving at {ARRIVAL}/s, cap {CAP}: "
              f"{'in session':>10s} {'cap binds':>9s} {'in flight':>9s} {'requests/min':>12s} {'output tok/s':>12s} {'of capacity':>11s} {'live context K':>14s}")
        for scale in SCALES:
            waits = [[min(w * scale, GAP_CAP_S) for w in u[4]] for u in us]
            dur = [u[2] + sum(w) for u, w in zip(us, waits)]
            mean_dur = sum(dur) / n
            in_session = min(CAP, ARRIVAL * mean_dur)
            rate = in_session / mean_dur   # users a second that get through
            every = [w for ws in waits for w in ws]
            print(f"  {'x' + str(scale):>5s} {mean_dur / 60:21.1f} {q(dur, .5) / 60:6.1f} {100 * sum(every) / sum(dur):12.1f}% "
                  f"{100 * sum(w > 60 for w in every) / max(1, len(every)):10.1f}% | {'':>28s}{in_session:10.1f} {'yes' if ARRIVAL * mean_dur > CAP else 'no':>9s} "
                  f"{rate * sum(u[3] for u in us) / n:9.1f} {60 * rate * sum(u[0] for u in us) / n:12.1f} {rate * sum(u[1] for u in us) / n:12.0f} "
                  f"{100 * rate * sum(u[1] for u in us) / n / CAPACITY:10.0f}% {in_session * sum(u[5] for u in us) / n / 1000:14.0f}")
