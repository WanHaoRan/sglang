"""What load a lane of each trace puts on a server, at the recorded pace: requests a minute, the share of time a
program is waiting, the context it keeps, and how long its waits are. Programs are MORI's (sessions cut at idle
periods over 30 min and at compactions), dropped when a request exceeds the cap, as replay_programs.py does.
In the container, from agent_cache/scripts/:
  python3 ../results/campaign27/checks/trace_load.py > ../results/campaign27/checks/trace_load.out
"Own time" replaces each request's recorded duration by its output tokens times TPOT_S, this box's decode pace.
Stdlib only."""
import sys

sys.path.insert(0, ".")
from program_trace import iter_trees, read_header, split_programs  # noqa: E402

TRACES = (("AgentX 256K", "../data/converted/agentx_062126_256k.programs.jsonl.gz", "wait"),
          ("TraceLab, tool and human waits", "../data/converted/tracelab_v0.0.2.programs.jsonl.gz", "wait"),
          ("TraceLab, tool waits only", "../data/converted/tracelab_v0.0.2.programs.jsonl.gz", "wait_tool"),
          ("FreeInference", "../data/converted/freeinference_2026-07-12.programs.jsonl.gz", "wait"))
CAPS = (65536, 131072, 262144)
TPOT_S = 0.05   # s per output token, between campaign 26's 36-63 ms a step and the 110 ms of 100K-token contexts


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


for name, path, field in TRACES:
    header = read_header(path)
    block = header.get("block_tokens") or 0
    progs = []  # (peak, requests, recorded busy s, waits s list, own busy s, mean prompt)
    for tree in iter_trees(path):
        for prog in split_programs(tree, idle_s=1800.0, compaction=True, content=header["content"]):
            peak, n, busy, own, waits, sizes = 0, 0, 0.0, 0.0, [], []
            for st in prog["streams"]:
                for k, r in enumerate(st["requests"]):
                    size = (r["keep"] + len(r["new"])) * block + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]
                    peak = max(peak, size + r["out"])
                    n += 1
                    busy += float(r.get("dur") or 0)
                    own += r["out"] * TPOT_S
                    sizes.append(size)
                    if k > 0:
                        waits.append(float(r.get(field, r["wait"]) or 0))
            progs.append((peak, n, busy, waits, own, sum(sizes) / len(sizes)))
    print(f"===== {name}: {len(progs)} programs")
    print(f"  {'cap':>7s} {'programs':>8s} {'requests':>8s} | per program: {'requests p50':>12s} {'span min p50':>12s} {'waiting share':>13s} "
          f"| {'req/min a lane, recorded':>24s} {'own pace':>9s} | {'waits >60 s':>11s} {'>300 s':>7s} | {'mean prompt K':>13s}")
    for cap in CAPS:
        kept = [p for p in progs if p[0] <= cap]
        if not kept:
            continue
        n = sum(p[1] for p in kept)
        wait = sum(sum(p[3]) for p in kept)
        busy, own = sum(p[2] for p in kept), sum(p[4] for p in kept)
        waits = [w for p in kept for w in p[3]]
        spans = [(p[2] + sum(p[3])) / 60 for p in kept]
        prompt = sum(p[5] * p[1] for p in kept) / n
        print(f"  {cap:7d} {len(kept):8d} {n:8d} | {q([p[1] for p in kept], .5):25d} {q(spans, .5):12.1f} {100 * wait / (wait + busy):12.1f}% "
              f"| {60 * n / (wait + busy):24.2f} {60 * n / (wait + own):9.2f} | {100 * sum(w > 60 for w in waits) / len(waits):10.1f}% "
              f"{100 * sum(w > 300 for w in waits) / len(waits):6.1f}% | {prompt / 1000:13.1f}")
