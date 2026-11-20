"""How large the sessions of each converted trace are: trees, requests and output tokens left under a context cap.
In the container, from agent_cache/scripts/:
  python3 ../results/campaign27/checks/trace_sizes.py > ../results/campaign27/checks/trace_sizes.out
A tree's peak is its largest prompt plus output, the quantity --max-context compares. Stdlib only."""
import sys

sys.path.insert(0, ".")
from program_trace import iter_trees, read_header  # noqa: E402

TRACES = (("AgentX 256K", "../data/converted/agentx_062126_256k.programs.jsonl.gz"),
          ("TraceLab", "../data/converted/tracelab_v0.0.2.programs.jsonl.gz"),
          ("FreeInference", "../data/converted/freeinference_2026-07-12.programs.jsonl.gz"))
CAPS = (32768, 65536, 131072, 262144)


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


for name, path in TRACES:
    header = read_header(path)
    block = header.get("block_tokens") or 0
    trees = []  # (peak, requests, prompt sizes, outputs, keyed requests, recorded durations)
    for tree in iter_trees(path):
        peak, sizes, outs, keyed, durs = 0, [], [], 0, []
        for st in tree["streams"]:
            for r in st["requests"]:
                size = (r["keep"] + len(r["new"])) * block + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]
                sizes.append(size)
                outs.append(r["out"])
                durs.append(float(r.get("dur") or 0))
                keyed += bool(r.get("key"))
                peak = max(peak, size + r["out"])
        trees.append((peak, len(sizes), sizes, outs, keyed, durs))
    total = sum(t[1] for t in trees)
    outs = [o for t in trees for o in t[3]]
    durs = [d for t in trees for d in t[5] if d > 0]
    print(f"===== {name}: {len(trees)} trees, {total} requests; output tokens per request mean {sum(outs) / len(outs):.0f}, "
          f"p50 {q(outs, .5)}, p90 {q(outs, .9)}; recorded request duration p50 {q(durs, .5):.1f} s, p90 {q(durs, .9):.1f} s")
    print(f"  {'cap':>8s} {'trees':>7s} {'share':>6s} {'requests':>9s} {'share':>6s} {'prompt p50':>10s} {'p90':>8s} {'keyed':>6s}")
    for cap in CAPS:
        kept = [t for t in trees if t[0] <= cap]
        n = sum(t[1] for t in kept)
        sizes = [s for t in kept for s in t[2]]
        keyed = sum(t[4] for t in kept)
        print(f"  {cap:8d} {len(kept):7d} {100 * len(kept) / len(trees):5.1f}% {n:9d} {100 * n / total:5.1f}% "
              f"{q(sizes, .5) if sizes else 0:10d} {q(sizes, .9) if sizes else 0:8d} {100 * keyed / n if n else 0:5.1f}%")
