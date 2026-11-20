"""Tool keys, gaps and prompt sizes of the converted program traces: what a per-key gap table would hold.
In the container, from agent_cache/scripts/:
  python3 ../results/campaign27/checks/trace_keys.py > ../results/campaign27/checks/trace_keys.out
A gap is filed under the key of the request it follows, as replay_programs.py --hints dist does. Stdlib only."""
import collections
import sys

sys.path.insert(0, ".")
from program_trace import iter_trees, read_header  # noqa: E402

TRACES = (("AgentX 256K", "../data/converted/agentx_062126_256k.programs.jsonl.gz"),
          ("TraceLab", "../data/converted/tracelab_v0.0.2.programs.jsonl.gz"),
          ("FreeInference", "../data/converted/freeinference_2026-07-12.programs.jsonl.gz"))
CTX = 262144


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def size_of(r, block):
    if "new" in r:
        return (r["keep"] + len(r["new"])) * block + int(r.get("tail") or 0)
    return r["prefix"] + r["append"]


for name, path in TRACES:
    header = read_header(path)
    block = header.get("block_tokens") or 0
    keyed, kinds, systems = collections.defaultdict(list), collections.Counter(), collections.Counter()
    requests = with_key = 0
    peaks, sizes, streams, per_tree = [], [], [], []
    for tree in iter_trees(path):
        systems[tuple(tree.get("system") or ())] += 1
        streams.append(len(tree["streams"]))
        peak = count = 0
        for st in tree["streams"]:
            rs = st["requests"]
            for k, r in enumerate(rs):
                requests += 1
                count += 1
                size = size_of(r, block)
                sizes.append(size)
                peak = max(peak, size + r["out"])
                kinds[r.get("gap")] += 1
                with_key += bool(r.get("key"))
                if k + 1 < len(rs):
                    keyed[r.get("key")].append(float(rs[k + 1]["wait"] or 0))
        peaks.append(peak)
        per_tree.append(count)
    gaps = [w for v in keyed.values() for w in v]
    named = {k: v for k, v in keyed.items() if k}
    tools = {k.split("/")[0] for k in named}
    print(f"===== {name}: {len(peaks)} trees, {requests} requests, content {header['content']}, block {block}")
    print(f"  agent systems (harness, model): {len(systems)}; largest {systems.most_common(3)}")
    print(f"  requests with a key: {100 * with_key / requests:.1f} %; distinct keys {len(named)}, tools {len(tools)}, "
          f"keys with 5 or more gaps {sum(1 for v in named.values() if len(v) >= 5)}")
    print(f"  gap kinds: {dict(kinds.most_common(6))}")
    print(f"  prompt tokens per request: p50 {q(sizes, .5)}, p90 {q(sizes, .9)}, p99 {q(sizes, .99)}, max {max(sizes)}; "
          f"trees with a request over {CTX}: {sum(p > CTX for p in peaks)}")
    print(f"  requests per tree: p50 {q(per_tree, .5)}, p90 {q(per_tree, .9)}; streams per tree p90 {q(streams, .9)}")
    print(f"  all gaps: n {len(gaps)}, p10 {q(gaps, .1):.2f}, p50 {q(gaps, .5):.2f}, p90 {q(gaps, .9):.1f} s, "
          f"over 60 s {100 * sum(w > 60 for w in gaps) / len(gaps):.1f} %")
    print(f"  {'key':30s} {'gaps':>7s} {'p10':>7s} {'p50':>7s} {'p90':>8s} {'>60 s':>6s}")
    for key, v in sorted(keyed.items(), key=lambda kv: -len(kv[1]))[:14]:
        print(f"  {str(key)[:30]:30s} {len(v):7d} {q(v, .1):7.2f} {q(v, .5):7.2f} {q(v, .9):8.1f} "
              f"{100 * sum(w > 60 for w in v) / len(v):5.1f}%")
