"""What an SSD read costs a turn when the GPU queue is short, and how much context the sessions in progress hold.
From agent_cache/:  nice -n 10 python3 -I -B results/campaign26/checks/calm_reads.py arm [arm ...]
Per arm on the returning turns the arms share, inside and outside the two load episodes: turns that read from SSD, their
mean TTFT and queue wait, and the others' mean TTFT. Then, for the first arm, the context (prompt + completion tokens of
the latest turn) of all sessions in progress, sampled every 30 s."""
import os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = sys.argv[1:]
source = open(os.path.join(HERE, "..", "..", "..", "frozen", "oracle_scripts_staging", "c26_ec", "compare_ec.py")).read()
ns = {"__name__": "compare_ec_loader"}
sys.argv = [sys.argv[0], *ARMS]
exec(compile(source[:source.index("\nD = {a: load(")], "compare_ec.py", "exec"), ns)
RUNS, load, client, ORDER_RUN, pct = ns["RUNS"], ns["load"], ns["client"], ns["ORDER_RUN"], ns["pct"]
D = {a: load(RUNS[a]) for a in ARMS}
W = max(len(a) for a in ARMS)
rank = {(r["conv"], r["turn"]): i for i, r in enumerate(client(ORDER_RUN)[1])}
keys = sorted(set.intersection(*(set(k for k in D[a]["by"] if k[1] > 0) for a in ARMS)))
episode = lambda k: any(lo <= rank.get(k, -1) < hi for lo, hi in ((5500, 7000), (8750, 11000)))
mean = lambda v: st.mean(v) if v else float("nan")
for name, ks in (("outside the load episodes", [k for k in keys if not episode(k)]), ("inside them", [k for k in keys if episode(k)])):
    print(f"{name}: {len(ks)} shared returning turns | per arm: turns reading from SSD (share), their mean TTFT, median, mean queue wait | other turns' mean TTFT | all")
    for a in ARMS:
        d = D[a]
        rd = [k for k in ks if d["pages"][k] > 0]
        no = [k for k in ks if d["pages"][k] == 0]
        t = lambda k: d["by"][k]["ttft"]
        print(f"  {a:{W}s} {len(rd):5d} ({100 * len(rd) / len(ks):4.1f} %) {mean([t(k) for k in rd]):6.2f} {st.median([t(k) for k in rd]) if rd else float('nan'):6.2f} "
              f"{mean([d['queue'].get(k, 0.0) for k in rd]):6.2f} | {mean([t(k) for k in no]):6.2f} | {mean([t(k) for k in ks]):6.2f}")
a = ARMS[0]
turns = sorted(D[a]["turns"], key=lambda r: r["t_send"])
conv = {}
for r in turns:
    conv.setdefault(r["conv"], []).append(r)
ctx = []
for t in range(600, int(D[a]["end"]), 30):
    total = 0
    for v in conv.values():
        if v[0]["t_send"] <= t < v[-1]["t_send"] + v[-1]["latency"]:
            last = max((r for r in v if r["t_send"] <= t), key=lambda r: r["t_send"])
            total += (last.get("prompt_tokens") or 0) + (last.get("completion_tokens") or 0)
    ctx.append(total)
print(f"\n{a}: context of all sessions in progress, every 30 s from minute 10 (K tokens): median {st.median(ctx) / 1000:.0f}, p10 {pct(ctx, .1) / 1000:.0f}, "
      f"p90 {pct(ctx, .9) / 1000:.0f}, max {max(ctx) / 1000:.0f}; samples above 712K (70 GB pool) {100 * sum(c > 712128 for c in ctx) / len(ctx):.0f} %, "
      f"above 1,017K (100 GB) {100 * sum(c > 1017326 for c in ctx) / len(ctx):.0f} %, above 1,628K (160 GB) {100 * sum(c > 1627721 for c in ctx) / len(ctx):.0f} %")
