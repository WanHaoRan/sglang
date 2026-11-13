"""Turn latency (send to last token) and throughput beside TTFT, on the turns the named arms share. From agent_cache/:
  nice -n 10 python3 -I -B results/campaign26/checks/latency.py arm [arm ...]
Per arm: mean TTFT, mean and median latency, mean of latency minus TTFT, completion tokens per turn; then turns and
completion tokens sent before minute 115 on the arm's own clock; then paired latency differences against the first arm."""
import os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = sys.argv[1:]
source = open(os.path.join(HERE, "..", "..", "..", "frozen", "oracle_scripts_staging", "c26_ec", "compare_ec.py")).read()
ns = {"__name__": "compare_ec_loader"}
sys.argv = [sys.argv[0], *ARMS]
exec(compile(source[:source.index("\nD = {a: load(")], "compare_ec.py", "exec"), ns)
RUNS, load = ns["RUNS"], ns["load"]
D = {a: load(RUNS[a]) for a in ARMS}
W = max(len(a) for a in ARMS)
keys = sorted(set.intersection(*(set(k for k in D[a]["by"] if k[1] > 0) for a in ARMS)))
print(f"{len(keys)} returning turns shared by {', '.join(ARMS)}")
print("per arm on them: mean TTFT, mean latency, median latency, mean latency - TTFT, completion tokens per turn | own run: turns and completion tokens (M) sent before minute 115")
for a in ARMS:
    r = [D[a]["by"][k] for k in keys]
    own = [t for t in D[a]["turns"] if t["t_send"] < 6900]
    print(f"  {a:{W}s} {st.mean(x['ttft'] for x in r):6.2f} {st.mean(x['latency'] for x in r):7.2f} {st.median(x['latency'] for x in r):7.2f} "
          f"{st.mean(x['latency'] - x['ttft'] for x in r):7.2f} {st.mean((x.get('completion_tokens') or 0) for x in r):6.0f} | "
          f"{len(own):6d} {sum((t.get('completion_tokens') or 0) for t in own) / 1e6:6.2f}")
ref = ARMS[0]
print(f"against {ref}, paired by turn: TTFT, latency, latency - TTFT (seconds per shared turn)")
for a in ARMS[1:]:
    d = [(D[a]["by"][k]["ttft"] - D[ref]["by"][k]["ttft"], D[a]["by"][k]["latency"] - D[ref]["by"][k]["latency"]) for k in keys]
    print(f"  {a:{W}s} {st.mean(x[0] for x in d):+7.3f} {st.mean(x[1] for x in d):+7.3f} {st.mean(x[1] - x[0] for x in d):+7.3f}")
