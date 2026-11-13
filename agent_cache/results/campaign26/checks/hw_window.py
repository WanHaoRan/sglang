"""Hardware control inside a time window, so a cut run compares with the same minutes of full runs. From agent_cache/:
  nice -n 10 python3 -I -B results/campaign26/checks/hw_window.py <end minute> arm [arm ...]
Decode ms per step with no queue by running requests, first-turn prefill ms per token, SSD ms per page in reads of 500
pages or more: medians (n), as compare_ec.py's hardware table but on the first <end minute> minutes of each run."""
import os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
END = float(sys.argv[1]) * 60
ARMS = sys.argv[2:]
source = open(os.path.join(HERE, "..", "..", "..", "frozen", "oracle_scripts_staging", "c26_ec", "compare_ec.py")).read()
ns = {"__name__": "compare_ec_loader"}
sys.argv = [sys.argv[0], *ARMS]
exec(compile(source[:source.index("\nD = {a: load(")], "compare_ec.py", "exec"), ns)
RUNS, load = ns["RUNS"], ns["load"]
W = max(len(a) for a in ARMS)
med = lambda v: f"{st.median(v):6.2f} ({len(v):4d})" if v else "     - (   0)"
for a in ARMS:
    d = load(RUNS[a])
    dec = [(n, 1000 * n / tps) for t, n, _, tps, q in d["dec"] if t < END and q == 0 and tps > 0]
    cells = " ".join(f"{lo}-{hi - 1} {med([ms for n, ms in dec if lo <= n < hi])}" for lo, hi in ((8, 16), (16, 24), (24, 32)))
    first = [1000 * (r["ttft"] - d["queue"].get((r["conv"], r["turn"]), 0.0)) / (r["prompt_tokens"] - (r.get("cached_tokens") or 0))
             for r in d["turns"] if r["turn"] == 0 and r["t_send"] < END and d["queue"].get((r["conv"], r["turn"]), 0.0) < 0.05
             and (r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0) > 2000]
    ssd = [ms / pages for t, pages, ms in d["io"] if t < END and pages >= 500]
    print(f"  {a:{W}s} decode {cells} | first-turn prefill {st.median(first) if first else float('nan'):6.3f} ({len(first):4d}) | SSD {med(ssd)}")
