"""Same requests, two GPUs: TTFT of cold requests (cached 0) and decode time per output token, minutes 0-10 (no queue).
usage: gpu_unit.py <label>=<server run>/<programs dir> ...   (paths under agent_cache/results/)"""
import json, sys, statistics as st
runs = {}
for a in sys.argv[1:]:
    lab, path = a.split("=", 1)
    d = {}
    with open(f"{path}/client.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r.get("kind") == "req" and r.get("ttft") is not None and r["t_send"] < 600:
                d[(r["play"], r["stream"], r["k"])] = r
    runs[lab] = d
labs = list(runs)
base = labs[0]
for lab in labs[1:]:
    common = [k for k in runs[base] if k in runs[lab]]
    cold = [k for k in common if runs[base][k]["cached_tokens"] == 0 and runs[lab][k]["cached_tokens"] == 0 and runs[base][k]["prompt_tokens"] >= 4000]
    rt = [runs[lab][k]["ttft"] / runs[base][k]["ttft"] for k in cold]
    dec = [k for k in common if runs[base][k]["completion_tokens"] >= 100 and runs[lab][k]["completion_tokens"] == runs[base][k]["completion_tokens"]]
    rd = [((runs[lab][k]["latency"] - runs[lab][k]["ttft"]) / (runs[lab][k]["completion_tokens"] - 1)) /
          ((runs[base][k]["latency"] - runs[base][k]["ttft"]) / (runs[base][k]["completion_tokens"] - 1)) for k in dec]
    print(f"{lab} / {base}: cold prefill (>=4K tokens, n={len(cold)}): TTFT ratio median {st.median(rt):.3f} "
          f"[p25 {st.quantiles(rt, n=4)[0]:.3f}, p75 {st.quantiles(rt, n=4)[2]:.3f}], mean TTFT {st.mean(runs[base][k]['ttft'] for k in cold):.3f} -> {st.mean(runs[lab][k]['ttft'] for k in cold):.3f} s; "
          f"decode s/token (n={len(dec)}): ratio median {st.median(rd):.3f}")
