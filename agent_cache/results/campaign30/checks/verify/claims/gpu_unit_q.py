"""gpu_unit.py's comparison (cold >=4K-token prompts, minutes 0-10; decode s/token), plus where the mean gap comes from:
the cold requests split by their TTFT ratio into quartiles, and each quartile's share of the summed TTFT difference."""
import json, sys, statistics as st
runs = {}
for a in sys.argv[1:]:
    lab, path = a.split("=", 1)
    d = {}
    for l in open(f"/home/wanhr/sglang/agent_cache/results/{path}/client.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "req" and r.get("ttft") is not None and r["t_send"] < 600:
            d[(r["play"], r["stream"], r["k"])] = r
    runs[lab] = d
labs = list(runs)
for i in range(len(labs)):
    for j in range(i + 1, len(labs)):
        A, B = runs[labs[i]], runs[labs[j]]
        common = [k for k in A if k in B]
        cold = [k for k in common if A[k]["cached_tokens"] == 0 and B[k]["cached_tokens"] == 0 and A[k]["prompt_tokens"] >= 4000]
        rt = sorted(((B[k]["ttft"] / A[k]["ttft"]), k) for k in cold)
        dec = [k for k in common if A[k]["completion_tokens"] >= 100 and B[k]["completion_tokens"] == A[k]["completion_tokens"]]
        rd = [((B[k]["latency"] - B[k]["ttft"]) / (B[k]["completion_tokens"] - 1)) / ((A[k]["latency"] - A[k]["ttft"]) / (A[k]["completion_tokens"] - 1)) for k in dec]
        tot = sum(B[k]["ttft"] - A[k]["ttft"] for _, k in rt)
        n = len(rt); q = [rt[int(n*a/4):int(n*(a+1)/4)] for a in range(4)]
        shares = [sum(B[k]["ttft"] - A[k]["ttft"] for _, k in qq) / tot if tot else float('nan') for qq in q]
        mA = st.mean(A[k]["ttft"] for k in cold); mB = st.mean(B[k]["ttft"] for k in cold)
        print(f"{labs[j]} / {labs[i]}: cold n={n} ratio median {st.median([x for x,_ in rt]):.3f}; mean {mA:.3f} -> {mB:.3f} s ({100*(mB/mA-1):+.1f} %); "
              f"decode n={len(dec)} median {st.median(rd):.3f}; share of the summed difference by ratio quartile (low..high): {', '.join(f'{100*s:.0f} %' for s in shares)}")
