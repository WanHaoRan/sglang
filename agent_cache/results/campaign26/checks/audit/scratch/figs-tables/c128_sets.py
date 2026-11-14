"""Read-only check: the shared returning-turn set of the five 128-user design runs under the two cut rules."""
import glob, json, os, statistics as st
RES = "/home/wanhr/sglang/agent_cache/results"
RUNS = {"baseline_c128": "20261018_103544_three_tier_wc_NAT160", "oracle_v2_c128": "20261018_154254_three_tier_wc_oracle_v2_nu_NAT160",
        "jit_p50_c128": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160", "jit_exact_c128": "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
        "ec_r134_c128": "20261112_194449_three_tier_wc_oracle_v2_nu_owned_NAT160"}
def turns(run):
    path = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get("kind") == "turn" and r.get("ttft") is not None]
T = {a: turns(r) for a, r in RUNS.items()}
for a, t in T.items():
    print(a, "turns with ttft", len(t), "last send", round(max(r["t_send"] for r in t), 1))
# rule A (plot_pricing_vs_jit.py): t_send < 7200, returning, not mid_start, t_send < last - 300
A = {}
for a, t in T.items():
    last = max(r["t_send"] for r in t)
    A[a] = {(r["conv"], r["turn"]): r for r in t if r["t_send"] < 7200 and r["turn"] > 0 and not r.get("mid_start") and r["t_send"] < last - 300}
sa = set.intersection(*(set(v) for v in A.values()))
print("rule A shared", len(sa), [round(st.mean(A[a][k]["ttft"] for k in sa), 2) for a in RUNS])
# rule B (compare_c128.py as the record words it): t_send < min(6900, last - 300), returning
B = {}
for a, t in T.items():
    last = max(r["t_send"] for r in t)
    lim = min(6900, last - 300)
    B[a] = {(r["conv"], r["turn"]): r for r in t if r["t_send"] < lim and r["turn"] > 0}
sb = set.intersection(*(set(v) for v in B.values()))
print("rule B shared", len(sb), [round(st.mean(B[a][k]["ttft"] for k in sb), 2) for a in RUNS])
