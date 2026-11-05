import sys, json, statistics as st
R = "/home/wanhr/sglang/agent_cache/results/"
RUNS = {
 "c20": "20261031_204526_three_tier_wc_oracle_v2_nu_NAT160/client_204850_cmp_three_tier_wc_oracle_v2_nu",
 "c21": "20261101_062014_three_tier_wc_oracle_v2_nu_custom_L1nat_H70/client_062142_cmp_three_tier_wc_oracle_v2_nu",
 "c23_44": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70/client_222615_cmp_three_tier_wc_oracle_v2_nu_owned",
 "c23_128": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160/client_023345_cmp_three_tier_wc_oracle_v2_nu_owned",
}
def load(n):
    out = {}
    for l in open(R + RUNS[n] + "/client.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "turn" and r.get("dist_hint") and r["dist_hint"].get("p50") is not None:
            out[(r["conv"], r["turn"])] = (r["dist_hint"]["p50"], r["dist_hint"]["matched"], r["dist_hint"]["true_gap_s"])
    return out
for a, b in (("c21", "c23_44"), ("c20", "c23_128")):
    A, B = load(a), load(b)
    shared = sorted(set(A) & set(B))
    diff = [k for k in shared if A[k][0] != B[k][0]]
    rel = sorted(abs(A[k][0] / B[k][0] - 1) for k in diff)
    keydiff = sum(A[k][1] != B[k][1] for k in shared)
    # does it change which side of the true gap the p50 falls on?
    flips = sum((A[k][0] >= A[k][2]) != (B[k][0] >= B[k][2]) for k in shared if A[k][2] is not None)
    print(f"{a} vs {b}: shared turns with a p50 {len(shared)}, p50 differs {len(diff)} ({len(diff)/len(shared):.1%}), "
          f"relative difference median {rel[len(rel)//2]:.2%} p90 {rel[int(len(rel)*0.9)]:.2%}, picked entry differs {keydiff}, "
          f"p50 on the other side of the true gap {flips} ({flips/len(shared):.2%})")
