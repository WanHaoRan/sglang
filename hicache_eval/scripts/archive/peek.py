import sys, pandas as pd
B = "/sgl-workspace/sglang/hicache_eval/results/20260908_nixl_exp234/"
key, arm = sys.argv[1], sys.argv[2]
d = pd.read_csv(B + f"exp2_{key}/interference.csv")
h = d[d.control == arm]
r = d[d.control == arm + "_recompute"]
print(f"=== {key} / {arm}: by rep (rep2 sits deepest in the queue ramp) ===")
for rate in sorted(h.requested_rps.unique()):
    a = h[h.requested_rps == rate].sort_values("rep")
    b = r[r.requested_rps == rate].sort_values("rep")
    print(f"  R={rate}")
    for i in range(len(a)):
        L3 = a.ttft_s.iloc[i]
        rc = b.ttft_s.iloc[i] if i < len(b) else float("nan")
        print(f"     rep{i}  L3 {L3:8.2f}s  recompute {rc:8.2f}s  speedup {rc/L3:6.2f}x"
              f"  storage={a.cached_storage.iloc[i]}  unfulfilled={a.unfulfilled_tokens.iloc[i]}")
hs, rs = h[h.rep < 2], r[r.rep < 2]
if not hs.empty:
    g = hs.groupby("requested_rps").agg(L3=("ttft_s", "median"), ach=("achieved_rps", "median"))
    g["recompute"] = rs.groupby("requested_rps").ttft_s.median()
    g["speedup"] = (g.recompute / g.L3).round(2)
    print("\n=== stable reps only (0,1) ===")
    print(g.round(3).to_string())
