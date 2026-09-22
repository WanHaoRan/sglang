"""Cross-model tier analysis: fits, crossovers, and the comparison figure."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GiB = 2 ** 30
BASE = "/sgl-workspace/sglang/hicache_eval/results"
NEW = os.environ.get("RESULTS", f"{BASE}/20260908_llama70b_awq_fp8kv")

# name -> (csv path, KV bytes/token, label)
MODELS = {
    "Qwen3-8B bf16KV": (f"{BASE}/20260907_203035/nixl/exp1_nixl/ttft_by_tier.csv", 147456),
    "Llama-3.3-70B AWQ fp8KV": (f"{NEW}/exp1_70b/ttft_by_tier.csv", 163840),
    "Qwen3-32B FP8 fp8KV": (f"{NEW}/exp1_32b/ttft_by_tier.csv", 131072),
}


def load(p):
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.ttft_s.notna()]
    hit = (d.cached_device.fillna(0) + d.cached_host.fillna(0)
           + d.cached_storage.fillna(0)) > 0
    return d[~((d.tier == "recompute") & hit)]


out = {}
fig, ax = plt.subplots(figsize=(10, 6.5))
cols = {"Qwen3-8B bf16KV": "#999999", "Llama-3.3-70B AWQ fp8KV": "#b2182b",
        "Qwen3-32B FP8 fp8KV": "#2166ac"}
for name, (path, b) in MODELS.items():
    d = load(path)
    if d is None or d.empty:
        print(f"  (skipping {name}: no data yet)")
        continue
    med = d.groupby(["tier", "L"]).ttft_s.median()
    rec = med.loc["recompute"] if "recompute" in med.index.get_level_values(0) else None
    l3 = med.loc["L3"] if "L3" in med.index.get_level_values(0) else None
    row = {"kv_bytes_per_token": b}
    if rec is not None:
        s, i = np.polyfit(rec.index, rec.values, 1)
        row["P_marginal_tok_s"] = 1 / s
        row["recompute_bar_GiBps"] = b / s / GiB
    if l3 is not None:
        s3, i3 = np.polyfit(l3.index, l3.values, 1)
        row["L3_GiBps"] = b / s3 / GiB
        row["L3_intercept_s"] = i3
        if rec is not None:
            ds = s3 - s
            row["fitted_crossover_L"] = (i - i3) / ds if ds < 0 else None
            wins = [int(L) for L in l3.index if L in rec.index and l3[L] < rec[L]]
            row["L_where_L3_wins"] = wins
            row["speedup_at_max_L"] = float(rec.iloc[-1] / l3.iloc[-1])
    out[name] = row
    if rec is not None and l3 is not None:
        ax.plot(rec.index, rec.values, "--", color=cols[name], lw=1.6,
                label=f"{name} — recompute")
        ax.plot(l3.index, l3.values, "o-", color=cols[name], lw=2,
                label=f"{name} — L3 hit")

ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("prompt tokens"); ax.set_ylabel("TTFT (s)")
ax.set_title("L3 hit vs recompute, by model (nixl backend, same disk)")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(NEW, "model_comparison.png"), dpi=130)
json.dump(out, open(os.path.join(NEW, "model_comparison.json"), "w"), indent=2)

print(f"{'model':<26}{'b B/tok':>10}{'P tok/s':>10}{'bar GiB/s':>11}"
      f"{'L3 GiB/s':>10}{'L3 wins from':>14}{'max speedup':>13}")
for n, r in out.items():
    w = r.get("L_where_L3_wins") or []
    print(f"{n:<26}{r['kv_bytes_per_token']:>10,}{r.get('P_marginal_tok_s',0):>10,.0f}"
          f"{r.get('recompute_bar_GiBps',0):>11.3f}{r.get('L3_GiBps',0):>10.2f}"
          f"{(min(w) if w else 0):>14,}{r.get('speedup_at_max_L',0):>13.1f}x")
print("\nwrote model_comparison.{png,json}")
