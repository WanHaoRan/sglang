"""Exp 0 / Exp 1 on nixl, and the file-vs-nixl comparison."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = os.environ["RESULTS"]
KV = 147456
OUT = os.path.join(R, "exp1_nixl")


def load(sub):
    d = pd.read_csv(os.path.join(R, sub, "ttft_by_tier.csv"))
    d = d[d.ttft_s.notna()]
    hit = (d.cached_device.fillna(0) + d.cached_host.fillna(0)
           + d.cached_storage.fillna(0)) > 0
    d = d[~((d.tier == "recompute") & hit)]           # same filter as the file run
    if "background_load" in d.columns:
        d = d[(d.background_load == "none") | d.background_load.isna()]
    if "prefetch_policy" in d.columns:
        d = d[(d.prefetch_policy == "wait_complete") | d.prefetch_policy.isna()]
    return d


def fit(d, tier):
    g = d[d.tier == tier].groupby("L").ttft_s.median().reset_index()
    if len(g) < 2:
        return None
    slope, icept = np.polyfit(g.L, g.ttft_s, 1)
    r = {"tier": tier, "slope_s_per_token": slope, "intercept_s": icept,
         "n_points": len(g)}
    if slope > 0:
        r["effective_GBps"] = KV / slope / 2 ** 30
        r["tokens_per_s"] = 1.0 / slope
    return r


def breakeven(rows, rec):
    out = []
    for r in rows:
        if r["tier"] == "recompute":
            continue
        ds = r["slope_s_per_token"] - rec["slope_s_per_token"]
        di = rec["intercept_s"] - r["intercept_s"]
        be = di / ds if ds else float("inf")
        out.append({"tier": r["tier"], "break_even_L": be,
                    "beats_recompute_above_L": be if ds < 0 else None,
                    "beats_recompute_below_L": be if ds > 0 else None})
    return out


nx = load("exp1_nixl")
fl = load("exp1")

rows = [r for r in (fit(nx, t) for t in ["recompute", "L1", "L2", "L3"]) if r]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tier_fits.csv"), index=False)
rec = next(r for r in rows if r["tier"] == "recompute")
be = breakeven(rows, rec)
pd.DataFrame(be).to_csv(os.path.join(OUT, "break_even.csv"), index=False)
summ = (nx.groupby(["tier", "L"]).ttft_s
          .agg(median="median", mn="min", mx="max", n="count").reset_index())
summ.to_csv(os.path.join(OUT, "ttft_summary.csv"), index=False)

print("=== Exp 1 on nixl — fits ===")
print(pd.DataFrame(rows).to_string(index=False))
print("\n=== break-even vs recompute ===")
print(pd.DataFrame(be).to_string(index=False))

# --- figure 1: nixl tier curves ------------------------------------------
colors = {"recompute": "#444", "L1": "#1b7837", "L2": "#2166ac", "L3": "#b2182b"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, logy in zip(axes, (False, True)):
    for t in ["recompute", "L1", "L2", "L3"]:
        d = summ[summ.tier == t]
        if d.empty:
            continue
        ax.plot(d.L, d["median"], "o-", label=t, color=colors[t])
        ax.fill_between(d.L, d.mn, d.mx, alpha=0.15, color=colors[t])
    ax.set_xlabel("prompt tokens"); ax.set_ylabel("TTFT (s)")
    ax.set_xscale("log", base=2); ax.grid(alpha=0.3); ax.legend()
    if logy:
        ax.set_yscale("log"); ax.set_title("nixl backend (log-log)")
    else:
        ax.set_title("nixl backend (linear)")
fig.suptitle("HiCache tier cost curves — nixl storage backend")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ttft_vs_len.png"), dpi=130)
print("\nwrote exp1_nixl/ttft_vs_len.png")

# --- figure 2: the backend comparison ------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for lbl, d, ls, col in [("recompute", fl, "--", "#444"),
                        ("L3 — file backend", fl, "-", "#b2182b"),
                        ("L3 — nixl backend", nx, "-", "#2166ac")]:
    t = "recompute" if lbl == "recompute" else "L3"
    src = d[d.tier == t].groupby("L").ttft_s.median()
    ax.plot(src.index, src.values, "o" + ls, label=lbl, color=col, lw=2)
n3 = nx[nx.tier == "L3"].groupby("L").ttft_s.median()
r0 = fl[fl.tier == "recompute"].groupby("L").ttft_s.median()
# The SUSTAINED crossover: walk down from the largest L while nixl L3 still
# beats recompute. Taking the first crossing instead would report L=512, which
# is an artifact of an inflated recompute@512 warm-up sample, not a real win.
common = [L for L in sorted(n3.index) if L in r0.index]
sustained = None
for L in reversed(common):
    if n3[L] < r0[L]:
        sustained = L
    else:
        break
if sustained is not None:
    ax.axvline(sustained, color="green", ls=":", lw=2,
               label=f"nixl L3 beats recompute from L={sustained}")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("prompt tokens"); ax.set_ylabel("TTFT (s)")
ax.set_title("L3 hit cost: file vs nixl, against the recompute bar")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(R, "exp1_backend_comparison.png"), dpi=130)
print("wrote exp1_backend_comparison.png")

cmp_rows = []
for L in sorted(set(n3.index) | set(r0.index)):
    f3 = fl[(fl.tier == "L3") & (fl.L == L)].ttft_s.median()
    cmp_rows.append({"L": int(L), "recompute_s": round(r0.get(L, np.nan), 4),
                     "L3_file_s": round(f3, 4) if f3 == f3 else None,
                     "L3_nixl_s": round(n3.get(L, np.nan), 4),
                     "nixl_speedup_vs_file": round(f3 / n3[L], 1) if (f3 == f3 and L in n3) else None,
                     "nixl_L3_vs_recompute": round(n3.get(L, np.nan) / r0.get(L, np.nan), 2)})
pd.DataFrame(cmp_rows).to_csv(os.path.join(R, "exp1_backend_comparison.csv"), index=False)
print("\n=== file vs nixl L3, against recompute ===")
print(pd.DataFrame(cmp_rows).to_string(index=False))
