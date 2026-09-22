"""Fit per-tier cost curves and emit the Exp 1 table and figure."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
KV = 147456
OUT = os.path.join(os.environ["RESULTS"], "exp1")


def fit(df, tier):
    """Least-squares TTFT = intercept + L/rate, per tier."""
    d = df[df.tier == tier]
    if d.empty:
        return None
    g = d.groupby("L").ttft_s.median().reset_index()
    if len(g) < 2:
        return None
    slope, intercept = np.polyfit(g.L, g.ttft_s, 1)
    row = {
        "tier": tier, "n_points": len(g),
        "slope_s_per_token": slope, "intercept_s": intercept,
        "median_ttft_at_4096": float(
            d[d.L == 4096].ttft_s.median()) if (d.L == 4096).any() else float("nan"),
    }
    # For the cache tiers the slope is a transfer rate; for recompute it is 1/P.
    if slope > 0:
        row["effective_GBps"] = KV / slope / 2 ** 30
        row["tokens_per_s"] = 1.0 / slope
    return row


def main():
    csv = os.path.join(OUT, "ttft_by_tier.csv")
    df = pd.read_csv(csv)
    df = df[df.ttft_s.notna()]
    # A "recompute" sample that reports ANY cached tokens is not a recompute.
    # Three L=512 rows from the pooled L2 pass were device hits; they dragged
    # the 512 median from 0.232 s to 0.048 s and biased the fitted intercept.
    hit = (df.cached_device.fillna(0) + df.cached_host.fillna(0)
           + df.cached_storage.fillna(0)) > 0
    contaminated = (df.tier == "recompute") & hit
    if contaminated.any():
        print(f"dropping {int(contaminated.sum())} contaminated recompute rows "
              f"(cache hits mislabelled as recompute)")
    df = df[~contaminated]
    rows = [r for r in (fit(df, t) for t in ["recompute", "L1", "L2", "L3"]) if r]
    fits = pd.DataFrame(rows)
    fits.to_csv(os.path.join(OUT, "tier_fits.csv"), index=False)

    # Break-even vs recompute: the L at which a tier hit costs what recompute costs.
    rec = next((r for r in rows if r["tier"] == "recompute"), None)
    be = []
    if rec:
        for r in rows:
            if r["tier"] == "recompute":
                continue
            ds = r["slope_s_per_token"] - rec["slope_s_per_token"]
            di = rec["intercept_s"] - r["intercept_s"]
            be.append({
                "tier": r["tier"],
                "faster_than_recompute_everywhere": ds < 0 and di > 0,
                "break_even_L": (di / ds) if ds != 0 else float("inf"),
            })
    pd.DataFrame(be).to_csv(os.path.join(OUT, "break_even.csv"), index=False)

    summ = (df.groupby(["tier", "L"]).ttft_s
              .agg(median="median", mn="min", mx="max", n="count").reset_index())
    summ.to_csv(os.path.join(OUT, "ttft_summary.csv"), index=False)

    print(fits.to_string(index=False))
    print()
    print(pd.DataFrame(be).to_string(index=False))
    print()
    print(summ.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = {"recompute": "#444", "L1": "#1b7837", "L2": "#2166ac", "L3": "#b2182b"}
        for ax, logy in zip(axes, (False, True)):
            for t in ["recompute", "L1", "L2", "L3"]:
                d = summ[summ.tier == t]
                if d.empty:
                    continue
                ax.plot(d.L, d["median"], "o-", label=t, color=colors[t])
                ax.fill_between(d.L, d.mn, d.mx, alpha=0.15, color=colors[t])
            ax.set_xlabel("prompt tokens")
            ax.set_ylabel("TTFT (s)")
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
                ax.set_title("TTFT by tier (log-log)")
            else:
                ax.set_title("TTFT by tier (linear)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("HiCache tier cost curves - Qwen3-8B, H100, L3 on virtio-blk")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "ttft_vs_len.png"), dpi=130)
        print("\nwrote ttft_vs_len.png")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
