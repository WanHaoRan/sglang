#!/usr/bin/env python3
"""c12_summary.py --scale x70=DIR --scale x10=DIR --scale x1=DIR --out PNG

One panel per gap scale: mean TTFT of returning turns per arm, burst baseline (hollow) -> campaign 12 (filled), log x.
Reads the stats.csv that vs_baseline.py writes into each DIR (campaign 12 rows labelled c12, the baseline its own label).
"""
import argparse, csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
# Validated as a set (OKLab x100, Machado CVD): worst normal-vision dE 17.6, worst CVD dE 9.2.
ARM_COLOR = {"hbm_host": "#eb6834", "three_tier_to": "#1baf7a", "three_tier_wc": "#2a78d6", "hbm_lru": "#6b6a66"}
ARMS = list(ARM_COLOR)
CAMPAIGN = {"c7": "campaign 7", "c8": "campaign 8", "c9": "campaign 9", "c11": "campaign 11"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", action="append", required=True, help="LABEL=DIR with stats.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scales = [s.split("=", 1) for s in a.scale]
    fig, axes = plt.subplots(1, len(scales), figsize=(4.3 * len(scales), 3.4), facecolor=SURF, sharey=True)
    for ax, (label, d) in zip(axes, scales):
        rows = list(csv.DictReader(open(os.path.join(d, "stats.csv"))))
        base = next(r["campaign"] for r in rows if r["campaign"] != "c12")
        val = {(r["campaign"], r["arm"]): float(r["ttft_mean"]) for r in rows}
        for y, arm in enumerate(ARMS):
            col = ARM_COLOR[arm]
            b, c = val.get((base, arm)), val.get(("c12", arm))
            if b is not None and c is not None:
                ax.annotate("", xy=(c, y), xytext=(b, y),
                            arrowprops=dict(arrowstyle="-|>", color=col, linewidth=1.6, shrinkA=5, shrinkB=5))
            if b is not None:
                ax.plot([b], [y], "o", markersize=8, markerfacecolor=SURF, markeredgecolor=col, markeredgewidth=1.6)
                ax.text(b, y + 0.28, f"{b:.1f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
            if c is not None:
                ax.plot([c], [y], "o", markersize=8, color=col)
                ax.text(c, y - 0.28, f"{c:.1f}", ha="center", va="top", fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 400)
        ax.set_ylim(len(ARMS) - 0.4, -0.6)
        ax.set_yticks(range(len(ARMS)), ARMS)
        ax.set_facecolor(SURF)
        ax.set_title(f"{label}: {CAMPAIGN.get(base, base)} (hollow) to campaign 12", loc="left", fontsize=9.5, color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_xlabel("mean TTFT of returning turns (s, log scale)", fontsize=8.5, color=MUTED)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(a.out, dpi=140, facecolor=SURF, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
