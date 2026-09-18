"""TTFT of a prefix-cache hit by tier vs prompt length: three models, the A100 box and the old H100 box.

  python3 plot_tier_ttft.py [--mode light|dark] [--out <png>]

Reads the raw Exp 1 probes (ttft_by_tier.csv) of the old and new campaigns; median of the 3 reps per point,
band = min to max. Recompute rows that hit any tier are dropped, as in analyze_nixl.py. Each panel header gives
slope_recompute / slope_L3 from a line fit through the 7 per-length medians (the L3 delivered rate over the
recompute bar b*P, as in compare_campaigns.py) and, separately, at how many lengths the measured L3 median beat
the measured recompute median, with the best measured ratio: recompute is superlinear, so the slope fit is set
by the two longest lengths and overstates the speed-up at shorter ones.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE = "/sgl-workspace/sglang/hicache_eval/results"
NEW8, NEWC2 = f"{BASE}/20260917_a100_gcp_qwen8b", f"{BASE}/20260917_a100_gcp_32b70b_fp8kv"
OLD8, OLDC2 = f"{BASE}/20260907_203035/nixl", f"{BASE}/20260908_llama70b_awq_fp8kv"
# column title, KV bytes per token, new Exp 1 dir, old Exp 1 dir
MODELS = [("Qwen3-8B  (bf16 KV)", 147456, f"{NEW8}/exp1_nixl", f"{OLD8}/exp1_nixl"),
          ("Qwen3-32B-FP8  (fp8 KV)", 131072, f"{NEWC2}/exp1_32b", f"{OLDC2}/exp1_32b"),
          ("Llama-3.3-70B-AWQ  (fp8 KV)", 163840, f"{NEWC2}/exp1_70b", f"{OLDC2}/exp1_70b")]
ROWS = [("A100 box, L3 on a local NVMe", 2), ("H100 box, L3 on a virtio disk", 3)]
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32512]

# Surfaces, ink and the first three categorical slots of the reference palette; the three hues validate
# all-pairs (small multiples) in both modes. Aqua is below 3:1 on the light surface, hence the end labels.
# Muted tick ink is 4.85:1 on the dark surface but under 4.5:1 on the light one, so light ticks use ink2.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", tick="#52514e", grid="#e1e0d9",
                  axis="#c3c2b7", L1="#2a78d6", L2="#eb6834", L3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", tick="#898781", grid="#2c2c2a",
                 axis="#383835", L1="#3987e5", L2="#d95926", L3="#199e70"),
}
# tier, legend text, marker, marker size, z-order. L1 is a small circle drawn over a larger L2 square,
# so both stay visible where the two tiers coincide.
TIERS = [("L1", "L1: hit in GPU memory", "o", 5.5, 5), ("L2", "L2: hit in host memory", "s", 8.5, 4),
         ("L3", "L3: hit on the disk tier", "^", 8.0, 3)]
DASH = (0, (4, 2))


def load_summary(exp1_dir):
    d = pd.read_csv(os.path.join(exp1_dir, "ttft_by_tier.csv"))
    d = d[d.ttft_s.notna()]
    hit = (d.cached_device.fillna(0) + d.cached_host.fillna(0) + d.cached_storage.fillna(0)) > 0
    d = d[~((d.tier == "recompute") & hit)]
    return d.groupby(["tier", "L"]).ttft_s.agg(median="median", lo="min", hi="max").reset_index()


def fit_slopes(summary):
    return {t: np.polyfit(g.L, g["median"], 1)[0] for t, g in summary.groupby("tier")}


def panel_note(summary):
    slope = fit_slopes(summary)
    med = summary.pivot(index="L", columns="tier", values="median")
    speedup = med["recompute"] / med["L3"]
    return (f"L3 rate / recompute bar = {slope['recompute'] / slope['L3']:.1f}x (slope fit)\n"
            f"L3 faster at {int((speedup > 1).sum())} of {len(speedup)} lengths (best {speedup.max():.1f}x)")


def l3_rate_gibps(summary, kv_bytes_per_token):
    return kv_bytes_per_token / fit_slopes(summary)["L3"] / 2 ** 30


def end_labels(ax, ends, theme):
    """Label each line at its right end; lines that finish within ~48 % of each other share one stacked label."""
    groups = []
    for name, y in sorted(ends, key=lambda e: e[1]):
        if groups and np.log10(y / groups[-1][1][-1]) < 0.17:
            groups[-1][0].append(name); groups[-1][1].append(y)
        else:
            groups.append(([name], [y]))
    for names, ys in groups:
        ax.text(LENGTHS[-1] * 1.15, float(np.exp(np.mean(np.log(ys)))), "\n".join(reversed(names)),
                color=theme["ink2"], fontsize=9.5, va="center", ha="left", linespacing=1.05)


def draw_panel(ax, summary, theme):
    rec = summary[summary.tier == "recompute"]
    ax.fill_between(rec.L, rec.lo, rec.hi, color=theme["ink2"], alpha=0.10, lw=0, zorder=1)
    ax.plot(rec.L, rec["median"], color=theme["ink2"], lw=2, ls=DASH, dash_capstyle="round", zorder=2)
    ax.plot(rec.L.iloc[-1:], rec["median"].iloc[-1:], ls="", marker="o", ms=5.5, color=theme["ink2"],
            mec=theme["surface"], mew=1.4, zorder=2.5)
    ends = [("recompute", rec["median"].iloc[-1])]
    for tier, _, marker, size, z in TIERS:
        g = summary[summary.tier == tier]
        ax.fill_between(g.L, g.lo, g.hi, color=theme[tier], alpha=0.10, lw=0, zorder=1)
        ax.plot(g.L, g["median"], color=theme[tier], lw=2, marker=marker, ms=size, mec=theme["surface"], mew=1.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=z)
        ends.append((tier, g["median"].iloc[-1]))
    end_labels(ax, ends, theme)


def style_axes(ax, theme):
    ax.set_facecolor(theme["surface"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlim(430, LENGTHS[-1] * 3.4); ax.set_ylim(0.022, 70)
    ax.set_xticks(LENGTHS); ax.set_xticklabels(["512", "1K", "2K", "4K", "8K", "16K", "32K"])
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3, 10, 30]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.grid(True, which="major", color=theme["grid"], lw=0.8, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"]); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["tick"], labelsize=9.5, length=0)


def draw_headers(ax, model_title, note, theme):
    ax.text(0, 1.035, note, transform=ax.transAxes, color=theme["ink2"], fontsize=9.5, va="bottom", ha="left",
            linespacing=1.35)
    ax.text(0, 1.185, model_title, transform=ax.transAxes, color=theme["ink"], fontsize=10.5, fontweight="bold",
            va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(THEMES), default="light")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    theme = THEMES[a.mode]
    out = a.out or os.path.join(NEWC2, "ttft_by_tier_3models.png" if a.mode == "light" else "ttft_by_tier_3models_dark.png")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.8), sharex=True, sharey=True, facecolor=theme["surface"])
    fig.subplots_adjust(left=0.06, right=0.985, top=0.765, bottom=0.065, wspace=0.07, hspace=0.46)
    for r, (row_label, which) in enumerate(ROWS):
        summaries = [load_summary(m[which]) for m in MODELS]
        rates = [l3_rate_gibps(s, m[1]) for s, m in zip(summaries, MODELS)]
        for c, (model, summary) in enumerate(zip(MODELS, summaries)):
            ax = axes[r][c]
            style_axes(ax, theme)
            draw_panel(ax, summary, theme)
            draw_headers(ax, model[0], panel_note(summary), theme)
            if c == 0:
                ax.set_ylabel("TTFT (s)", color=theme["ink2"], fontsize=9.5, labelpad=6)
                lo, hi = (f"{v:.2f}" if v < 1 else f"{v:.1f}" for v in (min(rates), max(rates)))
                ax.text(0, 1.30, f"{row_label}: L3 delivers {lo} to {hi} GiB/s through HiCache",
                        transform=ax.transAxes, color=theme["ink"], fontsize=12, fontweight="bold", va="bottom", ha="left")
            if r == 1:
                ax.set_xlabel("prompt length (tokens)", color=theme["ink2"], fontsize=9.5, labelpad=6)

    handles = [Line2D([], [], color=theme["ink2"], lw=2, ls=DASH, dash_capstyle="round",
                      label="recompute: no cache hit (the line a tier must stay under)")]
    handles += [Line2D([], [], color=theme[t], lw=2, marker=m, ms=size, mec=theme["surface"], mew=1.4, label=lbl)
                for t, lbl, m, size, _ in TIERS]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.036, 0.925), ncol=4, frameon=False,
                     fontsize=9.5, handlelength=2.6, columnspacing=2.2)
    for text in leg.get_texts():
        text.set_color(theme["ink2"])
    fig.text(0.04, 0.975, "An L3 (disk) hit pays only where the disk outruns recompute: on the A100 box it does for the 32B and 70B, not the 8B",
             color=theme["ink"], fontsize=13.5, fontweight="bold", ha="left", va="center")
    fig.text(0.04, 0.951, "Time to first token of a prefix-cache hit, by tier and prompt length. Median of 3 probes, band = min to max; "
             "one idle request at a time; both axes log; same scale in every panel.",
             color=theme["ink2"], fontsize=9.5, ha="left", va="center")
    fig.text(0.04, 0.932, "Panel notes: L3 read rate over the recompute bar (KV bytes per token x marginal prefill rate, from line fits), "
             "then how often the measured L3 median beat recompute.",
             color=theme["ink2"], fontsize=9.5, ha="left", va="center")
    fig.savefig(out, dpi=160, facecolor=theme["surface"])
    print("wrote", out)


if __name__ == "__main__":
    main()
