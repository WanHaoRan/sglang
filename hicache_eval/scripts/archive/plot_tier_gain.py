#!/usr/bin/env python3
"""Plot per-tier TTFT gain (recompute TTFT / tier TTFT) vs input length.

Sources: Exp 1 `ttft_by_tier.csv` for Qwen3-8B (campaign 1, nixl), Qwen3-32B-FP8
and Llama-3.3-70B-AWQ (campaign 2). Medians of N=3 reps per (tier, L); recompute
rows that reported any cached tokens are dropped (DEVIATIONS.md D16.1).

Usage: python3 plot_tier_gain.py [--out DIR]
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

MODELS = [
    ("Qwen3-8B, bf16 KV", os.path.join(RES, "20260907_203035/nixl/exp1_nixl/ttft_by_tier.csv")),
    ("Qwen3-32B-FP8, fp8 KV", os.path.join(RES, "20260908_llama70b_awq_fp8kv/exp1_32b/ttft_by_tier.csv")),
    ("Llama-3.3-70B-AWQ, fp8 KV", os.path.join(RES, "20260908_llama70b_awq_fp8kv/exp1_70b/ttft_by_tier.csv")),
]
TIERS = [("L1", "L1  GPU HBM", "#2a78d6"), ("L2", "L2  host DRAM", "#eb6834"), ("L3", "L3  SSD via nixl", "#1baf7a")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(path):
    df = pd.read_csv(path)
    cached = df[["cached_device", "cached_host", "cached_storage"]].fillna(0).sum(axis=1)
    df = df[~((df.tier == "recompute") & (cached > 0))]
    med = df.groupby(["tier", "L"]).ttft_s.median().unstack("tier")
    n = df.groupby(["tier", "L"]).ttft_s.size().unstack("tier")
    gain = med[["L1", "L2", "L3"]].rdiv(med["recompute"], axis=0)
    return med, n, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "20260908_llama70b_awq_fp8kv"))
    a = ap.parse_args()

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (name, path) in zip(axes, MODELS):
        med, n, gain = load(path)
        Ls = list(gain.index)
        # Recompute baseline: neutral ink, dashed, labelled in seconds.
        ax.plot(Ls, med["recompute"], color=INK2, lw=1.6, ls=(0, (4, 3)), marker="o", ms=4.5,
                mec="#fcfcfb", mew=1.0, label="recompute (baseline)", zorder=2)
        for tier, label, color in TIERS:
            ax.plot(Ls, med[tier], color=color, lw=2, marker="o", ms=5.5,
                    mec="#fcfcfb", mew=1.2, label=label, zorder=3)
        # End labels in seconds (plus gain for tiers); nudge apart when within 15%.
        ends = [(med["recompute"].iloc[-1], "recompute", None)]
        ends += [(med[t].iloc[-1], t, gain[t].iloc[-1]) for t, _, _ in TIERS]
        ends.sort(reverse=True)
        offsets = {t: 0 for _, t, _ in ends}
        for (hi, th, _), (lo, tl, _) in zip(ends, ends[1:]):
            if hi / lo < 1.15:
                offsets[th] += 6
                offsets[tl] -= 6
        for y, tier, g in ends:
            txt = f"{tier} {y:.2f} s" if g is None else f"{tier} {y:.2f} s  ({g:.1f}x)"
            ax.annotate(txt, (Ls[-1], y), xytext=(6, offsets[tier]), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        for L in Ls:
            for tier, _, _ in TIERS:
                rows.append(dict(model=name, L=L, tier=tier, tier_ttft_s=med.loc[L, tier],
                                 recompute_ttft_s=med.loc[L, "recompute"], gain=gain.loc[L, tier],
                                 n_tier=n.loc[L, tier], n_recompute=n.loc[L, "recompute"]))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ls)
        ax.set_xticklabels([f"{L//1024}K" if L >= 1024 else str(L) for L in Ls], fontsize=9)
        yt = [0.03, 0.1, 0.3, 1, 3, 10, 30]
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{v:g} s" for v in yt], fontsize=9)
        ax.set_xlim(Ls[0] * 0.85, Ls[-1] * 2.6)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.set_xlabel("input length (tokens)", fontsize=9.5, color=INK2)
    axes[0].set_ylabel("TTFT (seconds, log scale)", fontsize=9.5, color=INK2)
    axes[0].set_ylim(0.02, 60)
    fig.suptitle("Prefix-cache hit TTFT by tier against the recompute baseline",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.905, "Exp 1, nixl backend, wait_complete, single in-flight probe, median of N=3, "
             "1x H100 PCIe, L3 on a virtio-blk VM disk (not NVMe). End labels: TTFT and gain vs baseline.",
             fontsize=9, color=INK2)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, "tier_gain_vs_len.png")
    csv = os.path.join(a.out, "tier_gain_vs_len.csv")
    fig.savefig(png, dpi=160, facecolor=fig.get_facecolor())
    pd.DataFrame(rows).round(4).to_csv(csv, index=False)
    print(png)
    print(csv)
    print(pd.DataFrame(rows).pivot_table(index=["model", "L"], columns="tier", values="gain").round(2))


if __name__ == "__main__":
    main()
