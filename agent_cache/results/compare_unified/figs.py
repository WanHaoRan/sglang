"""The unified design's arms against the current design, per setting, in the campaign figure style
(scripts/jit_vs_hicache.py: 5-minute bins by send time, one legend above the panels, a footnote with the numbers).

In the container, from agent_cache/scripts/ (no GPU is used):
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python3 -B ../results/compare_unified/figs.py [setting folder ...]

Reads results/compare_unified/<setting>/manifest.txt (update.sh writes it: label, run folder, programs folder) and
settings.env, and each arm's client.jsonl. Writes into the setting's folder:
  timeline.png/.pdf, timeline.csv   mean TTFT, requests completed a second, and the share of prompt tokens recomputed
                                    although the client's history says they were cached before, per 5-minute bin
  ttft_cdf.png/.pdf                 the TTFT distribution over the requests every arm shown sent before the cut
Footnote numbers are over those shared requests (cut: 600 s before the end, as compare_arms.py); pairs are by play,
stream and index; the interval is a 95 % bootstrap over plays.
"""
import collections
import csv
import json
import os
import random
import statistics as st
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

import jit_vs_hicache as jv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..")
BIN_S, MARGIN_S = 300, 600
# label: (legend name, short name, colour, dashes); the designs set's colours, the current design in its orange.
STYLE = {
    "ec_full": ("Current design: the four rules", "current", "#eb6834", "-"),
    "ec_full_load": ("Current design + load hold", "current + hold", "#2a78d6", "-"),
    "unified": ("Unified cache rules, no hold", "cache rules", "#1baf7a", "-"),
    "unified_load": ("Unified design: cache rules + load hold", "unified", "#4a3aa7", "-"),
    "unified_load_po": ("Unified design + admission pass-over", "+ pass-over", "#a0307a", "--"),
    "unified_load_fixedprice": ("Unified design, fixed price of host room", "fixed price", "#8a8a80", "--"),
}
TITLES = {
    "A_128u_160g": "128 users, 160 GB host, 2 h",
    "B_cleaner50": "128 users, 160 GB host, SSD cleaner at 50 %, 2 h",
    "C_112g": "128 users, 112 GB host, 2 h",
    "D_64u_88g": "64 users, 88 GB host, 1 h",
}


def settings(folder):
    out = {}
    for line in open(os.path.join(folder, "settings.env")):
        k, _, v = line.strip().partition("=")
        if k and not k.startswith("#"):
            out[k] = v
    return out


def load_arm(path):
    reqs = {}
    with open(path) as f:
        for line in f:
            if '"kind":"req"' not in line:
                continue
            r = json.loads(line)
            if r.get("phase") != "profile" or r.get("error") or r.get("ttft") is None:
                continue
            reqs[(r["play"], r["stream"], r["k"])] = r
    return reqs


def paired(a, b, keys, rng):
    by_play = collections.defaultdict(lambda: [0.0, 0])
    for k in keys:
        cell = by_play[k[0]]
        cell[0] += b[k]["ttft"] - a[k]["ttft"]
        cell[1] += 1
    plays = list(by_play)
    boots = []
    for _ in range(2000):
        s = n = 0.0
        for p in rng.choices(plays, k=len(plays)):
            s += by_play[p][0]
            n += by_play[p][1]
        boots.append(s / n)
    boots.sort()
    total = sum(c[0] for c in by_play.values()) / sum(c[1] for c in by_play.values())
    return total, boots[50], boots[1949]


def bins_of(reqs, end_s):
    """Per 5-minute bin by send time: mean TTFT, completions a second (by finish time), recomputed share."""
    n = int(end_s // BIN_S)
    ttft = [[] for _ in range(n)]
    done = [0] * n
    lost = [0] * n
    prompt = [0] * n
    for r in reqs.values():
        b = int(r["t_send"] // BIN_S)
        if 0 <= b < n:
            ttft[b].append(r["ttft"])
            lost[b] += max(0, (r.get("ideal_cached") or 0) - (r.get("cached_tokens") or 0))
            prompt[b] += r.get("prompt_tokens") or 0
        e = int((r["t_send"] + r["latency"]) // BIN_S)
        if 0 <= e < n:
            done[e] += 1
    nan = float("nan")
    return ([st.mean(v) if v else nan for v in ttft], [d / BIN_S for d in done],
            [lost[i] / prompt[i] if prompt[i] else nan for i in range(n)])


def one(name):
    folder = os.path.join(RES, "compare_unified", name)
    manifest = [line.split() for line in open(os.path.join(folder, "manifest.txt")) if line.strip()]
    arms = [(label, run, prog) for label, run, prog in manifest if label in STYLE]
    if len(arms) < 2:
        print(f"{name}: {len(arms)} arm(s) with a style, nothing to draw")
        return
    cfg = settings(folder)
    end_s = float(cfg["DURATION"])
    cut = end_s - MARGIN_S
    data = {label: load_arm(os.path.join(RES, run, prog, "client.jsonl")) for label, run, prog in arms}
    order = [label for label in STYLE if label in data]
    shared = set.intersection(*(set(k for k, r in d.items() if r["t_send"] < cut) for d in data.values()))
    rng = random.Random(0)
    minutes = [i * BIN_S / 60 + BIN_S / 120 for i in range(int(end_s // BIN_S))]
    series = {label: bins_of(data[label], end_s) for label in order}

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.6), sharex=True, facecolor=jv.SURFACE)
    panels = (("Mean time to first token (log scale)", "seconds"), ("Requests completed per second", "requests/s"),
              ("Prompt tokens recomputed although cached before", "share of prompt tokens"))
    for i, (ax, (title, unit)) in enumerate(zip(axes, panels)):
        jv.style(ax, title, unit)
        for label in order:
            full, short, color, dash = STYLE[label]
            ax.plot(minutes, series[label][i], color=color, ls=dash, lw=1.8, label=full, marker="o", ms=2.6)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.1f} %"))
    axes[2].set_xlabel("minutes since the first request (5-minute bins by send time)")
    axes[2].set_xlim(0, end_s / 60)
    means = {label: st.mean(data[label][k]["ttft"] for k in shared) for label in order}
    lines = [f"Requests every arm sent before minute {cut / 60:.0f}: {len(shared):,}. Mean TTFT: "
             + "; ".join(f"{STYLE[label][1]} {means[label]:.2f} s" for label in order) + "."]
    base = order[0]
    rows = []
    for label in order[1:]:
        d, lo, hi = paired(data[base], data[label], sorted(shared), rng)
        rows.append((label, d, lo, hi))
    lines.append("Paired with the current design (same user, stream and request; 95 % interval over users): "
                 + "; ".join(f"{STYLE[label][1]} {d:+.2f} s [{lo:+.2f}, {hi:+.2f}]" for label, d, lo, hi in rows) + ".")
    lines.append("FreeInference sessions replayed as users at 0.1 users/s with recorded waits, Qwen3-30B-A3B on one H200, "
                 "FA3 attention; every arm on the same users.")
    foot = "\n".join(textwrap.fill(line, 132) for line in lines)
    legend_rows = (len(order) + 1) // 2
    fig.subplots_adjust(left=0.09, right=0.97, top=0.905 - 0.022 * (legend_rows - 1),
                        bottom=0.075 + 0.016 * foot.count("\n"), hspace=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.085, 0.958), ncol=2, frameon=False,
               fontsize=9.5, handlelength=1.8, columnspacing=1.6, labelcolor=jv.INK)
    fig.text(0.09, 0.985, f"Unified against current design: {TITLES.get(name, name)}", fontsize=12, color=jv.INK,
             va="top", weight="semibold")
    jv.footnote(fig, 0.09, foot)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"timeline.{ext}"), dpi=180, facecolor=jv.SURFACE)
    plt.close(fig)
    with open(os.path.join(folder, "timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute"] + [f"{label}_{c}" for label in order for c in ("ttft_mean_s", "done_per_s", "recomputed_share")])
        for i, m in enumerate(minutes):
            w.writerow([f"{m:.1f}"] + [f"{series[label][j][i]:.5g}" for label in order for j in range(3)])

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.2), facecolor=jv.SURFACE)
    jv.style(ax, "Time to first token, share of requests at or under", "share of requests")
    table = []
    for label in order:
        full, short, color, dash = STYLE[label]
        v = sorted(data[label][k]["ttft"] for k in shared)
        ax.plot(v, [(i + 1) / len(v) for i in range(len(v))], color=color, ls=dash, lw=1.8, label=full)
        table.append((short, means[label], jv.q(v, 0.5), jv.q(v, 0.9), jv.q(v, 0.99)))
    ax.set_xscale("log")
    ax.set_xlim(0.03, 300)
    ax.set_xlabel("seconds (log scale)")
    jv.percent(ax.yaxis)
    text = f"{'':<16s}{'mean':>6s} {'p50':>6s} {'p90':>6s} {'p99':>6s}\n" + "\n".join(
        f"{s:<16s}{m:6.2f} {a:6.2f} {b:6.2f} {c:6.2f}" for s, m, a, b, c in table)
    ax.text(0.98, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, family="monospace", color=jv.INK2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80 - 0.035 * (legend_rows - 1), bottom=0.17)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.085, 0.935), ncol=2, frameon=False,
               fontsize=9.5, handlelength=1.8, columnspacing=1.6, labelcolor=jv.INK)
    fig.text(0.09, 0.975, f"TTFT distribution: {TITLES.get(name, name)}", fontsize=12, color=jv.INK, va="top", weight="semibold")
    jv.footnote(fig, 0.09, textwrap.fill(lines[0] + " The table is in seconds.", 132))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"ttft_cdf.{ext}"), dpi=180, facecolor=jv.SURFACE)
    plt.close(fig)
    print(f"{name}: {len(order)} arms, {len(shared):,} shared requests; " + "; ".join(f"{label} {means[label]:.2f} s" for label in order))
    for label, d, lo, hi in rows:
        print(f"   {label} - {base}: {d:+.2f} s [{lo:+.2f}, {hi:+.2f}]")


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": jv.AXIS, "xtick.color": jv.INK2, "ytick.color": jv.INK2,
                         "axes.labelcolor": jv.INK2})
    names = sys.argv[1:] or sorted(d for d in os.listdir(os.path.join(RES, "compare_unified"))
                                   if os.path.isfile(os.path.join(RES, "compare_unified", d, "manifest.txt")))
    for name in names:
        one(name)


if __name__ == "__main__":
    main()
