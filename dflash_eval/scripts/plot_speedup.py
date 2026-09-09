#!/usr/bin/env python3
"""Plot decode speedup over the autoregressive baseline from a dflash_eval summary.csv.

    python3 scripts/plot_speedup.py                       # results/$(cat .current_results)/summary.csv
    python3 scripts/plot_speedup.py --run 20260909_1855   # a specific results dir
    python3 scripts/plot_speedup.py --with-accept-length  # add a second panel: acceptance length
    python3 scripts/plot_speedup.py --sampling think      # pick another sampling preset
    python3 scripts/plot_speedup.py -o /tmp/x.png

Speedup for a cell (dataset x concurrency x sampling) is the arm's client throughput
divided by the same track's baseline arm (A0 for A*, B0 for B*) in the same cell.
Cells without a baseline row are skipped and reported. If a cell was re-run and
appended to summary.csv, the latest row wins and a warning is printed.
Needs only matplotlib (the host python3 has it; the dflash venv does not).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent

# Fixed order, never cycled: slots 1-3 of the validated reference palette (blue, orange,
# aqua) validate all-pairs; slots 4-8 are adjacent-pair safe. More than 8 arms is refused.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

ARM_LABELS = {
    "A0": "baseline", "B0": "baseline",
    "A1": "EAGLE3 (3/4/16)",
    "A1-auto": "EAGLE3 (auto 3/1/4)",
    "A1-w8": "EAGLE3 (7/1/8)",
    "A1-card": "EAGLE3 (6/10/32)",
    "A2": "DFlash-b16",
    "A2-w8": "DFlash-b16 (window 8)",
    "A3": "DSpark block7",
    "B1-4": "MTP (3/1/4)",
    "B1-8": "MTP (7/1/8)",
    "B2": "DSpark",
    "B3": "DFlash2",
}
TARGET_LABELS = {"A": "Qwen3-8B", "B": "Qwen3.8-27B-FP8"}


def baseline_arm(arm: str) -> str:
    """A1, A2-w8, ... -> A0; B3 -> B0."""
    return arm[0] + "0"


def is_baseline(arm: str) -> bool:
    return arm == baseline_arm(arm)


def read_summary(path: Path) -> list[dict]:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("client_throughput_tok_s")]
    for r in rows:
        r["throughput"] = float(r["client_throughput_tok_s"])
        r["concurrency"] = int(r["concurrency"])
        al = r.get("server_avg_spec_accept_length") or r.get("client_accept_len") or ""
        r["accept_len"] = float(al) if al else None
    return rows


def build_table(rows: list[dict], sampling: str):
    """-> (cells, arms, speedup[arm][cell], accept[arm][cell], baseline_tps[cell], window[arm], rows_used)."""
    rows = [r for r in rows if r["sampling"] == sampling]
    tps: dict = {}
    for r in rows:                                  # later rows win; say so
        key = (r["arm"], r["dataset"], r["concurrency"])
        if key in tps:
            print(f"warning: duplicate cell {key}; using the later row ({r['timestamp']})", file=sys.stderr)
        tps[key] = r
    cells = sorted(OrderedDict.fromkeys((k[1], k[2]) for k in tps))
    arms = sorted(a for a in OrderedDict.fromkeys(k[0] for k in tps) if not is_baseline(a))
    if len(arms) > len(PALETTE):
        raise SystemExit(f"{len(arms)} arms exceed the {len(PALETTE)}-slot palette; split the chart")
    speedup: dict = {a: {} for a in arms}
    accept: dict = {a: {} for a in arms}
    baseline: dict = {}
    for a in arms:
        for c in cells:
            row = tps.get((a, *c))
            base = tps.get((baseline_arm(a), *c))
            if row is None:
                continue
            if base is None:
                print(f"warning: no {baseline_arm(a)} row for {c}; skipping {a} there", file=sys.stderr)
                continue
            baseline[c] = base["throughput"]
            speedup[a][c] = row["throughput"] / base["throughput"]
            accept[a][c] = row["accept_len"]
    cells = [c for c in cells if c in baseline]
    window = {a: next((tps[(a, *c)]["window"] for c in cells if (a, *c) in tps), "?") for a in arms}
    return cells, arms, speedup, accept, baseline, window, rows


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, cells, arms, values, labels, fmt, ylabel, ref_line=None):
    n = len(arms)
    width = min(0.66 / n, 0.19)       # cap bar thickness; the band's leftover is air
    gap = width * 0.12                # surface gap between adjacent bars
    xs = range(len(cells))
    for i, arm in enumerate(arms):
        offs = (i - (n - 1) / 2) * width
        vals = [values[arm].get(c) for c in cells]
        bars = ax.bar([x + offs for x in xs], [v or 0 for v in vals], width - gap,
                      color=PALETTE[i], label=labels[arm], linewidth=0)
        for b, v in zip(bars, vals):
            x = b.get_x() + b.get_width() / 2
            if v is None:
                ax.annotate("n/a", (x, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=TEXT_MUTED)
                continue
            ax.annotate(fmt(v), (x, b.get_height()), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=TEXT_PRIMARY)
    if ref_line is not None:          # a legend entry, so it never sits on top of a bar
        ax.axhline(ref_line[0], color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)), label=ref_line[1])
    ax.set_xticks(list(xs))
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.margins(x=0.04)
    style_axis(ax)


def finish_panel(ax, n_series, top_value):
    ncol = min(n_series, 4)
    rows = math.ceil(n_series / ncol)
    ax.set_ylim(0, top_value * (1.18 + 0.10 * (rows - 1)))   # headroom grows with legend rows
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, ncol=ncol)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="results dir name under dflash_eval/results (default: .current_results)")
    p.add_argument("--summary", help="explicit path to a summary.csv (overrides --run)")
    p.add_argument("--sampling", default="greedy", help="sampling preset to plot (default: greedy)")
    p.add_argument("--with-accept-length", action="store_true", help="add a second panel with acceptance length")
    p.add_argument("-o", "--output", help="output image path (default: <results dir>/speedup_<sampling>.png)")
    p.add_argument("--title", help="override the figure title")
    p.add_argument("--note", default="1x H100 PCIe 80GB, SGLang, radix cache off, fa3",
                   help="hardware/config fragment for the subtitle")
    args = p.parse_args()

    if args.summary:
        summary = Path(args.summary)
    else:
        run = args.run or (EVAL_DIR / ".current_results").read_text().strip()
        summary = EVAL_DIR / "results" / run / "summary.csv"
    if not summary.exists():
        print(f"no such file: {summary}", file=sys.stderr)
        return 1

    cells, arms, speedup, accept, baseline, window, used = build_table(read_summary(summary), args.sampling)
    if not cells or not arms:
        print(f"nothing to plot for sampling={args.sampling} in {summary}", file=sys.stderr)
        return 1
    track = arms[0][0]
    labels = {a: ARM_LABELS.get(a, a) for a in arms}
    accept_labels = {a: f"{labels[a]}, window {window[a]}" for a in arms}
    have_accept = any(v for a in arms for v in accept[a].values())
    panels = 2 if (args.with_accept_length and have_accept) else 1
    if args.with_accept_length and not have_accept:
        print("warning: no acceptance-length values for this preset; dropping that panel", file=sys.stderr)

    # ---- print the table that the chart shows ----------------------------------
    print(f"{'cell':<18}{'baseline tok/s':>15}" + "".join(f"{labels[a]:>26}" for a in arms))
    for c in cells:
        line = f"{c[0]} c={c[1]:<11}{baseline[c]:>15.1f}"
        for a in arms:
            s, al = speedup[a].get(c), accept[a].get(c)
            cell = "" if s is None else f"{s:.2f}x" + ("" if al is None else f" (AL {al:.2f})")
            line += f"{cell:>26}"
        print(line)

    # ---- figure -----------------------------------------------------------------
    fig_w = max(9.5, 0.45 * len(arms) * len(cells) + 1.5)      # keep ~0.4 in of bar pitch
    fig, axes = plt.subplots(panels, 1, figsize=(fig_w, 4.2 * panels + 0.6), dpi=200,
                             sharex=True, facecolor=SURFACE, constrained_layout=True)
    axes = [axes] if panels == 1 else list(axes)

    grouped_bars(axes[0], cells, arms, speedup, labels, lambda v: f"{v:.2f}x",
                 "speedup over baseline (output tok/s ratio)", ref_line=(1.0, "baseline = 1.00x"))
    axes[0].yaxis.set_major_locator(MultipleLocator(1))
    finish_panel(axes[0], len(arms) + 1, max(v for a in arms for v in speedup[a].values()))

    if panels == 2:
        grouped_bars(axes[1], cells, arms, accept, accept_labels, lambda v: f"{v:.2f}",
                     "mean acceptance length (tokens per verify step)")
        finish_panel(axes[1], len(arms), max(v for a in arms for v in accept[a].values() if v))

    axes[-1].set_xticklabels([f"{d}, concurrency {c}\nbaseline {baseline[(d, c)]:.0f} tok/s" for d, c in cells],
                             color=TEXT_SECONDARY, fontsize=9)

    n_prompts = "/".join(sorted({r["num_prompts"] for r in used}))
    max_new = "/".join(sorted({r["max_new_tokens"] for r in used}))
    title = args.title or f"{TARGET_LABELS.get(track, 'target')}: speculative decoding speedup over autoregressive baseline"
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11.5, fontweight="semibold")
    axes[0].set_title(f"{args.note}; sampling={args.sampling}, {n_prompts} prompts per cell, "
                      f"max_new_tokens={max_new}; run {summary.parent.name}",
                      color=TEXT_SECONDARY, fontsize=9, loc="left")

    out = Path(args.output) if args.output else summary.parent / f"speedup_{args.sampling}{'_accept' if panels == 2 else ''}.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
