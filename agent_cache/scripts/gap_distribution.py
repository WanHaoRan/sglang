#!/usr/bin/env python3
"""gap_distribution.py --set x1=M --set x10=M --set x70=M --this x70 [--iostat LOG] --out PNG

ECDF of the idle gap before each returning turn (the client's gap_slept: end of the previous reply to the next send),
one line per campaign, in the form of results/compare_20260922_023714/gap_distribution.png. The gaps of a campaign are
identical across its arms, so the first arm of each manifest is read. --this names the campaign the figure belongs to.

Reference lines, from the --this campaign: the SSD read time of its median restore (median storage-hit tokens of the
three-tier arms' restores, as in campaign_summary.py, x --bytes-per-token) at the SSD's measured read ceiling and at
its mean read rate while active (iostat samples reading > 1 MiB/s inside the three-tier arms' client windows).
A text block lists p10 / p50 / p90 / max per line and the share of gaps at or above each reference, and the arrival
process and conversation lengths of the --this campaign.
"""
import argparse
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_baseline import load_iostat, q, tier_of  # noqa: E402

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
# First three slots of the dataviz reference palette (validated all-pairs), in the order of the example figure.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
GIB = 2 ** 30


def read_client(results, run, client):
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    return next(r for r in rows if r.get("kind") == "run"), rows


def manifest_arms(path):
    results = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return results, [l.split()[:3] for l in open(path) if l.strip()]


def campaign_gaps(path):
    """Slept gaps of the returning turns and the client args of the manifest's first arm."""
    results, arms = manifest_arms(path)
    _, run, client = arms[0]
    run_rec, rows = read_client(results=results, run=run, client=client)
    gaps = [r.get("gap_slept") or 0.0 for r in rows if r.get("kind") == "turn" and r.get("turn", 0) > 0 and "t_send" in r]
    return gaps, run_rec["args"], rows


def restore_reference(path, iostat_path):
    """Median storage tokens per SSD restore and the mean active read rate (bytes/s) of the three-tier arms."""
    results, arms = manifest_arms(path)
    iostat = load_iostat(iostat_path) if iostat_path else []
    tokens, rates = [], []
    for arm, run, client in arms:
        if not arm.startswith("three_tier"):
            continue
        run_rec, rows = read_client(results=results, run=run, client=client)
        ret = [r for r in rows if r.get("kind") == "turn" and r.get("turn", 0) > 0 and "t_send" in r and r.get("ttft")]
        tokens += [(r.get("cached_details") or {}).get("storage") or 0 for r in ret if tier_of(r) == "storage"]
        summ = next((r for r in rows if r.get("kind") == "summary"), {})
        t0, wall = run_rec["t_start"], summ.get("elapsed_s") or 0
        rates += [x[1] * 1024 for x in iostat if t0 < x[0] <= t0 + wall + 10 and x[1] > 1024]
    return (q(xs=tokens, f=.5) if tokens else None), (st.mean(rates) if rates else None)


def workload_line(rows, args):
    """Arrivals and conversation lengths of one client run."""
    starts = [r.get("arrival_s", r.get("start_s", 0.0)) for r in rows if r.get("kind") == "conv_start"]
    per_conv = {}
    for r in rows:
        if r.get("kind") == "turn" and "t_send" in r:
            per_conv[r["conv"]] = per_conv.get(r["conv"], 0) + 1
    lens = list(per_conv.values())
    rate = float(args.get("arrival_rate") or 0)
    arrive = (f"Poisson arrivals at {rate:g}/s, last at {max(starts) / 60:.1f} min" if rate > 0
              else f"all {len(starts)} start at t = 0")
    return (f"{len(starts)} conversations, {arrive}; turns per conversation p10 {q(xs=lens, f=.1)}, "
            f"p50 {q(xs=lens, f=.5)}, p90 {q(xs=lens, f=.9)}, range {min(lens)}-{max(lens)} ({sum(lens):,} turns)")


def draw(sets, this, refs, wl, out):
    fig, ax = plt.subplots(figsize=(14.3, 8.0))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for (label, gaps, args), color in zip(sets, COLORS):
        xs = sorted(max(0.01, g) for g in gaps)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        scale, cap = float(args.get("gap_scale") or 1), float(args.get("gap_cap") or 0)
        name = f"{label}: recorded gaps x{scale:g}" + (f", cap {cap:,.0f} s" if cap else "")
        ax.plot(xs, ys, color=color, linewidth=2.6 if label == this else 1.8, label=name, drawstyle="steps-post")
        p50 = q(xs=xs, f=.5)
        ax.scatter([p50], [0.5], s=60, color=color, edgecolor=SURF, linewidth=2, zorder=5)
        ax.annotate(f"{label}\np50 {p50:.1f} s", (p50, 0.5), xytext=(-10, -30), textcoords="offset points",
                    fontsize=10, color=INK, ha="right")
    for x, _ in refs:
        ax.axvline(x, color=MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 3000)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1 s", "1 s", "10 s", "100 s", "1,000 s"])
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c1")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlabel("idle gap before the turn (end of previous reply to send)", fontsize=11, color=MUTED)
    ax.set_ylabel("share of returning turns with a gap <= x", fontsize=11, color=MUTED)
    n = len(sets[0][1])
    ax.set_title(f"Gap distribution, {wl['title']}, {n:,} returning turns per scale; this directory: {this}",
                 fontsize=12, color=INK)
    handles = ax.get_legend_handles_labels()[0]
    if refs:
        handles.append(Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (3, 3)), label=wl["ref_note"]))
    ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=False)
    stats = []
    width = max(len(label) for label, _, _ in sets)
    for label, gaps, _ in sets:
        cells = [f"p10 {q(xs=gaps, f=.1):7.2f}  p50 {q(xs=gaps, f=.5):7.2f}  p90 {q(xs=gaps, f=.9):7.2f}  max {max(gaps):6.0f} s"]
        cells += [f">= {x:.1f} s: {100 * sum(g >= x for g in gaps) / len(gaps):3.0f} %" for x, _ in refs]
        cells += [f">= {x:.0f} s: {100 * sum(g >= x for g in gaps) / len(gaps):3.0f} %" for x in (60, 300)]
        stats.append(f"{label:>{width}}: " + "   ".join(cells))
    stats.append(wl["line"])
    fig.text(0.01, 0.005, "\n".join(stats), fontsize=8.6, family="monospace", color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.21)
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="label=manifest, in legend order (up to three)")
    ap.add_argument("--this", required=True, help="label of the campaign this figure belongs to")
    ap.add_argument("--title", default="", help="workload description for the title")
    ap.add_argument("--iostat", default="", help="iostat log of the --this campaign, for the loaded read rate")
    ap.add_argument("--bytes-per-token", type=int, default=98_304, help="KV bytes per token (Qwen3-30B-A3B bf16)")
    ap.add_argument("--ceiling-gibs", type=float, default=1.88, help="measured SSD read ceiling, GiB/s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = [s.split("=", 1) for s in a.set]
    sets, this_rows, this_args, this_path = [], None, None, None
    for label, path in specs:
        gaps, args, rows = campaign_gaps(path)
        sets.append((label, gaps, args))
        if label == a.this:
            this_rows, this_args, this_path = rows, args, path
    if this_path is None:
        ap.error(f"--this {a.this} is not one of the --set labels")
    tokens, rate = restore_reference(path=this_path, iostat_path=a.iostat)
    refs, note = [], ""
    if tokens:
        size = tokens * a.bytes_per_token
        refs.append((size / (a.ceiling_gibs * GIB), f"{size / (a.ceiling_gibs * GIB):.1f} s at the {a.ceiling_gibs:.2f} GiB/s ceiling"))
        if rate:
            refs.append((size / rate, f"{size / rate:.1f} s at the loaded {rate / GIB:.2f} GiB/s"))
        note = (f"SSD read of a median {a.this} restore ({tokens / 1e3:.1f}K tokens, {size / 1e9:.1f} GB):\n"
                + ", ".join(t for _, t in refs))
    wl = dict(title=a.title or "agent replay", line=workload_line(rows=this_rows, args=this_args), ref_note=note)
    draw(sets=sets, this=a.this, refs=refs, wl=wl, out=a.out)
    print("wrote", a.out)
    for label, gaps, _ in sets:
        print(f"{label}: n {len(gaps)} p10 {q(xs=gaps, f=.1):.2f} p50 {q(xs=gaps, f=.5):.2f} p90 {q(xs=gaps, f=.9):.2f} "
              f"max {max(gaps):.0f}")
    print("refs:", [(round(x, 2), t.replace("\n", " ")) for x, t in refs])
    print(wl["line"])


if __name__ == "__main__":
    main()
