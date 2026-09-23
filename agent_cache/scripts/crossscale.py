#!/usr/bin/env python3
"""crossscale.py --out DIR <label>=<manifest> [...]   (labels in x-axis order, e.g. "x70=..." "x10=..." "x1=...")

One row per (campaign, arm) in crossscale.csv and a four-panel figure: mean / p90 returning-turn TTFT, share of returning
turns served from each tier, and the queue-wait share of the per-turn time, each against the gap scale. Everything is
recomputed from the raw client.jsonl + server.log through delay_components.load_arm (same tier classes and delay split).
"""
import argparse, csv, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delay_components import load_calibrated
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ARM_COLOR = {"hbm_lru": "#9a9891", "hbm_host": "#eb6834", "three_tier_to": "#1baf7a", "three_tier_wc": "#2e7d32"}
ARM_MARK = {"hbm_lru": "s", "hbm_host": "o", "three_tier_to": "^", "three_tier_wc": "v"}
def q(xs, f): xs = sorted(xs); return xs[min(len(xs) - 1, int(f * len(xs)))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("sets", nargs="+"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []; labels = []
    sets = [tuple(spec.rsplit("=", 1)) for spec in a.sets]; labels = [l for l, _ in sets]
    for label, arm, T in load_calibrated(sets):
        if True:
            n = len(T); tt = [t["ttft"] for t in T]
            tiers = {k: sum(1 for t in T if t["tier"] == k) / n for k in ("device", "host", "storage", "cold")}
            tot = sum(t["queue"] + t["prefill"] + t["decode"] for t in T) / n
            rows.append(dict(scale=label, arm=arm, turns=n, ttft_mean=round(st.mean(tt), 3), ttft_p50=round(q(tt, .5), 3), ttft_p90=round(q(tt, .9), 3), ttft_p99=round(q(tt, .99), 3),
                             share_device=round(tiers["device"], 4), share_host=round(tiers["host"], 4), share_storage=round(tiers["storage"], 4), share_recompute=round(tiers["cold"], 4),
                             storage_hits=sum(1 for t in T if t["tier"] == "storage"), queue_mean=round(sum(t["queue"] for t in T) / n, 3), l3_read_mean=round(sum(t["l3_read"] for t in T) / n, 3),
                             prefill_mean=round(sum(t["prefill"] for t in T) / n, 3), decode_mean=round(sum(t["decode"] for t in T) / n, 3), queue_share=round(sum(t["queue"] for t in T) / n / tot, 4) if tot else 0,
                             measured_frac=round(sum(1 for t in T if t["how"] != "estimated") / n, 3)))
            r = rows[-1]; print(f"{label:>6} {arm:14s} n {n:5d}  ttft mean {r['ttft_mean']:7.2f} p50 {r['ttft_p50']:7.2f} p90 {r['ttft_p90']:7.2f}  dev/host/sto/recomp {100*r['share_device']:3.0f}/{100*r['share_host']:3.0f}/{100*r['share_storage']:4.1f}/{100*r['share_recompute']:3.0f} %  queue {r['queue_mean']:6.1f} s ({100*r['queue_share']:.0f} %)")
    with open(os.path.join(a.out, "crossscale.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    arms = [ar for ar in ARM_COLOR if any(r["arm"] == ar for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5)); fig.patch.set_facecolor(SURF)
    panels = [("ttft_mean", "returning-turn TTFT, mean (s, log)", True), ("ttft_p90", "returning-turn TTFT, p90 (s, log)", True),
              ("share_storage", "share of returning turns restored from the SSD", False), ("queue_share", "queue wait as a share of the per-turn time", False)]
    for ax, (key, title, log) in zip(axes.flatten(), panels):
        ax.set_facecolor(SURF)
        for ar in arms:
            ys = [next((r[key] for r in rows if r["scale"] == lab and r["arm"] == ar), None) for lab in labels]
            xs = [i for i, y in enumerate(ys) if y is not None]; ys = [y for y in ys if y is not None]
            ax.plot(xs, ys, color=ARM_COLOR[ar], marker=ARM_MARK[ar], ms=7, linewidth=2, label=ar, markeredgecolor=SURF, markeredgewidth=1.2)
            for x, y in zip(xs, ys): ax.annotate(f"{y:.2f}" if key in ("share_storage", "queue_share") else f"{y:.1f}", (x, y), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=MUTED, va="center")
        if log: ax.set_yscale("log")
        if key in ("share_storage", "queue_share"): ax.set_ylim(0, 1.0 if key == "queue_share" else max(0.3, 1.1 * max((r[key] for r in rows), default=0.3)))
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9); ax.set_xlim(-0.4, len(labels) - 0.4)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8); ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[0][0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("The same four arms across the gap scale (128 live sessions, swebench replay, H200 + 160 GB host pool + SSD)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = os.path.join(a.out, "crossscale.png"); fig.savefig(out, dpi=130, facecolor=SURF); print("wrote", out)

if __name__ == "__main__": main()
