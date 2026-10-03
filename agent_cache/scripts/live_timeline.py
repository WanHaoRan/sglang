#!/usr/bin/env python3
"""live_timeline.py --manifest M [--ref M2 --ref-label c7] --label x70 --out DIR

Users over time for open-loop runs (campaign 12: Poisson arrivals, varied lengths): per arm, the number of live
conversations (arrived and not yet ended) and the TTFT of returning turns over time, plus per-phase statistics.
Phases per arm: arrival = before the last conversation arrives; peak = from then until the live count first drops below
90 % of the arm's own peak; departure = after that. A conversation is live from its arrival_s to its end_s
(conv_end.end_s; for runs that predate end_s, the end of its last reply). Writes live_timeline.png and live_phases.csv.
--ref adds another campaign's arms (e.g. campaign 7, all conversations at t = 0) to the phase table only.
"""
import argparse, csv, json, os, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SH = 7808  # shared system prompt (tokens), cached for every returning turn
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
# Validated as a set (OKLab x100, Machado CVD): worst normal-vision dE 17.6, worst CVD dE 9.2.
ARM_COLOR = {"hbm_host": "#eb6834", "three_tier_to": "#1baf7a", "three_tier_wc": "#2a78d6", "hbm_lru": "#6b6a66"}
TIERS = ("device", "host", "storage", "recompute")


def tier_of(r):
    cd = r.get("cached_details") or {}
    p = r.get("prompt_tokens") or 1
    if p - (r.get("cached_tokens") or 0) > 0.5 * max(p - SH, 1):
        return "recompute"
    return "storage" if (cd.get("storage") or 0) > 0 else "host" if (cd.get("host") or 0) > 0 else "device"


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def load(results, run, client):
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    arrive = {r["conv"]: r.get("arrival_s", r.get("start_s", 0.0)) for r in rows if r.get("kind") == "conv_start"}
    end = {r["conv"]: r["end_s"] for r in rows if r.get("kind") == "conv_end" and "end_s" in r}
    turns = [r for r in rows if r.get("kind") == "turn" and "t_send" in r and r.get("latency") is not None]
    last_reply = {}
    for r in turns:
        last_reply[r["conv"]] = max(last_reply.get(r["conv"], 0.0), r["t_send"] + r["latency"])
    for c, t in last_reply.items():
        end.setdefault(c, t)   # runs that predate conv_end.end_s: a conversation ends with its last reply
    return arrive, end, turns


def live_at(arrive, end, t):
    return sum(1 for c, a in arrive.items() if a <= t < end.get(c, float("inf")))


def phases(arrive, end):
    last = max(arrive.values())
    grid = [i * 5.0 for i in range(int(max(end.values()) / 5.0) + 2)]
    live = [live_at(arrive, end, t) for t in grid]
    peak = max(live)
    drop = next((t for t, n in zip(grid, live) if t >= last and n < 0.9 * peak), grid[-1])
    return last, drop, peak, grid, live


def phase_rows(label, arm, arrive, end, turns):
    last, drop, peak, _, _ = phases(arrive, end)
    out = []
    for name, lo, hi in (("arrival", 0.0, last), ("peak", last, drop), ("departure", drop, float("inf")), ("all", 0.0, float("inf"))):
        sel = [r for r in turns if r["turn"] > 0 and r.get("ttft") and lo <= r["t_send"] < hi]
        if not sel:
            continue
        tt = [r["ttft"] for r in sel]
        row = dict(label=label, arm=arm, phase=name, start_min=round(lo / 60, 1),
                   end_min=round(min(hi, max(end.values())) / 60, 1), peak_live=peak, returning_turns=len(sel),
                   ttft_mean=round(st.mean(tt), 3), ttft_p50=round(q(tt, .5), 3), ttft_p90=round(q(tt, .9), 3),
                   ttft_p99=round(q(tt, .99), 3))
        row.update({f"share_{k}": round(sum(tier_of(r) == k for r in sel) / len(sel), 3) for k in TIERS})
        out.append(row)
    return out


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--ref", default="", help="another campaign's manifest, added to the phase table")
    ap.add_argument("--ref-label", default="reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", type=float, default=120.0, help="TTFT bin width, s")
    a = ap.parse_args()
    results = os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    os.makedirs(a.out, exist_ok=True)
    arms = [l.split() for l in open(a.manifest) if l.strip()]
    data = {arm: load(results, run, client) for arm, run, client in arms}

    rows = []
    for arm, (arrive, end, turns) in data.items():
        rows += phase_rows(a.label, arm, arrive, end, turns)
    if a.ref:
        rres = os.path.dirname(os.path.dirname(os.path.abspath(a.ref)))
        for arm, run, client in (l.split() for l in open(a.ref) if l.strip()):
            if arm in ARM_COLOR:
                rows += phase_rows(a.ref_label, arm, *load(rres, run, client))
    with open(os.path.join(a.out, "live_phases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['label']:5s} {r['arm']:14s} {r['phase']:9s} {r['start_min']:6.1f}-{r['end_min']:6.1f} min  peak {r['peak_live']:3d}  "
              f"n {r['returning_turns']:5d}  TTFT mean {r['ttft_mean']:7.2f} p50 {r['ttft_p50']:6.2f} p90 {r['ttft_p90']:7.2f} "
              f"p99 {r['ttft_p99']:7.2f}  dev/host/ssd/rec {r['share_device']:.2f}/{r['share_host']:.2f}/"
              f"{r['share_storage']:.2f}/{r['share_recompute']:.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.4), facecolor=SURF, sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.18))
    style(ax1, f"Live conversations (arrived, not yet ended){' - ' + a.label if a.label else ''}; dotted line: last arrival")
    style(ax2, f"TTFT of returning turns, median per {a.bin / 60:.0f}-min bin (log scale; band = p10-p90)")
    xmax, labels, handles = 0.0, [], []
    for arm, (arrive, end, turns) in data.items():
        col = ARM_COLOR.get(arm, MUTED)
        last, drop, peak, grid, live = phases(arrive, end)
        xs = [t / 60 for t in grid]
        xmax = max(xmax, xs[-1])
        handles.append(ax1.step(xs, live, where="post", color=col, linewidth=2, label=arm)[0])
        labels.append((ax1, xs[-1], live[-1], arm))
        bins = {}
        for r in turns:
            if r["turn"] > 0 and r.get("ttft"):
                bins.setdefault(int(r["t_send"] // a.bin), []).append(r["ttft"])
        ks = sorted(k for k, v in bins.items() if len(v) >= 3)
        bx = [(k + 0.5) * a.bin / 60 for k in ks]
        ax2.plot(bx, [st.median(bins[k]) for k in ks], color=col, linewidth=2)
        ax2.fill_between(bx, [q(bins[k], .1) for k in ks], [q(bins[k], .9) for k in ks], color=col, alpha=0.12, linewidth=0)
        if ks:
            labels.append((ax2, bx[-1], st.median(bins[ks[-1]]), arm))
        ax1.axvline(last / 60, color=MUTED, linewidth=0.8, linestyle=":")
    ax1.set_ylabel("conversations", fontsize=8.5, color=MUTED)
    ax1.set_ylim(0, 136)
    ax2.set_yscale("log")
    ax2.set_ylabel("TTFT (s)", fontsize=8.5, color=MUTED)
    ax2.set_xlabel("minutes since the client started", fontsize=8.5, color=MUTED)
    ax2.set_xlim(0, xmax * 1.1)
    fig.canvas.draw()
    placed = {}
    for ax, x, y, text in sorted(labels, key=lambda l: -l[2]):   # top label first, later ones pushed down
        px, py = ax.transData.transform((x, y))
        for qx, qy in placed.get(ax, []):
            if abs(px - qx) < 90 and abs(py - qy) < 11:
                py = qy - 11
        placed.setdefault(ax, []).append((px, py))
        x2, y2 = ax.transData.inverted().transform((px + 5, py))
        ax.text(x2, y2, text, fontsize=8, color=INK, va="center", clip_on=False)
    fig.legend(handles=handles, labels=list(data), loc="upper center", ncol=len(data),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.99))
    path = os.path.join(a.out, "live_timeline.png")
    fig.savefig(path, dpi=130, facecolor=SURF, bbox_inches="tight")
    print("wrote", path, "and", os.path.join(a.out, "live_phases.csv"))


if __name__ == "__main__":
    main()
