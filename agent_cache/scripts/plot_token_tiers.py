"""Chat-turn token sizes vs. the three tiers' token capacity, from one replay run's client.jsonl (RUNBOOK 4.6) and
its server.log (RUNBOOK 5.2 boot facts).

  python3 plot_token_tiers.py --client ../results/<boot>/client_<...>/client.jsonl [--server-log <boot>/server.log] [--out <png>]

Panel A: what one turn sends, by turn index, as the MEAN over the replayed conversations (means add up, so the stack height
         is the mean turn size): the cross-session shared prefix (the mode of turn-0 cached_tokens), the session's own
         history re-sent from earlier turns, and the new input of this turn (tool results + user message = delta_tokens).
         The reply (completion_tokens, ~90 tokens) is too small to draw and is given as text. This is content anatomy, not
         a cache outcome: in this run most of the re-sent history was re-prefilled. The dot is the p90 of prompt + reply.
Panel B: this boot's L1 (--max-total-tokens) and L2 (--hicache-size) pools next to the workload's demand at several
         granularities. Under write_through the host pool mirrors everything resident on device, so its usable overflow
         is L2 - L1. The running batch's device footprint is the server's own '#token' figure from its Decode batch lines
         (non-evictable device tokens), not a model. L3 is a stat tile: bytes x cleaner watermark / bytes per token.
Panel C: over the run, the context of the live sessions (a reply counts once generated), the context requested by the
         in-flight turns (queued + running; a queued request pins nothing), and the device tokens the running batch pins
         (server '#token'), against the pool sizes.
Colors: demand in the dataviz reference palette's blue ordinal ramp (older -> newer, smaller -> larger), pool capacity in
        the palette's muted ink (emphasis form); statistics (p90) in primary ink.
"""
import argparse
import collections
import datetime as dt
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAPACITY = MUTED                                                     # documented muted ink, 3.5:1 on the surface
BLUE = {250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6", 600: "#184f95"}   # validated ordinal ramp (palette.md)
FIG_W, FIG_H, DPI = 13.5, 11.5, 170
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] Decode batch, #running-req: (\d+), #token: (\d+)")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q / 100 * len(v)))] if v else 0


def mean(v):
    return sum(v) / len(v) if v else 0.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="client.jsonl of one replay run")
    ap.add_argument("--server-log", default="", help="server.log of the boot (default: <client dir>/../server.log)")
    ap.add_argument("--l3-bytes", type=int, default=395_188_764_672, help="L3 filesystem size in bytes (RUNBOOK 1: 368.0 GiB)")
    ap.add_argument("--l3-watermark", type=float, default=0.80, help="nixl cleaner high watermark (fraction)")
    ap.add_argument("--bytes-per-token", type=int, default=131_072, help="KV bytes per token (Qwen3-32B fp8_e5m2)")
    ap.add_argument("--out", default="", help="PNG path (default: token_sizes_vs_tiers.png next to the client.jsonl)")
    return ap.parse_args()


def load_run(args):
    rows = [json.loads(l) for l in open(args.client)]
    run = next(r for r in rows if r["kind"] == "run")
    turns = [r for r in rows if r["kind"] == "turn" and "t_send" in r]
    by_conv = collections.defaultdict(dict)
    for r in turns:
        by_conv[r["conv"]][r["turn"]] = r
    log = args.server_log or os.path.join(os.path.dirname(os.path.abspath(args.client)), "..", "server.log")
    text = open(log, errors="replace").read()
    l1 = int(re.search(r"max_total_num_tokens=(\d+)", text).group(1))
    m = re.search(r"host pool: (\d+) tokens", text)
    l2 = int(m.group(1)) if m else 0
    t_end = max(r["t_send"] + r["latency"] for r in turns)
    pinned = []   # (seconds since the client started, running requests, device tokens pinned by the running batch)
    for line in text.splitlines():
        m = TS_RE.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() - run["t_start"]
        if 0 <= t <= t_end:
            pinned.append((t, int(m.group(2)), int(m.group(3))))
    return run, turns, by_conv, l1, l2, pinned


def turn_anatomy(by_conv):
    """Per turn index: means of (shared, carried, new) and of the reply, p90 of prompt+reply, n; shared = mode of turn-0 cached."""
    t0 = [c[0]["cached_tokens"] for c in by_conv.values() if 0 in c and c[0]["cached_tokens"] > 0]
    shared = collections.Counter(t0).most_common(1)[0][0] if t0 else 0
    parts = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in by_conv.values():
        for k in sorted(c):
            r = c[k]
            prompt, reply = r["prompt_tokens"], r["completion_tokens"]
            if k == 0 or (k - 1) not in c:
                new = max(0, prompt - shared)
            else:
                new = max(0, prompt - c[k - 1]["prompt_tokens"] - c[k - 1]["completion_tokens"])   # = delta_tokens
            carried = max(0, prompt - shared - new)
            parts[k]["shared"].append(min(shared, prompt)); parts[k]["carried"].append(carried)
            parts[k]["new"].append(new); parts[k]["reply"].append(reply); parts[k]["total"].append(prompt + reply)
    ks = sorted(parts)
    avg = {name: [mean(parts[k][name]) for k in ks] for name in ("shared", "carried", "new", "reply")}
    p90 = [pct(parts[k]["total"], 90) for k in ks]
    n = [len(parts[k]["total"]) for k in ks]
    return shared, ks, avg, p90, n


def demand_rows(by_conv, turns, shared, pinned):
    ret = [r for r in turns if r["turn"] > 0]
    turn_p50 = pct([r["prompt_tokens"] + r["completion_tokens"] for r in ret], 50)
    ctx_mean = mean([r["prompt_tokens"] + r["completion_tokens"] for r in turns])
    last = [c[max(c)] for c in by_conv.values()]
    n_conv = len(by_conv)
    pinned_p50 = pct([p for _, _, p in pinned], 50)
    running_p50 = pct([r for _, r, _ in pinned], 50)
    return [
        ("one returning turn, p50 (prompt + reply)", turn_p50),
        (f"running batch pins (server p50, {running_p50} running)", pinned_p50),
        (f"{n_conv} sessions: own history, run mean", n_conv * max(0.0, ctx_mean - shared)),
        (f"{n_conv} sessions: own history, at end", sum(max(0, r["prompt_tokens"] + r["completion_tokens"] - shared) for r in last)),
        (f"{n_conv} sessions: full context, at end", sum(r["prompt_tokens"] + r["completion_tokens"] for r in last)),
    ]


def working_set(by_conv, turns, step=5.0):
    """Every `step` s: context of live sessions (reply counted once generated) and context requested by in-flight turns."""
    t_end = max(r["t_send"] + r["latency"] for r in turns)
    starts = {c: min(r["t_send"] for r in v.values()) for c, v in by_conv.items()}
    ends = {c: max(r["t_send"] + r["latency"] for r in v.values()) for c, v in by_conv.items()}
    ordered = {c: sorted(v.values(), key=lambda r: r["t_send"]) for c, v in by_conv.items()}
    ts, live, requested = [], [], []
    t = 0.0
    while t <= t_end:
        lv = req = 0
        for c, rs in ordered.items():
            if not (starts[c] <= t <= ends[c]):
                continue
            cur = None
            for r in rs:
                if r["t_send"] <= t:
                    cur = r
                    if t < r["t_send"] + r["latency"]:
                        req += r["prompt_tokens"] + r["completion_tokens"]
            if cur is not None:
                done = t >= cur["t_send"] + cur["latency"]
                lv += cur["prompt_tokens"] + (cur["completion_tokens"] if done else 0)
        ts.append(t / 60); live.append(lv); requested.append(req)
        t += step
    return ts, live, requested


def style(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.6)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def kfmt(x, _=None):
    return f"{x:,.0f}"


def compact(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def main():
    args = parse_args()
    run, turns, by_conv, l1, l2, pinned = load_run(args)
    shared, ks, avg, p90, n = turn_anatomy(by_conv)
    rows = demand_rows(by_conv, turns, shared, pinned)
    ts, live, requested = working_set(by_conv, turns)
    l3_tokens = int(args.l3_bytes * args.l3_watermark // args.bytes_per_token)
    overflow = max(0, l2 - l1)
    tag = run["args"].get("tag") or "replay"
    n_conv = len(by_conv)
    max_turns = max(len(c) for c in by_conv.values())

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
                         "text.color": INK, "axes.labelcolor": SECONDARY, "axes.titlecolor": INK})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], width_ratios=[1.0, 1.05], hspace=0.55, wspace=0.62,
                          left=0.06, right=0.90, top=0.885, bottom=0.07)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    fig.suptitle(f"Chat-turn token sizes vs. the tiers' token capacity  -  {tag}: {n_conv} sessions, {len(turns)} turns "
                 f"(up to {max_turns} each); L1 {l1:,} / L2 {l2:,} / L3 {compact(l3_tokens)} tokens",
                 fontsize=11.5, color=INK, x=0.06, ha="left", y=0.965)
    fig.text(0.06, 0.940, "Blue = tokens the workload demands (one hue, older to newer / smaller to larger).  "
             "Gray = tokens a pool can hold: L1 = --max-total-tokens, L2 = --hicache-size, L3 = NVMe below the cleaner watermark.",
             fontsize=9, color=SECONDARY)
    fig.text(0.06, 0.922, "Under write_through L2 mirrors everything resident on L1, so its host-only overflow is L2 - L1.  "
             "Source: client.jsonl (per turn) and server.log (boot facts, Decode batch '#token').", fontsize=9, color=SECONDARY)

    # ---- A: turn anatomy ---------------------------------------------------------------------------------------------
    style(ax_a, grid_axis="y")
    order = [("shared", BLUE[250], "shared prefix (identical across sessions; resident in L1 throughout this run)"),
             ("carried", BLUE[350], "own history re-sent from earlier turns (content; mostly re-prefilled in this run)"),
             ("new", BLUE[450], "new input this turn (tool result + user message)")]
    y_top = max(p90) * 1.12
    ax_a.set_ylim(0, y_top)
    gap_tok = 2.0 * y_top / (ax_a.get_position().height * FIG_H * DPI)   # a 2 px surface gap, in tokens
    bottom = [0.0] * len(ks)
    for name, color, _ in order:
        for i, k in enumerate(ks):
            h = avg[name][i]
            if h >= 3 * gap_tok:
                ax_a.bar(k, h - gap_tok, bottom=bottom[i] + gap_tok, width=0.42, color=color, linewidth=0)
            elif h > 0:
                ax_a.bar(k, h, bottom=bottom[i], width=0.42, color=color, linewidth=0)
        bottom = [b + v for b, v in zip(bottom, avg[name])]
    ax_a.scatter(ks, p90, s=46, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    for k in (ks[0], ks[len(ks) // 2], ks[-1]):
        i = ks.index(k)
        ax_a.annotate(f"{bottom[i]:,.0f}", (k, bottom[i]), xytext=(0, 5), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8, color=SECONDARY)
    ax_a.set_xticks(ks); ax_a.set_xlabel("turn index within a session", fontsize=9)
    ax_a.set_ylabel("tokens", fontsize=9); ax_a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
    ax_a.set_title(f"A. What one turn sends (mean per turn, n = {min(n)}-{max(n)} sessions)", fontsize=10, loc="left", pad=8)
    handles = [Patch(facecolor=c, label=l) for _, c, l in reversed(order)]
    handles.append(Patch(facecolor="none", edgecolor="none",
                         label=f"reply generated per turn: ~{mean(avg['reply']):.0f} tokens mean (max_tokens), too small to draw"))
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor=INK, markeredgecolor=SURFACE, markersize=7,
                          label="p90 of prompt + reply at that turn"))
    ax_a.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.14), ncol=1, fontsize=7.6, frameon=False,
                labelcolor=SECONDARY)

    # ---- B: pools vs demand ------------------------------------------------------------------------------------------
    style(ax_b, grid_axis="x")
    cap = [("L1 device pool", l1), ("L2 host pool, total", l2), ("L2 host-only overflow (L2 - L1)", overflow)]
    labels = [c[0] for c in cap] + [r[0] for r in rows]
    values = [c[1] for c in cap] + [r[1] for r in rows]
    colors = [CAPACITY] * len(cap) + [BLUE[450]] * len(rows)
    y = list(range(len(labels)))[::-1]
    ax_b.barh(y, values, height=0.42, color=colors, linewidth=0)
    for yi, v in zip(y, values):
        ax_b.annotate(f"{v:,.0f}", (v, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=SECONDARY)
    ax_b.set_yticks(y); ax_b.set_yticklabels(labels, fontsize=8, color=SECONDARY)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlim(0, max(values) * 1.16)
    ax_b.xaxis.set_visible(False); ax_b.grid(False); ax_b.spines["bottom"].set_visible(False)
    ax_b.set_title("B. This boot's pools next to the workload's demand (tokens)", fontsize=10, loc="left", pad=8)
    ax_b.axvline(shared, color=BLUE[250], linewidth=1.2, zorder=0)
    ax_b.annotate(f"shared prefix {shared:,} (resident in L1 throughout this run)", (shared, y[2] - 0.5), xytext=(4, 0),
                  textcoords="offset points", fontsize=7.5, color=SECONDARY, va="center")
    # stat tile for the tier that is off this scale
    ax_b.text(0.985, 0.985, f"L3 NVMe pool at {args.l3_watermark:.0%} cleaner watermark", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=8, color=SECONDARY)
    ax_b.text(0.985, 0.925, f"{compact(l3_tokens)} tokens", transform=ax_b.transAxes, ha="right", va="top",
              fontsize=14, fontweight="semibold", color=INK)
    ax_b.text(0.985, 0.835, f"= {args.l3_bytes / 2**30:.0f} GiB x {args.l3_watermark:.0%} / {args.bytes_per_token:,} B per token\n"
              f"{l3_tokens / rows[-1][1]:.1f} x the {n_conv} sessions' full contexts at end", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=7.5, color=MUTED, linespacing=1.4)
    ax_b.legend(handles=[Patch(facecolor=CAPACITY, label="pool capacity"), Patch(facecolor=BLUE[450], label="workload demand")],
                loc="upper right", bbox_to_anchor=(1.0, 0.58), fontsize=8, frameon=False, labelcolor=SECONDARY)

    # ---- C: working set over time -------------------------------------------------------------------------------------
    style(ax_c)
    ax_c.plot(ts, live, color=BLUE[600], linewidth=2, solid_joinstyle="round", solid_capstyle="round",
              label="context of live sessions (started, not finished; a reply counts once generated): the working set")
    ax_c.plot(ts, requested, color=BLUE[450], linewidth=2, solid_joinstyle="round", solid_capstyle="round",
              label="context requested by in-flight turns (queued + running; a queued request pins nothing)")
    if pinned:
        ax_c.plot([t / 60 for t, _, _ in pinned], [p for _, _, p in pinned], color=BLUE[250], linewidth=2,
                  solid_joinstyle="round", solid_capstyle="round",
                  label="device tokens pinned by the running batch (server '#token')")
    for v, name in ((l1, f"L1 device pool {l1:,}"), (l2, f"L2 host pool {l2:,}"), (overflow, f"L2 overflow {overflow:,}")):
        ax_c.axhline(v, color=CAPACITY, linewidth=1.2, zorder=1)
        ax_c.annotate(name, (ts[-1], v), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                      fontsize=7.8, color=SECONDARY, annotation_clip=False)
    i_max = max(range(len(live)), key=lambda i: live[i])
    ax_c.annotate(f"peak {live[i_max]:,}", (ts[i_max], live[i_max]), xytext=(6, 4), textcoords="offset points",
                  fontsize=8, color=SECONDARY)
    ax_c.set_xlim(0, ts[-1]); ax_c.set_ylim(0, max(live) * 1.12)
    ax_c.annotate(f"L3 NVMe pool {l3_tokens:,} tokens = {l3_tokens / ax_c.get_ylim()[1]:.1f} x this axis (off scale)",
                  (ts[-1], ax_c.get_ylim()[1]), xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                  fontsize=7.8, color=SECONDARY)
    ax_c.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
    ax_c.set_xlabel("minutes since the client started", fontsize=9); ax_c.set_ylabel("tokens", fontsize=9)
    ax_c.set_title("C. Working set over the run vs. the pools", fontsize=10, loc="left", pad=8)
    ax_c.legend(handles=ax_c.get_legend_handles_labels()[0] + [Line2D([], [], color=CAPACITY, linewidth=1.2, label="pool capacity")],
                loc="upper left", fontsize=8, frameon=False, labelcolor=SECONDARY)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.client)), "token_sizes_vs_tiers.png")
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"shared prefix {shared:,} tokens; per-turn means shared/carried/new/reply (n): "
          + ", ".join(f"t{k}: {a:.0f}/{b:.0f}/{c:.0f}/{d:.0f} ({m})" for k, a, b, c, d, m in
                      zip(ks, avg["shared"], avg["carried"], avg["new"], avg["reply"], n)))
    for name, v in cap + rows:
        print(f"  {v:>9,.0f}  {name}")
    print(f"  {l3_tokens:>9,}  L3 tokens at watermark; working-set peak {max(live):,} at {ts[i_max]:.1f} min; "
          f"requested peak {max(requested):,}; pinned p50 {pct([p for _, _, p in pinned], 50):,} max {max(p for _, _, p in pinned):,} "
          f"over {len(pinned)} decode lines")


if __name__ == "__main__":
    main()
