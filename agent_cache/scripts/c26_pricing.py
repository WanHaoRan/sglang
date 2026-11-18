"""pricing_vs_jit.png / .pdf / .csv of two campaign 26 figure sets, redrawn here from the runs:

  python3 c26_pricing.py --set {ec,h100} [--arms LABEL ...] [--out DIR]

  ec    results/compare_c26_ec/: 44 users, 70 GB; original HiCache, JIT p50 and exact, the no-hint arm, pricing the room
  h100  results/compare_c26_designs_h100/: 44 users, a 100 GB host pool against 70 GB; drawn without the no-hint runs
        unless --arms names them

The box drew both with frozen/oracle_scripts_staging/c26_ec/plot_pricing_vs_jit.py, which git excludes. This script takes
the runs from each folder's manifest.txt and computes every column of the box's CSVs the same way (byte-identical on the
box's runs), with jit_vs_hicache.py's helpers. Per 5-minute bin by send time: the mean TTFT of all turns, turns completed
per second (binned by send time plus latency), history recomputed per returning turn (with the turn's own delta_tokens),
SSD-read history not used per returning turn, SSD-read KV held in host memory until admission, the share of returning
turns with an SSD read, and the SSD read rate by landing time. The footnote is over the returning turns every drawn run
sent at least 300 s before its last send (about minute 115), the box's turn set. Its no-hint sentence describes the arm
in full; the box's names only its read timing and eviction.
"""
import argparse
import collections
import csv
import json
import os
import statistics as st
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import jit_vs_hicache as J

Spec = collections.namedtuple("Spec", "label short foot color ls")   # legend label, end label, footnote name
DASHED, DOTTED = (0, (4, 2.2)), (0, (1, 2))
SHARED_MARGIN_S = 300
HOST_ROOM_70GB = 44_000   # jit_vs_hicache's c44 host_room: tokens free beyond the GPU's copy
DESIGNS = {   # key: (legend label, short label, color), as in the box's figures
    "nofc_noend": ("No hints, fetch at arrival", "No hints", "#4a3aa7"),
    "jit_p50": ("Just-in-time, p50 hints", "JIT p50", "#a0307a"),
    "jit_exact": ("Just-in-time, exact hints", "JIT exact", "#1baf7a"),
}
NO_HINT_FOOT = ("The no-hint arm uses the hints only to tell which session a turn belongs to. It reads nothing before a turn "
                "arrives, times each read by the turn's place in the queue (at once near the head) so it lands shortly before "
                "admission, pins landed reads in host memory until admission (up to 30 % of the pool), re-reads up to twice "
                "whatever part of a queued turn's history is missing (its read cut short or refused for lack of host room, or "
                "the history evicted), skips the SSD lookup when the session's history is already in GPU or host memory, and "
                "evicts least recently used history first, queued turns' history last.")
EC_CONTRAST = ("The just-in-time and priced arms also time, pin, re-read and skip lookups this way, but read before a turn "
               "arrives and rank eviction by the hints; HiCache reads at arrival in arrival order, with no pins or lookup "
               "skip, retries a missed read up to 8 times, and evicts by plain LRU.")
PANELS = {"ttft": J.PANELS["ttft"], "tps": J.PANELS["tps"], "recomputed": J.PANELS["recomputed"],
          "held": ("Host memory needed to hold SSD-loaded KV until admission", "tokens"),
          "reading": ("Returning turns that read from SSD", "share of turns")}
# the box figures' geometry in inches, measured from its h100 and ec PNGs: the header with three legend rows, the four
# panels, the space under them for a six-line footnote; the title, legend and footnote positions from the figure's edges
HEAD_IN, PANELS_IN, FOOT_IN, LEGEND_ROW_IN, FOOT_LINE_IN = 1.508, 7.439, 1.642, 0.217, 0.161
TITLE_Y_IN, LEGEND_Y_IN, FOOT_BASE_IN = 0.139, 0.411, 0.187   # FOOT_BASE_IN: the last footnote line's baseline


def pool_arm(design, gb):
    label, short, color = DESIGNS[design]
    return Spec(f"{label}, {gb} GB", f"{short}, {gb} GB", f"{short} {gb} GB", color, DASHED if gb == 70 else "-")


def no_hint(arms, contrast=""):
    return " " + NO_HINT_FOOT + contrast if any(a.startswith("nofc_noend") for a in arms) else ""


def listing(values, fmt):
    v = [fmt(x) for x in values]
    return v[0] if len(v) == 1 else f"{', '.join(v[:-1])} and {v[-1]}"


def ec_notes(arms, data):
    late = {a: data[a]["held_s"][45 * 60:J.END_S] for a in arms}
    held = listing([st.mean(late[a]) for a in arms], lambda v: f"{v / 1000:.0f}K")
    over = listing([sum(v > HOST_ROOM_70GB for v in late[a]) / len(late[a]) for a in arms], lambda v: f"{v * 100:.0f}")
    return [f"Host memory needed counts every SSD read from landing to its turn's admission, whether or not the engine kept "
            f"the data. Minutes 45-120: mean {held} tokens, above the {HOST_ROOM_70GB // 1000}K for {over} % of the seconds.",
            "x10 gaps, one run per line. HiCache and JIT exact are campaign 21's and 23's runs, the others campaign 26's, on "
            "other GPU units of the same measured speed. " + J.FOOT_HINTS + " Pricing the room here is its rules 1 and 3 at a "
            "fixed price, given that table's p10, p50 and p90 and an end-of-session signal." + no_hint(arms, " " + EC_CONTRAST)]


def h100_notes(arms, data):
    return ["x10 gaps, one run per line. The host pool holds 712K tokens at 70 GB and 1,017K at 100 GB; 668K of either copy "
            "what the GPU holds.", J.FOOT_HINTS + no_hint(arms)]


SETS = {
    "ec": dict(
        out="compare_c26_ec", title="44 users, 70 GB of host memory: pricing the room against just-in-time fetching",
        arms={"baseline": Spec("Original SGLang HiCache", "HiCache", "HiCache", "#2a78d6", "-"),
              "jit_p50_ctl": Spec("Just-in-time, p50 hints", "JIT, p50", "JIT p50", "#a0307a", "-"),
              "nofc_noend": Spec("No hints, fetch at arrival", "No hints", "No hints", "#4a3aa7", "-"),
              "ec_r13": Spec("Pricing the room, rules 1 and 3", "Priced", "Priced", "#eb6834", "-"),
              "ec_r13_rep": Spec("Pricing the room, repeat", "Priced, repeat", "Priced repeat", "#eb6834", DOTTED),
              "jit_exact": Spec("Just-in-time, exact hints", "JIT, exact", "JIT exact", "#1baf7a", "-")},
        drop=(), panels=("ttft", "tps", "recomputed", "held"), legend_cols=3,
        columns=("ttft", "tps", "recomputed", "unused", "held", "reading"), notes=ec_notes),
    "h100": dict(
        out="compare_c26_designs_h100", title="44 users: a 100 GB host pool against 70 GB",
        arms={"nofc_noend_rep": pool_arm("nofc_noend", 70), "nofc_noend_h100": pool_arm("nofc_noend", 100),
              "jit_p50_d": pool_arm("jit_p50", 70), "jit_p50_h100": pool_arm("jit_p50", 100),
              "jit_exact_d": pool_arm("jit_exact", 70), "jit_exact_h100": pool_arm("jit_exact", 100)},
        drop=("nofc_noend_rep", "nofc_noend_h100"), panels=("ttft", "tps", "recomputed", "reading"), legend_cols=2,
        columns=("ttft", "tps", "recomputed", "unused", "held", "reading", "ssd"), notes=h100_notes),
}


# ----------------------------------------------------------------------------- data
def load(cfg, arms):
    folder = os.path.join(J.RES, cfg["out"])
    runs = {l.split()[0]: l.split()[1:] for l in open(os.path.join(folder, "manifest.txt")) if l.strip()}
    data = {}
    for name in arms:
        run, client = runs[name]
        spec = cfg["arms"][name]
        arm = J.Arm(spec.label, spec.short, spec.color, name, run, client)
        t_start, turns = J.client_turns(arm)
        reads, admitted, _ = J.server_events(arm)
        rows = map(json.loads, open(os.path.join(J.RES, run, client, "client.jsonl")))
        latency = {(r["conv"], r["turn"]): r.get("latency") for r in rows
                   if r.get("kind") == "turn" and (r["conv"], r["turn"]) in turns}
        d = dict(run=run, turns=turns, returning={k: t for k, t in turns.items() if not t["first"]}, reads=reads,
                 admitted=admitted, t_start=t_start, latency=latency, last_send=max(t["t0"] for t in turns.values()),
                 read_tok={k: sum(t for _, t in v) for k, v in reads.items()})
        d["held_s"] = J.held_tokens(d)
        data[name] = d
    return data


def bin_of(s):
    return int(s // J.BIN_S) * J.BIN_S // 60


def series(d):
    """The CSV's columns per 5-minute bin, keyed by the bin's first minute."""
    minutes = range(0, J.END_S // 60, J.BIN_S // 60)
    ttft, rec = collections.defaultdict(list), collections.defaultdict(list)
    done, n, reading, read = (collections.Counter() for _ in range(4))
    for k, t in d["turns"].items():
        b = bin_of(t["t0"])
        ttft[b].append(t["ttft"])
        if d["latency"][k] is not None:
            done[bin_of(t["t0"] + d["latency"][k])] += 1
        if not t["first"]:
            n[b] += 1
            rec[b].append(J.recomputed(t, t["new"]) if t["new"] is not None else 0)
            reading[b] += d["read_tok"].get(k, 0) > 0
    for v in d["reads"].values():
        for t, tok in v:
            if 0 <= t - d["t_start"] < J.END_S:
                read[bin_of(t - d["t_start"])] += tok
    held, unused = d["held_s"], J.per_bin(d, {}, minutes)["unused"]
    mean = lambda v: st.mean(v) if v else float("nan")   # as the box: a whole mean of ints stays an int in the CSV
    return {m: dict(ttft=mean(ttft[m]), tps=done[m] / J.BIN_S, recomputed=mean(rec[m]), unused=u,
                    held=sum(held[m * 60:m * 60 + J.BIN_S]) / J.BIN_S,
                    reading=reading[m] / n[m] if n[m] else float("nan"), ssd=read[m] * J.BYTES_PER_TOKEN / J.BIN_S / 1e9)
            for m, u in zip(minutes, unused)}


def write_csv(cfg, arms, data, rows, out):
    with open(os.path.join(out, "pricing_vs_jit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("arm", "label", "run", "minute") + cfg["columns"])
        for a in arms:
            for m, r in rows[a].items():
                w.writerow((a, cfg["arms"][a].label, data[a]["run"], m) + tuple(round(r[c], 4) for c in cfg["columns"]))


# ----------------------------------------------------------------------------- figure
def shared_turns(arms, data):
    keep = lambda d: {k for k, t in d["returning"].items() if t["t0"] <= d["last_send"] - SHARED_MARGIN_S}
    return set.intersection(*(keep(data[a]) for a in arms))


def footnote_lines(cfg, arms, data):
    shared = shared_turns(arms, data)
    ttft = [st.mean(data[a]["returning"][k]["ttft"] for k in shared) for a in arms]
    used = [sum(data[a]["returning"][k]["storage"] for k in shared) / sum(data[a]["read_tok"].get(k, 0) for k in shared)
            for a in arms]
    names = ", ".join(cfg["arms"][a].foot.replace(" ", " ") for a in arms)   # a name never breaks across lines
    count = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(len(arms), str(len(arms)))
    first = (f"Over the {len(shared):,} returning turns all {count} runs sent before about minute 115 ({names}): mean TTFT "
             f"{listing(ttft, lambda v: f'{v:.2f}')} s; share of SSD reads used {listing(used, lambda v: f'{v * 100:.0f}')} "
             f"%. Turns served in 2 h: {listing([len(data[a]['turns']) for a in arms], lambda v: f'{v:,}')}.")
    return [line for p in [first] + cfg["notes"](arms, data) for line in textwrap.wrap(p, width=132)]


def end_labels(fig, ax, x, ends, gap=0.108):
    """Labels right of the line ends, at least `gap` axes heights apart: crowded labels are stacked around their mean
    height, and a leader joins a moved label to its dot."""
    to_frac = lambda y: (ax.transScale + ax.transLimits).transform((x, y))[1]
    groups = [[(to_frac(y), y, spec)] for spec, y in sorted(ends, key=lambda e: e[1])]
    span = lambda g: (st.mean(f for f, _, _ in g) - (len(g) - 1) * gap / 2, st.mean(f for f, _, _ in g) + (len(g) - 1) * gap / 2)
    i = 0
    while i < len(groups) - 1:
        if span(groups[i + 1])[0] - span(groups[i])[1] < gap:
            groups[i:i + 2] = [groups[i] + groups[i + 1]]
            i = max(0, i - 1)
        else:
            i += 1
    height_pt = ax.get_position().height * fig.get_figheight() * 72
    for g in groups:
        for j, (f, y, spec) in enumerate(g):
            ax.plot([x], [y], "o", ms=5, color=spec.color, mec=J.SURFACE, mew=1.2, clip_on=False, zorder=3)
            dy = (span(g)[0] + j * gap - f) * height_pt
            leader = dict(arrowstyle="-", color=J.AXIS, lw=0.8, shrinkA=1, shrinkB=3.5, relpos=(0, 0.5))
            ax.annotate(spec.short, (x, y), xytext=(8, dy), textcoords="offset points", va="center", fontsize=9,
                        color=J.INK2, annotation_clip=False, arrowprops=leader if abs(dy) > 2 else None)


def plot(cfg, arms, data, rows, out):
    specs, ncol = cfg["arms"], cfg["legend_cols"]
    legend_rows = -(-len(arms) // ncol)
    lines = footnote_lines(cfg, arms, data)
    head = HEAD_IN - LEGEND_ROW_IN * (3 - legend_rows)
    foot = FOOT_IN + FOOT_LINE_IN * (len(lines) - 6)
    height = head + PANELS_IN + foot
    fig, axes = plt.subplots(len(cfg["panels"]), 1, figsize=(8.6, height), sharex=True, facecolor=J.SURFACE)
    minutes = sorted(rows[arms[0]])
    x = [m + J.BIN_S / 120 for m in minutes]   # bin centres in minutes
    for ax, key in zip(axes, cfg["panels"]):
        J.style(ax, *PANELS[key])
        for a in arms:
            ax.plot(x, [rows[a][m][key] for m in minutes], color=specs[a].color, lw=1.8, label=specs[a].label,
                    ls=specs[a].ls, solid_capstyle="round")
        if key == "ttft":
            ax.set_yscale("log")
            top = max(rows[a][m]["ttft"] for a in arms for m in minutes)
            ticks = [t for t in (0.3, 1, 3, 10, 30, 100) if t <= top]
            ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(bottom=0)
        if key in ("recomputed", "held"):
            J.kilo(ax)
        if key == "reading":
            J.percent(ax.yaxis)
        if key == "held":
            room = ax.axhline(HOST_ROOM_70GB, color=J.INK, lw=0.9)
            ax.legend([room], [f"host memory free beyond the GPU's copy: {HOST_ROOM_70GB // 1000}K tokens"],
                      loc="upper left", frameon=False, fontsize=8.5, handlelength=1.8, labelcolor=J.INK)
    axes[-1].set_xlabel("minutes since the client started (send time, 5-minute bins)")
    axes[-1].set_xlim(0, 120)
    fig.subplots_adjust(top=1 - head / height, bottom=foot / height, left=0.085, right=0.86, hspace=0.42)
    axes[0].set_ylim(axes[0].get_ylim())
    end_labels(fig, axes[0], x[-1], [(specs[a], rows[a][minutes[-1]]["ttft"]) for a in arms])
    handles = dict(zip(*reversed(axes[0].get_legend_handles_labels())))
    order = [arms[r * ncol + c] for c in range(ncol) for r in range(legend_rows) if r * ncol + c < len(arms)]
    fig.legend([handles[specs[a].label] for a in order], [specs[a].label for a in order], loc="upper left",   # rows read in arm order
               bbox_to_anchor=(0.085, 1 - LEGEND_Y_IN / height), ncol=ncol, frameon=False, fontsize=9.5,
               handlelength=1.8, columnspacing=1.6, labelcolor=J.INK)
    fig.text(0.085, 1 - TITLE_Y_IN / height, cfg["title"], fontsize=12, color=J.INK, va="top", weight="semibold")
    for i, line in enumerate(reversed(lines)):   # one text per line: multi-line spacing differs across Matplotlib versions
        fig.text(0.085, (FOOT_BASE_IN + i * FOOT_LINE_IN) / height, line, fontsize=8, color=J.INK2, va="baseline")
    J.save(fig, out, "pricing_vs_jit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--arms", nargs="+", help="manifest labels to draw (default: the set's arms, less its dropped ones)")
    ap.add_argument("--out", help="folder for the figure and its CSV (default: the set's folder)")
    args = ap.parse_args()
    cfg = SETS[args.set]
    unknown = set(args.arms or ()) - set(cfg["arms"])
    if unknown:
        ap.error(f"not in set {args.set}: {', '.join(sorted(unknown))}")
    arms = [a for a in cfg["arms"] if a in args.arms] if args.arms else [a for a in cfg["arms"] if a not in cfg["drop"]]
    out = args.out or os.path.join(J.RES, cfg["out"])
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": J.AXIS, "xtick.color": J.INK2, "ytick.color": J.INK2,
                         "axes.labelcolor": J.INK2})
    data = load(cfg, arms)
    rows = {a: series(data[a]) for a in arms}
    write_csv(cfg, arms, data, rows, out)
    plot(cfg, arms, data, rows, out)


if __name__ == "__main__":
    main()
