"""results/compare_c34_designs/pricing_vs_jit.png / .csv with campaign 32's runs drawn beside campaign 34's:

  python3 c34_pricing.py [--arms LABEL ...] [--out DIR] [--check]

Campaign 34 reran three of campaign 32's arms (jit_p50, ec_full, jit_exact) on the engine with the four FA3 fixes;
original HiCache and the oracle (ec_exact) ran only in campaign 32, without them. Same users, trace, settings and GPU,
so requests pair by (conv, turn), but the engine differs, and the figure marks campaign 32's runs dashed. The box drew
each campaign's figure with a script git excludes; this one computes its four columns as the box did (equal to 4 decimals
on all five runs), and --check compares them with each campaign's CSV. This script's output replaced the box's
compare_c34_designs figure and CSV; the box's are at commit a7791bd20e. The footnote is over the returning requests
every drawn run sent at least 600 s before the earliest of the runs' last sends (the slowest request took 727 s). The
box called that set "sent before minute 110"; the rule gives its 14,767 on campaign 34's three runs, and 9,155 against
its 9,158 on campaign 32's six.
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

import c26_pricing as C
import jit_vs_hicache as J

OUT = "compare_c34_designs"
MARGIN_S = 600   # the footnote set ends this long before the earliest last send (the box's rule, inferred)
ARRIVING_USERS = 729   # both box footnotes: users that arrive in 2 h at 0.1 a second, seed 0
COLUMNS = ("ttft", "tps", "recomputed", "reading")
Spec = collections.namedtuple("Spec", "campaign label short foot color ls")
ARMS = {   # label: where its run is, legend label, end label, footnote name; colors as in the box's figures
    "hicache": Spec(32, "Original SGLang HiCache, campaign 32", "HiCache", "HiCache", "#2a78d6", C.DASHED),
    "jit_p50": Spec(34, "Just-in-time, p50 hints, campaign 34", "JIT, p50", "JIT p50", "#a0307a", "-"),
    "ec_full": Spec(34, "Pricing the room, all four rules, campaign 34", "All four rules", "All four rules", "#eb6834", "-"),
    "ec_exact": Spec(32, "Oracle: the four rules, exact hints, campaign 32", "Oracle", "Oracle", "#eb6834", C.DASHED),
    "jit_exact": Spec(34, "Just-in-time, exact hints, campaign 34", "JIT, exact", "JIT exact", "#1baf7a", "-"),
}
DEFAULT_ARMS = ("hicache", "jit_p50", "ec_full", "ec_exact")
PANELS = {"ttft": ("Mean time to first token (log scale)", "seconds"),
          "tps": ("Requests completed per second", "requests/s"),
          "recomputed": ("History recomputed per returning request", "tokens"),
          "reading": ("Returning requests that read from SSD", "share of requests")}
SETTING = ("Recorded waits at x1, one run per arm. 0.1 users arrive a second and at most 128 are in session; the others "
           "wait for a slot, a wait TTFT leaves out. Prompts are synthetic token ids of the recorded lengths and outputs "
           "have the recorded length; sessions with a request over 65,536 tokens are left out. Exact hints are the true "
           "next-request times; p50 hints are the median wait after the tool call the reply makes, from a table the client "
           "keeps online for each harness and model. The rules arm gets that table's p10, p50 and p90 and an "
           "end-of-session signal: it starts reads by expected cost at a learned price on host memory, evicts by chance "
           "of return, and lets an arrived request's read take room from early reads. The oracle is the rules arm given "
           "each true next-request time as its distribution. Attention runs on FlashAttention 3, and every run's engine "
           "has the eviction fix: a request queued for over 60 s is no longer taken for one not coming.")
ENGINE = ("Dashed runs are campaign 32's, solid ones campaign 34's: the same users, trace, settings and GPU, but campaign "
          "34's engine adds four fixes: pages the SSD already holds are not written again; queued writes are merged into "
          "transfers of up to 128 pages; timed reads wait while pending writes hold 50,000 host tokens; a read refused "
          "host memory is retried without spending the request's lookups. Without them every hinted arm of campaign 32 "
          "fell into a loop of SSD reads refused host memory (the oracle from minute 79), and on the same requests the "
          "fixes alone took 5.79 s off the rules arm's mean and 9.52 s off JIT p50's. An outside request also stopped "
          "the oracle's server three times in minutes 46 to 48.")


def run_of(label, campaign=None):
    """The run's folder; campaign overrides ARMS, for runs of a campaign's own figure (oracle_v2 is in none here)."""
    run = os.path.join(f"compare_c{campaign or ARMS[label].campaign}_designs", "runs", label)
    return J.Arm(label, label, None, label, run, f"client_{label}")   # only run and client are read


# ----------------------------------------------------------------------------- data
def load(label, campaign=None):
    arm = run_of(label, campaign)
    t_start, turns = J.client_turns(arm)
    reads, admitted, _ = J.server_events(arm)
    rows = [json.loads(line) for line in open(os.path.join(J.RES, arm.run, arm.client, "client.jsonl")) if line.strip()]
    sent = [r for r in rows if r.get("kind") == "turn" and (r["conv"], r["turn"]) in turns]
    plays = next(r["plays"] for r in rows if r.get("kind") == "summary")
    d = dict(run=arm.run, turns=turns, returning={k: t for k, t in turns.items() if not t["first"]}, reads=reads,
             admitted=admitted, t_start=t_start, latency={(r["conv"], r["turn"]): r.get("latency") for r in sent},
             reprefilled={(r["conv"], r["turn"]): r.get("reply_reprefilled") or 0 for r in sent}, plays=plays,
             read_tok={k: sum(tok for _, tok in v) for k, v in reads.items()})
    d["held_s"] = J.held_tokens(d)
    return d


def series(d):
    """c26_pricing.series, but history recomputed is the client's reply_reprefilled: these program replays log no
    delta_tokens, so the c26 column would be 0 (the box's CSVs equal this one exactly)."""
    rows = C.series(d)
    rec = collections.defaultdict(list)
    for k in d["returning"]:
        rec[C.bin_of(d["turns"][k]["t0"])].append(d["reprefilled"][k])
    for m, r in rows.items():
        r["recomputed"] = st.mean(rec[m]) if rec[m] else float("nan")
    return rows


def cut_s(arms, data):
    return min(max(t["t0"] for t in data[a]["turns"].values()) for a in arms) - MARGIN_S


def shared_requests(arms, data):
    cut = cut_s(arms, data)
    return set.intersection(*({k for k, t in data[a]["returning"].items() if t["t0"] < cut} for a in arms))


def write_csv(arms, data, rows, out):
    with open(os.path.join(out, "pricing_vs_jit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("arm", "label", "minute") + COLUMNS + ("campaign", "run"))   # the box's columns first
        for a in arms:
            for m, r in rows[a].items():
                w.writerow((a, ARMS[a].label, m) + tuple(round(r[c], 4) for c in COLUMNS) + (ARMS[a].campaign, data[a]["run"]))


def check(arms, rows):
    """Largest difference from the box's own CSV of the run's campaign, per arm and column."""
    for a in arms:
        path = os.path.join(J.RES, f"compare_c{ARMS[a].campaign}_designs", "pricing_vs_jit.csv")
        box = {int(r["minute"]): r for r in csv.DictReader(open(path)) if r["arm"] == a}
        worst = {c: max(abs(rows[a][m][c] - float(box[m][c])) for m in box
                        if box[m][c] not in ("", "nan") and rows[a][m][c] == rows[a][m][c]) for c in COLUMNS}
        print(f"check {a} against {os.path.relpath(path, J.RES)}: " + ", ".join(f"{c} {v:.4g}" for c, v in worst.items()))


def windows(arms, data, shared):
    """Mean TTFT of the shared requests by send-time window, printed for the record."""
    for lo, hi in ((0, 1800), (1800, 3600), (3600, cut_s(arms, data))):
        keys = [k for k in shared if lo <= data[arms[0]]["returning"][k]["t0"] < hi]
        means = ", ".join(f"{a} {st.mean(data[a]['returning'][k]['ttft'] for k in keys):.2f}" for a in arms)
        print(f"minutes {lo / 60:.0f}-{hi / 60:.1f}, {len(keys):,} shared returning requests: {means} s")


# ----------------------------------------------------------------------------- figure
def footnote_lines(arms, data, shared):
    ttft = [st.mean(data[a]["returning"][k]["ttft"] for k in shared) for a in arms]
    count = {2: "two", 3: "three", 4: "four", 5: "five"}.get(len(arms), str(len(arms)))
    names = ", ".join(ARMS[a].foot.replace(" ", " ") for a in arms)   # a name never breaks across lines
    first = (f"Over the {len(shared):,} returning requests all {count} runs sent before minute {cut_s(arms, data) / 60:.1f} "
             f"(10 minutes before the earliest run's last request; {names}): mean TTFT "
             f"{C.listing(ttft, lambda v: f'{v:.2f}')} s. Requests served in 2 h: "
             f"{C.listing([len(data[a]['turns']) for a in arms], lambda v: f'{v:,}')}; users admitted: "
             f"{C.listing([data[a]['plays'] for a in arms], str)} of the {ARRIVING_USERS} that arrive.")
    return [line for p in (first, ENGINE, SETTING) for line in textwrap.wrap(p, width=132)]


def draw_panel(ax, key, arms, rows, minutes, x):
    J.style(ax, *PANELS[key])
    for a in arms:
        ax.plot(x, [rows[a][m][key] for m in minutes], color=ARMS[a].color, lw=1.8, label=ARMS[a].label, ls=ARMS[a].ls,
                solid_capstyle="round")
    if key == "ttft":
        ax.set_yscale("log")
        top = max(rows[a][m]["ttft"] for a in arms for m in minutes)
        ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100) if t <= top]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.yaxis.set_minor_locator(NullLocator())
        return
    ax.set_ylim(bottom=0)
    if key == "recomputed":
        J.kilo(ax)
    if key == "reading":
        J.percent(ax.yaxis)


def plot(arms, data, rows, shared, out):
    ncol = 2
    legend_rows = -(-len(arms) // ncol)
    lines = footnote_lines(arms, data, shared)
    head = C.HEAD_IN - C.LEGEND_ROW_IN * (3 - legend_rows)
    foot = C.FOOT_IN + C.FOOT_LINE_IN * (len(lines) - 6)
    height = head + C.PANELS_IN + foot
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.6, height), sharex=True, facecolor=J.SURFACE)
    minutes = sorted(rows[arms[0]])
    x = [m + J.BIN_S / 120 for m in minutes]   # bin centres in minutes
    for ax, key in zip(axes, PANELS):
        draw_panel(ax, key, arms, rows, minutes, x)
    axes[-1].set_xlabel("minutes since the client started (send time, 5-minute bins)")
    axes[-1].set_xlim(0, 120)
    fig.subplots_adjust(top=1 - head / height, bottom=foot / height, left=0.085, right=0.86, hspace=0.42)
    axes[0].set_ylim(axes[0].get_ylim())
    C.end_labels(fig, axes[0], x[-1], [(ARMS[a], rows[a][minutes[-1]]["ttft"]) for a in arms])
    handles = dict(zip(*reversed(axes[0].get_legend_handles_labels())))
    order = [arms[r * ncol + c] for c in range(ncol) for r in range(legend_rows) if r * ncol + c < len(arms)]
    fig.legend([handles[ARMS[a].label] for a in order], [ARMS[a].label for a in order], loc="upper left",   # rows read in arm order
               bbox_to_anchor=(0.085, 1 - C.LEGEND_Y_IN / height), ncol=ncol, frameon=False, fontsize=9.5,
               handlelength=2.4, columnspacing=1.6, labelcolor=J.INK)
    fig.text(0.085, 1 - C.TITLE_Y_IN / height, "FreeInference, 128 users, 160 GB: from original HiCache to pricing the room",
             fontsize=12, color=J.INK, va="top", weight="semibold")
    for i, line in enumerate(reversed(lines)):   # one text per line: multi-line spacing differs across Matplotlib versions
        fig.text(0.085, (C.FOOT_BASE_IN + i * C.FOOT_LINE_IN) / height, line, fontsize=8, color=J.INK2, va="baseline")
    path = os.path.join(out, "pricing_vs_jit.png")
    fig.savefig(path, dpi=180, facecolor=J.SURFACE)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(DEFAULT_ARMS), help="runs to draw, in order")
    ap.add_argument("--out", default=os.path.join(J.RES, OUT), help="folder for the figure and its CSV")
    ap.add_argument("--check", action="store_true", help="compare each run's columns with its campaign's box CSV")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": J.AXIS, "xtick.color": J.INK2, "ytick.color": J.INK2,
                         "axes.labelcolor": J.INK2})
    data = {a: load(a) for a in args.arms}
    rows = {a: series(data[a]) for a in args.arms}
    shared = shared_requests(args.arms, data)
    if args.check:
        check(args.arms, rows)
    windows(args.arms, data, shared)
    write_csv(args.arms, data, rows, args.out)
    plot(args.arms, data, rows, shared, args.out)


if __name__ == "__main__":
    main()
