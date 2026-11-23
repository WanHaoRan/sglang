"""results/compare_c34_designs/'s other figures with campaign 32's runs beside campaign 34's, as c34_pricing.py draws
pricing_vs_jit.png:

  python3 c34_figures.py [--arms LABEL ...] [--out DIR] [--check-split]

  compare.png / .csv, ttft_over_time_60s.png / .csv   timeline.py's (tier rule "prompt", which gives the box's
                                                     compare.csv exactly); campaign 32's runs dashed. The runs/ copies
                                                     log 0 waiting users, so the waiting panel counts users arrived and
                                                     not started from each run's own client log (the box's curve)
  ec_wait_split.png / .csv                           where a returning request's TTFT goes, over c34_pricing's set:
                                                     the returning requests every drawn run sent at least 600 s before
                                                     the earliest last send
  timeline_<arm>.png                                 for the campaign 32 runs only; their events_ and turns_ CSVs stay
                                                     in compare_c32_designs/

The box drew ec_wait_split with a script git excludes. This one splits the wait as campaign 26's verifier did
(results/campaign26/checks/verify/server/analyze.py, Run.stall and item23): from the server log, a request's queue wait
(the last queue_duration line) and its read stall, the last SSD read landed by admission + 0.1 s, minus arrival, within
[0, queue wait]. A request with a read whose stall is under 0.05 s landed by arrival; one whose read landed within 0.1 s
of admission waited for its read only; any other read stall also waited for the GPU. The rest of the queue wait is not
tied to the request's own read, and TTFT minus the queue wait is after admission. --check-split compares the split with
the box CSVs of both campaigns, over each campaign's own runs: equal to 4 decimals on campaign 34's; campaign 32's set
is 9,155 requests against the box's 9,158.
"""
import argparse
import bisect
import collections
import csv
import json
import os
import random
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import c34_pricing as P
import jit_vs_hicache as J
import timeline as tl

SLACK_S, LANDED_S, LAST_WAIT_S = 0.1, 0.05, 0.1   # campaign 26's verifier: item23(slack, thr, bound)
TIMELINE_LABEL = {   # the box's "label (description)" legends, with the campaign
    "hicache": "hicache (original HiCache, campaign 32)", "jit_p50": "jit_p50 (JIT, p50 hints, campaign 34)",
    "ec_full": "ec_full (all four rules, campaign 34)", "ec_exact": "ec_exact (oracle: four rules, exact hints, campaign 32)",
    "jit_exact": "jit_exact (JIT, exact hints, campaign 34)"}
SPLIT_LABEL = {   # the box's bar labels; the run's campaign is added
    "hicache": "original HiCache", "jit_p50": "JIT, p50 hints", "ec_full": "all four rules",
    "ec_exact": "oracle: the four rules on exact hints", "jit_exact": "JIT, exact hints",
    "oracle_v2": "old oracle v2, exact hints"}
PARTS = (("after_admission_s", "after admission", "#2a78d6"),
         ("queue_other_s", "queue wait not tied to the turn's own read", "#eb6834"),
         ("stall_gpu_also_s", "waiting for its SSD read, and for the GPU after it", "#1baf7a"),
         ("stall_read_last_s", "waiting for its SSD read only", "#eda100"))
ADMIT = re.compile(r"rid=\S+?-c(\d+)-t(\d+)[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")


# ----------------------------------------------------------------------------- timeline.py's figures
def load_timeline(label, campaign):
    run = os.path.join(J.RES, f"compare_c{campaign}_designs", "runs", label)
    cl = tl.parse_client(os.path.join(run, f"client_{label}", "client.jsonl"), "prompt")
    sv = tl.parse_server(os.path.join(run, "server.log"), cl["t_start"], {t["rid"]: t["t0"] for t in cl["turns"]})
    return cl, sv, tl.arm_metrics(label, cl, sv)


def waiting_users(label, campaign, caps):
    """Fills each users record's waiting count: users arrived by then and not started. The program client
    (replay_programs.py run_arrivals) draws arrival gaps from Random("<seed>:arrivals"), and a user starts when one of
    max_active slots frees; starts are the run's play records, else its first request (they differ by at most 0.37 s)."""
    run, client = next(line.split()[1:] for line in open(os.path.join(J.RES, f"campaign{campaign}", "manifest.txt"))
                       if line.split()[0] == label)
    start, args = {}, None
    for line in open(os.path.join(J.RES, run, client, "client.jsonl")):
        r = json.loads(line)
        if r.get("kind") == "run":
            args = r["args"]
        elif r.get("kind") == "play" and "start" in r:
            start[r["play"]] = r["start"]
        elif r.get("kind") == "req" and r.get("phase") == "profile" and "t_send" in r and r["play"] not in start:
            start[r["play"]] = min(start.get(r["play"], float("inf")), r["t_send"])
    rng, t, arrivals = random.Random(f"{args['seed']}:arrivals"), 0.0, []
    while t < args["duration"]:
        arrivals.append(t)
        t += rng.expovariate(args["arrival_rate"])
    starts = sorted(start.values())
    return [{**c, "waiting": bisect.bisect_right(arrivals, c["t"]) - bisect.bisect_right(starts, c["t"])} for c in caps]


def draw_timeline_set(arms, loaded, out):
    """compare.png / .csv and ttft_over_time_60s.png / .csv; the figures name each run with its campaign."""
    for a in arms:
        tl.ARM_COLOR[TIMELINE_LABEL[a]] = P.ARMS[a].color
        if P.ARMS[a].campaign == 32:
            tl.ARM_DASH[TIMELINE_LABEL[a]] = (0, (4, 2.2))
    for a in arms:
        cl = loaded[a][0]
        cl["caps"] = waiting_users(a, P.ARMS[a].campaign, cl["caps"])
    shown = [(TIMELINE_LABEL[a], *loaded[a]) for a in arms]
    tl.draw_compare(shown, os.path.join(out, "compare.png"))
    with open(os.path.join(out, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loaded[arms[0]][2].keys()))
        w.writeheader(); w.writerows([loaded[a][2] for a in arms])
    rows = tl.draw_ttft_over_time(shown, os.path.join(out, "ttft_over_time_60s.png"), 60.0)
    short = {TIMELINE_LABEL[a]: a for a in arms}
    with open(os.path.join(out, "ttft_over_time_60s.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "t_from_s", "turns", "ttft_mean_s"])
        w.writeheader(); w.writerows([{**r, "arm": short[r["arm"]]} for r in rows])
    print("wrote", os.path.join(out, "compare.png"), "and", os.path.join(out, "ttft_over_time_60s.png"))


# ----------------------------------------------------------------------------- the wait split
def server_waits(run):
    """Per (conv, turn): arrival (entry_time) and queue wait from the last queue_duration line, and SSD read landings."""
    arrival, queue, lands = {}, {}, collections.defaultdict(list)
    for line in open(os.path.join(J.RES, run, "server.log"), errors="replace"):
        if "s2h_io" in line and (m := J.READ.match(line)):
            lands[(int(m[2]), int(m[3]))].append(J.stamp(m[1]))
        elif "queue_duration=" in line and (m := ADMIT.search(line)):
            k = (int(m[1]), int(m[2]))
            arrival[k], queue[k] = float(m[5]), float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
    return arrival, queue, lands


def wait_split(run, ttft, keys):
    arrival, queue, lands = server_waits(run)
    sums, reading, landed = collections.Counter(), 0, 0
    for k in keys:
        qw = queue[k]
        sums["after_admission_s"] += ttft[k] - qw
        ends = [t for t in lands.get(k, ()) if t <= arrival[k] + qw + SLACK_S]
        stall = min(qw, max(0.0, max(ends) - arrival[k])) if ends else 0.0
        if k in lands:
            reading += 1
            if stall < LANDED_S:
                landed += 1
            elif qw - stall < LAST_WAIT_S:
                sums["stall_read_last_s"] += stall
            else:
                sums["stall_gpu_also_s"] += stall
        sums["queue_other_s"] += qw
    n = len(keys)
    sums["queue_other_s"] -= sums["stall_read_last_s"] + sums["stall_gpu_also_s"]
    out = {p: sums[p] / n for p, _, _ in PARTS}
    return dict(shared_returning_turns=n, ttft_mean_s=st.mean(ttft[k] for k in keys), **out, reading_turns=reading,
                landed_by_arrival=landed)


def split_rows(arms, data):
    keys = sorted(P.shared_requests(arms, data))
    for a in arms:
        missing = [k for k in keys if k not in data[a]["admitted"]]
        if missing:
            raise SystemExit(f"{a}: {len(missing)} shared requests have no queue_duration line, e.g. {missing[:3]}")
    return {a: wait_split(data[a]["run"], {k: data[a]["returning"][k]["ttft"] for k in keys}, keys) for a in arms}


def split_label(a):
    return f"{SPLIT_LABEL[a]}, campaign {P.ARMS[a].campaign}"


def write_split(arms, rows, out):
    cols = ("shared_returning_turns", "ttft_mean_s") + tuple(p for p, _, _ in PARTS) + ("reading_turns", "landed_by_arrival")
    with open(os.path.join(out, "ec_wait_split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("arm", "run") + cols + ("campaign",))
        for a in arms:
            w.writerow((SPLIT_LABEL[a], a) + tuple(round(rows[a][c], 4) if isinstance(rows[a][c], float) else rows[a][c]
                                                   for c in cols) + (P.ARMS[a].campaign,))


def draw_split(arms, rows, out, cut):
    n = rows[arms[0]]["shared_returning_turns"]
    fig, ax = plt.subplots(figsize=(8.9, 1.7 + 0.47 * len(arms)), facecolor=J.SURFACE)
    ax.set_facecolor(J.SURFACE)
    for i, a in enumerate(arms):
        left = 0.0
        for p, _, color in PARTS:
            ax.barh(i, rows[a][p], left=left, color=color, height=0.55, edgecolor=J.SURFACE, linewidth=1)
            left += rows[a][p]
        ax.text(left + 0.25, i, f"{rows[a]['ttft_mean_s']:.2f} s", va="center", fontsize=10, color=J.INK)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([split_label(a) for a in arms], fontsize=10, color=J.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["ttft_mean_s"] for r in rows.values()) * 1.12)
    ax.set_xlabel("mean TTFT per returning turn, seconds", color=J.INK2)
    ax.grid(True, axis="x", color=J.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(J.AXIS)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in PARTS], loc="upper left", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.01, 0.86))
    fig.suptitle(f"FreeInference, 128 users, 160 GB: where the wait is\n({n:,} returning requests every arm sent before "
                 f"minute {cut / 60:.1f})", x=0.01, ha="left", fontsize=11, color=J.INK)
    fig.text(0.01, 0.015, "Campaign 32's runs ran on the engine without the four FA3 fixes, campaign 34's with them.",
             fontsize=8, color=J.INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.75))
    path = os.path.join(out, "ec_wait_split.png")
    fig.savefig(path, dpi=130, facecolor=J.SURFACE)
    plt.close(fig)
    print("wrote", path)


def check_split():
    """The split of each campaign's own runs against its box CSV (each over its own shared set)."""
    for campaign, arms in ((34, ["jit_p50", "jit_exact", "ec_full"]),
                           (32, ["hicache", "oracle_v2", "jit_p50", "jit_exact", "ec_full", "ec_exact"])):
        rows = split_rows(arms, {a: P.load(a, campaign) for a in arms})
        box = {r["run"]: r for r in csv.DictReader(open(os.path.join(J.RES, f"compare_c{campaign}_designs", "ec_wait_split.csv")))}
        for a in arms:
            diff = {c: (round(v, 4) if isinstance(v, float) else v, box[a][c]) for c, v in rows[a].items()
                    if abs(float(v) - float(box[a][c])) > (5e-5 if isinstance(v, float) else 0)}
            print(f"check c{campaign} {a}: " + ("equal" if not diff else f"differs {diff}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(P.ARMS), default=list(P.DEFAULT_ARMS), help="runs to draw, in order")
    ap.add_argument("--out", default=os.path.join(J.RES, P.OUT), help="folder for the figures and their CSVs")
    ap.add_argument("--check-split", action="store_true", help="compare the wait split with both campaigns' box CSVs")
    args = ap.parse_args()
    if args.check_split:
        check_split()
    loaded = {a: load_timeline(a, P.ARMS[a].campaign) for a in args.arms}
    for a in args.arms:
        if P.ARMS[a].campaign == 32:
            cl, sv, _ = loaded[a]
            tl.draw_timeline(a, cl, sv, os.path.join(args.out, f"timeline_{a}.png"))
    draw_timeline_set(args.arms, loaded, args.out)
    plt.rcParams.update({"font.size": 9.5})
    data = {a: P.load(a) for a in args.arms}
    rows = split_rows(args.arms, data)
    write_split(args.arms, rows, args.out)
    draw_split(args.arms, rows, args.out, P.cut_s(args.arms, data))
    for a in args.arms:
        r = rows[a]
        print(f"{a}: {r['ttft_mean_s']:.2f} = after admission {r['after_admission_s']:.2f} + queue {r['queue_other_s']:.2f} "
              f"+ read and GPU {r['stall_gpu_also_s']:.2f} + read only {r['stall_read_last_s']:.2f}")


if __name__ == "__main__":
    main()
