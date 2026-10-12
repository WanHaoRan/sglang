#!/usr/bin/env python3
"""steady_figures.py --manifest M --arm ARM [--base M2 --base-label c12] [--warmup-users N] [--window-start S]
                   [--bin 900] --out DIR

Figures for one arm of a held-population run (campaign 13), each with its data as CSV:
  steady_<arm>_time.png   the run over time, in 15-min bins: users live and finished, the scheduler's queued and
                          running requests, turns completed per minute, returning-turn TTFT (p10-p50-p90), and where
                          returning turns found their prefix; the measured window (steady_state.py, with the same
                          --warmup-users / --window-start) is shaded. When the user cap is set by the server backlog
                          (kind=cap records), the users panel also draws the cap and the scheduler panel the backlog
                          each cap update measured (steady_<arm>_cap.csv).
  steady_<arm>_ttft.png   TTFT of returning turns in the window, as a CDF beside the same arm of a burst run (--base),
                          and the median TTFT and server queue wait by prompt size (what the TTFT is made of).
Run with matplotlib (the container, or a venv); reads the client JSONL and server.log only.
"""
import argparse
import csv
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from live_timeline import phases  # noqa: E402
from steady_state import cap_segments, default_warmup, live_waiting, load_arm, window_of  # noqa: E402
from timeline import parse_server  # noqa: E402
from vs_baseline import TIERS, is_returning, q, tier_of  # noqa: E402

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARM_COLOR = {"hbm_host": "#eb6834", "three_tier_to": "#1baf7a", "three_tier_wc": "#2a78d6", "hbm_lru": "#6b6a66",
             "three_tier_wc_oracle": "#e87ba4", "three_tier_to_oracle": "#4a3aa7",
             # campaign 16, as timeline.py
             "three_tier_wc_dl": "#56b4e9", "three_tier_wc_oracle_v2": "#882255", "three_tier_wc_oracle_v2_nu": "#4a3aa7"}
# Ordinal blue ramp for the cache tiers, cheapest to costliest (validated --ordinal: monotone L, light end 2.06:1);
# a recompute reused nothing, so it gets the neutral.
TIER_COLOR = {"device": "#86b6ef", "host": "#3987e5", "storage": "#256abf", "recompute": BASE}
TIER_LABEL = {"device": "GPU (L1) hit", "host": "host (L2) hit", "storage": "SSD (L3) restore",
              "recompute": "recompute (over half of the private prompt uncached)"}
CAP_FIELDS = ("t", "cap", "prev_cap", "backlog_mean", "target", "live", "waiting", "admitted", "finished")


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def label_end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", fontsize=8, color=INK2, va="center")


def time_bins(d: dict, sv: dict, width: float) -> list:
    stop = d["stop"]
    finished = sorted(r["end_s"] for r in d["ends"].values() if not r.get("capped") and r.get("end_s") is not None)
    rows = []
    for k in range(math.ceil(stop / width)):
        lo, hi = k * width, min((k + 1) * width, stop)   # the last bin ends at the stop
        ret = [r for r in d["turns"] if is_returning(r) and r.get("ttft") and lo <= r["t_send"] < hi]
        tt = [r["ttft"] for r in ret]
        done = sum(1 for r in d["turns"] if lo <= r["t_send"] + (r.get("latency") or 0) < hi)
        bs = [b for b in sv["batches"] if lo <= b["t"] < hi]
        live, waiting = live_waiting(d, hi - 1e-6)
        row = dict(t_h=round(lo / 3600, 3), t_mid_h=round((lo + hi) / 7200, 3), live_at_end=live, waiting_at_end=waiting,
                   finished_by_end=sum(1 for t in finished if t < hi), sched_queue_mean=round(st.mean(b["queue"] for b in bs), 1)
                   if bs else "", sched_running_mean=round(st.mean(b["running"] for b in bs), 1) if bs else "",
                   turns_per_min=round(done * 60 / (hi - lo), 2), returning=len(tt),
                   ttft_p10=round(q(tt, .1), 3) if tt else "", ttft_p50=round(q(tt, .5), 3) if tt else "",
                   ttft_p90=round(q(tt, .9), 3) if tt else "")
        row.update({f"share_{t}": round(sum(tier_of(r) == t for r in ret) / len(ret), 4) if ret else "" for t in TIERS})
        rows.append(row)
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def live_series(d: dict, step: float = 60.0) -> tuple:
    ts = [k * step for k in range(int(d["stop"] // step) + 1)] + [d["stop"] - 1e-6]
    return [t / 3600 for t in ts], [live_waiting(d, t)[0] for t in ts]


def held_texts(d: dict) -> dict:
    """How the live population was held, for the labels: at the run's fixed --concurrency, or at a cap that the
    server backlog set (kind=cap records)."""
    conc = d["args"].get("concurrency")
    if d["caps"] or not conc:
        return dict(live="the cap", keep="the cap filled", title="users held at the cap", series="held at the cap")
    return dict(live=f"the cap of {conc}", keep=f"{conc} live", title=f"{conc} users held live",
                series=f"held at {conc}")


def cap_lines(d: dict, stop: float) -> tuple:
    """Step lines (hours, values) up to the stop: the cap (each update's cap holds until the next) and the backlog
    each update measured (its time average over the interval before it)."""
    if not d["caps"]:
        return None, None
    seg = cap_segments(d)
    cs = [(a, v) for a, _, v in seg["cap"] if a <= stop and v is not None]
    cap = ([a / 3600 for a, _ in cs] + [stop / 3600], [v for _, v in cs] + [cs[-1][1]]) if cs else None
    bs = [(a, b, v) for a, b, v in seg["backlog"] if b <= stop and v is not None]
    backlog = ([bs[0][0] / 3600] + [b / 3600 for _, b, _ in bs], [bs[0][2]] + [v for _, _, v in bs]) if bs else None
    return cap, backlog


def draw_users(ax, col: str, live: tuple, finished_t: list, stop: float, rates: dict, held: dict,
               cap: tuple | None = None) -> None:
    style(ax, "Users: live and finished", "users")
    n = len(finished_t)
    ax.plot(live[0], live[1], color=col, linewidth=2, label=f"live (held at {held['live']})")
    if cap:
        ax.step(cap[0], cap[1], where="post", color=col, linewidth=1.2, linestyle="--",
                label="user cap, set by the server backlog")
    ax.step([0.0] + [t / 3600 for t in finished_t] + [stop / 3600], [0] + list(range(1, n + 1)) + [n], where="post",
            color=INK2, linewidth=2, label="finished, cumulative")
    label_end(ax, live[0][-1], live[1][-1], "live")
    label_end(ax, stop / 3600, n, "finished")
    ax.text(0.01, 0.95, f"client reservoir: {rates['waiting']:,} users waiting at the stop\n"
            f"arrivals {rates['arrivals_per_h']:.0f}/h vs about {rates['finished_per_h']:.0f}/h finishing keep "
            f"{held['keep']}", transform=ax.transAxes, fontsize=8, color=INK2, va="top", linespacing=1.4)
    ax.set_ylim(0, 1.08 * max([n] + (list(cap[1]) if cap else [])))
    ax.legend(loc="center left", frameon=False, fontsize=8, bbox_to_anchor=(0.0, 0.55))


def draw_scheduler(ax, col: str, x: list, rows: list, backlog: tuple | None = None) -> None:
    style(ax, "Server scheduler: requests queued and running (mean per 15-min bin)", "requests")
    ys = [(a, r) for a, r in zip(x, rows) if r["sched_queue_mean"] != ""]
    ax.plot([a for a, _ in ys], [r["sched_queue_mean"] for _, r in ys], color=col, linewidth=2,
            label="queued (queue-req)")
    ax.plot([a for a, _ in ys], [r["sched_running_mean"] for _, r in ys], color=INK2, linewidth=2, label="running")
    if backlog:   # what the cap was set from, beneath the bin means
        ax.step(backlog[0], backlog[1], where="pre", color=MUTED, linewidth=0.9, zorder=1.8,
                label="measured backlog per cap update")
    ax.set_ylim(bottom=0)
    if ys:
        qv, rv = ys[-1][1]["sched_queue_mean"], ys[-1][1]["sched_running_mean"]
        dy = 5 if abs(qv - rv) < 0.08 * ax.get_ylim()[1] else 0   # end labels within ~8% of the axis would overlap
        label_end(ax, ys[-1][0], qv, "queued", dy if qv >= rv else -dy)
        label_end(ax, ys[-1][0], rv, "running", -dy if qv >= rv else dy)
    # one column when the backlog is drawn: a wider row would run into the panel title
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1 if backlog else 2, frameon=False, fontsize=8,
              borderaxespad=0.2)


def draw_ttft_bins(ax, col: str, x: list, rows: list) -> None:
    style(ax, "TTFT of returning turns per 15-min bin (line p50, band p10-p90)", "seconds")
    ys = [(a, r) for a, r in zip(x, rows) if r["ttft_p50"] != ""]
    ax.fill_between([a for a, _ in ys], [r["ttft_p10"] for _, r in ys], [r["ttft_p90"] for _, r in ys], color=col,
                    alpha=0.25, linewidth=0)
    ax.plot([a for a, _ in ys], [r["ttft_p50"] for _, r in ys], color=col, linewidth=2)
    ax.set_ylim(0, 1.05 * max(r["ttft_p90"] for _, r in ys))


def draw_tiers(ax, rows: list) -> None:
    style(ax, "Where returning turns found their prefix (share of returning turns per 15-min bin)", "share of turns")
    bottom = [0.0] * len(rows)
    for t in TIERS:
        vals = [r[f"share_{t}"] or 0.0 for r in rows]
        if not any(vals):
            continue
        ax.bar([r["t_h"] for r in rows], vals, width=[2 * (r["t_mid_h"] - r["t_h"]) for r in rows], bottom=bottom,
               align="edge", color=TIER_COLOR[t], edgecolor=SURF, linewidth=1.0, label=TIER_LABEL[t])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.32))


def draw_time(arm: str, rows: list, window: tuple, opens: str, out: str, live: tuple, finished_t: list,
              rates: dict, held: dict, cap: tuple | None = None, backlog: tuple | None = None,
              label: str = "Campaign 13", setting: str = "x70 gaps") -> None:
    col = ARM_COLOR.get(arm, INK2)
    lo, hi = window
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), facecolor=SURF, sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 0.9, 1.1, 0.9], hspace=0.36))
    fig.suptitle(f"{label}, {arm}: {setting}, {held['title']}, stopped after {len(finished_t)} finished; "
                 f"shaded: the measured window", x=0.125, y=0.915, ha="left", fontsize=11.5, color=INK)
    x = [r["t_mid_h"] for r in rows]
    draw_users(axes[0], col, live, finished_t, hi, rates, held, cap)
    draw_scheduler(axes[1], col, x, rows, backlog)
    style(axes[2], "Turns completed per minute (15-min bins)", "turns / min")
    axes[2].plot(x, [r["turns_per_min"] for r in rows], color=col, linewidth=2)
    axes[2].set_ylim(bottom=0)
    draw_ttft_bins(axes[3], col, x, rows)
    draw_tiers(axes[4], rows)
    axes[4].set_xlabel("hours since the client started", fontsize=8.5, color=INK2)
    axes[4].xaxis.set_major_locator(MultipleLocator(1))
    for ax in axes:
        if lo is not None:
            ax.axvspan(lo / 3600, hi / 3600, color=BASE, alpha=0.25, linewidth=0, zorder=0)
            ax.axvline(lo / 3600, color=MUTED, linewidth=1, zorder=2.5)
        ax.axvline(hi / 3600, color=MUTED, linewidth=1, zorder=2.5)
    if lo is not None:
        axes[0].annotate(f"measured window ({(hi - lo) / 3600:.1f} h) opens: {opens}",
                         (lo / 3600, 1.0), xycoords=("data", "axes fraction"), xytext=(5, -3),
                         textcoords="offset points", fontsize=8, color=INK2, va="top")
    axes[0].annotate(f"stop: {len(finished_t)} finished", (hi / 3600, 0.0), xycoords=("data", "axes fraction"),
                     xytext=(-5, 4), textcoords="offset points", fontsize=8, color=INK2, va="bottom", ha="right")
    fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def cdf(xs: list) -> tuple:
    xs = sorted(xs)
    return xs, [(k + 1) / len(xs) for k in range(len(xs))]


def draw_ttft(arm: str, series: list, by_prompt: list, title: str, out: str) -> None:
    col = ARM_COLOR.get(arm, INK2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURF, gridspec_kw=dict(wspace=0.22))
    fig.suptitle(title, x=0.125, y=0.995, ha="left", fontsize=11.5, color=INK)
    style(a1, "CDF (log x)", "share of turns at or below")
    spots = ((0.5, (-6, 0), "right"), (0.45, (8, -10), "left"), (0.5, (-6, 0), "right"))   # where each label sits
    for k, (name, short, recs, color, width) in enumerate(series):
        xs = [r["ttft"] for r in recs]
        xv, yv = cdf(xs)
        a1.plot(xv, yv, color=color, linewidth=width, zorder=3 if k == 0 else 2,
                label=f"{name} (n {len(xs):,}, p50 {q(xs, .5):.3g} s)")
        y, off, ha = spots[k % 3]
        a1.annotate(short, (q(xs, y), y), xytext=off, textcoords="offset points", fontsize=8, color=INK2, ha=ha,
                    va="center")
    a1.set_xscale("log")
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.set_xlabel("TTFT (s)", fontsize=8.5, color=INK2)
    a1.set_ylim(0, 1.0)
    a1.legend(loc="upper left", frameon=False, fontsize=8, bbox_to_anchor=(0.0, -0.13))
    style(a2, "By prompt size: medians per decile of prompt tokens", "seconds")
    xs = [r["prompt_mid_k"] for r in by_prompt]
    a2.fill_between(xs, 0, [r["queue_p50"] or 0 for r in by_prompt], color=col, alpha=0.25, linewidth=0,
                    label="of which: server queue wait")
    a2.plot(xs, [r["ttft_p50"] for r in by_prompt], color=col, linewidth=2, marker="o", markersize=4, label="TTFT")
    a2.set_ylim(0, 1.3 * max(r["ttft_p50"] for r in by_prompt))
    a2.set_xlabel("prompt tokens (thousands; decile bins)", fontsize=8.5, color=INK2)
    a2.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    first, last = by_prompt[0], by_prompt[-1]
    if first["rest_p50"] != "" and last["rest_p50"] != "":
        a2.text(0.01, 0.88, f"TTFT minus queue wait (median): {first['rest_p50']:.1f} s at {first['prompt_mid_k']:.0f}K "
                f"tokens, {last['rest_p50']:.1f} s at {last['prompt_mid_k']:.0f}K", transform=a2.transAxes, fontsize=8,
                color=INK2, va="top")
    fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def base_series(path: str, arm: str, label: str, short: str) -> list:
    """The same arm of a burst run: its peak phase (all users arrived, live >= 90% of the peak) and the whole run."""
    bres = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    brun, bclient = next((r, c) for a, r, c in (l.split() for l in open(path) if l.strip()) if a == arm)
    b = load_arm(bres, brun, bclient)
    arrive = {c: s["arrival_s"] for c, s in b["starts"].items()}
    end = {c: e["end_s"] for c, e in b["ends"].items() if e.get("end_s") is not None}
    last, drop, peak, grid, live = phases(arrive, end)
    low = min(n for t, n in zip(grid, live) if last <= t < drop)
    bret = [r for r in b["turns"] if is_returning(r) and r.get("ttft")]
    return [(f"{label}, peak phase ({low}-{peak} live, {(drop - last) / 60:.0f} min)", f"{short} peak phase",
             [r for r in bret if last <= r["t_send"] < drop], INK2, 1.8),
            (f"{label}, whole run", f"{short} whole run", bret, MUTED, 1.8)]


def summary_row(name: str, recs: list) -> dict:
    xs = [r["ttft"] for r in recs]
    rc = [r["ttft"] for r in recs if tier_of(r) == "recompute"]
    hit = [r["ttft"] for r in recs if tier_of(r) != "recompute"]
    return dict(series=name, n=len(xs), mean=round(st.mean(xs), 3), min=round(min(xs), 3), p10=round(q(xs, .1), 3),
                p50=round(q(xs, .5), 3), p90=round(q(xs, .9), 3), p99=round(q(xs, .99), 3),
                share_recompute=round(len(rc) / len(xs), 4), recompute_p50=round(q(rc, .5), 3) if rc else "",
                hit_p50=round(q(hit, .5), 3) if hit else "")


def prompt_deciles(d: dict, ret: list) -> list:
    qkey = lambda r: f"{d['tag']}-c{r['conv']}-t{r['turn']}"   # noqa: E731
    sized = sorted((r["prompt_tokens"], r["ttft"], d["queue"].get(qkey(r))) for r in ret if r.get("prompt_tokens"))
    out = []
    for k in range(10):
        part = sized[k * len(sized) // 10:(k + 1) * len(sized) // 10]
        if not part:
            continue
        qs = [x[2] for x in part if x[2] is not None]
        rest = [x[1] - x[2] for x in part if x[2] is not None]
        out.append(dict(decile=k + 1, prompt_lo=part[0][0], prompt_hi=part[-1][0],
                        prompt_mid_k=round(q([x[0] for x in part], .5) / 1000, 1), n=len(part),
                        ttft_p50=round(q([x[1] for x in part], .5), 2), queue_p50=round(q(qs, .5), 2) if qs else "",
                        rest_p50=round(q(rest, .5), 2) if rest else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", default="", help="a burst campaign's manifest (same arm) for the CDF")
    ap.add_argument("--base-label", default="campaign 12")
    ap.add_argument("--base-short", default="c12", help="the base run's name in the direct labels")
    ap.add_argument("--label", default="Campaign 13", help="this run's name in the figure titles")
    ap.add_argument("--setting", default="x70 gaps", help="the workload, after the arm in the time figure's title")
    ap.add_argument("--series-label", default="campaign 13 x70", help="this run's name in the CDF legend and CSV")
    ap.add_argument("--series-short", default="campaign 13", help="this run's name in the CDF's direct label")
    ap.add_argument("--warmup-users", type=int, default=None,
                    help="users who must finish before the window opens (default: the run's --concurrency)")
    ap.add_argument("--window-end", type=float, default=None,
                    help="close every window this many seconds after the client started (default: the stop)")
    ap.add_argument("--window-start", type=float, default=None,
                    help="open the window this many seconds after the client started (overrides --warmup-users)")
    ap.add_argument("--bin", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = a.results or os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    os.makedirs(a.out, exist_ok=True)
    run, client = next((r, c) for arm, r, c in (l.split() for l in open(a.manifest) if l.strip()) if arm == a.arm)
    d = load_arm(results, run, client)
    sv = parse_server(os.path.join(results, run, "server.log"), d["t0"])
    if a.window_start is None:
        warmup = a.warmup_users if a.warmup_users is not None else default_warmup(d, a.arm)
        lo, hi = window_of(d, warmup)
        opens = f"{warmup} users finished"
    else:
        lo, hi = window_of(d, 0, a.window_start)
        opens = f"at {a.window_start / 3600:.2f} h (--window-start)"
    if a.window_end is not None and lo is not None:
        hi = min(hi, a.window_end)
    rows = time_bins(d, sv, a.bin)
    write_csv(os.path.join(a.out, f"steady_{a.arm}_time.csv"), rows)
    held = held_texts(d)
    cap, backlog = cap_lines(d, hi)
    cap_rows = [dict(t_h=round(r["t"] / 3600, 4), **{k: r.get(k, "") for k in CAP_FIELDS})
                for r in d["caps"] if r["t"] <= hi]
    if cap_rows:
        write_csv(os.path.join(a.out, f"steady_{a.arm}_cap.csv"), cap_rows)
    finished_t = sorted(r["end_s"] for r in d["ends"].values()
                        if not r.get("capped") and r.get("end_s") is not None and r["end_s"] <= hi)
    arrivals = sum(1 for s in list(d["starts"].values()) + d["skipped"] if s["arrival_s"] <= hi)
    rates = dict(waiting=live_waiting(d, hi - 1e-6)[1], arrivals_per_h=arrivals * 3600 / hi,
                 finished_per_h=sum(1 for t in finished_t if lo < t) * 3600 / (hi - lo) if lo is not None else 0.0)
    draw_time(a.arm, rows, (lo, hi), opens, os.path.join(a.out, f"steady_{a.arm}_time.png"), live_series(d),
              finished_t, rates, held, cap, backlog, label=a.label, setting=a.setting)
    if lo is None:
        print("warm-up not over: no window, no TTFT figure" if a.window_start is None
              else "--window-start is after the stop: no window, no TTFT figure")
        return
    ret = [r for r in d["turns"] if is_returning(r) and r.get("ttft") and lo <= r["t_send"] < hi]
    series = [(f"{a.series_label}, {a.arm}, {held['series']} (window, {(hi - lo) / 3600:.1f} h)",
               f"{a.series_short} window",
               ret, ARM_COLOR.get(a.arm, INK2), 2.4)]
    if a.base:
        series += base_series(a.base, a.arm, a.base_label, a.base_short)
    cdf_rows = [summary_row(name, recs) for name, _, recs, _, _ in series]
    write_csv(os.path.join(a.out, f"steady_{a.arm}_ttft_summary.csv"), cdf_rows)
    by_prompt = prompt_deciles(d, ret)
    write_csv(os.path.join(a.out, f"steady_{a.arm}_by_prompt.csv"), by_prompt)
    draw_ttft(a.arm, series, by_prompt, f"{a.label}, {a.arm}: TTFT of returning turns in the measured window "
              f"({len(ret):,} turns, {(hi - lo) / 3600:.1f} h)", os.path.join(a.out, f"steady_{a.arm}_ttft.png"))
    for r in cdf_rows:
        print(r)
    for r in by_prompt:
        print(r)
    print(rates)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
