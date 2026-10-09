#!/usr/bin/env python3
"""steady_state.py --manifest M [--results DIR] [--warmup-users N] [--window-start S] [--iostat LOG] [--bin 300]
                  --out DIR

Steady-state statistics of open-loop runs whose users keep arriving (replay_agentic.py --arrival-until, campaign 13).
Per arm, the measured window runs from the moment the --warmup-users-th user finished (one turnover of a held
population; default: the arm's --concurrency, read from its run record) to the run's stop (summary.stop_s, else its last
reply); --window-start S opens every arm's window S seconds after its client started instead. In the window:
returning-turn TTFT (mean, p50, p90, p99; vs_baseline.is_returning, so the first request of a user who starts
mid-session is not one), server queue wait, where the prefix was found (vs_baseline.tier_of), uncached tokens per turn,
throughput (turns/s, finished users/h), live and waiting users, SSD traffic, and the drift between the window's halves
(a stationarity check). Runs whose user cap is set by the server backlog (replay_agentic.py --cap-mode backlog: one
kind=cap record per cap update) add the time-weighted means over the window of the cap (cap_mean), of the live users
the updates counted (cap_live_mean) and of the backlog they measured (backlog_mean -> cap_backlog_mean). Then the same
(conv, turn) pairs against the first arm, both inside their windows. Writes steady.csv, steady_paired.csv,
steady_bins.csv (per --bin seconds: live, waiting, turns/s, TTFT p50) and, when matplotlib is installed, steady.png.
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

from vs_baseline import TIERS, TSL, boot_ci, is_returning, load_iostat, q, tier_of


def load_arm(results: str, run: str, client: str) -> dict:
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    run_rec = next(r for r in rows if r.get("kind") == "run")
    summ = next((r for r in rows if r.get("kind") == "summary"), {})
    tag = run_rec.get("tag") or ""
    turns = [r for r in rows if r.get("kind") == "turn" and r.get("t_send") is not None]
    starts = {r["conv"]: r for r in rows if r.get("kind") == "conv_start"}
    ends = {r["conv"]: r for r in rows if r.get("kind") == "conv_end"}
    skipped = [r for r in rows if r.get("kind") == "conv_skipped"]
    caps = sorted((r for r in rows if r.get("kind") == "cap" and r.get("t") is not None), key=lambda r: r["t"])
    queue = {}
    log = os.path.join(results, run, "server.log")
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            if "ReqTimeStats(rid=" + tag in line:
                m = TSL.search(line)
                if m:
                    queue[m[1]] = float(m[2]) / (1000.0 if m[3] == "ms" else 1.0)
    stop = summ.get("stop_s") or max((r["t_send"] + (r.get("latency") or 0) for r in turns), default=0.0)
    return dict(tag=tag, t0=run_rec["t_start"], args=run_rec.get("args") or {}, summ=summ, turns=turns, starts=starts,
                ends=ends, skipped=skipped, caps=caps, queue=queue, stop=stop)


def default_warmup(d: dict, arm: str) -> int:
    """The run's --concurrency (its run record): users who finish in one turnover of a population held at it."""
    c = d["args"].get("concurrency")
    if not c:
        sys.exit(f"{arm}: the run record has no --concurrency; pass --warmup-users or --window-start")
    return int(c)


def window_of(d: dict, warmup_users: int, start: float | None = None) -> tuple:
    """(start, stop) of the measured window: after the warmup_users-th user who finished on their own (from `start`
    seconds instead, when given), to the stop. The start is None when the window is empty."""
    if start is not None:
        return (start if start < d["stop"] else None), d["stop"]
    if warmup_users <= 0:
        return 0.0, d["stop"]
    finished = sorted(r["end_s"] for r in d["ends"].values() if not r.get("capped") and r.get("end_s") is not None)
    start = finished[warmup_users - 1] if len(finished) >= warmup_users else None
    return start, d["stop"]


def cap_segments(d: dict) -> dict:
    """The kind=cap records as piecewise-constant series [(from_s, to_s, value)], seconds since the client started:
    cap     the cap an update sets holds until the next update; before the first, that update's prev_cap (the start
            cap, --concurrency);
    live    the live users counted at an update, held until the next one;
    backlog backlog_mean, the time-averaged backlog over the interval since the previous update (the first: since 0)."""
    caps = d["caps"]
    if not caps:
        return dict(cap=[], live=[], backlog=[])
    ts = [r["t"] for r in caps]
    nxt, prv = ts[1:] + [float("inf")], [0.0] + ts[:-1]
    start_cap = caps[0].get("prev_cap", d["args"].get("concurrency"))
    return dict(cap=[(0.0, ts[0], start_cap)] + [(t, n, r.get("cap")) for t, n, r in zip(ts, nxt, caps)],
                live=[(t, n, r.get("live")) for t, n, r in zip(ts, nxt, caps)],
                backlog=[(p, t, r.get("backlog_mean")) for p, t, r in zip(prv, ts, caps)])


def segments_mean(segs: list, lo: float, hi: float):
    """Time-weighted mean over [lo, hi) of piecewise-constant values; None when no segment with a value overlaps it."""
    num = den = 0.0
    for a, b, v in segs:
        w = min(b, hi) - max(a, lo)
        if w > 0 and v is not None:
            num, den = num + w * v, den + w
    return num / den if den > 0 else None


def live_waiting(d: dict, t: float) -> tuple:
    live = sum(1 for c, s in d["starts"].items()
               if s["start_s"] <= t < (d["ends"].get(c) or {}).get("end_s", float("inf")))
    waiting = sum(1 for s in d["starts"].values() if s["arrival_s"] <= t < s["start_s"])
    waiting += sum(1 for s in d["skipped"] if s["arrival_s"] <= t < d["stop"])
    return live, waiting


def stats(d: dict, lo: float, hi: float, iostat: list) -> dict:
    ret = [r for r in d["turns"] if is_returning(r) and r.get("ttft") and lo <= r["t_send"] < hi]
    done = [r for r in d["turns"] if lo <= r["t_send"] + (r.get("latency") or 0) < hi]
    tt = [r["ttft"] for r in ret]
    qq = [d["queue"][k] for k in (f"{d['tag']}-c{r['conv']}-t{r['turn']}" for r in ret) if k in d["queue"]]
    fin = sum(1 for r in d["ends"].values() if not r.get("capped") and lo <= (r.get("end_s") or -1) < hi)
    span = max(1e-9, hi - lo)
    io = [x for x in iostat if d["t0"] + lo < x[0] <= d["t0"] + hi] if iostat else []
    mid = (lo + hi) / 2
    first = [r["ttft"] for r in ret if r["t_send"] < mid]
    second = [r["ttft"] for r in ret if r["t_send"] >= mid]
    thirds = [[r["ttft"] for r in ret if lo + j * span / 3 <= r["t_send"] < lo + (j + 1) * span / 3] for j in range(3)]
    hours = [[r["ttft"] for r in ret if lo + j * 3600 <= r["t_send"] < min(hi, lo + (j + 1) * 3600)]
             for j in range(int(span // 3600) + 1)]
    hours = [st.mean(h) for h in hours if len(h) >= 50]
    live = [live_waiting(d, lo + (k + 0.5) * span / 200) for k in range(200)]   # midpoints: waiting rises steadily
    out = dict(window_start_s=round(lo, 1), window_end_s=round(hi, 1), window_min=round(span / 60, 1),
               returning=len(tt), ttft_mean=round(st.mean(tt), 3) if tt else "", ttft_p50=round(q(tt, .5), 3) if tt else "",
               ttft_p90=round(q(tt, .9), 3) if tt else "", ttft_p99=round(q(tt, .99), 3) if tt else "",
               queue_mean=round(st.mean(qq), 3) if qq else "",
               turns_per_s=round(len(done) / span, 4), finished_users_per_h=round(fin * 3600 / span, 2),
               live_mean=round(st.mean(x[0] for x in live), 1), waiting_mean=round(st.mean(x[1] for x in live), 1))
    if d["caps"]:   # a cap set by the server backlog: what it was, how full it was kept, and what it was set from
        seg = cap_segments(d)
        for key, col in (("cap", "cap_mean"), ("live", "cap_live_mean"), ("backlog", "cap_backlog_mean")):
            v = segments_mean(seg[key], lo, hi)
            out[col] = round(v, 1) if v is not None else ""
    out.update(uncached_per_turn=round(st.mean((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0) for r in ret))
               if ret else "",
               **{f"share_{k}": round(sum(tier_of(r) == k for r in ret) / len(ret), 3) if ret else "" for k in TIERS},
               ttft_drift_pct=round(100 * (st.mean(second) / st.mean(first) - 1), 1) if first and second else "",
               ttft_thirds=" / ".join(f"{st.mean(t):.1f}" for t in thirds if t),
               ttft_hourly_min_max=f"{min(hours):.1f} / {max(hours):.1f}" if hours else "",
               ssd_read_gb=round(sum(x[1] for x in io) * 10 / 1e6, 1) if io else "",
               ssd_written_gb=round(sum(x[2] for x in io) * 10 / 1e6, 1) if io else "",
               ssd_util_mean=round(st.mean(x[3] for x in io), 1) if io else "")
    return out


def bins(arm: str, d: dict, width: float) -> list:
    out = []
    for k in range(int(d["stop"] // width) + 1):
        lo, hi = k * width, (k + 1) * width
        tt = [r["ttft"] for r in d["turns"] if is_returning(r) and r.get("ttft") and lo <= r["t_send"] < hi]
        done = sum(1 for r in d["turns"] if lo <= r["t_send"] + (r.get("latency") or 0) < hi)
        live, waiting = live_waiting(d, lo + width / 2)
        out.append(dict(arm=arm, t_min=round(lo / 60, 1), live=live, waiting=waiting, turns_per_s=round(done / width, 4),
                        ttft_p50=round(q(tt, .5), 3) if tt else ""))
    return out


def draw(data: dict, windows: dict, all_bins: list, out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colors = {"hbm_host": "#eb6834", "three_tier_to": "#1baf7a", "three_tier_wc": "#2a78d6", "hbm_lru": "#6b6a66"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, gridspec_kw=dict(hspace=0.16))
    for arm in data:
        b = [x for x in all_bins if x["arm"] == arm]
        xs = [x["t_min"] + 2.5 for x in b]
        col = colors.get(arm, "#52514e")
        axes[0].plot(xs, [x["live"] for x in b], color=col, linewidth=2, label=arm)
        axes[0].plot(xs, [x["waiting"] for x in b], color=col, linewidth=1, linestyle=":")
        axes[1].plot(xs, [x["turns_per_s"] for x in b], color=col, linewidth=2)
        ys = [(x0, x["ttft_p50"]) for x0, x in zip(xs, b) if x["ttft_p50"] != ""]
        axes[2].plot([a for a, _ in ys], [y for _, y in ys], color=col, linewidth=2)
        lo, hi = windows[arm]
        for ax in axes:
            if lo is not None:
                ax.axvspan(lo / 60, hi / 60, color=col, alpha=0.05)
    axes[0].set_ylabel("users (solid live, dotted waiting)")
    axes[0].set_yscale("symlog", linthresh=200)
    axes[1].set_ylabel("turns completed / s")
    axes[2].set_ylabel("TTFT p50 (s)")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("minutes since the client started (shaded: each arm's measured window)")
    axes[0].legend(loc="upper center", ncol=len(data), frameon=False, bbox_to_anchor=(0.5, 1.18))
    for ax in axes:
        ax.grid(True, color="#e6e5e0")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", default="", help="directory holding the run dirs; default: two levels above --manifest")
    ap.add_argument("--warmup-users", type=int, default=None,
                    help="users who must finish before the window opens (default: each arm's --concurrency)")
    ap.add_argument("--window-end", type=float, default=None,
                    help="close every window this many seconds after the client started (default: the stop)")
    ap.add_argument("--window-start", type=float, default=None,
                    help="open every arm's window this many seconds after its client started (overrides "
                         "--warmup-users)")
    ap.add_argument("--iostat", default="")
    ap.add_argument("--bin", type=float, default=300.0, help="seconds per row of steady_bins.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = a.results or os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    os.makedirs(a.out, exist_ok=True)
    arms = [l.split() for l in open(a.manifest) if l.strip()]
    data = {arm: load_arm(results, run, client) for arm, run, client in arms}
    iostat = load_iostat(a.iostat) if a.iostat else []
    rows, windows, all_bins = [], {}, []
    for arm, d in data.items():
        if a.window_start is None:
            lo, hi = window_of(d, a.warmup_users if a.warmup_users is not None else default_warmup(d, arm))
        else:
            lo, hi = window_of(d, 0, a.window_start)
        if a.window_end is not None and lo is not None:
            hi = min(hi, a.window_end)
        windows[arm] = (lo, hi)
        s = d["summ"]
        at_stop = sum(1 for r in d["ends"].values() if not r.get("capped") and (r.get("end_s") or 1e18) <= d["stop"])
        row = dict(arm=arm, stopped_by=s.get("stopped_by") or "", finished_at_stop=at_stop,
                   finished_incl_drain=s.get("finished_users", ""), users_started=len(d["starts"]), errors=s.get("errors", ""))
        empty = "warm-up not over" if a.window_start is None else "--window-start is after the stop"
        row.update(stats(d, lo, hi, iostat) if lo is not None else {"window_start_s": empty})
        rows.append(row)
        all_bins += bins(arm, d, a.bin)
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    # columns in the order of the widest row (an arm with cap records, or one whose warm-up is over), then any others
    keys = list(dict.fromkeys(k for r in sorted(rows, key=len, reverse=True) for k in r))
    with open(os.path.join(a.out, "steady.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "steady_bins.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_bins[0]))
        w.writeheader()
        w.writerows(all_bins)
    ref = arms[0][0]
    prs = []
    for arm in list(data)[1:]:
        (alo, ahi), (rlo, rhi) = windows[arm], windows[ref]
        if alo is None or rlo is None:
            continue
        ta = {(r["conv"], r["turn"]): r["ttft"] for r in data[arm]["turns"] if is_returning(r) and r.get("ttft") and alo <= r["t_send"] < ahi}
        tr = {(r["conv"], r["turn"]): r["ttft"] for r in data[ref]["turns"] if is_returning(r) and r.get("ttft") and rlo <= r["t_send"] < rhi}
        keys2 = sorted(set(ta) & set(tr))
        if len(keys2) < 2:
            continue
        diff = [ta[k] - tr[k] for k in keys2]
        lo_ci, hi_ci = boot_ci(diff)
        prs.append(dict(arm=arm, ref=ref, pairs=len(diff), diff_mean=round(st.mean(diff), 3), ci_lo=round(lo_ci, 3),
                        ci_hi=round(hi_ci, 3), diff_p50=round(q(diff, .5), 3)))
        print(f"{arm} - {ref}: {len(diff)} pairs, TTFT mean {st.mean(diff):+.2f} s [{lo_ci:+.2f}, {hi_ci:+.2f}]")
    if prs:
        with open(os.path.join(a.out, "steady_paired.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prs[0]))
            w.writeheader()
            w.writerows(prs)
    if draw(data, windows, all_bins, os.path.join(a.out, "steady.png")):
        print("wrote", os.path.join(a.out, "steady.png"))
    print("wrote", os.path.join(a.out, "steady.csv"))


if __name__ == "__main__":
    main()
