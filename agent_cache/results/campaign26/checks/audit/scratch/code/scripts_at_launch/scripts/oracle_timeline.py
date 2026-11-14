"""Zoomed timeline of the oracle next-turn prefetch (ORACLE_PLAN.md) over a short window of one run.

  python3 oracle_timeline.py --run results/<boot dir> [--client client_<...>] --start 7200 [--seconds 120]
                             [--lanes 40] [--out <png>] [--csv <csv>]

The clock is seconds since the client started (timeline.py's). Top panel: one row per conversation with a hinted turn in
the window; its turns are drawn left to right on that row, each labeled t<turn> above its admission (or arrival), and a
dark segment from a turn's admission to the next turn's plan tick marks the turn running (prefill + decode); --lanes
conversations are spread evenly over the window by their first admission in it.
Each turn has three lines:
  request  the previous turn ended (the plan is registered), think time, the predicted arrival (hollow diamond) and
           the arrival (triangle), the wait in the scheduler queue (gray band), the predicted admission at the trigger
           (hollow circle) and the admission (dot);
  plan     the oracle's decision at its trigger (tick): the read planned over [A - L-hat, A];
  read     each SSD read for the turn, waiting in the FIFO read queue (thin) then in I/O (thick); a read the
           `timeout` policy cut while it was still queued (the read thread then skips it: s2h_io ms=0) ends at the
           cut with a cross and leaves the queue there. A star marks each #39283 re-check (admission found the
           matched prefix shorter than at the lookup and reissued it), also when no read followed (a miss, or a
           decline at the prefetch capacity cap); a hollow square marks the request's 8-lookup retry budget running
           out ("reissue cap reached": no further SSD lookups).
Rows run off the window's edges where a turn started earlier or ends later.
Panels below: pages waiting in the SSD read queue (rebuilt from the completed reads; dots are the read backlog the
oracle saw at each trigger in the window), the SSD read rate (s2h_io, 1-s bins), requests waiting for admission.
Colors: blue = staged before arrival (case 1), orange = re-fetched while queued (case 2), aqua = the request's own
lookup or a paced retry, violet = the #39283 re-query; the dataviz reference palette's first three slots and its violet,
as timeline.py.
A run without the oracle (no ntp_* events, e.g. campaign 14's baselines) gets the same lanes for every returning turn,
timed by the server's ReqTimeStats lines (arrival = entry_time, admission = + queue_duration, end = + forward_duration;
the tick is the previous turn's end), without the plan line and the predictions; all of its reads are aqua.
"""
import argparse
import bisect
import collections
import csv
import datetime as dt
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from timeline import AQUA, BLUE, INK, MUTED, ORANGE, TS_RE, parse_ts, style  # noqa: E402

SURF, BAND, THINK, RUN = "#fcfcfb", "#d9d8d2", "#b9b8b1", "#52514e"
VIOLET = "#4a3aa7"   # the reference palette's violet, as timeline.py's three_tier_wc_norq
SOURCE_COLOR = {"staged": BLUE, "refetch": ORANGE, "own": AQUA, "requery": VIOLET}
SOURCE_LABEL = {"staged": "staged before arrival (case 1)", "refetch": "re-fetched while queued (case 2)",
                "own": "own lookup or paced retry", "requery": "#39283 re-query"}
EVT_RE = re.compile(r"HICACHE_EVT (\w+) rid=(\S+)(.*)")
KV_RE = re.compile(r"(\w+)=(-?[\w./-]+)")
SUCCESS_RE = re.compile(r"HiCache prefetch (?:success|dropped) req=(\S+) completed=(\d+)")
BATCH_RE = re.compile(r"(?:Prefill|Decode) batch.*#queue-req: (\d+)")
H2D_RE = re.compile(r"HICACHE_EVT h2d_done .*tokens=(\d+) bytes=(\d+)")
PAGE_RE = re.compile(r"'page_size': (\d+)")
RTS_RE = re.compile(r"ReqTimeStats\(rid=([^,]+),.*?: queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, "
                    r"entry_time=([\d.]+)")
RID_RE = re.compile(r"(.+-c\d+)-t(\d+)$")
RETRY_RE = re.compile(r"HiCache (device prefix shrank before admission|storage prefetch reissue cap reached) req=(\S+)")
KINDS = {"ntp_plan", "ntp_due", "ntp_stage", "ntp_refetch", "ntp_arrive", "ntp_admit", "ntp_expire", "s2h_query",
         "s2h_io", "prefetch_start"}
ACTION_WINDOW_S = 2.0   # a lookup that hits within this long after an oracle action is that action's read
LATE_S = 0.05           # as oracle_eval.py: admission within 50 ms of the read's end waited on the read


def load(run: str, client: str, w0: float, w1: float) -> dict:
    with open(os.path.join(run, client, "client.jsonl")) as f:
        t_start = next(json.loads(l) for l in f if '"kind": "run"' in l)["t_start"]
    ev = collections.defaultdict(lambda: collections.defaultdict(list))   # rid -> kind -> [(t, fields)]
    rts = {}   # rid -> (arrival, admission, end), from the ReqTimeStats line the server logs when the request ends
    queue_len, page_size, bytes_per_token = [], 64, None
    with open(os.path.join(run, "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT" in line:
                m = EVT_RE.search(line)
                if m is None:
                    continue
                if m[1] == "h2d_done" and bytes_per_token is None:
                    h = H2D_RE.search(line)
                    if h and int(h[1]):
                        bytes_per_token = int(h[2]) / int(h[1])
                if m[1] in KINDS:
                    ev[m[2]][m[1]].append((parse_ts(TS_RE.match(line)[1]) - t_start, dict(KV_RE.findall(m[3]))))
            elif "HiCache prefetch" in line:
                m = SUCCESS_RE.search(line)
                if m:
                    ev[m[1]]["insert"].append((parse_ts(TS_RE.match(line)[1]) - t_start, {"completed": m[2]}))
            elif "HiCache device prefix shrank" in line or "reissue cap reached" in line:
                m = RETRY_RE.search(line)
                if m:
                    kind = "requery_check" if m[1].startswith("device") else "budget_spent"
                    ev[m[2]][kind].append((parse_ts(TS_RE.match(line)[1]) - t_start, {}))
            elif "#queue-req" in line:
                ts = TS_RE.match(line)
                if ts:
                    t = parse_ts(ts[1]) - t_start
                    if w0 - 5 <= t <= w1 + 5:
                        queue_len.append((t, int(BATCH_RE.search(line)[1])))
            elif "ReqTimeStats(" in line:
                m = RTS_RE.search(line)
                if m:
                    arr = float(m[4]) - t_start
                    adm = arr + float(m[2]) / 1000
                    rts[m[1]] = (arr, adm, adm + float(m[3]) / 1000)
            elif "'page_size'" in line:
                m = PAGE_RE.search(line)
                page_size = int(m[1]) if m else page_size
    reads = {rid: reads_of(e) for rid, e in ev.items()}
    return dict(t_start=t_start, ev=ev, reads=reads, rts=rts, queue_len=queue_len,
                page_bytes=page_size * (bytes_per_token or 98304))


def reads_of(e: dict) -> list:
    """Each completed read of one rid: enqueue (its lookup hit), I/O start and end, pages, and who started it."""
    out = []
    hits = sorted(t for t, f in e.get("s2h_query", ()) if int(f.get("storage_hit", 0)) > 0)
    starts = sorted(t for t, _ in e.get("prefetch_start", ()))
    actions = sorted([(t, "staged") for t, f in e.get("ntp_stage", ()) if f.get("inflight") == "1"]
                     + [(t, "refetch") for t, f in e.get("ntp_refetch", ()) if f.get("issued") == "1"]
                     + [(t, "requery") for t, _ in e.get("requery_check", ())])
    cuts = sorted(t for t, f in e.get("insert", ()) if f["completed"] == "0")
    for t_end, f in e.get("s2h_io", ()):
        io_start = t_end - float(f.get("ms", 0)) / 1000
        cut = None
        if f.get("ms") == "0":   # cut while queued; the read thread skipped it
            k = bisect.bisect_right(cuts, t_end) - 1
            cut = cuts[k] if k >= 0 else t_end
            io_start = t_end = cut
        i = bisect.bisect_right(hits, io_start) - 1
        j = bisect.bisect_right(starts, io_start) - 1
        enq = hits[i] if i >= 0 else (starts[j] if j >= 0 else io_start)
        src = "own"
        for t_a, kind in actions:
            if t_a <= enq + 0.05 and enq - t_a <= ACTION_WINDOW_S:
                src = kind
        enq = min(enq, io_start)
        out.append(dict(enq=enq, io_start=io_start, io_end=t_end, pages=int(f.get("pages", 0)), src=src, cut=cut))
    return out


def first(e: dict, kind: str):
    return e[kind][0] if e.get(kind) else (None, {})


def retry_marks(e: dict) -> dict:
    """The turn's #39283 re-checks and the moment its retry budget ran out, for the read line's markers."""
    return {k: [t for t, _ in e.get(k, ())] for k in ("requery_check", "budget_spent")}


def plans_in_window(data: dict, w0: float, w1: float) -> list:
    rows = []
    for rid, e in data["ev"].items():
        if "ntp_plan" not in e:
            continue
        (t_plan, plan), (t_arr, arr), (t_adm, adm) = first(e, "ntp_plan"), first(e, "ntp_arrive"), first(e, "ntp_admit")
        (t_due, due), (t_exp, _) = first(e, "ntp_due"), first(e, "ntp_expire")
        pred_arr = t_plan + float(plan.get("pred_arr_ms", 0)) / 1000
        # The prefetcher logs an admission on its next pass, after that pass's forward (p50 36 ms, p90 80 ms late at
        # campaign 15's load); the request's own stamps give the true times: arrival = predicted + error, + the wait.
        if t_arr is not None and arr.get("arr_err_ms") not in (None, "na"):
            t_arr = pred_arr + float(arr["arr_err_ms"]) / 1000
            if t_adm is not None and adm.get("wait_ms") not in (None, "na"):
                t_adm = t_arr + float(adm["wait_ms"]) / 1000
        reads = data["reads"][rid]
        end = t_adm if t_adm is not None else (t_exp if t_exp is not None else w1)
        if t_plan > w1 or max([end] + [x["io_end"] for x in reads]) < w0:
            continue
        sess, turn = rid.rsplit("-t", 1)
        r = dict(rid=rid, label=f"{sess.rsplit('-', 1)[-1]} t{turn}", conv=sess.rsplit("-", 1)[-1], turn=int(turn),
                 next_rid=f"{sess}-t{int(turn) + 1}", t_plan=t_plan, pred_arr=pred_arr,
                 t_arr=t_arr, t_adm=t_adm, t_exp=t_exp,
                 t_due=t_due, case=int(due.get("case", 0)) if due else 0, reads=reads, **retry_marks(e))
        if t_due is not None:
            r["pred_adm"] = t_due + float(due["lead_ms"]) / 1000
            r["trigger"] = r["pred_adm"] - float(due["load_ms"]) / 1000
            r["backlog"] = int(due.get("backlog_pages", 0))
        ends = [x["io_end"] for x in reads if x["cut"] is None and t_adm is not None and x["io_end"] <= t_adm + 0.2]
        r["read_before_adm"] = bool(ends)
        r["late"] = bool(ends) and t_adm - max(ends) <= LATE_S
        rows.append(r)
    return rows


def turns_in_window(data: dict, w0: float, w1: float) -> list:
    """plans_in_window's rows for a run without the oracle: every turn that follows another turn of its conversation,
    timed by ReqTimeStats; the tick is the previous turn's end, and there is no plan and no prediction."""
    rows = []
    for rid, (t_arr, t_adm, t_end) in data["rts"].items():
        m = RID_RE.match(rid)
        prev = data["rts"].get(f"{m[1]}-t{int(m[2]) - 1}") if m else None
        reads = data["reads"].get(rid, [])
        if prev is None or prev[2] > w1 or max([t_end] + [x["io_end"] for x in reads]) < w0:
            continue
        sess, turn = m[1], int(m[2])
        ends = [x["io_end"] for x in reads if x["cut"] is None and x["io_end"] <= t_adm + 0.2]
        rows.append(dict(rid=rid, label=f"{sess.rsplit('-', 1)[-1]} t{turn}", conv=sess.rsplit("-", 1)[-1], turn=turn,
                         next_rid=f"{sess}-t{turn + 1}", t_plan=prev[2], pred_arr=None, t_arr=t_arr, t_adm=t_adm,
                         t_exp=None, t_due=None, case=0, reads=reads, t_end=t_end, read_before_adm=bool(ends),
                         late=bool(ends) and t_adm - max(ends) <= LATE_S, **retry_marks(data["ev"].get(rid, {}))))
    return rows


def select_lanes(rows: list, w0: float, w1: float, lanes: int) -> list:
    """Up to `lanes` conversations, each with every turn of its that has something in the window, spread evenly over the
    window by the conversation's first admission in it; conversations with no admission in the window fill the rest."""
    in_win = lambda t: t is not None and w0 <= t <= w1
    by_conv = collections.defaultdict(list)
    for r in rows:
        by_conv[r["conv"]].append(r)
    convs = []
    for conv, turns in by_conv.items():
        turns.sort(key=lambda r: (r["t_plan"], r["turn"]))
        adm = [r["t_adm"] for r in turns if in_win(r["t_adm"])]
        first = min([t for r in turns for t in (r["t_due"], r["t_arr"]) if in_win(t)] + [w1])
        convs.append(dict(conv=conv, turns=turns, first_adm=min(adm) if adm else None, first=first))
    admitted = sorted((c for c in convs if c["first_adm"] is not None), key=lambda c: c["first_adm"])
    others = sorted((c for c in convs if c["first_adm"] is None), key=lambda c: c["first"])
    if lanes <= 0 or len(convs) <= lanes:
        chosen = admitted + others
    elif len(admitted) >= lanes:
        step = (len(admitted) - 1) / max(lanes - 1, 1)
        chosen = [admitted[round(i * step)] for i in range(lanes)]
    else:
        chosen = admitted + others[:lanes - len(admitted)]
    return sorted(chosen, key=lambda c: (c["first_adm"] if c["first_adm"] is not None else float("inf"), c["first"]))


def queue_series(data: dict, w0: float, w1: float):
    """Pages waiting in the FIFO read queue as a step series over [w0, w1]."""
    deltas = []
    for reads in data["reads"].values():
        for x in reads:
            if x["enq"] <= w1:   # reads entirely before w0 net to zero
                deltas += [(x["enq"], x["pages"]), (x["io_start"], -x["pages"])]
    deltas.sort()
    t_s, waiting, cur = [w0], [0], 0
    for t, d in deltas:
        if t > w1:
            break
        cur += d
        if t <= w0:
            waiting[0] = cur
        else:
            t_s.append(t)
            waiting.append(cur)
    t_s.append(w1)
    waiting.append(cur)
    return t_s, waiting


def read_rate(data: dict, w0: float, w1: float, bin_s: float = 1.0):
    n = int((w1 - w0) / bin_s) + 1
    gb = [0.0] * n
    for reads in data["reads"].values():
        for x in reads:
            a, b = max(x["io_start"], w0), min(x["io_end"], w1)
            if b <= a:
                continue
            rate = x["pages"] * data["page_bytes"] / 1e9 / max(x["io_end"] - x["io_start"], 1e-3)
            k = int((a - w0) / bin_s)
            while a < b and k < n:
                edge = min(b, w0 + (k + 1) * bin_s)
                gb[k] += rate * (edge - a) / bin_s
                a, k = edge, k + 1
    return [w0 + (k + 0.5) * bin_s for k in range(n)], gb


def legend_keys(oracle: bool = True) -> list:
    key = [Line2D([], [], color=THINK, linewidth=1, label="think time: previous turn ended (tick) -> next request"),
           Line2D([], [], color=MUTED, marker="D", markerfacecolor="none", linestyle="", label="predicted arrival"),
           Line2D([], [], color=INK, marker="^", linestyle="", label="arrival"),
           Line2D([], [], color=BAND, linewidth=6, label="waiting in the scheduler queue"),
           Line2D([], [], color=RUN, linewidth=2,
                  label="running: admission -> turn ended" + (" (next plan registered)" if oracle else "")),
           Line2D([], [], color=MUTED, marker="o", markerfacecolor="none", linestyle="",
                  label="predicted admission, at the trigger"),
           Line2D([], [], color=INK, marker="o", linestyle="", label="admission"),
           Line2D([], [], color=MUTED, linewidth=1, marker="|", markersize=9,
                  label="oracle plan: trigger (tick), read planned to end at the predicted admission"),
           Line2D([], [], color=MUTED, linewidth=1.6, label="read waiting in the FIFO read queue (thin)"),
           Line2D([], [], color=MUTED, marker="x", linestyle="", markeredgewidth=1.6,
                  label="read cut by the timeout while queued"),
           Line2D([], [], color=VIOLET, marker="*", markersize=9, linestyle="",
                  label="#39283 re-check: prefix shorter at admission, lookup reissued"),
           Line2D([], [], color=INK, marker="s", markerfacecolor="none", linestyle="",
                  label="retry budget spent: no further SSD lookups")]
    if not oracle:   # no plans, no predictions, only the requests' own reads
        key = [k for k in key if not k.get_label().startswith(("predicted", "oracle plan"))]
    key += [Line2D([], [], color=SOURCE_COLOR[s], linewidth=6, label=f"read I/O: {SOURCE_LABEL[s]}")
            for s in (("staged", "refetch", "own", "requery") if oracle else ("own", "requery"))]
    return key


def draw_turn(a_lane, r: dict, y: float, w0: float, w1: float, clip, inside) -> None:
    """One turn's three lines on the lane at y: request (y + 0.27), plan (y), reads (y - 0.27)."""
    y_req, y_plan, y_read = y + 0.27, y, y - 0.27
    t_arr_or_end = r["t_arr"] if r["t_arr"] is not None else w1
    if inside(r["t_plan"], t_arr_or_end):
        a_lane.plot([clip(r["t_plan"]), clip(t_arr_or_end)], [y_req] * 2, color=THINK, linewidth=1, zorder=1)
    if r["t_arr"] is not None and inside(r["t_arr"], r["t_adm"] if r["t_adm"] is not None else w1):
        a_lane.plot([clip(r["t_arr"]), clip(r["t_adm"] if r["t_adm"] is not None else w1)], [y_req] * 2,
                    color=BAND, linewidth=5, solid_capstyle="butt", zorder=2)
    for t, mk, face in ((r["t_plan"], "|", INK), (r["pred_arr"], "D", "none"), (r["t_arr"], "^", INK),
                        (r.get("pred_adm"), "o", "none"), (r["t_adm"], "o", INK)):
        if t is not None and w0 <= t <= w1:
            a_lane.plot(t, y_req, marker=mk, markersize=8 if mk == "|" else 6, markerfacecolor=face,
                        markeredgecolor=INK if face != "none" else MUTED, markeredgewidth=1.2, linestyle="", zorder=5)
    if "trigger" in r:
        col = SOURCE_COLOR["staged" if r["case"] == 1 else "refetch"]
        if inside(r["trigger"], r["pred_adm"]):
            a_lane.plot([clip(r["trigger"]), clip(r["pred_adm"])], [y_plan] * 2, color=col, linewidth=1, alpha=0.6,
                        zorder=2)
        if w0 <= r["t_due"] <= w1:
            a_lane.plot(r["t_due"], y_plan, marker="|", markersize=9, color=col, markeredgewidth=2, zorder=4)
    for x in r["reads"]:
        col = SOURCE_COLOR[x["src"]]
        if inside(x["enq"], x["io_start"]):
            a_lane.plot([clip(x["enq"]), clip(x["io_start"])], [y_read] * 2, color=col, linewidth=1.4, zorder=3)
        if x["cut"] is not None:
            if w0 <= x["cut"] <= w1:
                a_lane.plot(x["cut"], y_read, marker="x", markersize=6, color=col, markeredgewidth=1.6, zorder=4)
        elif inside(x["io_start"], x["io_end"]):
            a_lane.plot([clip(x["io_start"]), clip(x["io_end"])], [y_read] * 2, color=col, linewidth=5,
                        solid_capstyle="butt", zorder=4)
    for t in r.get("requery_check", ()):
        if w0 <= t <= w1:
            a_lane.plot(t, y_read, marker="*", markersize=9, color=VIOLET, linestyle="", zorder=6)
    for t in r.get("budget_spent", ()):
        if w0 <= t <= w1:
            a_lane.plot(t, y_read, marker="s", markersize=5, markerfacecolor="none", markeredgecolor=INK,
                        markeredgewidth=1.2, linestyle="", zorder=6)


def draw(shown: list, all_rows: list, data: dict, w0: float, w1: float, title: str, out_png: str,
         oracle: bool = True) -> None:
    n = len(shown)
    lane_h = max(3.0, 0.32 * n)
    keys = legend_keys(oracle)
    head = 1.0 + 0.17 * max(0, (len(keys) + 4) // 5 - 3)   # inches above the lanes: the title and the legend rows
    fig_h = lane_h + 6.4 + head
    fig, axes = plt.subplots(4, 1, figsize=(16, fig_h), sharex=True, facecolor=SURF,
                             gridspec_kw=dict(height_ratios=[lane_h, 1.7, 1.2, 1.1], hspace=0.3))
    fig.subplots_adjust(top=1 - head / fig_h)
    a_lane, a_queue, a_rate, a_sched = axes
    clip = lambda t: min(max(t, w0), w1)
    inside = lambda a, b: b > w0 and a < w1
    for i, c in enumerate(shown):
        y = n - 1 - i
        turns = c["turns"]
        for k, r in enumerate(turns):
            draw_turn(a_lane, r, y, w0, w1, clip, inside)
            # the turn running (prefill + decode): its admission -> its end (the next turn's plan tick, or ReqTimeStats)
            nxt = (r["t_end"] if "t_end" in r else turns[k + 1]["t_plan"] if k + 1 < len(turns)
                   else first(data["ev"].get(r["next_rid"], {}), "ntp_plan")[0])
            if r["t_adm"] is not None and nxt is not None and inside(r["t_adm"], nxt):
                a_lane.plot([clip(r["t_adm"]), clip(nxt)], [y + 0.27] * 2, color=RUN, linewidth=2, zorder=3)
            at = next((t for t in (r["t_adm"], r["t_arr"], r["t_due"], r["t_plan"]) if t is not None and w0 <= t <= w1), None)
            if at is not None:
                a_lane.text(at, y + 0.42, f"t{r['turn']}", ha="center", va="bottom", fontsize=6, color=MUTED, clip_on=True)
    a_lane.set_yticks(range(n))
    a_lane.set_yticklabels([c["conv"] for c in reversed(shown)], fontsize=7)
    a_lane.set_ylim(-0.6, n - 0.25)
    style(a_lane)
    left = a_lane.get_position().x0
    fig.text(left, 1 - 0.12 / fig_h, title, ha="left", va="top", fontsize=10, color=INK)
    fig.legend(handles=keys, loc="upper left", bbox_to_anchor=(left, 1 - 0.38 / fig_h), ncol=5,
               fontsize=7.5, frameon=False, borderaxespad=0)

    t_s, waiting = queue_series(data, w0, w1)
    a_queue.fill_between(t_s, waiting, step="post", color=MUTED, alpha=0.12, linewidth=0)
    a_queue.step(t_s, waiting, where="post", color=MUTED, linewidth=1.5, label="pages waiting in the read queue")
    for case in (1, 2):
        pts = [(r["t_due"], r["backlog"]) for r in all_rows
               if r["case"] == case and "backlog" in r and w0 <= r["t_due"] <= w1]
        if pts:
            a_queue.plot(*zip(*pts), marker="o", linestyle="", markersize=5, markeredgecolor=SURF,
                         markeredgewidth=1, color=SOURCE_COLOR["staged" if case == 1 else "refetch"],
                         label=f"read backlog the oracle saw at a case-{case} trigger")
    style(a_queue, ylabel=f"pages ({data['page_bytes'] / 2**20:.0f} MiB each)")
    a_queue.set_ylim(bottom=0)
    a_queue.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5, frameon=False, ncol=3)

    t_c, gb = read_rate(data, w0, w1)
    a_rate.step(t_c, gb, where="mid", color=MUTED, linewidth=1.5)
    style(a_rate, ylabel="SSD reads\n(GB/s, 1-s bins)")
    a_rate.set_ylim(bottom=0)
    if data["queue_len"]:
        a_sched.step(*zip(*data["queue_len"]), where="post", color=MUTED, linewidth=1.5)
    style(a_sched, ylabel="requests waiting\nfor admission")
    a_sched.set_ylim(bottom=0)
    a_sched.set_xlim(w0, w1)
    a_sched.set_xlabel("seconds since the client started", fontsize=8, color=MUTED)
    fig.savefig(out_png, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rid", "event", "t", "t_end", "detail"])
        for r in rows:
            for k in ("t_plan", "pred_arr", "t_arr", "t_due", "trigger", "pred_adm", "t_adm", "t_exp"):
                if r.get(k) is not None:
                    detail = f"case {r['case']}, backlog {r['backlog']} pages" if k == "t_due" else ""
                    w.writerow([r["rid"], k, f"{r[k]:.3f}", "", detail])
            for x in r["reads"]:
                w.writerow([r["rid"], f"read_{x['src']}", f"{x['enq']:.3f}", f"{x['io_end']:.3f}",
                            f"io_start {x['io_start']:.3f}, {x['pages']} pages"])
            for k in ("requery_check", "budget_spent"):
                for t in r.get(k, ()):
                    w.writerow([r["rid"], k, f"{t:.3f}", "", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="server run dir (results/<UTC>_<arm>_<level>)")
    ap.add_argument("--client", default=None, help="client dir inside --run (default: the last client_* dir)")
    ap.add_argument("--start", type=float, required=True, help="window start, seconds since the client started")
    ap.add_argument("--seconds", type=float, default=120.0, help="window length (default 120)")
    ap.add_argument("--lanes", type=int, default=40, help="conversations to draw, spread over the window (0: all)")
    ap.add_argument("--out", default=None, help="png path (default: oracle_timeline_<start>_<seconds>.png in --run)")
    ap.add_argument("--csv", default=None, help="also write the drawn rows' events as CSV")
    a = ap.parse_args()
    client = a.client or sorted(d for d in os.listdir(a.run) if d.startswith("client_"))[-1]
    w0, w1 = a.start, a.start + a.seconds
    data = load(a.run, client, w0, w1)
    oracle = any("ntp_plan" in e for e in data["ev"].values())
    rows = plans_in_window(data, w0, w1) if oracle else turns_in_window(data, w0, w1)
    shown = select_lanes(rows, w0, w1, a.lanes)
    admitted = [r for r in rows if r["t_adm"] is not None and w0 <= r["t_adm"] <= w1]
    with_read = [r for r in admitted if r["read_before_adm"]]
    utc = lambda t: dt.datetime.fromtimestamp(data["t_start"] + t, tz=dt.timezone.utc).strftime("%H:%M:%S")
    title = (f"{os.path.basename(os.path.normpath(a.run))}: {w0:.0f}-{w1:.0f} s after the client started "
             f"({utc(w0)}-{utc(w1)} UTC{'' if oracle else ', no oracle'}). {len(admitted)} "
             f"{'hinted' if oracle else 'returning'} turns admitted in the window, {len(with_read)} after an "
             f"SSD read, {sum(r['late'] for r in with_read)} of those within {LATE_S * 1000:.0f} ms of its end (late); "
             f"{len(shown)} conversations drawn ({sum(len(c['turns']) for c in shown)} turns)")
    out = a.out or os.path.join(a.run, f"oracle_timeline_{w0:.0f}_{a.seconds:.0f}.png")
    draw(shown, rows, data, w0, w1, title, out, oracle)
    if a.csv:
        write_csv([r for c in shown for r in c["turns"]], a.csv)
    marks = {k: sum(w0 <= t <= w1 for e in data["ev"].values() for t, _ in e.get(k, ()))
             for k in ("requery_check", "budget_spent")}
    print(f"{marks['requery_check']} #39283 re-checks and {marks['budget_spent']} spent retry budgets in the window")
    print(f"wrote {out}" + (f" and {a.csv}" if a.csv else "")
          + f"; {len(shown)} conversations, {sum(len(c['turns']) for c in shown)} of {len(rows)} turns drawn")


if __name__ == "__main__":
    main()
