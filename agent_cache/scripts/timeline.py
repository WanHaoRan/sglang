"""Timeline and comparison plots for agent-replay runs (RUNBOOK 6.6 events + 4.6 client JSONL).

Joins one boot's server.log (millisecond timestamps, HICACHE_EVT lines, Prefill/Decode batch lines)
with one client run's client.jsonl (per-turn send/first-token/end, tier split) on a common clock:
seconds since the client started (client `run.t_start` is epoch; server stamps are UTC wall clock).

  python3 timeline.py --manifest results/compare_<t>/manifest.txt        # every arm + a comparison figure
  python3 timeline.py --run results/<boot dir> [--client client_<...>]    # one arm
  --tier-rule private: classify by the private history (vs_baseline.tier_of, the rule of the READMEs since campaign 12)
  --steady-state [SECONDS]: also compare_steady.png / compare_steady.csv over the turns sent once every arm's cached data
  had outgrown device + host memory (the latest of the arms' first host evictions; the first device eviction on a server
  without a host tier), or compare_steady_<SECONDS>s.* over the turns sent from SECONDS on (since the client started).
  --paired-cdf REF [--paired-max N]: also paired_cdf_<arm>.png for every other arm: REF's TTFT ECDF over the turns both
  arms completed, with the arm's TTFT of each same (user, turn) at the same height, joined by a dashed line (N: draw at
  most N pairs per panel, every k-th by rank; default all); with --steady-state also paired_cdf_<arm>_steady*.png.
  --ttft-over-time [SECONDS]: also ttft_over_time_<SECONDS>s.png / .csv: every arm's mean TTFT of the turns sent in each
  SECONDS of client time (default 60), with the turns sent per minute and the users waiting for admission.

Outputs next to the manifest (or in the run dir): timeline_<arm>.png, events_<arm>.csv, turns_<arm>.csv (its last
column, mid_start, marks the first request of a user who started mid-session), compare.png, compare.csv. A user's first
request (turn 0 or mid_start, vs_baseline.is_first) is not a returning turn.
Colors: blue = writes (d2h, h2s) / device hit, orange = restores (h2d, s2h) / host hit,
aqua = evictions / storage hit, gray = cold/recompute (default: less than half the prompt cached; private: over half of
the prompt beyond the shared system prompt uncached); the three hues are the validated first three slots of the
dataviz reference palette.
"""
import argparse
import bisect
import collections
import csv
import datetime as dt
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from vs_baseline import is_first, is_returning, tier_of  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9891", "#0b0b0b", "#52514e"
GREEN = "#2e7d32"  # three_tier_wc, distinct from the AQUA of three_tier_to
TIER_COLOR = {"cold": GRAY, "device": BLUE, "host": ORANGE, "storage": AQUA}
COLD_LABEL = {"prompt": "cold/recompute", "private": "recompute (over half of the private prompt uncached)"}
ARM_COLOR = {"hbm_lru": GRAY, "hbm_host": ORANGE, "three_tier": AQUA,
             "three_tier_to": AQUA, "three_tier_wc": GREEN,   # campaign 6: the two prefetch policies
             "three_tier_wc_norq": "#4a3aa7",   # campaign 11: wait_complete with the #39283 re-query disabled
             # the oracle (ORACLE_PLAN.md): next reference-palette slots, validated with GREEN and the violet above
             "three_tier_wc_ntpobs": "#eda100", "three_tier_wc_oracle": "#e87ba4",
             "three_tier_wc_ntpobs_norq": BLUE, "three_tier_wc_oracle_norq": "#e34948",
             # campaign 15: violet again, never in a figure with three_tier_wc_norq (campaign 11)
             "three_tier_to_oracle": "#4a3aa7",
             # campaign 16: Okabe-Ito sky, Tol wine, violet again (never with three_tier_to_oracle); with GREEN,
             # worst normal-vision dE 18.2 and worst CVD dE 11.4 (OKLab x100, Machado)
             "three_tier_wc_dl": "#56b4e9", "three_tier_wc_oracle_v2": "#882255", "three_tier_wc_oracle_v2_nu": "#4a3aa7"}
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d{3})?)\]\s*(.*)$")
KV_RE = re.compile(r"(\w+)=(-?[\w./-]+)")
HOST_POOL_RE = re.compile(r"Allocating kv hierarchical KV host pool: (\d+) tokens, ([\d.]+) GB")


def parse_ts(s: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S"
    return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).timestamp()


def host_path(p: str) -> str:
    """Manifest paths are container paths; map them onto the host checkout if needed."""
    if os.path.exists(p):
        return p
    alt = p.replace("/sgl-workspace/sglang", "/home/wanhr/sglang")
    return alt if os.path.exists(alt) else p


# ----------------------------------------------------------------------------- parsing
def parse_client(path: str, tier_rule: str = "prompt") -> dict:
    rows = [json.loads(l) for l in open(path)]
    run = next(r for r in rows if r["kind"] == "run")
    t_start = run["t_start"]
    turns = []
    for r in rows:
        if r["kind"] != "turn" or "t_send" not in r:
            continue
        cd = r.get("cached_details") or {}
        host, storage = int(cd.get("host") or 0), int(cd.get("storage") or 0)
        cached = int(r.get("cached_tokens") or 0)
        prompt = int(r.get("prompt_tokens") or 0)
        # private: over half of the prompt beyond the shared system prompt uncached is cold/recompute. prompt (older):
        # a turn that finds less than half its prompt lost its own prefix (only the shared system prompt hit)
        if tier_rule == "private":
            tier = tier_of(r).replace("recompute", "cold")
        elif cached < 0.5 * prompt:
            tier = "cold"
        else:
            tier = "storage" if storage > 0 else "host" if host > 0 else "device"
        turns.append(dict(conv=r["conv"], turn=r["turn"], t0=r["t_send"], ttft=r.get("ttft") or 0.0,
                          lat=r["latency"], tier=tier, prompt=r.get("prompt_tokens", 0), cached=cached,
                          uncached=r.get("prompt_tokens", 0) - cached, host=host, storage=storage,
                          gap=r.get("gap_slept", 0.0), rid=f"{run['args'].get('tag') or 'replay'}-c{r['conv']}-t{r['turn']}",
                          mid_start=bool(r.get("mid_start"))))   # a mid-session start's first request (is_first)
    summary = next((r for r in rows if r["kind"] == "summary"), None)
    caps = [dict(t=r["t"], waiting=r["waiting"], live=r["live"], cap=r["cap"]) for r in rows if r["kind"] in ("cap", "users")]
    return dict(t_start=t_start, turns=turns, summary=summary, args=run["args"], tier_rule=tier_rule, caps=caps)


def parse_server(path: str, t_start: float, sends: dict = None) -> dict:
    """Everything relative to t_start (seconds). Submit/done pairs are matched FIFO for d2h/h2d
    (the ack queues are FIFO), by op id for h2s, by rid for s2h. sends maps a rid to its client send time: storage
    events of a rid stamped before it was sent were staged by the oracle and go to s2h_staged, not to the turn."""
    batches, evd, evh, lbi = [], [], [], []
    d2h_sub, h2d_sub = collections.deque(), collections.deque()
    d2h, h2d = [], []
    h2s = collections.defaultdict(dict)
    s2h = collections.defaultdict(dict)
    s2h_staged = collections.defaultdict(dict)
    ntp = []
    posts = []
    host_pool, host_moves = None, []   # host KV pool size; (t, tokens) written or loaded to host (+), evicted (-)
    sends = sends or {}
    planned = set()   # rids the oracle planned for: one never sent had all its reads staged
    def s2h_of(rid, t):
        staged = t < sends[rid] if rid in sends else rid in planned
        return s2h_staged[rid] if staged else s2h[rid]
    for line in open(path, errors="replace"):
        m = TS_RE.match(line)
        if not m:
            continue
        t = parse_ts(m.group(1)) - t_start
        msg = m.group(2)
        if msg.startswith("Prefill batch") or msg.startswith("Decode batch"):
            kv = {k.strip(): v for k, v in re.findall(r"#?([\w -]+?): ([0-9.]+)", msg)}
            usage = float(kv.get("token usage", 0))
            batches.append(dict(t=t, kind="prefill" if msg.startswith("Prefill") else "decode", usage=usage,
                                running=int(kv.get("running-req", 0)), queue=int(kv.get("queue-req", 0)),
                                new=int(kv.get("new-token", 0)), cached=int(kv.get("cached-token", 0))))
        elif "HICACHE_EVT" in msg:
            ev = msg.split("HICACHE_EVT ", 1)[1].split()[0]
            kv = dict(KV_RE.findall(msg))
            toks = int(kv.get("tokens", 0))
            if ev == "d2h_submit":
                d2h_sub.append((t, toks))
                host_moves.append((t, toks))   # the host slots are taken at submit, not when the copy is done
            elif ev == "d2h_done":
                ts, _ = d2h_sub.popleft() if d2h_sub else (t, toks)
                d2h.append(dict(t0=ts, t1=t, tokens=toks, ms=float(kv.get("ms", -1))))
            elif ev == "h2d_submit":
                h2d_sub.append((t, toks))
            elif ev == "h2d_done":
                ts, _ = h2d_sub.popleft() if h2d_sub else (t, toks)
                h2d.append(dict(t0=ts, t1=t, tokens=toks, ms=float(kv.get("ms", -1))))
            elif ev == "h2s_submit":
                h2s[kv["op"]].update(t_submit=t, tokens=toks)
            elif ev == "h2s_io":
                h2s[kv["op"]].update(t_io_end=t, io_ms=float(kv.get("ms", 0)), tokens_io=toks)
            elif ev == "h2s_done":
                h2s[kv["op"]].update(t_done=t, tokens_done=toks)
            elif ev == "prefetch_start":
                s2h_of(kv["rid"], t).update(t_start=t, tokens_req=toks)
            elif ev == "s2h_query":
                s2h_of(kv["rid"], t).update(t_query=t, storage_hit=int(kv.get("storage_hit", 0)))
            elif ev == "s2h_io":
                s2h_of(kv["rid"], t).update(t_io_end=t, io_ms=float(kv.get("ms", 0)), tokens_io=toks)
            elif ev == "ntp_plan":
                planned.add(kv.get("rid"))
            elif ev in ("ntp_due", "ntp_stage", "ntp_refetch", "ntp_admit"):
                ntp.append(dict(kind=ev, t=t, rid=kv.get("rid"), fields=kv))
            elif ev == "evict_device":
                evd.append(dict(t=t, tokens=toks, requested=int(kv.get("requested", 0))))
            elif ev == "evict_host":
                evh.append(dict(t=t, tokens=toks))
                host_moves.append((t, -toks))
            elif ev == "load_back_init":
                lbi.append(dict(t=t, rid=kv.get("rid"), tokens=toks, host_hit=int(kv.get("host_hit", 0))))
        elif "HiCache prefetch" in msg:
            kv = dict(KV_RE.findall(msg))
            if "req" in kv:
                s2h_of(kv["req"], t).update(t_end=t, completed=int(kv.get("completed", 0)),
                                            loaded=int(kv.get("loaded", 0)), dropped="dropped" in msg)
                host_moves.append((t, int(kv.get("loaded", 0))))   # only loaded tokens stay; the read's other slots are freed
        elif '"POST /v1/chat/completions' in msg:
            posts.append(t)
        elif msg.startswith("Allocating kv hierarchical KV host pool"):
            tokens, gb = HOST_POOL_RE.match(msg).groups()
            host_pool = dict(tokens=int(tokens), gb=float(gb))
    return dict(batches=batches, d2h=d2h, h2d=h2d, h2s=dict(h2s), s2h=dict(s2h), evict_device=evd, evict_host=evh,
                load_back_init=lbi, posts=posts, s2h_staged=dict(s2h_staged), ntp=ntp, host_pool=host_pool,
                host_moves=host_moves)


# ----------------------------------------------------------------------------- figures
def style(ax, title=None, ylabel=None):
    ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.8)
    ax.grid(False, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c1")
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def draw_timeline(arm: str, cl: dict, sv: dict, out_png: str) -> None:
    turns = cl["turns"]
    t_end = max([t["t0"] + t["lat"] for t in turns] + [b["t"] for b in sv["batches"]] + [1.0])
    convs = sorted({t["conv"] for t in turns})
    # the layout was drawn for 128 rows; more rows (users of a held-population run) get up to twice the lane panel
    lane = 4.2 * min(2.0, max(1.0, len(convs) / 128))
    ratios = [lane, 1.2, 1.2, 1.2, 2.2, 1.6]
    # 14 / 10.4 inches per ratio unit: the scale the panel heights were tuned at
    fig, axes = plt.subplots(len(ratios), 1, figsize=(16, sum(ratios) * 14 / 10.4), sharex=True,
                             gridspec_kw=dict(height_ratios=ratios, hspace=0.28))
    fig.patch.set_facecolor("#fcfcfb")
    a_req, a_pool, a_host, a_queue, a_evt, a_cum = axes

    # A. request lanes: one row per conversation; prefill/queue segment thick, decode segment thin
    row = {c: i for i, c in enumerate(convs)}
    pitch = a_req.get_position().height * fig.get_figheight() * 72 / max(1, len(convs))   # points per row
    scale = min(1.0, pitch / 2.0)   # 2.0 pt: the 128-row panel, where the 4 pt / 1.2 pt lanes were chosen
    segs, colors, widths = [], [], []
    for t in turns:
        y, c = row[t["conv"]], TIER_COLOR[t["tier"]]
        segs += [[(t["t0"], y), (t["t0"] + t["ttft"], y)], [(t["t0"] + t["ttft"], y), (t["t0"] + t["lat"], y)]]
        colors += [to_rgba(c), to_rgba(c, 0.7)]
        widths += [4 * scale, 1.2 * scale]
    a_req.add_collection(LineCollection(segs, colors=colors, linewidths=widths, capstyle="butt"))
    a_req.autoscale_view()
    step = max(1, len(convs) // 32)  # at most ~32 labels: 128 conversations would otherwise print as a black bar
    a_req.set_yticks(range(0, len(convs), step))
    a_req.set_yticklabels([f"c{c}" for c in convs[::step]], fontsize=7)
    a_req.invert_yaxis()
    n_conv = int((cl.get("args") or {}).get("num_conversations") or len(convs))
    who = "conversation" if len(convs) <= n_conv else f"user ({len(convs)} users replay {n_conv} conversations)"
    style(a_req, f"{arm}: requests per {who} (thick = send to first token, thin = decode); color = where the prefix was found",
          "users")
    a_req.legend(handles=[Patch(color=TIER_COLOR[k], label=(COLD_LABEL[cl["tier_rule"]] if k == "cold" else k))
                          for k in ("cold", "device", "host", "storage")],
                 loc="upper right", fontsize=8, frameon=False, ncol=4)

    # B. device pool usage (one measure per axis)
    bt = [b["t"] for b in sv["batches"]]
    a_pool.plot(bt, [b["usage"] for b in sv["batches"]], color=BLUE, linewidth=1.5)
    a_pool.set_ylim(0, 1.05)
    style(a_pool, "device KV pool usage (fraction of --max-total-tokens)", "usage")

    # C. host pool usage, summed from the host events: the server logs no host usage line
    if sv["host_pool"]:
        pool, used, ts, fills = sv["host_pool"], 0, [], []
        for t, tokens in sv["host_moves"]:
            used += tokens
            ts.append(t)
            fills.append(used / pool["tokens"])
        a_host.plot(ts, fills, color=ORANGE, linewidth=1.5, drawstyle="steps-post")
        style(a_host, f"host KV pool usage (fraction of the {pool['gb']:g} GB pool, {pool['tokens']:,} tokens; cached KV = "
                      "d2h writes + storage loads - host evictions; storage reads in flight not counted)", "usage")
    else:
        style(a_host, "host KV pool usage: no host pool in this run", "usage")
    a_host.set_ylim(0, 1.05)

    # D. queue depth and running requests
    a_queue.plot(bt, [b["queue"] for b in sv["batches"]], color=ORANGE, linewidth=1.5, label="queued (#queue-req)")
    a_queue.plot(bt, [b["running"] for b in sv["batches"]], color=BLUE, linewidth=1.5, label="running (#running-req)")
    style(a_queue, "scheduler queue and running requests", "requests")
    a_queue.set_ylim(0, 1.3 * max([b["queue"] for b in sv["batches"]] + [b["running"] for b in sv["batches"]] + [1]))
    a_queue.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # E. tier events: one lane per type; bars where a duration is known, ticks otherwise (size ~ tokens)
    lanes = ["d2h", "h2s", "s2h", "h2d", "evict_device", "evict_host"]
    oracle = bool(sv["ntp"] or sv["s2h_staged"])   # runs with --hicache-storage-next-turn-prefetch observe | on
    if oracle:
        lanes.append("oracle")
    ypos = {k: i for i, k in enumerate(lanes)}
    def bars(lane, spans, color):
        y = ypos[lane]
        if spans:
            a_evt.add_collection(LineCollection([[(t0, y), (max(t1, t0 + 0.25), y)] for t0, t1 in spans],
                                                colors=[to_rgba(color, 0.85)], linewidths=6, capstyle="butt"))
    def ticks(lane, ts, color, alpha=1.0):
        if ts:
            a_evt.plot(ts, [ypos[lane]] * len(ts), linestyle="none", marker="|", color=color, markersize=8, alpha=alpha)
    def copy_spans(evs):
        return [(e["t1"] - e["ms"] / 1000 if e["ms"] > 0 else e["t0"], e["t1"]) for e in evs]
    def io_spans(ops):
        return [(op["t_io_end"] - op["io_ms"] / 1000, op["t_io_end"]) for op in ops if "t_io_end" in op]
    bars("d2h", copy_spans(sv["d2h"]), BLUE)
    bars("h2s", io_spans(sv["h2s"].values()), BLUE)
    ticks("h2s", [op["t_submit"] for op in sv["h2s"].values() if "t_io_end" not in op and "t_submit" in op], BLUE)
    bars("s2h", io_spans(sv["s2h"].values()), ORANGE)
    ticks("s2h", [op["t_start"] for op in sv["s2h"].values() if "t_io_end" not in op and "t_start" in op], ORANGE, 0.6)
    bars("h2d", copy_spans(sv["h2d"]), ORANGE)
    for lane in ("evict_device", "evict_host"):
        if sv[lane]:
            a_evt.scatter([e["t"] for e in sv[lane]], [ypos[lane]] * len(sv[lane]),
                          s=[6 + e["tokens"] / 200 for e in sv[lane]], color=AQUA, alpha=0.7, linewidths=0)
    title = "HiCache tier events (bars = measured copy/IO time ending at completion; dots = evictions, area ~ tokens)"
    if oracle:
        # the oracle lane: staged reads (I/O before the turn was sent), trigger times, and re-issued prefetches
        bars("oracle", io_spans(sv["s2h_staged"].values()), ORANGE)
        ticks("oracle", [e["t"] for e in sv["ntp"] if e["kind"] == "ntp_due"], GRAY)
        ticks("oracle", [e["t"] for e in sv["ntp"] if e["kind"] == "ntp_stage"], ORANGE)
        refetched = [e["t"] for e in sv["ntp"] if e["kind"] == "ntp_refetch" and e["fields"].get("issued") == "1"]
        if refetched:
            a_evt.plot(refetched, [ypos["oracle"]] * len(refetched), linestyle="none", marker="x", color=BLUE, markersize=6)
        title += "; oracle: | trigger due (gray), staged read (orange), x re-issued prefetch"
    a_evt.set_yticks(range(len(lanes)))
    a_evt.set_yticklabels(lanes, fontsize=8)
    a_evt.set_ylim(-0.7, len(lanes) - 0.3)
    a_evt.invert_yaxis()
    style(a_evt, title)

    # F. cumulative tokens moved per event type
    def cum(xs, ys, label, color, ls="-"):
        if not xs:
            return
        pts = sorted(zip(xs, ys))
        run, out = 0, []
        for x, y in pts:
            run += y
            out.append((x, run))
        a_cum.plot([p[0] for p in out], [p[1] / 1000 for p in out], color=color, linewidth=1.5, linestyle=ls,
                   drawstyle="steps-post", label=f"{label} {run/1000:.0f}K")
    cum([e["t1"] for e in sv["d2h"]], [e["tokens"] for e in sv["d2h"]], "d2h", BLUE)
    h2s_done = [op for op in sv["h2s"].values() if "t_done" in op]
    cum([op["t_done"] for op in h2s_done], [op.get("tokens_done", 0) for op in h2s_done], "h2s", BLUE, "--")
    s2h_done = [op for op in sv["s2h"].values() if "t_end" in op]
    cum([op["t_end"] for op in s2h_done], [op.get("loaded", 0) for op in s2h_done], "s2h", ORANGE, "--")
    cum([e["t1"] for e in sv["h2d"]], [e["tokens"] for e in sv["h2d"]], "h2d", ORANGE)
    cum([e["t"] for e in sv["evict_device"]], [e["tokens"] for e in sv["evict_device"]], "evict_device", AQUA)
    cum([e["t"] for e in sv["evict_host"]], [e["tokens"] for e in sv["evict_host"]], "evict_host", AQUA, "--")
    style(a_cum, "cumulative tokens moved (K tokens; solid = device-side, dashed = storage/host-side)", "K tokens")
    if a_cum.get_legend_handles_labels()[0]:
        a_cum.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    a_cum.set_xlabel("seconds since the client started", fontsize=8, color=MUTED)
    a_cum.set_xlim(0, t_end * 1.02)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def arm_metrics(arm: str, cl: dict, sv: dict) -> dict:
    turns = cl["turns"]
    ret = [t for t in turns if is_returning(t)]
    first = [t for t in turns if is_first(t)]   # turn 0, or a mid-session start's first request
    tiers = collections.Counter(t["tier"] for t in ret)
    pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
    ttft_ret = [t["ttft"] for t in ret]
    return dict(arm=arm, turns=len(turns), conversations=len({t["conv"] for t in turns}),
                wall_s=round(max(t["t0"] + t["lat"] for t in turns), 1) if turns else 0,
                turn0_ttft_p50=round(pct([t["ttft"] for t in first], .5), 2),
                ret_ttft_mean=round(sum(ttft_ret) / len(ttft_ret), 3) if ttft_ret else float("nan"),
                ret_ttft_p50=round(pct(ttft_ret, .5), 3), ret_ttft_p90=round(pct(ttft_ret, .9), 3),
                ret_ttft_p99=round(pct(ttft_ret, .99), 3),
                ret_uncached_mean=round(sum(t["uncached"] for t in ret) / len(ret)) if ret else 0,
                ret_uncached_total=sum(t["uncached"] for t in ret),
                t95_done_s=round(pct([t["t0"] + t["lat"] for t in turns], .95), 1),
                uncached_total=sum(t["uncached"] for t in turns),
                hits_device=tiers["device"], hits_host=tiers["host"], hits_storage=tiers["storage"], cold_returns=tiers["cold"],
                d2h_tokens=sum(e["tokens"] for e in sv["d2h"]), h2d_tokens=sum(e["tokens"] for e in sv["h2d"]),
                h2s_tokens=sum(op.get("tokens_done", 0) for op in sv["h2s"].values()),
                s2h_tokens=sum(op.get("loaded", 0) for op in sv["s2h"].values()),
                evict_device_tokens=sum(e["tokens"] for e in sv["evict_device"]),
                evict_host_tokens=sum(e["tokens"] for e in sv["evict_host"]),
                peak_queue=max((b["queue"] for b in sv["batches"]), default=0))


def first_overflow(sv: dict):
    """Seconds since the client started at which the cached data first outgrew device + host memory: the first host
    eviction (the first device eviction on a server without a host tier, which writes nothing d2h); None if never."""
    return next((e["t"] for e in (sv["evict_host"] if sv["d2h"] else sv["evict_device"]) if e["t"] >= 0), None)


def steady_view(cl: dict, sv: dict, t_from: float) -> tuple:
    """The client and server records cut to the turns sent, and the events stamped, from t_from on."""
    turns = [t for t in cl["turns"] if t["t0"] >= t_from]
    rids = {t["rid"] for t in turns}
    since = lambda xs, key: [x for x in xs if x[key] >= t_from]
    return dict(cl, turns=turns), dict(
        sv, batches=since(sv["batches"], "t"), d2h=since(sv["d2h"], "t0"), h2d=since(sv["h2d"], "t0"),
        h2s={op_id: op for op_id, op in sv["h2s"].items() if op.get("t_submit", -1.0) >= t_from},
        s2h={rid: op for rid, op in sv["s2h"].items() if rid in rids},
        evict_device=since(sv["evict_device"], "t"), evict_host=since(sv["evict_host"], "t"),
        load_back_init=since(sv["load_back_init"], "t"))


def write_steady(results: list, out_dir: str, start: float = None, paired_ref: str = None, paired_max: int = 0) -> None:
    """compare_steady.png / .csv over the turns sent from the latest arm's first_overflow on, so every arm is compared
    over the same span and only once its cached data had outgrown device + host memory; compare_steady_<start>s.* when
    `start` is given."""
    firsts = {arm: first_overflow(sv) for arm, _, sv, _ in results}
    if start is None:
        never = [arm for arm, t in firsts.items() if t is None]
        if never:
            print(f"warning: the cached data of {', '.join(never)} never outgrew device + host memory; "
                  "left out of the steady-state comparison", file=sys.stderr)
        kept = [r for r in results if firsts[r[0]] is not None]
        if not kept:
            print("warning: no arm outgrew device + host memory; compare_steady not written", file=sys.stderr)
            return
        t_from = max(firsts[r[0]] for r in kept)
        why = "once every arm's cached data had outgrown device + host memory (its first host eviction)"
    else:
        kept, t_from, why = results, start, "the start set by --steady-state"
    steady = []
    for arm, cl, sv, _ in kept:
        cl_s, sv_s = steady_view(cl, sv, t_from)
        first = firsts[arm]
        m = dict(arm_metrics(arm, cl_s, sv_s), first_overflow_s=round(first, 1) if first is not None else "",
                 steady_from_s=round(t_from, 1), turns_skipped=len(cl["turns"]) - len(cl_s["turns"]))
        steady.append((arm, cl_s, sv_s, m))
        own = f"{first:.0f} s" if first is not None else "never"
        print(f"{arm} steady state (from {t_from:.0f} s; its own first overflow {own}): {m['turns']} turns kept, "
              f"{m['turns_skipped']} skipped; returning TTFT mean {m['ret_ttft_mean']} p50 {m['ret_ttft_p50']} "
              f"p90 {m['ret_ttft_p90']} s; hits device/host/storage/cold "
              f"{m['hits_device']}/{m['hits_host']}/{m['hits_storage']}/{m['cold_returns']}")
    name = "compare_steady" if start is None else f"compare_steady_{start:g}s"
    if len(steady) > 1:
        draw_compare(steady, os.path.join(out_dir, f"{name}.png"), steady_from=t_from, steady_why=why)
    ref = next((r for r in steady if r[0] == paired_ref), None)
    for r in steady if ref else ():
        if r is not ref:
            draw_paired_cdf(ref, r, os.path.join(out_dir, f"paired_cdf_{r[0]}_{name[len('compare_'):]}.png"),
                            max_pairs=paired_max, steady_from=t_from, steady_why=why)
    with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(steady[0][3].keys()))
        w.writeheader(); w.writerows([r[3] for r in steady])


def arm_colors(arms: list) -> dict:
    """ARM_COLOR per arm; an arm missing from it gets its own color from matplotlib's cycle and a warning."""
    missing = [a for a in arms if a not in ARM_COLOR]
    if missing:
        print(f"warning: no ARM_COLOR entry for {', '.join(missing)}; add one to timeline.py", file=sys.stderr)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {a: ARM_COLOR[a] if a in ARM_COLOR else cycle[missing.index(a) % len(cycle)] for a in arms}


def draw_compare(results: list, out_png: str, steady_from: float = None, steady_why: str = "") -> None:
    """steady_from: the turns were cut to those sent from this time on (seconds since the client started), for steady_why."""
    colors = arm_colors([r[0] for r in results])
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#fcfcfb")
    a_done, a_cdf, a_cdf0, a_tier = axes.flatten()
    for arm, cl, sv, m in results:
        ends = sorted(t["t0"] + t["lat"] for t in cl["turns"])
        a_done.plot(ends, range(1, len(ends) + 1), color=colors[arm], linewidth=1.8, label=arm, drawstyle="steps-post")
    if steady_from is not None:
        a_done.axvline(steady_from, color=MUTED, linestyle="--", linewidth=1)
    style(a_done, "turns completed over time (all conversations)" if steady_from is None
          else f"turns completed over time (turns sent from {steady_from:.0f} s on, dashed)", "turns done")
    a_done.set_xlabel("seconds since the client started", fontsize=8, color=MUTED)
    a_done.legend(fontsize=8, frameon=False, loc="lower right")
    # log x when the arms span more than ~2 decades (a collapsed arm's tail would squash the healthy ones against 0)
    sel_ret, sel_first = is_returning, is_first
    def spans_decades(sel):
        per_arm = [sorted(t["ttft"] for t in cl["turns"] if sel(t)) for _, cl, _, _ in results]
        per_arm = [xs for xs in per_arm if xs]
        if not per_arm:
            return False
        lo = min(xs[len(xs) // 2] for xs in per_arm)
        hi = max(xs[min(len(xs) - 1, int(0.99 * len(xs)))] for xs in per_arm)
        return lo > 0 and hi / lo > 50
    for arm, cl, sv, m in results:
        color = colors[arm]
        for ax, sel, in ((a_cdf, sel_ret), (a_cdf0, sel_first)):
            xs = sorted(max(0.01, t["ttft"]) for t in cl["turns"] if sel(t))
            if xs:
                ax.plot(xs, [(i + 1) / len(xs) for i in range(len(xs))], color=color, linewidth=1.8, label=arm,
                        drawstyle="steps-post")
    for ax, sel in ((a_cdf, sel_ret), (a_cdf0, sel_first)):
        log = spans_decades(sel)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel("seconds (log)" if log else "seconds", fontsize=8, color=MUTED)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(a_cdf, "TTFT of returning turns (ECDF)", "fraction of turns")
    # the start: a closed-loop cohort (arrival_rate 0: all C sessions start at once) or sessions arriving at a rate
    args0 = results[0][1].get("args") or {}
    conc, rate = args0.get("concurrency"), float(args0.get("arrival_rate") or 0)
    live = (f"a cap from {conc} set by the server backlog" if args0.get("cap_mode") == "backlog"
            else f"up to {conc} live sessions" if conc else "uncapped sessions")
    start = (f"{live}, arriving at {rate:g}/s" if rate > 0
             else f"cold prefill under the {conc}-way start" if conc else "cold prefill")
    style(a_cdf0, f"TTFT of first turns (ECDF; {start})", "fraction of turns")
    # tier share of returning turns, one bar per arm, 2px gaps between segments
    for i, (arm, cl, sv, m) in enumerate(results):
        n = max(1, m["hits_device"] + m["hits_host"] + m["hits_storage"] + m["cold_returns"])
        left = 0
        for key, tier in (("hits_device", "device"), ("hits_host", "host"), ("hits_storage", "storage"), ("cold_returns", "cold")):
            w = m[key] / n
            if w > 0:
                a_tier.barh(i, w, left=left, color=TIER_COLOR[tier], height=0.55, edgecolor="#fcfcfb", linewidth=2)
                if w > 0.06:
                    a_tier.text(left + w / 2, i, f"{100*w:.0f}%", ha="center", va="center", fontsize=7, color="#fcfcfb")
            left += w
        a_tier.text(1.01, i, f"uncached/turn {m['ret_uncached_mean']}", fontsize=7, color=MUTED, va="center")
    a_tier.set_yticks(range(len(results)))
    a_tier.set_yticklabels([r[0] for r in results], fontsize=8)
    a_tier.set_xlim(0, 1.0)
    a_tier.invert_yaxis()
    style(a_tier, "where returning turns found their prefix", None)
    a_tier.legend(handles=[Patch(color=TIER_COLOR[k], label=(COLD_LABEL[results[0][1]["tier_rule"]] if k == "cold" else k))
                           for k in ("device", "host", "storage", "cold")],
                  fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4)  # below the bars: any arm count
    if steady_from is not None:
        fig.suptitle(f"Steady state only: turns sent from {steady_from:.0f} s on, {steady_why}", x=0.01, ha="left",
                     fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_paired_cdf(ref: tuple, other: tuple, out_png: str, max_pairs: int = 0, log_x: bool = None,
                    steady_from: float = None, steady_why: str = "") -> None:
    """The reference arm's TTFT ECDF over the turns both arms completed, and at the height of each of its turns the other
    arm's TTFT of the same (user, turn), joined to the curve by a light dashed line: left of the curve the other arm was
    faster on that turn. ref, other: result tuples (arm, cl, sv, m) as main() builds them. max_pairs: draw at most this
    many pairs per panel, every k-th by rank (0: all); the curve always uses every turn. log_x: None picks the scale
    by draw_compare's rule."""
    (ref_arm, ref_cl), (arm, cl) = ref[:2], other[:2]
    colors = arm_colors([ref_arm, arm])
    theirs = {(t["conv"], t["turn"]): t for t in cl["turns"]}
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), gridspec_kw={"width_ratios": [5, 4, 4]})
    fig.patch.set_facecolor("#fcfcfb")
    for sel, panels in ((is_returning, ((axes[0], "returning turns", 0.0),
                                        (axes[1], f"returning turns, the slowest 15 % in {ref_arm}", 0.85))),
                        (is_first, ((axes[2], "first turns", 0.0),))):
        pairs = sorted((max(0.01, t["ttft"]), max(0.01, theirs[(t["conv"], t["turn"])]["ttft"]))
                       for t in ref_cl["turns"] if sel(t) and (t["conv"], t["turn"]) in theirs)
        if not pairs:
            for ax, _, _ in panels:
                ax.set_visible(False)
            continue
        n = len(pairs)
        ys = [(i + 1) / n for i in range(n)]
        mine, others = [a for a, _ in pairs], sorted(b for _, b in pairs)
        lo, hi = min(q(mine, .5), q(others, .5)), max(q(mine, .99), q(others, .99))
        log = log_x if log_x is not None else lo > 0 and hi / lo > 50
        for ax, what, y_lo in panels:
            shown = [i for i in range(n) if ys[i] > y_lo]
            k = max(1, -(-len(shown) // max_pairs)) if max_pairs else 1
            drawn = shown[k - 1::k]
            ax.add_collection(LineCollection([((pairs[i][0], ys[i]), (pairs[i][1], ys[i])) for i in drawn], colors=GRAY,
                                             linewidths=0.6, linestyles=(0, (3, 2)), alpha=0.5, zorder=1))
            ax.scatter([pairs[i][1] for i in drawn], [ys[i] for i in drawn], s=6 if k > 1 else 2, color=colors[arm],
                       alpha=0.75 if k > 1 else 0.5, linewidths=0, zorder=2, label=f"{arm}, the same turn")
            ax.plot(mine, ys, color=colors[ref_arm], linewidth=1.8, drawstyle="steps-post", zorder=3,
                    label=f"{ref_arm} (ECDF)")
            if log:
                ax.set_xscale("log")
            ax.set_xlabel("TTFT, seconds" + (" (log)" if log else ""), fontsize=8, color=MUTED)
            ax.set_ylim(y_lo, 1.0 + (1.0 - y_lo) * 0.01)
            diffs = sorted(pairs[i][1] - pairs[i][0] for i in shown)
            m = len(diffs)
            stats = (f"{m} turns" + (f", one in {k} drawn" if k > 1 else "") + f"\n{arm}\n"
                     f"faster on {100 * sum(d < -0.5 for d in diffs) / m:.0f} %, slower on "
                     f"{100 * sum(d > 0.5 for d in diffs) / m:.0f} % (by over 0.5 s)\n"
                     f"paired difference: mean {sum(diffs) / m:+.2f} s, median {diffs[m // 2]:+.2f} s")
            leg = ax.legend(handles=None if ax is axes[0] else [], title=stats, loc="best", fontsize=8,
                            title_fontsize=7.5, frameon=False, alignment="left", markerscale=2)
            leg.get_title().set_color(MUTED)
            style(ax, f"TTFT of {what}", "fraction of turns" if y_lo == 0 else f"fraction of turns (from {y_lo:g})")
    title = (f"{arm} against {ref_arm}, turn by turn: each dot is one (user, turn) of {arm}, at the height of the same "
             f"turn on the {ref_arm} ECDF, joined to it by a dashed line; left of the curve, {arm} was faster")
    if steady_from is not None:
        title += f"\nsteady state only: turns both arms sent from {steady_from:.0f} s on, {steady_why}"
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_ttft_over_time(results: list, out_png: str, window_s: float) -> list:
    """Mean TTFT of the turns sent in each window_s of client time, per arm, so the arms are compared at the same
    moment rather than turn by turn; below it the turns sent per minute (how many turns each mean covers) and the users
    waiting for admission under the client's cap (a wait TTFT leaves out). Returns the binned rows for the CSV."""
    colors = arm_colors([r[0] for r in results])
    fig, (a_ttft, a_n, a_wait) = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True,
                                              gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.patch.set_facecolor("#fcfcfb")
    rows = []
    for arm, cl, _, _ in results:
        bins = collections.defaultdict(list)
        for t in cl["turns"]:
            bins[int(t["t0"] // window_s)].append(t["ttft"])
        idx = range(max(bins) + 1) if bins else range(0)
        mid = [(i + 0.5) * window_s / 60 for i in idx]
        mean = [sum(bins[i]) / len(bins[i]) if bins[i] else float("nan") for i in idx]
        a_ttft.plot(mid, mean, color=colors[arm], linewidth=1.3, label=arm)
        a_n.plot(mid, [len(bins[i]) * 60 / window_s for i in idx], color=colors[arm], linewidth=1.1)
        caps = cl.get("caps") or []
        a_wait.plot([c["t"] / 60 for c in caps], [c["waiting"] for c in caps], color=colors[arm], linewidth=1.3,
                    drawstyle="steps-post")
        rows += [dict(arm=arm, t_from_s=round(i * window_s, 1), turns=len(bins[i]),
                      ttft_mean_s=round(mean[k], 3) if bins[i] else "") for k, i in enumerate(idx)]
    w = f"{window_s / 60:g}-min" if window_s % 60 == 0 else f"{window_s:g}-s"
    style(a_ttft, f"mean TTFT of the turns sent in each {w} window (all turns)", "seconds")
    a_ttft.legend(fontsize=8, frameon=False, loc="upper right")
    style(a_n, "turns sent per minute", "turns")
    style(a_wait, "users waiting for admission under the client's cap (not in TTFT)", "users")
    a_wait.set_xlabel("minutes since the client started (send time)", fontsize=8, color=MUTED)
    for ax in (a_ttft, a_n, a_wait):
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return rows


def write_csvs(arm: str, cl: dict, sv: dict, out_dir: str) -> None:
    with open(os.path.join(out_dir, f"turns_{arm}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cl["turns"][0].keys()) if cl["turns"] else ["conv"])
        w.writeheader(); w.writerows(cl["turns"])
    rows = []
    for e in sv["d2h"]:
        rows.append(dict(kind="d2h", t0=e["t0"], t1=e["t1"], tokens=e["tokens"], ms=e["ms"], key=""))
    for e in sv["h2d"]:
        rows.append(dict(kind="h2d", t0=e["t0"], t1=e["t1"], tokens=e["tokens"], ms=e["ms"], key=""))
    for op_id, op in sv["h2s"].items():
        rows.append(dict(kind="h2s", t0=op.get("t_submit"), t1=op.get("t_done"), tokens=op.get("tokens_done", op.get("tokens")),
                         ms=op.get("io_ms"), key=op_id))
    for rid, op in sv["s2h"].items():
        rows.append(dict(kind="s2h", t0=op.get("t_start"), t1=op.get("t_end"), tokens=op.get("loaded"), ms=op.get("io_ms"), key=rid))
    for e in sv["evict_device"]:
        rows.append(dict(kind="evict_device", t0=e["t"], t1=e["t"], tokens=e["tokens"], ms="", key=""))
    for e in sv["evict_host"]:
        rows.append(dict(kind="evict_host", t0=e["t"], t1=e["t"], tokens=e["tokens"], ms="", key=""))
    for e in sv["load_back_init"]:
        rows.append(dict(kind="load_back_init", t0=e["t"], t1=e["t"], tokens=e["tokens"], ms="", key=e["rid"]))
    for rid, op in sv["s2h_staged"].items():
        rows.append(dict(kind="s2h_staged", t0=op.get("t_start"), t1=op.get("t_end"), tokens=op.get("loaded"),
                         ms=op.get("io_ms"), key=rid))
    for e in sv["ntp"]:
        rows.append(dict(kind=e["kind"], t0=e["t"], t1=e["t"], tokens=e["fields"].get("tokens", ""), ms="", key=e["rid"]))
    rows.sort(key=lambda r: (r["t0"] if r["t0"] is not None else 1e18))
    with open(os.path.join(out_dir, f"events_{arm}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "t0", "t1", "tokens", "ms", "key"])
        w.writeheader(); w.writerows(rows)


def steady_start(v: str):
    """--steady-state value: "auto" or seconds since the client started."""
    try:
        return v if v == "auto" else float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds since the client started, got {v!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="compare_<t>/manifest.txt: lines '<arm> <run dir name> <client dir name>'")
    ap.add_argument("--run", help="one server run dir")
    ap.add_argument("--client", help="client dir name inside --run (default: the last client_* dir)")
    ap.add_argument("--arm", default=None, help="label for --run (default: from run.txt)")
    ap.add_argument("--out", default=None, help="output dir (default: next to the manifest, or the run dir)")
    ap.add_argument("--tier-rule", choices=("prompt", "private"), default="prompt",
                    help="recompute when under half the prompt is cached (prompt), or when over half of the prompt beyond "
                         "the shared system prompt is uncached (private)")
    ap.add_argument("--steady-state", nargs="?", const="auto", default=None, metavar="SECONDS",
                    type=steady_start,
                    help="also write compare_steady.png / compare_steady.csv over the turns sent once every arm's cached "
                         "data had outgrown device + host memory (the latest first host eviction), or "
                         "compare_steady_<SECONDS>s.png / .csv over the turns sent from SECONDS on (since the client started)")
    ap.add_argument("--paired-cdf", default=None, metavar="REF",
                    help="also write paired_cdf_<arm>.png for every other arm: REF's TTFT ECDF over the turns both arms "
                         "completed, and the arm's TTFT of each same (user, turn) at the same height, joined by a dashed line")
    ap.add_argument("--paired-max", type=int, default=0, metavar="N",
                    help="--paired-cdf: draw at most N pairs per panel, every k-th by rank (default 0: all)")
    ap.add_argument("--ttft-over-time", nargs="?", const=60.0, default=None, type=float, metavar="SECONDS",
                    help="also write ttft_over_time_<SECONDS>s.png / .csv: each arm's mean TTFT of the turns sent in "
                         "each SECONDS of client time (default 60), with turns sent per minute and users waiting")
    args = ap.parse_args()

    jobs = []
    if args.manifest:
        mpath = host_path(args.manifest)
        root = os.path.dirname(os.path.dirname(os.path.abspath(mpath)))
        for line in open(mpath):
            parts = line.split()
            if len(parts) >= 3:
                jobs.append((parts[0], os.path.join(root, parts[1]), os.path.join(root, parts[1], parts[2])))
        out_dir = args.out or os.path.dirname(os.path.abspath(mpath))
    elif args.run:
        run = host_path(args.run)
        client = os.path.join(run, args.client) if args.client else sorted(
            os.path.join(run, d) for d in os.listdir(run) if d.startswith("client_"))[-1]
        arm = args.arm
        if arm is None:
            arm = next((l.split("=", 1)[1].strip() for l in open(os.path.join(run, "run.txt")) if l.startswith("arm=")), "run")
        jobs.append((arm, run, client))
        out_dir = args.out or run
    else:
        ap.error("--manifest or --run is required")
    if args.paired_cdf is not None and args.paired_cdf not in [j[0] for j in jobs]:
        ap.error(f"--paired-cdf {args.paired_cdf}: not one of the arms ({', '.join(j[0] for j in jobs)})")
    if args.ttft_over_time is not None and args.ttft_over_time <= 0:
        ap.error(f"--ttft-over-time {args.ttft_over_time:g}: the window must be positive (seconds)")
    if args.paired_cdf is not None and len(jobs) < 2:
        print("warning: --paired-cdf needs a second arm; no paired figure written", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for arm, run, client in jobs:
        cl = parse_client(os.path.join(client, "client.jsonl"), args.tier_rule)
        sv = parse_server(os.path.join(run, "server.log"), cl["t_start"], {t["rid"]: t["t0"] for t in cl["turns"]})
        m = arm_metrics(arm, cl, sv)
        results.append((arm, cl, sv, m))
        draw_timeline(arm, cl, sv, os.path.join(out_dir, f"timeline_{arm}.png"))
        write_csvs(arm, cl, sv, out_dir)
        print(f"{arm}: {m['turns']} turns / {m['conversations']} conv, wall {m['wall_s']} s; returning TTFT mean {m['ret_ttft_mean']} "
              f"p50 {m['ret_ttft_p50']} p90 {m['ret_ttft_p90']} s; hits device/host/storage/cold "
              f"{m['hits_device']}/{m['hits_host']}/{m['hits_storage']}/{m['cold_returns']}; uncached/turn {m['ret_uncached_mean']}; "
              f"d2h {m['d2h_tokens']} h2d {m['h2d_tokens']} h2s {m['h2s_tokens']} s2h {m['s2h_tokens']} "
              f"evicted dev/host {m['evict_device_tokens']}/{m['evict_host_tokens']} tokens; peak queue {m['peak_queue']}")
    if len(results) > 1:
        draw_compare(results, os.path.join(out_dir, "compare.png"))
    ref = next((r for r in results if r[0] == args.paired_cdf), None)
    for r in results if ref else ():
        if r is not ref:
            draw_paired_cdf(ref, r, os.path.join(out_dir, f"paired_cdf_{r[0]}.png"), max_pairs=args.paired_max)
    if args.ttft_over_time is not None:
        name = os.path.join(out_dir, f"ttft_over_time_{args.ttft_over_time:g}s")
        rows = draw_ttft_over_time(results, name + ".png", args.ttft_over_time)
        with open(name + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "t_from_s", "turns", "ttft_mean_s"])
            w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0][3].keys()))
        w.writeheader(); w.writerows([r[3] for r in results])
    if args.steady_state is not None:
        write_steady(results, out_dir, start=None if args.steady_state == "auto" else args.steady_state,
                     paired_ref=args.paired_cdf, paired_max=args.paired_max)
    print(f"wrote {out_dir}: " + ", ".join(sorted(x for x in os.listdir(out_dir) if x.endswith((".png", ".csv")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
