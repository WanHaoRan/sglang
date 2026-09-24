"""Timeline and comparison plots for agent-replay runs (RUNBOOK 6.6 events + 4.6 client JSONL).

Joins one boot's server.log (millisecond timestamps, HICACHE_EVT lines, Prefill/Decode batch lines)
with one client run's client.jsonl (per-turn send/first-token/end, tier split) on a common clock:
seconds since the client started (client `run.t_start` is epoch; server stamps are UTC wall clock).

  python3 timeline.py --manifest results/compare_<t>/manifest.txt        # every arm + a comparison figure
  python3 timeline.py --run results/<boot dir> [--client client_<...>]    # one arm

Outputs next to the manifest (or in the run dir): timeline_<arm>.png, events_<arm>.csv, turns_<arm>.csv,
compare.png, compare.csv. Colors: blue = writes (d2h, h2s) / device hit, orange = restores (h2d, s2h) / host hit,
aqua = evictions / storage hit, gray = cold/recompute (less than half the prompt cached); the three hues are the validated first three slots of the
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
from matplotlib.patches import Patch  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9891", "#0b0b0b", "#52514e"
GREEN = "#2e7d32"  # three_tier_wc, distinct from the AQUA of three_tier_to
TIER_COLOR = {"cold": GRAY, "device": BLUE, "host": ORANGE, "storage": AQUA}
ARM_COLOR = {"hbm_lru": GRAY, "hbm_host": ORANGE, "three_tier": AQUA,
             "three_tier_to": AQUA, "three_tier_wc": GREEN}   # campaign 6: the two prefetch policies
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d{3})?)\]\s*(.*)$")
KV_RE = re.compile(r"(\w+)=(-?[\w./-]+)")


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
def parse_client(path: str) -> dict:
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
        # a turn that finds less than half its prompt in any tier lost its own prefix (only the shared system prompt hit):
        # count it as cold/recompute, not as a device hit
        if cached < 0.5 * prompt:
            tier = "cold"
        else:
            tier = "storage" if storage > 0 else "host" if host > 0 else "device"
        turns.append(dict(conv=r["conv"], turn=r["turn"], t0=r["t_send"], ttft=r.get("ttft") or 0.0,
                          lat=r["latency"], tier=tier, prompt=r.get("prompt_tokens", 0), cached=cached,
                          uncached=r.get("prompt_tokens", 0) - cached, host=host, storage=storage,
                          gap=r.get("gap_slept", 0.0), rid=f"{run['args'].get('tag') or 'replay'}-c{r['conv']}-t{r['turn']}"))
    summary = next((r for r in rows if r["kind"] == "summary"), None)
    return dict(t_start=t_start, turns=turns, summary=summary, args=run["args"])


def parse_server(path: str, t_start: float) -> dict:
    """Everything relative to t_start (seconds). Submit/done pairs are matched FIFO for d2h/h2d
    (the ack queues are FIFO), by op id for h2s, by rid for s2h."""
    batches, evd, evh, lbi = [], [], [], []
    d2h_sub, h2d_sub = collections.deque(), collections.deque()
    d2h, h2d = [], []
    h2s = collections.defaultdict(dict)
    s2h = collections.defaultdict(dict)
    posts = []
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
                s2h[kv["rid"]].update(t_start=t, tokens_req=toks)
            elif ev == "s2h_query":
                s2h[kv["rid"]].update(t_query=t, storage_hit=int(kv.get("storage_hit", 0)))
            elif ev == "s2h_io":
                s2h[kv["rid"]].update(t_io_end=t, io_ms=float(kv.get("ms", 0)), tokens_io=toks)
            elif ev == "evict_device":
                evd.append(dict(t=t, tokens=toks, requested=int(kv.get("requested", 0))))
            elif ev == "evict_host":
                evh.append(dict(t=t, tokens=toks))
            elif ev == "load_back_init":
                lbi.append(dict(t=t, rid=kv.get("rid"), tokens=toks, host_hit=int(kv.get("host_hit", 0))))
        elif "HiCache prefetch" in msg:
            kv = dict(KV_RE.findall(msg))
            if "req" in kv:
                s2h[kv["req"]].update(t_end=t, completed=int(kv.get("completed", 0)), loaded=int(kv.get("loaded", 0)),
                                      dropped="dropped" in msg)
        elif '"POST /v1/chat/completions' in msg:
            posts.append(t)
    return dict(batches=batches, d2h=d2h, h2d=h2d, h2s=dict(h2s), s2h=dict(s2h), evict_device=evd, evict_host=evh,
                load_back_init=lbi, posts=posts)


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
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw=dict(height_ratios=[4.2, 1.2, 1.2, 2.2, 1.6], hspace=0.28))
    fig.patch.set_facecolor("#fcfcfb")
    a_req, a_pool, a_queue, a_evt, a_cum = axes

    # A. request lanes: one row per conversation; prefill/queue segment thick, decode segment thin
    convs = sorted({t["conv"] for t in turns})
    for t in turns:
        y = convs.index(t["conv"])
        c = TIER_COLOR[t["tier"]]
        a_req.plot([t["t0"], t["t0"] + t["ttft"]], [y, y], color=c, linewidth=4, solid_capstyle="butt")
        a_req.plot([t["t0"] + t["ttft"], t["t0"] + t["lat"]], [y, y], color=c, linewidth=1.2, alpha=0.7)
    step = max(1, len(convs) // 32)  # at most ~32 labels: 128 conversations would otherwise print as a black bar
    a_req.set_yticks(range(0, len(convs), step))
    a_req.set_yticklabels([f"c{c}" for c in convs[::step]], fontsize=7)
    a_req.invert_yaxis()
    style(a_req, f"{arm}: requests per conversation (thick = send to first token, thin = decode); color = where the prefix was found",
          "conversation")
    a_req.legend(handles=[Patch(color=TIER_COLOR[k], label=("cold/recompute" if k == "cold" else k)) for k in ("cold", "device", "host", "storage")],
                 loc="upper right", fontsize=8, frameon=False, ncol=4)

    # B. device pool usage (one measure per axis)
    bt = [b["t"] for b in sv["batches"]]
    a_pool.plot(bt, [b["usage"] for b in sv["batches"]], color=BLUE, linewidth=1.5)
    a_pool.set_ylim(0, 1.05)
    style(a_pool, "device KV pool usage (fraction of --max-total-tokens)", "usage")

    # C. queue depth and running requests
    a_queue.plot(bt, [b["queue"] for b in sv["batches"]], color=ORANGE, linewidth=1.5, label="queued (#queue-req)")
    a_queue.plot(bt, [b["running"] for b in sv["batches"]], color=BLUE, linewidth=1.5, label="running (#running-req)")
    style(a_queue, "scheduler queue and running requests", "requests")
    a_queue.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # D. tier events: one lane per type; bars where a duration is known, ticks otherwise (size ~ tokens)
    lanes = ["d2h", "h2s", "s2h", "h2d", "evict_device", "evict_host"]
    ypos = {k: i for i, k in enumerate(lanes)}
    def bar(y, t0, t1, color):
        a_evt.plot([t0, max(t1, t0 + 0.25)], [y, y], color=color, linewidth=6, solid_capstyle="butt", alpha=0.85)
    def copy_bar(lane, e, color):
        t0 = e["t1"] - e["ms"] / 1000 if e["ms"] > 0 else e["t0"]
        bar(ypos[lane], t0, e["t1"], color)
    for e in sv["d2h"]:
        copy_bar("d2h", e, BLUE)
    for op in sv["h2s"].values():
        if "t_io_end" in op:
            bar(ypos["h2s"], op["t_io_end"] - op["io_ms"] / 1000, op["t_io_end"], BLUE)
        elif "t_submit" in op:
            a_evt.plot([op["t_submit"]], [ypos["h2s"]], marker="|", color=BLUE, markersize=8)
    for rid, op in sv["s2h"].items():
        if "t_io_end" in op:
            bar(ypos["s2h"], op["t_io_end"] - op["io_ms"] / 1000, op["t_io_end"], ORANGE)
        elif "t_start" in op:
            a_evt.plot([op["t_start"]], [ypos["s2h"]], marker="|", color=ORANGE, markersize=8, alpha=0.6)
    for e in sv["h2d"]:
        copy_bar("h2d", e, ORANGE)
    for e in sv["evict_device"]:
        a_evt.scatter([e["t"]], [ypos["evict_device"]], s=6 + e["tokens"] / 200, color=AQUA, alpha=0.7, linewidths=0)
    for e in sv["evict_host"]:
        a_evt.scatter([e["t"]], [ypos["evict_host"]], s=6 + e["tokens"] / 200, color=AQUA, alpha=0.7, linewidths=0)
    a_evt.set_yticks(range(len(lanes)))
    a_evt.set_yticklabels(lanes, fontsize=8)
    a_evt.set_ylim(-0.7, len(lanes) - 0.3)
    a_evt.invert_yaxis()
    style(a_evt, "HiCache tier events (bars = measured copy/IO time ending at completion; dots = evictions, area ~ tokens)")

    # E. cumulative tokens moved per event type
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
    ret = [t for t in turns if t["turn"] > 0]
    first = [t for t in turns if t["turn"] == 0]
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


def draw_compare(results: list, out_png: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#fcfcfb")
    a_done, a_cdf, a_cdf0, a_tier = axes.flatten()
    for arm, cl, sv, m in results:
        ends = sorted(t["t0"] + t["lat"] for t in cl["turns"])
        a_done.plot(ends, range(1, len(ends) + 1), color=ARM_COLOR.get(arm, INK), linewidth=1.8, label=arm, drawstyle="steps-post")
    style(a_done, "turns completed over time (all conversations)", "turns done")
    a_done.set_xlabel("seconds since the client started", fontsize=8, color=MUTED)
    a_done.legend(fontsize=8, frameon=False, loc="lower right")
    # log x when the arms span more than ~2 decades (a collapsed arm's tail would squash the healthy ones against 0)
    sel_ret, sel_first = (lambda t: t["turn"] > 0), (lambda t: t["turn"] == 0)
    def spans_decades(sel):
        per_arm = [sorted(t["ttft"] for t in cl["turns"] if sel(t)) for _, cl, _, _ in results]
        per_arm = [xs for xs in per_arm if xs]
        if not per_arm:
            return False
        lo = min(xs[len(xs) // 2] for xs in per_arm)
        hi = max(xs[min(len(xs) - 1, int(0.99 * len(xs)))] for xs in per_arm)
        return lo > 0 and hi / lo > 50
    for arm, cl, sv, m in results:
        color = ARM_COLOR.get(arm, INK)
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
    start = (f"up to {conc} live sessions, arriving at {rate:g}/s" if rate > 0
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
    a_tier.legend(handles=[Patch(color=TIER_COLOR[k], label=("cold/recompute" if k == "cold" else k)) for k in ("device", "host", "storage", "cold")],
                  fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4)  # below the bars: any arm count
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


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
    rows.sort(key=lambda r: (r["t0"] if r["t0"] is not None else 1e18))
    with open(os.path.join(out_dir, f"events_{arm}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "t0", "t1", "tokens", "ms", "key"])
        w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="compare_<t>/manifest.txt: lines '<arm> <run dir name> <client dir name>'")
    ap.add_argument("--run", help="one server run dir")
    ap.add_argument("--client", help="client dir name inside --run (default: the last client_* dir)")
    ap.add_argument("--arm", default=None, help="label for --run (default: from run.txt)")
    ap.add_argument("--out", default=None, help="output dir (default: next to the manifest, or the run dir)")
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
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for arm, run, client in jobs:
        cl = parse_client(os.path.join(client, "client.jsonl"))
        sv = parse_server(os.path.join(run, "server.log"), cl["t_start"])
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
    with open(os.path.join(out_dir, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0][3].keys()))
        w.writeheader(); w.writerows([r[3] for r in results])
    print(f"wrote {out_dir}: " + ", ".join(sorted(x for x in os.listdir(out_dir) if x.endswith((".png", ".csv")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
