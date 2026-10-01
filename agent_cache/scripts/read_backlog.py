#!/usr/bin/env python3
"""read_backlog.py --manifest compare_<t>/manifest.txt [--ref compare_<t>/manifest.txt] [--out DIR]

How the SSD read backlog shapes the TTFT of turns restored from the SSD (L3), reconstructed from the server log.
The server runs one storage-query thread and one storage-read thread, both FIFO. Per request it logs
`prefetch_start` (arrival), `s2h_query` (existence query answered; the read is then handed to the read thread),
`s2h_io` (read finished, with its I/O time in ms) and `load_back_init` (admission). A read's start is s2h_io - ms, so
the read thread's schedule, and for every read the I/O still queued ahead of it when its query was answered (the
backlog), follow from these lines. Turns use the client's tier (cached_details.storage > 0 = restored from the SSD).
The head-of-queue time (the request becomes the oldest one waiting, FCFS) is bounded from the admissions of earlier
arrivals: exact for host/SSD loads (load_back_init), and between arrival and first token for the others.
Writes read_backlog.png and read_backlog.csv; --ref adds another campaign's SSD-restored turns as a reference.
"""
import argparse, bisect, calendar, csv, json, os, re, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ARM_COLOR = {"three_tier_to": "#1baf7a", "three_tier_wc": "#2a78d6"}
C_QUERY, C_WAIT, C_IO, C_ADMIT, C_PREFILL = "#9a9891", "#eb6834", "#1baf7a", "#d6b22a", "#2a78d6"
EVT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.(\d{3})\] HICACHE_EVT (\w+) rid=(\S+)(.*)")
BYTES_PER_TOKEN, PAGE = 98304, 64  # bf16 KV of Qwen3-30B-A3B; pages of 64 tokens
BINS = [(0, 0.05), (0.05, 2), (2, 8), (8, 20), (20, 1e9)]  # GB queued ahead


def ts(day, ms):
    # Log timestamps are UTC wall clock; the client's t_start is time.time().
    return calendar.timegm(tuple(map(int, re.split(r"[- :]", day))) + (0, 0, 0)) + int(ms) / 1000.0


def kv(s):
    return dict(x.split("=", 1) for x in s.split() if "=" in x)


def load_arm(results, arm, run, client):
    ev = {}
    reads = []
    for line in open(os.path.join(results, run, "server.log"), errors="replace"):
        m = EVT.match(line)
        if not m:
            continue
        t, kind, rid, rest = ts(m[1], m[2]), m[3], m[4], kv(m[5])
        d = ev.setdefault(rid, {})
        if kind == "s2h_io":
            ms = float(rest["ms"]) / 1000.0
            reads.append((t - ms, t, rid, int(rest["pages"])))
        if kind in ("prefetch_start", "s2h_query", "s2h_io", "load_back_init") and kind not in d:
            d[kind] = (t, rest)
    reads.sort()
    starts = [r[0] for r in reads]
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r.get("kind") == "run")
    head = head_of_queue_times(arm, rows, ev, t0)
    out = []
    for r in rows:
        if r.get("kind") != "turn" or "t_send" not in r or r["turn"] == 0 or not r.get("ttft"):
            continue
        rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
        e = ev.get(rid, {})
        if "s2h_query" not in e or int(e["s2h_query"][1].get("storage_hit", 0)) <= 0 or "s2h_io" not in e:
            continue
        sto = (r.get("cached_details") or {}).get("storage") or 0
        sq = e["s2h_query"][0]
        io_end = e["s2h_io"][0]
        io_ms = float(e["s2h_io"][1]["ms"]) / 1000.0
        io_start = io_end - io_ms
        # I/O still queued or running ahead of this read when its query was answered (single FIFO read thread).
        i = bisect.bisect_left(starts, io_start)
        ahead = [(s, f, p) for s, f, _, p in reads[:i] if f > sq]
        backlog_s = sum(f - max(s, sq) for s, f, _ in ahead)
        # Bytes still to read ahead: the read in service counts by its unread fraction.
        backlog = sum(p * (f - max(s, sq)) / max(f - s, 1e-6) for s, f, p in ahead) * PAGE * BYTES_PER_TOKEN / 1e9
        send = t0 + r["t_send"]
        ps = e["prefetch_start"][0] if "prefetch_start" in e else send
        lb = e["load_back_init"][0] if "load_back_init" in e else None
        out.append(dict(arm=arm, conv=r["conv"], turn=r["turn"], restored=sto > 0, t=r["t_send"], ttft=r["ttft"],
                        backlog=backlog, backlog_s=backlog_s, reads_ahead=len(ahead), pages=int(e["s2h_io"][1]["pages"]),
                        query=max(0.0, sq - ps), wait=max(0.0, io_start - sq), io=io_ms,
                        to_admit=max(0.0, lb - io_end) if lb else None,
                        admit_to_ft=max(0.0, r["ttft"] - (lb - send)) if lb else None, send_to_ps=ps - send,
                        head_lo=head[rid][0] - ps, head_hi=head[rid][1] - ps))
    return out, reads, t0


def head_of_queue_times(arm, rows, ev, t0):
    """Per rid, (earliest, latest) time it became the oldest request in the FCFS waiting queue.

    The head is reached once every earlier arrival has been admitted. Admission is logged only for host/SSD loads
    (load_back_init); any other earlier request is bounded by its arrival (earliest) and its first token (latest).
    """
    turns = [r for r in rows if r.get("kind") == "turn" and "t_send" in r and r.get("ttft")]
    rid_of = lambda r: f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
    offs = [ev[rid_of(r)]["prefetch_start"][0] - (t0 + r["t_send"]) for r in turns if "prefetch_start" in ev.get(rid_of(r), {})]
    off = st.median(offs) if offs else 0.0
    reqs = []
    for r in turns:
        e = ev.get(rid_of(r), {})
        arr = e["prefetch_start"][0] if "prefetch_start" in e else t0 + r["t_send"] + off
        adm = e["load_back_init"][0] if "load_back_init" in e else None
        reqs.append((arr, rid_of(r), adm, t0 + r["t_send"] + r["ttft"]))
    reqs.sort()
    lo = hi = float("-inf")
    head = {}
    for arr, rid, adm, ft in reqs:
        head[rid] = (max(arr, lo), max(arr, hi))
        lo = max(lo, adm if adm is not None else arr)
        hi = max(hi, adm if adm is not None else ft)
    return head


def bandwidth(reads):
    return sum(p for _, _, _, p in reads) * PAGE * BYTES_PER_TOKEN / 1e9 / sum(f - s for s, f, _, _ in reads)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def binned(rows, key):
    res = []
    for lo, hi in BINS:
        b = [x for x in rows if lo <= x["backlog"] < hi]
        res.append((lo, hi, b))
    return res


def draw_read_thread(ax, arm, data, reads_of, t0_of, prefix=""):
    """One row per SSD read in service order, in a window around the arm's deepest backlog."""
    peak = max(data[arm], key=lambda x: x["backlog"])
    t0 = t0_of[arm]
    c = peak["t"] + peak["send_to_ps"] + peak["query"]
    w0, w1 = c - 60, c + 45
    win = [r for r in reads_of[arm] if r[1] - t0 > w0 and r[0] - t0 < w1]
    sq_of = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}": x for x in data[arm]}
    style(ax, f"{prefix}{arm}: the single SSD read thread around its deepest backlog ({peak['backlog']:.1f} GB queued) "
              "— each row is one read, in service order")
    for k, (s, f, rid, pages) in enumerate(win):
        x = sq_of.get(rid)
        if x is not None:
            ax.barh(k, x["wait"], left=s - x["wait"] - t0, color=C_WAIT, height=0.8, alpha=0.55, linewidth=0)
        ax.barh(k, f - s, left=s - t0, color=C_IO, height=0.8, linewidth=0)
        if x is not None and x["restored"]:
            ax.plot(x["t"] + x["ttft"], k, marker="|", color=INK, markersize=6)
    ax.set_xlim(w0, w1)
    ax.set_ylim(len(win), -1)
    ax.set_xlabel("seconds since the client started", fontsize=8.5, color=MUTED)
    ax.set_ylabel("reads (service order)", fontsize=8.5, color=MUTED)
    ax.legend(handles=[Patch(color=C_WAIT, alpha=0.55, label="query answered, waiting for the read thread"),
                       Patch(color=C_IO, label="own read (I/O)"),
                       Line2D([], [], color=INK, marker="|", linestyle="", markersize=8, label="first token of that turn")],
              loc="lower left", fontsize=8, frameon=False)
    return len(win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref", default="", help="another campaign's manifest, plotted as reference points")
    ap.add_argument("--ref-label", default="reference")
    ap.add_argument("--out", default="")
    ap.add_argument("--thread", nargs="*", default=[],
                    help="also write read_backlog_thread_<arm>.png, panel (a) alone, for these arms")
    a = ap.parse_args()
    results = os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    out_dir = a.out or os.path.dirname(os.path.abspath(a.manifest))
    arms = [l.split() for l in open(a.manifest) if l.strip() and l.split()[0] in ARM_COLOR]
    data, reads_of, t0_of = {}, {}, {}
    for arm, run, client in arms:
        data[arm], reads_of[arm], t0_of[arm] = load_arm(results, arm, run, client)
    ref = []
    if a.ref:
        rres = os.path.dirname(os.path.dirname(os.path.abspath(a.ref)))
        for arm, run, client in (l.split() for l in open(a.ref) if l.strip()):
            if arm in ARM_COLOR:
                ref += [x for x in load_arm(rres, arm, run, client)[0] if x["restored"]]

    with open(os.path.join(out_dir, "read_backlog.csv"), "w", newline="") as f:
        allrows = [x for arm in data for x in data[arm]]
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)

    for arm in data:
        rs = [x for x in data[arm] if x["restored"]]
        nr = [x for x in data[arm] if not x["restored"]]
        print(f"== {arm}: {len(rs)} SSD-restored turns, {len(nr)} with an SSD hit that ended recomputed/host")
        for lo, hi, b in binned(rs, "backlog"):
            if b:
                m = lambda k: st.mean(x[k] for x in b if x[k] is not None)
                print(f"   backlog {lo:>4}-{hi if hi < 1e9 else 'inf':>4} GB  n {len(b):4d}  TTFT mean {m('ttft'):6.2f} p50 "
                      f"{st.median(x['ttft'] for x in b):6.2f}  query {m('query'):5.2f} wait {m('wait'):6.2f} io {m('io'):5.2f} "
                      f"io->admit {m('to_admit'):5.2f} admit->ft {m('admit_to_ft'):5.2f}")
        tot = sum(x["ttft"] for x in rs)
        print(f"   share of restored-turn TTFT: wait for the read thread {sum(x['wait'] for x in rs) / tot:.0%}, own read "
              f"{sum(x['io'] for x in rs) / tot:.0%}; read bandwidth {bandwidth(reads_of[arm]):.2f} GB/s; corr(TTFT, GB ahead) "
              f"{st.correlation([x['ttft'] for x in rs], [x['backlog'] for x in rs]):.3f}")
        sp = [x["send_to_ps"] for x in data[arm]]
        print(f"   clock check send->prefetch_start p50 {st.median(sp) * 1000:.1f} ms")
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
        hw = [x["head_lo"] for x in rs]
        hr = [x["head_lo"] - (x["query"] + x["wait"]) for x in rs]
        print(f"   arrival -> head of the scheduler queue p50 {q(hw, .5):.2f} p90 {q(hw, .9):.2f} s; head - own read start "
              f"p50 {q(hr, .5):+.2f} p90 {q(hr, .9):+.2f} s; head uncertain by > 0.3 s for "
              f"{sum(1 for x in rs if x['head_hi'] - x['head_lo'] > 0.3) / len(rs):.0%}")

    fig = plt.figure(figsize=(17, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.22)

    # (a) the read thread around the busiest moment of the first arm
    draw_read_thread(fig.add_subplot(gs[0, :]), arms[0][0], data, reads_of, t0_of, "(a) ")

    # (b) TTFT vs backlog
    ax = fig.add_subplot(gs[1, 0])
    style(ax, "(b) TTFT of SSD-restored turns vs the SSD reads queued ahead of them")
    for arm in data:
        rs = [x for x in data[arm] if x["restored"]]
        ax.scatter([x["backlog"] for x in rs], [x["ttft"] for x in rs], s=7, alpha=0.35, color=ARM_COLOR[arm], linewidths=0,
                   label=f"{arm} (n {len(rs)})")
        nr = [x for x in data[arm] if not x["restored"]]
        if nr:
            ax.scatter([x["backlog"] for x in nr], [x["ttft"] for x in nr], s=14, marker="x", alpha=0.6,
                       color=ARM_COLOR[arm], linewidths=0.8, label=f"{arm}: SSD hit, but recomputed (n {len(nr)})")
        xs, ys = [], []
        for lo, hi, b in binned(rs, "backlog"):
            if len(b) >= 5:
                xs.append(st.median(x["backlog"] for x in b)); ys.append(st.median(x["ttft"] for x in b))
        ax.plot(xs, ys, color=ARM_COLOR[arm], linewidth=2)
    if ref:
        ax.scatter([x["backlog"] for x in ref], [x["ttft"] for x in ref], s=9, alpha=0.5, color=MUTED, linewidths=0,
                   label=f"{a.ref_label} (n {len(ref)})")
    lim = max(x["backlog"] for arm in data for x in data[arm]) * 1.05
    bw = st.mean(bandwidth(reads_of[arm]) for arm in data)
    ax.plot([0, lim], [0, lim / bw], color=MUTED, linestyle=":", linewidth=1)
    ax.text(lim * 0.6, lim / bw * 0.5, f"queued GB / {bw:.2f} GB/s\n(time to drain the queue)", color=MUTED, fontsize=7.5)
    ax.set_xlabel("GB of SSD reads queued ahead when the turn's query was answered", fontsize=8.5, color=MUTED)
    ax.set_ylabel("TTFT (s)", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (c) where the time goes, per backlog bin
    ax = fig.add_subplot(gs[1, 1])
    style(ax, "(c) mean TTFT of SSD-restored turns by phase, per GB of reads queued ahead")
    ys, labels, k = [], [], 0.0
    for arm in data:
        rs = [x for x in data[arm] if x["restored"] and x["to_admit"] is not None]
        for lo, hi, b in binned(rs, "backlog"):
            if not b:
                continue
            m = lambda key: st.mean(x[key] for x in b)
            left = 0.0
            for val, col in ((m("query"), C_QUERY), (m("wait"), C_WAIT), (m("io"), C_IO), (m("to_admit"), C_ADMIT),
                             (m("admit_to_ft"), C_PREFILL)):
                ax.barh(k, val, left=left, color=col, height=0.7, edgecolor=SURF, linewidth=1)
                left += val
            ax.text(left + 0.15, k, f"{left:.1f} s (n {len(b)})", va="center", fontsize=7.5, color=MUTED)
            ys.append(k)
            labels.append(f"{arm} · " + ("nothing ahead" if lo == 0 else f"{lo:g}-{hi:g} GB" if hi < 1e9 else f"≥{lo:g} GB"))
            k += 1
        k += 0.5
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds (mean per turn)", fontsize=8.5, color=MUTED)
    ax.legend(handles=[Patch(color=C_QUERY, label="arrival -> query answered"),
                       Patch(color=C_WAIT, label="waiting for the read thread"),
                       Patch(color=C_IO, label="own read (I/O)"),
                       Patch(color=C_ADMIT, label="read done -> admission"),
                       Patch(color=C_PREFILL, label="admission -> first token")],
              fontsize=7.5, frameon=False, loc="upper right")
    fig.text(0.01, 0.005, "Reconstructed from HICACHE_EVT s2h_query / s2h_io (ms) / load_back_init in server.log: one FIFO read "
             "thread, so a read starts at s2h_io - ms and the backlog is the data earlier reads still had to read when its "
             "query was answered (64-token pages x 98,304 B/token). TTFT from the client.", fontsize=7.5, color=MUTED)
    path = os.path.join(out_dir, "read_backlog.png")
    fig.savefig(path, dpi=130, facecolor=SURF, bbox_inches="tight")
    print("wrote", path)
    for arm in a.thread:
        fig = plt.figure(figsize=(17, 5.4), facecolor=SURF)
        draw_read_thread(fig.add_subplot(1, 1, 1), arm, data, reads_of, t0_of)
        path = os.path.join(out_dir, f"read_backlog_thread_{arm}.png")
        fig.savefig(path, dpi=130, facecolor=SURF, bbox_inches="tight")
        print("wrote", path)


if __name__ == "__main__":
    main()
