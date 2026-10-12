#!/usr/bin/env python3
"""delay_components.py --out DIR <label>=<manifest> [<label>=<manifest> ...]

Per returning turn, split the client-measured latency into: queue wait (send -> admission), of which waiting on SSD
reads; admission -> first token ("prefill", includes the layer-wise H2D load-back); decode (first token -> last token).
Admission is measured where the server log has a `queue_duration=` time-stats line (entry_time + queue_duration, so the
queue wait also covers HTTP, tokenization and the hand-off before the scheduler's queue) or `HICACHE_EVT load_back_init
rid=...` (host and storage hits); elsewhere (device hits, recomputes on servers without
--enable-request-time-stats-logging) the admission -> first-token time is estimated from the measured turns of the same
tier and load regime (server queue depth at arrival >= 10 = saturated), pooled over every campaign given, and the queue
wait is the remainder. With no measured turn to calibrate from, 0.05 s + uncached / P is used.
SSD wait (l3_read): each read (s2h_io) counts from the last lookup before it (prefetch_start: the turn's own, a #39283
re-query at admission, or the oracle's staging lookup when the read was still in flight at the send) to its landing, and
the union of those spans inside send -> admission is the column: time queued with a read outstanding (parked or in
flight), whether or not the turn was also waiting for its place or for GPU room. A read the oracle (ORACLE_PLAN.md)
staged under the next turn's rid and that landed before the send is outside the turn's latency; its time is the
staged_l3 column (ntp_stage -> its s2h_io). Server logs are stamped in UTC, client sends in epoch seconds; the logs are
parsed as UTC so the two line up on any machine. Arms that the config.txt next to a manifest lists but the manifest
lacks are reported, not plotted.
Writes turn_components_<label>_<arm>.csv and delay_components.png (one row per label: per arm, then per arm x tier).
"""
import argparse, csv, datetime as dt, json, os, re, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from vs_baseline import is_first   # turn 0, or the first request of a user who started mid-session

P_TOK_S = 4903.0     # single-stream prefill rate measured on this box (results/probe_moe), estimate only
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C_QUEUE, C_L3, C_PREFILL, C_DECODE = "#eb6834", "#1baf7a", "#2a78d6", "#9a9891"
EVT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] HICACHE_EVT (\w+) rid=(\S+)")
BATCH = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] (?:Prefill|Decode) batch.*?#running-req: (\d+).*?#queue-req: (\d+)")
SATURATED_QUEUE = 10   # queue depth at arrival from which a turn is treated as running under saturation
import bisect
TSL = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\].*?rid=(\S+?)[,)].*?queue_duration=([0-9.]+)(ms|s), forward_duration=([0-9.]+)(ms|s)(?:, entry_time=([0-9.]+))?")
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()   # server logs are UTC

def load_arm(results, arm, run, client):
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r.get("kind") == "run")
    ev = {}; qt, qd = [], []
    for line in open(os.path.join(results, run, "server.log"), errors="replace"):
        m = EVT.match(line)
        if m:
            d = ev.setdefault(m[3], {}); d.setdefault(m[2], ts(m[1]))
            if m[2] in ("prefetch_start", "s2h_io"): d.setdefault("_" + m[2], []).append(ts(m[1]))
            continue
        m = BATCH.match(line)
        if m:
            qt.append(ts(m[1])); qd.append(int(m[3])); continue
        if "queue_duration=" in line:
            m = TSL.match(line)
            if m:
                d = ev.setdefault(m[2], {}); d["queue_duration"] = float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
                if m[7]: d["admitted"] = float(m[7]) + d["queue_duration"]   # entry_time: epoch seconds
    out = []
    for r in rows:
        if r.get("kind") != "turn" or "t_send" not in r or is_first(r) or not r.get("ttft"): continue
        rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"; e = ev.get(rid, {})
        prompt = int(r.get("prompt_tokens") or 0); cached = int(r.get("cached_tokens") or 0); cd = r.get("cached_details") or {}
        tier = "cold" if cached < 0.5 * prompt else ("storage" if (cd.get("storage") or 0) > 0 else "host" if (cd.get("host") or 0) > 0 else "device")
        send = t0 + r["t_send"]; ttft = r["ttft"]; lat = r["latency"]; unc = max(0, prompt - cached)
        i = bisect.bisect_right(qt, send) - 1; qdepth = qd[i] if i >= 0 else 0
        if "queue_duration" in e:
            queue = min(ttft, max(e["queue_duration"], e.get("admitted", 0.0) - send)); how = "time_stats"
        elif "load_back_init" in e:
            queue = min(ttft, max(0.0, e["load_back_init"] - send)); how = "load_back_init"
        else:
            queue = None; how = "estimated"
        # Read spans in seconds from the send; finish() clips them to send -> admission once the queue wait is known.
        looks = e.get("_prefetch_start", [])
        spans = [(max([t for t in looks if t <= land], default=land) - send, land - send) for land in e.get("_s2h_io", ()) if land > send]
        stage = e.get("ntp_stage")
        staged_io = [t for t in e.get("_s2h_io", ()) if stage is not None and t >= stage]
        staged_l3 = staged_io[0] - stage if stage is not None and stage < send and staged_io else 0.0
        out.append(dict(arm=arm, conv=r["conv"], turn=r["turn"], tier=tier, t_send=round(r["t_send"], 3), prompt=prompt, uncached=unc,
                        ttft=round(ttft, 4), latency=round(lat, 4), queue=queue, l3_read=None, prefill=None,
                        decode=round(max(0.0, lat - ttft), 4), how=how, qdepth=qdepth, staged_l3=round(staged_l3, 4), _spans=spans))
    return out

def calibrate(all_turns):
    """Mean admission -> first-token time per (tier, regime) over measured turns; fallback per tier over both regimes."""
    tab = {}
    for t in all_turns:
        if t["how"] == "estimated": continue
        key = (t["tier"], t["qdepth"] >= SATURATED_QUEUE); tab.setdefault(key, []).append(t["ttft"] - t["queue"])
    return {k: (st.mean(v), len(v)) for k, v in tab.items()}

def covered(spans, hi):
    """Seconds of [0, hi] inside the union of the (start, end) spans."""
    tot = cur = 0.0
    for s, e in sorted(spans):
        s, e = max(s, cur), min(e, hi)
        if e > s: tot += e - s; cur = e
    return tot

def finish(turns, cal):
    for t in turns:
        if t["queue"] is None:
            key = (t["tier"], t["qdepth"] >= SATURATED_QUEUE)
            if key in cal and cal[key][1] >= 20: pre = cal[key][0]
            elif any(k[0] == t["tier"] and cal[k][1] >= 20 for k in cal): pre = st.mean(cal[k][0] for k in cal if k[0] == t["tier"] and cal[k][1] >= 20)
            else: pre = 0.05 + t["uncached"] / P_TOK_S
            pre = min(t["ttft"], pre); t["queue"] = round(max(0.0, t["ttft"] - pre), 4)
        else:
            t["queue"] = round(t["queue"], 4)
        t["prefill"] = round(max(0.0, t["ttft"] - t["queue"]), 4); t["l3_read"] = round(covered(t.pop("_spans"), t["queue"]), 4)
    return turns

def means(ts_):
    n = len(ts_); m = lambda k: sum(t[k] for t in ts_) / n
    return dict(n=n, queue=m("queue"), l3=m("l3_read"), prefill=m("prefill"), decode=m("decode"),
                measured=sum(1 for t in ts_ if t["how"] != "estimated") / n, q50=st.median(t["queue"] for t in ts_))

def draw_rows(ax, rows, title):
    for i, (lab, m) in enumerate(rows):
        left = 0.0
        for val, c in ((m["queue"] - m["l3"], C_QUEUE), (m["l3"], C_L3), (m["prefill"], C_PREFILL), (m["decode"], C_DECODE)):
            if val > 0: ax.barh(i, val, left=left, color=c, height=0.62, edgecolor=SURF, linewidth=1.2); left += val
        tot = left; qs = 100 * m["queue"] / tot if tot else 0
        note = f"{tot:.1f} s, queue {qs:.0f} %" + ("" if m["measured"] > 0.99 else f" ({100*m['measured']:.0f} % measured)")
        ax.text(tot + ax.get_xlim()[1] * 0.01 if ax.get_xlim()[1] > 1 else tot, i, note, va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5); ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8); ax.tick_params(colors=MUTED, labelsize=8); ax.set_xlabel("seconds per returning turn (mean)", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, max(sum((m["queue"], m["prefill"], m["decode"])) for _, m in rows) * 1.42)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("sets", nargs="+"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); fig_rows = []
    loaded = []
    for spec in a.sets:
        label, manifest = spec.rsplit("=", 1); results = os.path.dirname(os.path.dirname(os.path.abspath(manifest)))
        lines = [l.split() for l in open(manifest) if l.strip()]
        cfg = os.path.join(os.path.dirname(os.path.abspath(manifest)), "config.txt")
        m = re.search(r"\barms=(.*?)\s+\w+=", open(cfg).read()) if os.path.exists(cfg) else None
        missing = [x for x in (m[1].split() if m else []) if x not in [l[0] for l in lines]]
        if missing: print(f"{label}: {cfg} lists {' '.join(missing)}, which {manifest} has no line for; not plotted")
        for arm, run, client in lines:
            loaded.append((label, arm, load_arm(results, arm, run, client)))
    cal = calibrate([t for _, _, T in loaded for t in T])
    for (tier, sat), (m, n) in sorted(cal.items()): print(f"calibration: {tier:8s} {'saturated' if sat else 'light    '} admission->first token mean {m:6.2f} s (n {n})")
    for label, arm, T in loaded: finish(T, cal)
    for label in dict.fromkeys(l for l, _, _ in loaded):
        per_arm, per_tier = [], []
        for _, arm, T in [x for x in loaded if x[0] == label]:
            with open(os.path.join(a.out, f"turn_components_{label}_{arm}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(T[0].keys())); w.writeheader(); w.writerows(T)
            m = means(T); per_arm.append((arm, m))
            print(f"{label} {arm:14s} n {m['n']:5d}  mean queue {m['queue']:7.2f} (of which SSD wait {m['l3']:5.2f})  prefill {m['prefill']:5.2f}  decode {m['decode']:5.2f} s;  queue p50 {m['q50']:6.2f};  measured {100*m['measured']:.0f} %")
            for tier in ("device", "host", "storage", "cold"):
                tt = [t for t in T if t["tier"] == tier]
                if len(tt) >= 20:
                    mt = means(tt); per_tier.append((f"{arm} · {tier} (n={len(tt)})", mt))
                    print(f"    {tier:8s} n {len(tt):5d}  queue {mt['queue']:7.2f} (SSD {mt['l3']:5.2f})  prefill {mt['prefill']:5.2f}  decode {mt['decode']:5.2f}  measured {100*mt['measured']:.0f} %")
        fig_rows.append((label, per_arm, per_tier))
    n = len(fig_rows); fig_h = 1.2 + sum(0.9 + 0.34 * max(len(pa), len(pt)) for _, pa, pt in fig_rows); fig = plt.figure(figsize=(16, fig_h))
    fig.patch.set_facecolor(SURF); gs = fig.add_gridspec(n, 2, width_ratios=[1, 1.35], height_ratios=[max(len(pa), len(pt)) + 2 for _, pa, pt in fig_rows], hspace=0.45, wspace=0.55)
    for i, (label, pa, pt) in enumerate(fig_rows):
        axA = fig.add_subplot(gs[i, 0]); axA.set_facecolor(SURF); draw_rows(axA, pa, f"{label}: where a returning turn's time goes, per arm")
        axB = fig.add_subplot(gs[i, 1]); axB.set_facecolor(SURF); draw_rows(axB, pt, f"{label}: per arm and tier (tiers with >= 20 turns)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_QUEUE, label="queue wait (send -> admission) with no SSD read outstanding"), Patch(color=C_L3, label="queued with an SSD read outstanding (lookup -> data in host memory)"),
                        Patch(color=C_PREFILL, label="admission -> first token (prefill + layer-wise H2D)"), Patch(color=C_DECODE, label="decode (first -> last token)")],
               loc="upper left", bbox_to_anchor=(0.01, 0.995), fontsize=8.5, frameon=False, ncol=2)
    fig.text(0.01, 0.003, "Admission time measured from ReqTimeStats (entry_time + queue_duration) or HICACHE_EVT load_back_init (host/storage hits) where logged; otherwise the admission -> first-token time is the mean of the measured turns of the same tier and load regime (server queue depth at arrival >= 10 = saturated), pooled over all campaigns shown, and the queue wait is the remainder ('% measured' = share of turns with a measured admission). A read the oracle staged that landed before the send is outside the turn's latency. TTFT = queue + first-token segment.",
             fontsize=7.5, color=MUTED, wrap=True)
    fig.subplots_adjust(top=0.90, bottom=1.0 / fig_h, left=0.16, right=0.98)   # 1 in below the axes for the x labels and the note
    out = os.path.join(a.out, "delay_components.png"); fig.savefig(out, dpi=130, facecolor=SURF); print("wrote", out)

def load_calibrated(sets):
    """For other tools: [(label, arm, turns)] with the same estimates as the figure."""
    loaded = []
    for label, manifest in sets:
        results = os.path.dirname(os.path.dirname(os.path.abspath(manifest)))
        for arm, run, client in (l.split() for l in open(manifest) if l.strip()):
            loaded.append((label, arm, load_arm(results, arm, run, client)))
    cal = calibrate([t for _, _, T in loaded for t in T])
    for _, _, T in loaded: finish(T, cal)
    return loaded

if __name__ == "__main__": main()
