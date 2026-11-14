"""Replay one run's SSD reads through the single FIFO read thread with every read issued earlier
(agent_cache/ORACLE_V2_PLAN.md section 1.5 and 4.8).

  python3 read_shift_replay.py --run results/<boot dir> [--client client_<...>] --window 7200 7320
                               [--shifts 0 5 10 20 40] [--alloc enqueue|dequeue] [--cap-pages N]

A read is issued at (its lookup hit minus the shift) but never before its plan existed (the previous turn's end,
`ntp_plan`), keeps its measured service time, and holds host pages from issue until its logged admission under the
prefetch cap (default: half the host pool, parsed from server.log). Reads are served FIFO by issue time. The table
reports, for the reads that ended inside the window, how much earlier they finish than at shift 0, the net issue
shift after cap waits, and the SSD's position at the window start (pages read ahead of the real run, queued, parked).
`--alloc dequeue` charges host pages only when a read reaches the thread, the ORACLE_V2_PLAN lever c.
"""
import argparse
import collections
import datetime as dt
import heapq
import json
import os
import re

KV_RE = re.compile(r"(\w+)=(-?[\w./-]+)")
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")


def parse_ts(line: str) -> float:
    return (
        dt.datetime.strptime(TS_RE.match(line)[1], "%Y-%m-%d %H:%M:%S.%f")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )


def load_reads(run: str, client: str) -> tuple[list, int]:
    with open(os.path.join(run, client, "client.jsonl")) as f:
        t_start = next(json.loads(l) for l in f if '"kind": "run"' in l)["t_start"]
    reads, hits, plan_t, admit_t, host_tokens = [], collections.defaultdict(list), {}, {}, None
    for line in open(os.path.join(run, "server.log"), errors="replace"):
        if host_tokens is None and "host pool: " in line:
            host_tokens = int(re.search(r"host pool: (\d+) tokens", line)[1])
        if "HICACHE_EVT" not in line:
            continue
        t = parse_ts(line) - t_start
        kv = dict(KV_RE.findall(line))
        if "HICACHE_EVT s2h_query" in line:
            if int(kv.get("storage_hit", 0)) > 0:
                hits[kv["rid"]].append(t)
        elif "HICACHE_EVT s2h_io" in line:
            ms = float(kv.get("ms", 0))
            if ms > 0:
                reads.append(dict(rid=kv["rid"], io_start=t - ms / 1000, io_end=t, svc=ms / 1000, pages=int(kv.get("pages", 0))))
        elif "HICACHE_EVT ntp_plan" in line:
            plan_t[kv["rid"]] = t
        elif "HICACHE_EVT ntp_admit" in line:
            admit_t[kv["rid"]] = t
    reads.sort(key=lambda r: r["io_start"])
    for r in reads:
        before = [t for t in hits.get(r["rid"], []) if t <= r["io_start"]]
        r["enq"] = max(before) if before else r["io_start"]
        r["earliest"] = plan_t.get(r["rid"], r["enq"])
        r["adm"] = admit_t.get(r["rid"], r["io_end"])
    return reads, host_tokens or 0


def simulate(reads: list, shift: float, cap_pages: int, alloc_at_dequeue: bool) -> dict:
    order = sorted(range(len(reads)), key=lambda i: (max(reads[i]["earliest"], reads[i]["enq"] - shift), reads[i]["io_start"]))
    server_free, occ, held = 0.0, 0, []
    out = {}
    for i in order:
        r = reads[i]
        t = max(r["earliest"], r["enq"] - shift)
        if alloc_at_dequeue:
            t = max(t, server_free)
        while held and held[0][0] <= t:
            occ -= heapq.heappop(held)[1]
        cap_wait_from = t
        while occ + r["pages"] > cap_pages and held:
            t = max(t, held[0][0])
            occ -= heapq.heappop(held)[1]
        occ += r["pages"]
        start = max(t, server_free)
        end = start + r["svc"]
        server_free = end
        heapq.heappush(held, (max(r["adm"], end), r["pages"]))
        out[i] = dict(issue=t, cap_wait=t - cap_wait_from, start=start, end=end)
    return out


def p50(values: list) -> float:
    values = sorted(values)
    return values[len(values) // 2] if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--client", default=None)
    ap.add_argument("--window", type=float, nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--shifts", type=float, nargs="+", default=[0, 5, 10, 20, 40])
    ap.add_argument("--alloc", choices=["enqueue", "dequeue"], default="enqueue")
    ap.add_argument("--cap-pages", type=int, default=0, help="0 = half the host pool at 64 tokens per page")
    a = ap.parse_args()
    client = a.client or sorted(d for d in os.listdir(a.run) if d.startswith("client_"))[-1]
    reads, host_tokens = load_reads(a.run, client)
    cap = a.cap_pages or host_tokens // 2 // 64
    w0, w1 = a.window
    win = [i for i, r in enumerate(reads) if w0 <= r["io_end"] <= w1]
    base = simulate(reads, 0.0, cap, a.alloc == "dequeue")
    err = sum(abs(base[i]["end"] - reads[i]["io_end"]) for i in win) / max(1, len(win))
    idle = 0.0
    cur = 0.0
    for r in reads:
        if r["io_start"] > cur and r["io_start"] <= w0:
            idle += min(r["io_start"], w0) - cur
        cur = max(cur, r["io_end"])
    print(f"{len(reads)} reads, cap {cap:,} pages, alloc at {a.alloc}; window {w0:.0f}-{w1:.0f} s holds {len(win)} reads; "
          f"shift-0 replay vs real end: mean |diff| {err:.2f} s; SSD idle before the window {idle:.0f} s ({idle / w0:.0%})")
    real_done = sum(r["pages"] for r in reads if r["io_end"] <= w0)
    print(f"{'shift':>6} {'end gain p50':>13} {'net issue shift p50':>20} {'cap wait p50':>13} {'pages ahead at window start':>28} {'queued':>8} {'parked':>8}")
    for shift in a.shifts:
        sim = simulate(reads, shift, cap, a.alloc == "dequeue")
        gain = p50([base[i]["end"] - sim[i]["end"] for i in win])
        net = p50([reads[i]["enq"] - sim[i]["issue"] for i in win])
        cw = p50([sim[i]["cap_wait"] for i in win])
        done = sum(reads[i]["pages"] for i, s in sim.items() if s["end"] <= w0)
        queued = sum(reads[i]["pages"] for i, s in sim.items() if s["issue"] <= w0 < s["start"])
        parked = sum(reads[i]["pages"] for i, s in sim.items() if s["end"] <= w0 < reads[i]["adm"])
        print(f"{shift:>6.0f} {gain:>13.1f} {net:>20.1f} {cw:>13.1f} {done - real_done:>28,} {queued:>8,} {parked:>8,}")


if __name__ == "__main__":
    main()
