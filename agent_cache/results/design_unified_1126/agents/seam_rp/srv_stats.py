"""Server-side counts behind the Retain / Preempt seams, from one arm's event extract (ext/<arm>.evt).

Usage: python3 -I -B srv_stats.py ext/c34_ec_full.evt.gz   (extract.sh builds the extracts)
Prints: write-hold intervals (P2), write latency and queue (R2), short writes (R3), ended registrations joined
to their tail node's SSD write (R1), queued turns that lost pages again (P1), what a read asked eviction for
against what it was short of (bug B6), cleaner passes and failed page reads (R4).
All times are the server log's UTC stamps; minutes count from the first h2s_submit line.
"""
import gzip
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
KV = re.compile(r"(\w+)=([^\s]+)")


def open_text(path):
    """The extract, gzipped or not."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, errors="replace")


def ts(line):
    m = TS.match(line)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def q(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


def main(path):
    submit = {}          # op -> (t, node, tokens)
    done_lat = []        # (t_submit, latency_s, tokens)
    short = []           # (op, submitted, completed)
    pending_at_submit = []
    holds = []           # (t_on, t_off, pending_on, parked_on)
    hold_on = None
    reevict = []         # (t, rid, evicted, refetches, capped)
    ended_nodes = {}     # node -> (t, rid)
    node_write = defaultdict(int)   # node -> tokens submitted for SSD write
    node_write_t = {}
    evict = []           # (t, freed, requested)
    last_evict = None
    dispatch_after_evict = []   # (requested, pages*64, launched)
    cleanup = []
    read_failed = []
    gate = defaultdict(int)
    io_ms = []           # (t, pages, ms) per h2s_io
    skipped = 0
    pages_written = 0
    t0 = None
    pending_now = 0
    n_dispatch = n_dispatch_evict = 0
    for line in open_text(path):
        t = ts(line)
        if t is None:
            continue
        if "HICACHE_EVT " in line:
            ev, _, rest = line.split("HICACHE_EVT ", 1)[1].partition(" ")
            f = dict(KV.findall(rest))
        elif "NIXL L3 cleanup" in line:
            m = re.search(r"deleted (\d+) groups / (\d+) files \(([\d.]+) GiB\) in ([\d.]+)s", line)
            cleanup.append((t, int(m.group(1)), float(m.group(3)), float(m.group(4))))
            continue
        elif "failed to retrieve page" in line:
            read_failed.append(t)
            continue
        else:
            continue
        if ev == "h2s_submit":
            if t0 is None:
                t0 = t
            op = int(f["op"])
            node = int(f.get("node", -op - 1))     # campaign 32's lines carry no node and no pending
            submit[op] = (t, node, int(f["tokens"]))
            node_write[node] += int(f["tokens"])
            node_write_t.setdefault(node, t)
            pending_now += int(f["tokens"])
            pending_at_submit.append((t, int(f.get("pending", pending_now))))
        elif ev == "h2s_done":
            op = int(f["op"])
            if op in submit:
                ts_, _, tok = submit[op]
                pending_now -= tok
                done_lat.append((ts_, t - ts_, tok))
                if int(f["tokens"]) != tok:
                    short.append((op, tok, int(f["tokens"])))
        elif ev == "h2s_io":
            io_ms.append((t, int(f["pages"]), float(f["ms"])))
            pages_written += int(f["pages"])
            if "skipped" in f:
                skipped += int(f["skipped"])
        elif ev == "h2s_group":
            skipped += int(f["skipped"])
        elif ev == "srs_write_hold":
            if f["state"] == "on":
                hold_on = (t, int(f["pending_write_tokens"]), int(f["parked"]), int(f["parked_pages"]))
            elif hold_on is not None:
                holds.append((hold_on[0], t, hold_on[1], hold_on[2], hold_on[3], int(f["parked"])))
                hold_on = None
        elif ev == "ntp_reevict":
            reevict.append((t, f["rid"], int(f["evicted"]), int(f["refetches"]), int(f["capped"])))
        elif ev == "nue_register":
            if f.get("next_use_ms") == "ended":
                ended_nodes[int(f["node"])] = (t, f["rid"])
        elif ev == "evict_host":
            last_evict = (t, int(f["tokens"]), int(f["requested"]))
            evict.append(last_evict)
        elif ev == "srs_dispatch":
            n_dispatch += 1
            # The read's own eviction is logged in the same pass, just before its dispatch line.
            if last_evict is not None and 0 <= t - last_evict[0] < 0.05:
                n_dispatch_evict += 1
                dispatch_after_evict.append((last_evict[2], int(f["pages"]) * 64, int(f.get("launched", 1)), last_evict[1]))
            last_evict = None
        elif ev == "ntp_gate":
            gate[f["decision"]] += 1

    def minute(t):
        return (t - t0) / 60.0

    print(f"file {path}")
    print(f"first h2s_submit at {datetime.fromtimestamp(t0, timezone.utc):%H:%M:%S}Z; cleaner passes at minutes "
          + ", ".join(f"{minute(c[0]):.1f} ({c[1]} groups, {c[2]:.0f} GiB, {c[3]:.1f} s)" for c in cleanup))
    print(f"failed page reads: {len(read_failed)} at minutes " + ", ".join(f"{minute(t):.1f}" for t in read_failed))

    # ---- R2 / R3: the write path ----
    lat = [x[1] for x in done_lat]
    tok = sum(x[2] for x in done_lat)
    print(f"[R2] SSD writes: {len(done_lat)} ops, {tok / 1e6:.2f} M tokens; submit-to-done p50 {q(lat, .5):.2f} s, "
          f"p90 {q(lat, .9):.2f}, p99 {q(lat, .99):.2f}, max {max(lat):.2f}")
    for lo in range(0, 120, 30):
        w = [x[1] for x in done_lat if lo <= minute(x[0]) < lo + 30]
        p = [x[1] for x in pending_at_submit if lo <= minute(x[0]) < lo + 30]
        wt = sum(x[2] for x in done_lat if lo <= minute(x[0]) < lo + 30)
        print(f"     min {lo:3d}-{lo + 30:3d}: ops {len(w):5d}, tokens {wt / 1e6:5.2f} M ({wt / 1800:6.0f}/s), "
              f"submit-to-done p50 {q(w, .5):5.2f} p90 {q(w, .9):5.2f} max {max(w) if w else 0:5.2f} s; "
              f"pending at submit p50 {q(p, .5):6.0f} p90 {q(p, .9):6.0f} max {max(p) if p else 0:6.0f}")
    over = [x for x in done_lat if x[1] > 5.0]
    print(f"     ops waiting over 5 s: {len(over)} ({sum(x[2] for x in over) / 1e6:.2f} M tokens); "
          f"over 2 s: {sum(1 for x in done_lat if x[1] > 2.0)}")
    print(f"     pages written {pages_written}, skipped as stored {skipped} ({skipped / max(pages_written, 1):.1%})")
    print(f"[R3] writes acked short of their pages: {len(short)}" + (f" e.g. {short[:5]}" if short else ""))

    # ---- R1: ended registrations ----
    ended_tok = sum(node_write.get(n, 0) for n in ended_nodes)
    with_write = sum(1 for n in ended_nodes if n in node_write)
    print(f"[R1] ended registrations (distinct tail nodes): {len(ended_nodes)}; tail nodes with an SSD write: {with_write}, "
          f"{ended_tok / 1e6:.3f} M tokens = {ended_tok / max(tok, 1):.2%} of written tokens "
          f"(the tail node holds the last turn's output only; the prompt's new pages are another node)")
    # Was the tail's write submitted before or after the end was registered?
    before = sum(1 for n, (t, _) in ended_nodes.items() if n in node_write_t and node_write_t[n] <= t)
    print(f"     tail writes submitted before the end was registered: {before} of {with_write}")

    # ---- P2: the write hold ----
    dur = [h[1] - h[0] for h in holds]
    print(f"[P2] write holds: {len(holds)}; held {sum(dur):.1f} s in total ({sum(dur) / 7200:.2%} of 2 h); "
          f"duration p50 {q(dur, .5):.2f} s, p90 {q(dur, .9):.2f}, max {max(dur) if dur else 0:.2f}; "
          f"pending at on p50 {q([h[2] for h in holds], .5)} max {max([h[2] for h in holds]) if holds else 0}; "
          f"timed reads parked at on p50 {q([h[3] for h in holds], .5)}, p90 {q([h[3] for h in holds], .9)}, "
          f"max {max([h[3] for h in holds]) if holds else 0}")
    for lo in range(0, 120, 30):
        d = [h[1] - h[0] for h in holds if lo <= minute(h[0]) < lo + 30]
        print(f"     min {lo:3d}-{lo + 30:3d}: holds {len(d):3d}, held {sum(d):6.1f} s, parked reads at on "
              f"{[h[3] for h in holds if lo <= minute(h[0]) < lo + 30]}")

    # ---- P1: queued turns that lost pages again; what reads asked eviction for ----
    capped = [r for r in reevict if r[4]]
    rids = {r[1] for r in reevict}
    print(f"[P1] ntp_reevict events: {len(reevict)} on {len(rids)} requests, capped {len(capped)} "
          f"({sum(r[2] for r in capped) / 1e6:.2f} M tokens left to recompute at the cap); "
          f"evicted tokens per event p50 {q([r[2] for r in reevict], .5)}, p90 {q([r[2] for r in reevict], .9)}")
    for lo in range(0, 120, 30):
        w = [r for r in reevict if lo <= minute(r[0]) < lo + 30]
        print(f"     min {lo:3d}-{lo + 30:3d}: events {len(w):4d}, requests {len({r[1] for r in w}):4d}, capped "
              f"{sum(1 for r in w if r[4]):4d} ({sum(r[2] for r in w if r[4]) / 1e3:7.0f} K tokens)")
    print(f"     arrival gate decisions: {dict(gate)}")
    ev_tok = sum(e[1] for e in evict)
    short_ev = [e for e in evict if e[1] < e[2]]
    print(f"     evict_host calls that freed anything: {len(evict)}, {ev_tok / 1e6:.2f} M tokens; freed less than asked: {len(short_ev)}")
    full = [d for d in dispatch_after_evict if d[0] == d[1]]
    print(f"[B6] read dispatches: {n_dispatch}; with an eviction in the same pass: {n_dispatch_evict}; of those the read asked "
          f"eviction for its whole size: {len(full)} ({len(full) / max(len(dispatch_after_evict), 1):.1%}); "
          f"tokens freed by those calls {sum(d[3] for d in full) / 1e6:.2f} M for reads of {sum(d[1] for d in full) / 1e6:.2f} M")


if __name__ == "__main__":
    main(sys.argv[1])
