"""Q3: reads against writes on the one SSD, campaign 34 (fix 3's hold, the device, the draft's rule).

  python3 -I -B q3_rw.py <campaign> <label> [<label> ...]

A. fix 3: each hold (timed reads stop at >= 50 K pending write tokens, resume at <= 25 K): how long, which timed
   reads sat parked across it, how long the read thread was idle inside it.
B. what a hold protected: the census nearest its start (free, evictable, held by writes), the host allocation of
   the next seconds, and whether any eviction in the run came back short of what was asked.
C. the draft's rule on the same instants: hold only if free + evictable clean room < the allocation expected while
   the pending writes drain.
D. the device: ms per page of reads by how much of the read overlapped a write, and of writes by read overlap;
   iostat by window.
E. per 10 minutes: tokens read and written a second, thread busy shares, parked waits, write submit-to-done.
"""
from __future__ import annotations

import bisect
import calendar
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import BASE, PAGE, RESULTS, load_trees  # noqa: E402
from model import load  # noqa: E402


def q(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(p * len(ys)))]


def merge(iv: list) -> list:
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Busy:
    """Merged busy intervals with prefix sums, for overlap queries."""

    def __init__(self, iv: list):
        self.iv = merge(iv)
        self.starts = [a for a, _ in self.iv]
        self.cum = [0.0]
        for a, b in self.iv:
            self.cum.append(self.cum[-1] + b - a)

    def upto(self, t: float) -> float:
        i = bisect.bisect_right(self.starts, t)
        if i == 0:
            return 0.0
        a, b = self.iv[i - 1]
        return self.cum[i - 1] + min(b, t) - a

    def overlap(self, a: float, b: float) -> float:
        return max(0.0, self.upto(b) - self.upto(a))


def iostat(campaign: str, t_start: float) -> list:
    path = f"{RESULTS}/{campaign}/iostat_vdc.log"
    if not os.path.exists(path):
        return []
    rows, t = [], None
    ts = re.compile(r"^(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d) (AM|PM)")
    with open(path) as f:
        for line in f:
            g = ts.match(line)
            if g:
                mo, d, y, h, mi, s, ap = g.groups()
                h = int(h) % 12 + (12 if ap == "PM" else 0)
                t = calendar.timegm((int(y), int(mo), int(d), h, int(mi), int(s))) - t_start
            elif line.startswith("vdc") and t is not None:
                p = line.split()
                rows.append((t, float(p[2]) / 1024.0, float(p[8]) / 1024.0, float(p[-1])))   # rMB/s, wMB/s, util
    return [r for r in rows if -60 <= r[0] <= 7300]


def analyse(campaign: str, label: str, trees) -> dict:
    m = load(campaign, label, trees)
    res = {"campaign": campaign, "label": label}
    print(f"\n=== {campaign} {label}")
    # read and write busy intervals
    read_iv = [(r["t0"], r["t1"]) for r in m.reads]
    write_iv = []
    grouped_first = {g["op"] for g in m.groups}
    for o in m.ops.values():
        if o["t_io"] is None:
            continue
        if o["group"] is None:
            write_iv.append((o["t_io"] - o["io_ms"] / 1000.0, o["t_io"]))
    for g in m.groups:
        write_iv.append((g["t"] - g["ms"] / 1000.0, g["t"]))
    rb, wb = Busy(read_iv), Busy(write_iv)
    # ----- A. the holds -----
    holds = []
    on = None
    for t, state, pending, parked, ppages in m.holds:
        if state == "on":
            on = (t, pending, parked, ppages)
        elif on is not None:
            holds.append({"t0": on[0], "t1": t, "pending0": on[1], "pending1": pending, "parked0": on[2],
                          "parked1": parked, "ppages0": on[3], "ppages1": ppages})
            on = None
    timed = [d for d in m.dispatches if d["park"] is not None and d["park"]["deadline"] != "now"]
    now_reads = [d for d in m.dispatches if d["park"] is not None and d["park"]["deadline"] == "now"]
    hb = Busy([(h["t0"], h["t1"]) for h in holds])
    tot_overlap = 0.0
    n_hit = 0
    pages_hit = 0
    released = 0
    ov_list = []
    for d in timed:
        ov = hb.overlap(d["park"]["t"], d["t"])
        if ov > 0:
            n_hit += 1
            tot_overlap += ov
            pages_hit += d["pages"]
            ov_list.append(ov)
            # dispatched within 100 ms of a hold's end: the hold was what it waited for last
            i = bisect.bisect_right([h["t1"] for h in holds], d["t"])
            if i > 0 and 0 <= d["t"] - holds[i - 1]["t1"] < 0.1:
                released += 1
    idle_in_hold = sum((h["t1"] - h["t0"]) - rb.overlap(h["t0"], h["t1"]) for h in holds)
    idle_parked = sum((h["t1"] - h["t0"]) - rb.overlap(h["t0"], h["t1"]) for h in holds
                      if h["parked0"] > 0 or h["parked1"] > 0)
    # every timed read parked through a hold waits the read thread's idle time in it once more
    added = sum(((h["t1"] - h["t0"]) - rb.overlap(h["t0"], h["t1"])) * max(h["parked0"], h["parked1"]) for h in holds)
    dur = [h["t1"] - h["t0"] for h in holds]
    tot_timed_wait = sum(d["t"] - d["park"]["t"] for d in timed)
    print(f"A. fix 3 holds: {len(holds)}, {sum(dur):.1f} s in all (median {q(dur, .5):.1f} s, max {max(dur, default=0):.1f} s); "
          f"with timed reads parked at start or end: {sum(1 for h in holds if h['parked0'] or h['parked1'])}")
    print(f"   timed reads dispatched in the run: {len(timed)} (parked {tot_timed_wait:.0f} s in all, median "
          f"{q([d['t'] - d['park']['t'] for d in timed], .5):.1f} s); 'now' reads: {len(now_reads)}")
    print(f"   timed reads parked across a hold: {n_hit} ({pages_hit * PAGE / 1e6:.2f} M tokens); their time inside "
          f"holds {tot_overlap:.1f} s in all (median {q(ov_list, .5):.2f} s, max {max(ov_list, default=0):.2f} s) = "
          f"{tot_overlap / max(tot_timed_wait, 1e-9):.2%} of all timed-read parking; dispatched within 0.1 s of a "
          f"hold's end: {released}")
    print(f"   read thread idle inside holds: {idle_in_hold:.1f} s ({idle_parked:.1f} s in holds with timed reads "
          f"parked) of {sum(dur):.1f} s; at 17.4 K tokens/s that is at most {idle_parked * 17.4 / 1e3:.2f} M tokens of "
          f"reads put off; idle time x reads parked, summed over holds: {added:.0f} reader-seconds")
    res["holds"] = {"n": len(holds), "seconds": sum(dur), "reads_hit": n_hit, "overlap_s": tot_overlap,
                    "released_by_hold_end": released, "idle_s": idle_in_hold, "idle_parked_s": idle_parked,
                    "timed_reads": len(timed), "timed_parked_s": tot_timed_wait, "added_reader_s": added}
    # ----- B. what the hold protected -----
    cz = m.census
    ct = [c["t"] for c in cz]
    eh = m.evict_host
    eht = [e[0] for e in eh]
    d2t = [x[0] for x in m.d2h]
    launched = sorted((d["t"], d["pages"] * PAGE) for d in m.dispatches if d["launched"])
    lt = [x[0] for x in launched]

    def alloc(a: float, b: float) -> tuple:
        i0, i1 = bisect.bisect_left(d2t, a), bisect.bisect_right(d2t, b)
        new_kv = sum(x[2] for x in m.d2h[i0:i1])
        j0, j1 = bisect.bisect_left(lt, a), bisect.bisect_right(lt, b)
        rd = sum(x[1] for x in launched[j0:j1])
        k0, k1 = bisect.bisect_left(eht, a), bisect.bisect_right(eht, b)
        ev = sum(x[1] for x in eh[k0:k1])
        return new_kv, rd, ev

    short = [(t, tok, reqd) for t, tok, reqd in eh if tok < reqd]
    refused = sum(c.get("refused", 0) for c in cz)
    cut = sum(c.get("cut", 0) for c in cz)
    bref = sum(c.get("backup_refused", 0) for c in cz)
    print(f"B. evictions that freed less than asked: {len(short)} of {len(eh)}"
          + (f" (first at {short[0][0]:.0f} s)" if short else "")
          + f"; census counts over the run: reads refused room {refused}, cut {cut}, GPU-to-host copies refused {bref}")
    rows = []
    for h in holds:
        i = bisect.bisect_right(ct, h["t0"]) - 1
        if i < 0:
            continue
        c = cz[i]
        room = c["free"] + c["evict_leaf"] + c["evict_casc"]
        d = h["t1"] - h["t0"]
        new_kv, rd, ev = alloc(h["t0"], h["t1"])
        new10, rd10, ev10 = alloc(h["t0"], h["t0"] + 10.0)
        # the draft's rule: allocation expected while the pending writes drain, at the rates of the 30 s before
        nk30, rd30, _ = alloc(h["t0"] - 30.0, h["t0"])
        alloc_rate = (nk30 + rd30) / 30.0
        w0, w1 = h["t0"] - 30.0, h["t0"]
        done_tok = sum(o["tokens"] for o in m.ops.values() if o["t_done"] is not None and w0 <= o["t_done"] < w1)
        busy_w = wb.overlap(w0, w1)
        w_rate = done_tok / busy_w if busy_w > 0 else float("nan")       # tokens per second of write-thread work
        drain = h["pending0"] / w_rate if w_rate == w_rate and w_rate > 0 else float("nan")
        rows.append({"t0": h["t0"], "dur": d, "pending0": h["pending0"], "census_age": h["t0"] - c["t"],
                     "free": c["free"], "leaf": c["evict_leaf"], "casc": c["evict_casc"], "h_write": c["h_write"],
                     "room": room, "alloc_in_hold": new_kv + rd, "alloc_10s": new10 + rd10, "evicted_10s": ev10,
                     "alloc_rate": alloc_rate, "w_rate": w_rate, "drain_s": drain,
                     "expected_alloc": alloc_rate * drain if drain == drain else float("nan"),
                     "parked0": h["parked0"], "ppages0": h["ppages0"]})
    if rows:
        print("   per hold: start, length, pending at start, parked timed reads (pages) | census before it: free, "
              "leaf, cascade, held by writes only | host allocation inside the hold and in the next 10 s | draft "
              "rule: allocation expected over the drain, room / that")
        for r in rows:
            ratio = r["room"] / r["expected_alloc"] if r["expected_alloc"] == r["expected_alloc"] and r["expected_alloc"] > 0 else float("nan")
            print(f"   {r['t0']:7.0f} s {r['dur']:5.1f} s {r['pending0'] / 1e3:5.0f}K  {r['parked0']:2d} ({r['ppages0']:5d}) | "
                  f"{r['free'] / 1e3:5.0f}K {r['leaf'] / 1e3:5.0f}K {r['casc'] / 1e3:5.0f}K {r['h_write'] / 1e3:5.0f}K | "
                  f"{r['alloc_in_hold'] / 1e3:5.0f}K {r['alloc_10s'] / 1e3:5.0f}K | drain {r['drain_s']:5.1f} s x "
                  f"{r['alloc_rate'] / 1e3:5.1f}K/s = {r['expected_alloc'] / 1e3:6.0f}K; room/alloc {ratio:6.1f}")
        ratios = [r["room"] / r["expected_alloc"] for r in rows if r["expected_alloc"] == r["expected_alloc"] and r["expected_alloc"] > 0]
        fl = [(r["free"] + r["leaf"]) / r["expected_alloc"] for r in rows if r["expected_alloc"] == r["expected_alloc"] and r["expected_alloc"] > 0]
        print(f"C. draft rule at the {len(rows)} holds: free + evictable room over the allocation expected while the "
              f"writes drain: min {min(ratios):.1f}, median {q(ratios, .5):.1f}; holds where the rule would also hold "
              f"(ratio < 1): {sum(1 for x in ratios if x < 1)}; with free + leaf only: min {min(fl):.2f}, median "
              f"{q(fl, .5):.2f}, below 1: {sum(1 for x in fl if x < 1)}")
        res["rule"] = {"holds": len(rows), "ratio_min": min(ratios), "ratio_median": q(ratios, .5),
                       "rule_would_hold": sum(1 for x in ratios if x < 1),
                       "freeleaf_min": min(fl), "freeleaf_below1": sum(1 for x in fl if x < 1)}
    # the draft's rule at every census line
    if cz:
        n_rule = n_rule_fl = n_lines = 0
        worst = None
        for c in cz:
            if c["t"] < 60 or c["t"] > 7200:
                continue
            n_lines += 1
            nk30, rd30, _ = alloc(c["t"] - 30.0, c["t"])
            alloc_rate = (nk30 + rd30) / 30.0
            w0, w1 = c["t"] - 30.0, c["t"]
            done_tok = sum(o["tokens"] for o in m.ops.values() if o["t_done"] is not None and w0 <= o["t_done"] < w1)
            busy_w = wb.overlap(w0, w1)
            if busy_w <= 0 or done_tok <= 0:
                continue
            drain = c["pending_write"] / (done_tok / busy_w)
            exp = alloc_rate * drain
            room = c["free"] + c["evict_leaf"] + c["evict_casc"]
            if exp > 0:
                if room < exp:
                    n_rule += 1
                if c["free"] + c["evict_leaf"] < exp:
                    n_rule_fl += 1
                if worst is None or room / exp < worst[0]:
                    worst = (room / exp, c["t"], room, exp, c["pending_write"], drain, alloc_rate)
        print(f"   at the {n_lines} census lines: rule holds (room < expected allocation over the drain) at {n_rule}; "
              f"with free + leaf only at {n_rule_fl}; tightest line: room/alloc {worst[0]:.1f} at {worst[1]:.0f} s "
              f"(room {worst[2] / 1e3:.0f}K, expected {worst[3] / 1e3:.0f}K, pending {worst[4] / 1e3:.0f}K, drain "
              f"{worst[5]:.1f} s, allocation {worst[6] / 1e3:.1f}K/s)" if worst else "   no census line with writes")
        res["rule_census"] = {"lines": n_lines, "holds": n_rule, "holds_free_leaf": n_rule_fl,
                              "tightest": worst}
    # ----- D. the device -----
    print("D. ms per page by overlap with the other direction:")
    bins = [(0.0, 0.0, "none"), (0.0, 0.5, "under half"), (0.5, 1.01, "half or more")]
    out_d = {}
    for name, items, other in (("reads", [(r["t0"], r["t1"], r["pages"]) for r in m.reads if r["pages"] >= 16], wb),
                               ("writes", [(g["t"] - g["ms"] / 1000.0, g["t"], g["pages"] - g["skipped"]) for g in m.groups]
                                + [(o["t_io"] - o["io_ms"] / 1000.0, o["t_io"], o["io_pages"] - (o["skipped"] or 0))
                                   for o in m.ops.values() if o["group"] is None and o["t_io"] is not None], rb)):
        agg = collections.defaultdict(lambda: [0, 0.0, 0])
        for a, b, pages in items:
            if pages <= 0 or b <= a:
                continue
            fr = other.overlap(a, b) / (b - a)
            key = "none" if fr <= 0.001 else ("under half" if fr < 0.5 else "half or more")
            agg[key][0] += pages
            agg[key][1] += (b - a) * 1000.0
            agg[key][2] += 1
        line = []
        for _a, _b, key in bins:
            pg, ms, n = agg[key]
            line.append(f"{key}: {ms / max(pg, 1):.2f} ms/page ({n} ops, {pg * PAGE / 1e6:.1f} M tokens)")
            out_d[f"{name}|{key}"] = (ms / max(pg, 1), n, pg * PAGE)
        print(f"   {name} (ops of 16 pages or more for reads; pages actually written for writes): " + "; ".join(line))
    res["device"] = out_d
    io = iostat(campaign, m.t_start)
    # ----- E. per 10 minutes -----
    print("E. per 10 minutes: tokens/s read and written (physical: skipped pages out), thread busy shares, "
          "timed reads' parked wait p50/p90, write submit-to-done p50/p90, pending writes max, holds; iostat MB/s "
          "read, written, util")
    rows_e = []
    for w in range(12):
        a, b = w * 600.0, (w + 1) * 600.0
        rtok = sum(r["pages"] for r in m.reads if a <= r["t1"] < b) * PAGE
        wtok = sum(o["tokens"] for o in m.ops.values() if o["t_done"] is not None and a <= o["t_done"] < b)
        skipped = (sum(g["skipped"] for g in m.groups if a <= g["t"] < b)
                   + sum(o["skipped"] or 0 for o in m.ops.values()
                         if o["group"] is None and o["t_io"] is not None and a <= o["t_io"] < b)) * PAGE
        waits = [d["t"] - d["park"]["t"] for d in timed if a <= d["t"] < b]
        s2d = [o["t_done"] - o["t_sub"] for o in m.ops.values() if o["t_done"] is not None and a <= o["t_sub"] < b]
        pend = max((o["pending"] for o in m.ops.values() if a <= o["t_sub"] < b), default=0)
        nh = sum(1 for h in holds if a <= h["t0"] < b)
        sel = [r for r in io if a <= r[0] < b]
        rmb = sum(r[1] for r in sel) / max(len(sel), 1)
        wmb = sum(r[2] for r in sel) / max(len(sel), 1)
        ut = sum(r[3] for r in sel) / max(len(sel), 1)
        sat = [r[1] + r[2] for r in sel if r[3] >= 95.0]
        rows_e.append({"w": w * 10, "read_tok_s": rtok / 600.0, "write_tok_s": (wtok - skipped) / 600.0,
                       "submitted_tok_s": wtok / 600.0, "read_busy": rb.overlap(a, b) / 600.0,
                       "write_busy": wb.overlap(a, b) / 600.0, "wait_p50": q(waits, .5), "wait_p90": q(waits, .9),
                       "s2d_p50": q(s2d, .5), "s2d_p90": q(s2d, .9), "pending_max": pend, "holds": nh,
                       "io_r": rmb, "io_w": wmb, "io_util": ut, "io_sat_total": q(sat, .5), "io_sat_n": len(sat)})
        e = rows_e[-1]
        print(f"   {w * 10:3d}-{w * 10 + 10:<3d} read {e['read_tok_s'] / 1e3:5.1f}K/s written {e['write_tok_s'] / 1e3:4.1f}K/s "
              f"(submitted {e['submitted_tok_s'] / 1e3:4.1f}K/s) | read thread {e['read_busy']:5.1%} write thread "
              f"{e['write_busy']:5.1%} | parked {e['wait_p50']:5.1f}/{e['wait_p90']:5.1f} s | write lag "
              f"{e['s2d_p50']:4.2f}/{e['s2d_p90']:4.2f} s | pending max {pend / 1e3:4.0f}K holds {nh:2d} | iostat "
              f"{rmb:5.0f} + {wmb:4.0f} MB/s util {ut:4.0f}% ; at >=95% util: {e['io_sat_total']:.0f} MB/s over "
              f"{len(sat)} samples")
    res["windows"] = rows_e
    return res


def main() -> None:
    campaign, labels = sys.argv[1], sys.argv[2:]
    trees = load_trees()
    path = f"{BASE}/out/q3_{campaign.replace('campaign', 'c')}.json"
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    for label in labels:
        old[label] = analyse(campaign, label, trees)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
