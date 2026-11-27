"""F3, the line: where a read's time goes, and what the engine's read-time estimates said.

  python f3_line.py <label> [ec]

A. Per read of a queued turn: park -> release (the hold, replayed) -> dispatch (the line) -> I/O start (the read
   thread's FIFO) -> I/O end -> usable (pin). By window of minutes and by pool usage.
B. The read time the engine estimated against what happened:
   - planner: 0.904 s + ms/page x (own pages + pages in flight), the pre-arrival formula (parked reads and the write
     hold left out), evaluated when the read was parked;
   - hold: the same plus the parked reads ordered ahead by deadline, as read_not_before computes it;
   - line: the hold's plus the write hold (the rest of a hold that is on, at the run's median hold length) and reads
     counted in arrival order when the read thread is saturated (pages in flight >= the 512-page window);
   against release -> usable (reads released at once: park -> usable).
C. Staged reads and expected-cost decisions: the read time the planner used before arrival against the read's time.
"""
import bisect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import runlib as R  # noqa: E402

WINDOWS = ((0, 30), (30, 70), (70, 90), (90, 110))
WINDOW_PAGES = 512


def qs(xs, ps=(.1, .5, .9)):
    return " ".join(f"{C.q(xs, p):7.2f}" for p in ps) + (f" mean {sum(xs) / len(xs):7.2f}" if xs else " mean      na")


def main():
    label = sys.argv[1]
    ec = len(sys.argv) > 2 and sys.argv[2] == "ec"
    d = R.load(label)
    Q = R.Queue(d)
    RD = R.Reads(d, Q)
    usage = R.usage_series(d)
    D = R.gpu_turn_refined(Q, RD, usage)
    H = R.hold_replay(d, Q, RD, dt=0.1, expected_cost=ec)
    t0 = d["run_rec"]["t_start"]
    ev = d["evts"]
    wh_len = [b - a for a, b in RD.wh if b < math.inf]
    wh_med = C.q(wh_len, .5) if wh_len else 0.0
    print(f"==== {label} ====")
    print(f"write hold: {len(RD.wh)} holds, total {sum(wh_len):.1f} s, median {wh_med:.2f} s, max {max(wh_len) if wh_len else 0:.2f} s")

    # really-parked intervals, for the arrival-order count
    parked = [(r["park"], r["disp"] if r["disp"] is not None else r["park"], r["arr"] if r["arr"] is not None else r["park"],
               r["pages"]) for r in RD.reads]
    flight = []
    for r in RD.reads:
        if r["disp"] is not None and r.get("launched"):
            flight.append((r["disp"], r["pages"]))
            flight.append((r["land"] if r["land"] is not None else r["disp"], -r["pages"]))
    flight.sort()
    fl_t = [e[0] for e in flight]
    fl_cum, s = [], 0
    for e in flight:
        s += e[1]
        fl_cum.append(s)

    def inflight(t):
        i = bisect.bisect_right(fl_t, t) - 1
        return fl_cum[i] if i >= 0 else 0

    rows = []
    for r in RD.reads:
        if r["staged"] or r["arr"] is None or r["adm"] is None or r["land"] is None or r["disp"] is None:
            continue
        st = H[r["n"]]
        if not st:
            continue
        rel = R.release_time(st, r["pages"], R.HOLD_MARGIN_S)
        if rel is None:
            continue
        rel = min(rel, r["disp"])           # the replay's grid can lag the engine's pass by a step
        at_rel = next((s_ for s_ in st if s_[0] >= rel), st[-1])
        at_park = st[0]
        p_park, p_rel = at_park[4], at_rel[4]
        fl_park = inflight(r["park"])
        # arrival order: parked reads of turns that arrived before this one
        fifo_pages = sum(pg for (a, b, arr, pg) in parked if a <= r["park"] < b and arr < r["arr"])
        io_start = r["io_end"] - r["io_ms"] / 1e3 if r["io_end"] is not None else None
        pos_disp = Q.position(r["rid"], r["disp"] - 1e-3)
        row = {"win": next((f"{lo}-{hi}" for lo, hi in WINDOWS if lo * 60 <= r["park"] - t0 < hi * 60), None),
               "usage": R.usage_at(usage, r["park"]), "pages": r["pages"],
               "hold": rel - r["park"], "line": r["disp"] - rel,
               "fifo": (io_start - r["disp"]) if io_start is not None else float("nan"),
               "io": r["io_ms"] / 1e3 if r.get("io_ms") is not None else float("nan"),
               "drain": (r["land"] - r["io_end"]) if r["io_end"] is not None else float("nan"),
               "total_from_rel": r["land"] - rel, "total_from_park": r["land"] - r["park"],
               "pos_disp": pos_disp, "now_at_park": r["deadline_at_park"] == "now",
               "inflight_park": fl_park, "wh_park": RD.write_held(r["park"]),
               "wh_overlap": RD.write_hold_overlap(rel, r["disp"]),
               "est_plan_park": R.LOAD_FIXED_S + p_park * (r["pages"] + fl_park),
               "est_hold_park": R.LOAD_FIXED_S + p_park * (r["pages"] + at_park[3]),
               "est_hold_rel": R.LOAD_FIXED_S + p_rel * (r["pages"] + at_rel[3]),
               "est_fifo_park": R.LOAD_FIXED_S + p_park * (r["pages"] + fl_park + fifo_pages),
               "held": rel > r["park"] + 0.15, "stall": r["adm"] - D[r["rid"]], "t": r["park"] - t0}
        sat = fl_park >= WINDOW_PAGES
        line = row["est_fifo_park"] if sat else row["est_hold_park"]
        row["est_line_park"] = line + (wh_med / 2 if row["wh_park"] else 0.0)
        row["sat"] = sat
        rows.append(row)

    groups = [(f"minutes {lo}-{hi}", [x for x in rows if x["win"] == f"{lo}-{hi}"]) for lo, hi in WINDOWS]
    groups += [("pool < 0.90 at park", [x for x in rows if x["usage"] < 0.90]),
               ("pool >= 0.90 at park", [x for x in rows if x["usage"] >= 0.90]),
               ("read thread under the window at park (< 512 pages in flight), pool >= 0.90",
                [x for x in rows if not x["sat"] and x["usage"] >= 0.90]),
               ("read thread over the window at park (>= 512 pages in flight)", [x for x in rows if x["sat"]])]
    print("\n== A. Where the time goes (s; p10 p50 p90 mean) ==")
    for title, g in groups:
        if len(g) < 10:
            print(f"  -- {title}: {len(g)} reads, too few --")
            continue
        print(f"  -- {title}: {len(g)} reads, {sum(1 for x in g if x['held'])} held, median {C.q([x['pages'] for x in g], .5)} pages; "
              f"dispatched at positions 0-1: {100 * sum(1 for x in g if x['pos_disp'] is not None and x['pos_disp'] <= 1) / len(g):.0f} %; "
              f"parked as 'now': {100 * sum(1 for x in g if x['now_at_park']) / len(g):.0f} %; "
              f"with the write hold on in release->dispatch: {100 * sum(1 for x in g if x['wh_overlap'] > 0) / len(g):.0f} % "
              f"(mean {sum(x['wh_overlap'] for x in g) / len(g):.2f} s) --")
        for name, key in (("park -> release (hold)", "hold"), ("release -> dispatch (line)", "line"),
                          ("dispatch -> I/O start (thread FIFO)", "fifo"), ("I/O (own pages)", "io"),
                          ("I/O end -> usable", "drain"), ("release -> usable", "total_from_rel"),
                          ("park -> usable", "total_from_park")):
            xs = [x[key] for x in g if not math.isnan(x[key])]
            print(f"      {name:36s}: {qs(xs)}")
    print("\n== B. Read-time estimates against release -> usable (s) ==")
    print("   estimate: p10 p50 p90 mean | actual - estimate: p10 p50 p90 mean | actual / estimate p50 | |err| p50 p90")
    for title, g in groups:
        if len(g) < 10:
            continue
        print(f"  -- {title}: {len(g)} reads; actual release -> usable {qs([x['total_from_rel'] for x in g])}; "
              f"park -> usable {qs([x['total_from_park'] for x in g])} --")
        for name, key, act in (("planner (own + in flight), at park      ", "est_plan_park", "total_from_park"),
                               ("hold (+ parked ahead by deadline), park ", "est_hold_park", "total_from_park"),
                               ("hold (+ parked ahead), at release       ", "est_hold_rel", "total_from_rel"),
                               ("arrival order (+ parked earlier), park  ", "est_fifo_park", "total_from_park"),
                               ("line (order by regime + write hold)     ", "est_line_park", "total_from_park")):
            # for held reads the estimate at park is asked about a read released later: compare from the release
            if act == "total_from_park":
                pairs = [(x[key], x["total_from_rel"] if x["held"] else x["total_from_park"]) for x in g]
            else:
                pairs = [(x[key], x[act]) for x in g]
            est = [p[0] for p in pairs]
            err = [p[1] - p[0] for p in pairs]
            ratio = [p[1] / p[0] for p in pairs if p[0] > 0]
            print(f"      {name}: {qs(est)} | {qs(err)} | {C.q(ratio, .5):5.2f} | {C.q([abs(e) for e in err], .5):6.2f} "
                  f"{C.q([abs(e) for e in err], .9):6.2f}")
    # C. before arrival
    print("\n== C. The read time planned before arrival against the read's own time ==")
    by_rid = {}
    for r in RD.reads:
        by_rid.setdefault(r["rid"], []).append(r)
    # C1: expected-cost decisions (ec_start): the last decision before the turn arrived
    last = {}
    for t, rid, f in ev.get("ec_start", []):
        if rid in Q.arr and t <= Q.arr[rid] + 0.05:
            last[rid] = (t, f)
    rowsC = []
    for rid, (t, f) in last.items():
        rs = [r for r in by_rid.get(rid, []) if not r["staged"] and r["land"] is not None]
        if not rs:
            continue
        r = rs[0]
        # the read's own time runs from its release: the hold before it is waiting on purpose
        rel = R.release_time(H[r["n"]], r["pages"], R.HOLD_MARGIN_S) if H[r["n"]] else None
        start = r["park"] if rel is None or r["disp"] is None else min(max(rel, r["park"]), r["disp"])
        rowsC.append({"win": next((f"{lo}-{hi}" for lo, hi in WINDOWS if lo * 60 <= r["park"] - t0 < hi * 60), None),
                      "decision": f["decision"], "read_plan": f["read_ms"] / 1e3 - 1.0, "wait_plan": f["wait_ms"] / 1e3,
                      "actual": r["land"] - start, "lead": Q.arr[rid] - t, "pages": r["pages"],
                      "wait_actual": D[rid] - Q.arr[rid]})
    if rowsC:
        print("  expected cost (ec_start), turns that then read at arrival: planned read = read_ms - 1 s margin; "
              "actual = release -> usable of the turn's read; W planned against arrival -> GPU turn")
        for lo, hi in WINDOWS:
            g = [x for x in rowsC if x["win"] == f"{lo}-{hi}"]
            if len(g) < 10:
                continue
            print(f"    minutes {lo}-{hi}: n {len(g)}; decisions {dict((k, sum(1 for x in g if x['decision'] == k)) for k in ('arrival', 'wait', 'now'))}; "
                  f"planned read {qs([x['read_plan'] for x in g])} | actual {qs([x['actual'] for x in g])} | "
                  f"actual/planned p50 {C.q([x['actual'] / x['read_plan'] for x in g], .5):.2f} | planned W {qs([x['wait_plan'] for x in g])} | "
                  f"actual W {qs([x['wait_actual'] for x in g])} | decision made {C.q([x['lead'] for x in g], .5):.1f} s before arrival (p50) | "
                  f"read > W by plan: {100 * sum(1 for x in g if x['read_plan'] + 1.0 > x['wait_plan']) / len(g):.0f} %, in fact: "
                  f"{100 * sum(1 for x in g if x['actual'] > x['wait_actual']) / len(g):.0f} %")
    # C2: staged reads (ntp_due case 1 load_ms) against stage -> usable
    due1 = {}
    for t, rid, f in ev.get("ntp_due", []):
        if f.get("case") == 1 and rid not in due1:
            due1[rid] = (t, f)
    rowsS = []
    for rid, (t, f) in due1.items():
        rs = [r for r in by_rid.get(rid, []) if r["staged"] and r["land"] is not None]
        if not rs:
            continue
        r = rs[0]
        arr = Q.arr.get(rid)
        rowsS.append({"win": next((f"{lo}-{hi}" for lo, hi in WINDOWS if lo * 60 <= r["park"] - t0 < hi * 60), None),
                      "plan": f["load_ms"] / 1e3, "actual": r["land"] - r["park"], "line": r["disp"] - r["park"],
                      "wait_plan": f["wait_ms"] / 1e3, "pages": r["pages"],
                      "late_by": None if arr is None else r["land"] - arr,
                      "arrived": arr is not None,
                      "wait_actual": None if arr is None else D[rid] - arr})
    if rowsS:
        print("  staged reads (ntp_due case 1): planned load against stage -> usable")
        for lo, hi in WINDOWS:
            g = [x for x in rowsS if x["win"] == f"{lo}-{hi}"]
            if len(g) < 5:
                continue
            la = [x["late_by"] for x in g if x["late_by"] is not None]
            print(f"    minutes {lo}-{hi}: n {len(g)} ({sum(1 for x in g if x['arrived'])} whose turn came); planned {qs([x['plan'] for x in g])} | "
                  f"actual {qs([x['actual'] for x in g])} | of which parked {qs([x['line'] for x in g])} | actual/planned p50 "
                  f"{C.q([x['actual'] / x['plan'] for x in g], .5):.2f} | usable after the turn arrived by {qs(la)} "
                  f"(landed before the arrival: {100 * sum(1 for v in la if v <= 0) / max(1, len(la)):.0f} %)")
    with open(os.path.join(C.OUT, f"f3_line_{label}.json"), "w") as f:
        json.dump({"rows": rows, "ec": rowsC, "staged": rowsS}, f)


if __name__ == "__main__":
    main()
