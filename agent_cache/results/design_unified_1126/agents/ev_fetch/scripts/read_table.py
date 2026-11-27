"""Per-read table of one run for the landing-quantile (F1) and line (F3) checks.

  python read_table.py <label> [ec]     (ec: the arm stages reads by expected cost; only orders the staged reads)

For every storage read of a queued turn: when it was parked, released by the hold (replayed, margin 2 s and a grid of
other margins), dispatched and landed; its turn's arrival, GPU turn D (first pass-over, else the admission) and
admission; the engine's estimates at arrival, at the first hold and at the release; the line state. Written to
extracts/<label>.reads.json; a per-request table (planned turns: predicted wait at arrival against D and admission)
to extracts/<label>.requests.json.
"""
import bisect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import runlib as R  # noqa: E402

MARGINS = (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 1e9)


def main():
    label = sys.argv[1]
    ec = len(sys.argv) > 2 and sys.argv[2] == "ec"
    d = R.load(label)
    Q = R.Queue(d)
    RD = R.Reads(d, Q)
    usage = R.usage_series(d)
    D = R.gpu_turn_refined(Q, RD, usage)
    H = R.hold_replay(d, Q, RD, dt=0.1, expected_cost=ec)
    ready = R.ready_series(Q, RD)
    t0 = d["run_rec"]["t_start"]
    ev = d["evts"]
    admit = R.first_by_rid(ev.get("ntp_admit", []))
    arrive = R.first_by_rid(ev.get("ntp_arrive", []))
    due = R.all_by_rid(ev.get("ntp_due", []))
    ecs = R.all_by_rid(ev.get("ec_start", []))
    reqs = d["reqs"]

    # reads dispatched over time, for "pages served between release and dispatch"
    disp_sorted = sorted((r["disp"], r["pages"]) for r in RD.reads if r["disp"] is not None and r.get("launched"))
    disp_t = [x[0] for x in disp_sorted]
    disp_cum = [0]
    for _, p in disp_sorted:
        disp_cum.append(disp_cum[-1] + p)

    def pages_dispatched(a, b):
        return disp_cum[bisect.bisect_right(disp_t, b)] - disp_cum[bisect.bisect_right(disp_t, a)]

    rows = []
    for r in RD.reads:
        rid = r["rid"]
        if r["arr"] is None or r["adm"] is None or r["disp"] is None or r["land"] is None:
            continue
        st = H[r["n"]]
        row = {"rid": rid, "i": r["i"], "pages": r["pages"], "tokens": r["pages"] * R.PAGE, "staged": r["staged"],
               "park": r["park"] - t0, "disp": r["disp"] - t0, "land": r["land"] - t0, "arr": r["arr"] - t0,
               "adm": r["adm"] - t0, "D": D[rid] - t0, "io_ms": r.get("io_ms"), "inflight_after": r.get("inflight_after"),
               "pend_write": r.get("pend_write"), "deadline_at_park": r["deadline_at_park"],
               "held_logged": "hold" in r, "usage_park": R.usage_at(usage, r["park"]),
               "qlen_park": Q.qlen(r["park"]), "s_per_page_park": RD.s_per_page(r["park"]),
               "wh_park": RD.write_held(r["park"])}
        if "hold" in r:
            h = r["hold"]
            row["hold"] = {"t": h["t"] - t0, "pos": h["pos"], "wait": h["wait_ms"] / 1e3, "load": h["load_ms"] / 1e3,
                           "hold": h["hold_ms"] / 1e3}
        if rid in admit:
            f = admit[rid][1]
            if isinstance(f.get("pred_wait_ql_ms"), int):
                row["pred_wait_arr"] = f["pred_wait_ql_ms"] / 1e3
        if rid in arrive:
            row["pos_arr"] = arrive[rid][1]["pos"]
        # the planner's read time at arrival (ntp_due case 2: own pages and dispatched backlog only)
        dd = [(t, f) for t, f in due.get(rid, []) if f.get("case") == 2]
        if dd:
            row["due2"] = {"t": dd[0][0] - t0, "load": dd[0][1]["load_ms"] / 1e3, "wait": dd[0][1]["wait_ms"] / 1e3,
                           "l3_pages": dd[0][1]["l3_pages"], "backlog_pages": dd[0][1]["backlog_pages"]}
        # expected cost: the last start decision before the turn arrived
        ee = [(t, f) for t, f in ecs.get(rid, []) if t <= r["arr"] + 0.05]
        if ee:
            t_e, f = ee[-1]
            row["ec_last"] = {"t": t_e - t0, "decision": f["decision"], "read": f["read_ms"] / 1e3,
                              "wait": f["wait_ms"] / 1e3, "tokens": f["tokens"]}
        if st and not r["staged"]:
            rel = {}
            for m in MARGINS:
                x = R.release_time(st, r["pages"], m)
                rel[f"{m:g}"] = None if x is None else x - t0
            row["rel"] = rel
            # state at the replayed release (margin 2 s)
            x2 = R.release_time(st, r["pages"], R.HOLD_MARGIN_S)
            for (t, pos, w, ahead, p, wh) in st:
                if t == x2:
                    rate = Q.rate(t)
                    row["at_rel"] = {"pos": pos, "wait": None if rate is None else (pos + 0.5) / rate[0],
                                     "ahead": ahead, "p": p, "wh": wh,
                                     "load": R.LOAD_FIXED_S + p * (r["pages"] + ahead),
                                     "usage": R.usage_at(usage, t), "qlen": Q.qlen(t),
                                     "rate": None if rate is None else rate[0]}
                    break
            # first step (at park)
            t, pos, w, ahead, p, wh = st[0]
            rate = Q.rate(t)
            row["at_park"] = {"pos": pos, "wait": None if rate is None else (pos + 0.5) / rate[0], "ahead": ahead,
                              "load": R.LOAD_FIXED_S + p * (r["pages"] + ahead)}
            if x2 is not None:
                row["wh_rel_disp"] = RD.write_hold_overlap(x2, r["disp"])
                row["pages_disp_rel_disp"] = pages_dispatched(x2, r["disp"] - 1e-6)
        # the share of the stall (D to admission) with no other ready turn waiting
        if r["adm"] > D[rid] + 0.05:
            row["idle_share_stall"] = R.idle_share(ready, D[rid], r["adm"])
        cr = reqs.get(rid)
        if cr is not None:
            row["key"] = cr.get("key")
            row["gap"] = cr.get("wait_s")
            row["gap_kind"] = cr.get("gap")
            row["ttft"] = cr.get("ttft")
            row["k"] = cr.get("k")
        rows.append(row)
    with open(os.path.join(C.EXTRACTS, f"{label}.reads.json"), "w") as f:
        json.dump({"label": label, "margins": [f"{m:g}" for m in MARGINS], "rows": rows}, f)

    # per-request table: every request with its predicted wait at arrival (planned turns), D, admission, reader flag
    landed = {}
    for r in RD.reads:
        if r["land"] is not None and r["adm"] is not None and r["land"] <= r["adm"] + 0.05:
            landed.setdefault(r["rid"], []).append(r["land"])
    parked = set(r["rid"] for r in RD.reads)
    out = []
    for rid, a in Q.arr.items():
        row = {"rid": rid, "arr": a - t0, "adm": Q.adm[rid] - t0, "D": D[rid] - t0, "reader": rid in landed,
               "parked": rid in parked, "usage_arr": R.usage_at(usage, a), "qlen_arr": Q.qlen(a)}
        if rid in landed:
            row["last_land"] = max(landed[rid]) - t0
        if rid in admit and isinstance(admit[rid][1].get("pred_wait_ql_ms"), int):
            row["pred_wait_arr"] = admit[rid][1]["pred_wait_ql_ms"] / 1e3
        if rid in arrive:
            row["pos_arr"] = arrive[rid][1]["pos"]
        cr = reqs.get(rid)
        if cr is not None:
            for k_ in ("key", "wait_s", "gap", "ttft", "k", "prompt_tokens", "cached_tokens", "ideal_cached", "play",
                       "stream", "latency", "completion_tokens"):
                row[k_] = cr.get(k_)
            row["t_send"] = cr.get("t_send")
            row["hint"] = cr.get("hint")
            row["cached_details"] = cr.get("cached_details")
            row["oracle_gap_s"] = cr.get("oracle_gap_s")
        out.append(row)
    with open(os.path.join(C.EXTRACTS, f"{label}.requests.json"), "w") as f:
        json.dump({"label": label, "rows": out}, f)
    # the ready-turn series, sampled, for the lateness weight c by time
    ts, cs = ready
    with open(os.path.join(C.EXTRACTS, f"{label}.ready.json"), "w") as f:
        json.dump({"t": [x - t0 for x in ts], "ready": cs}, f)
    print(label, "reads", len(rows), "requests", len(out))


if __name__ == "__main__":
    main()
