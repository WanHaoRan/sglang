#!/usr/bin/env python3
"""Independent re-derivation (recheck.py's raw-log loader, exact microseconds) of the
figures quoted in the 'concerns' of the final answer, which were first produced by
supp.py / posthoc.py / posthoc2.py / check_clock.py on the cache.  LEAD5 arms, LEAD5
shared turns unless stated.  Computed after the existing outputs were opened.

Run as:  nice -n 19 python3 -I -B verify_concerns.py
"""
import os
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
sys.path.insert(0, OUT)
import recheck as RC  # noqa: E402

LEAD5 = list(RC.LEAD5)
out = []


def P(s=""):
    out.append(s)
    print(s)


with open(RC.RES + "/.current_run") as f:
    current = f.read().strip()
R = {a: RC.load(a, current) for a in LEAD5 + ["order_run"]}


def extra(label):
    """srs_dispatch lines per key [(t_us, waited_ms, launched)] and the keys with an ntp_hold line."""
    folder = RC.RUNS[label]
    assert folder != current
    disp, hold = {}, set()
    with open(os.path.join(RC.RES, folder, "server.log"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            i = line.find("HICACHE_EVT ")
            if i < 0:
                continue
            body = line[i + 12:]
            kind = body.split(" ", 1)[0]
            if kind == "srs_dispatch":
                k = RC.key_of(RC.tok_after(body, "rid"))
                disp.setdefault(k, []).append((RC.line_us(line), int(RC.tok_after(body, "waited_ms")),
                                               RC.tok_after(body, "launched") == "1"))
            elif kind == "ntp_hold":
                hold.add(RC.key_of(RC.tok_after(body, "rid")))
    return disp, hold


pos = {(t["conv"], t["turn"]): i for i, t in enumerate(R["order_run"].ok)}
shared = None
for a in LEAD5:
    shared = set(R[a].kept_ret) if shared is None else (shared & R[a].kept_ret)
shared = sorted(shared)
N = len(shared)
episode = [k for k in shared if k in pos and (5500 <= pos[k] <= 6999 or 8750 <= pos[k] <= 10999)]
ep2 = [k for k in shared if k in pos and 8750 <= pos[k] <= 10999]
absent = [k for k in shared if k not in pos]
P("LEAD5 shared turns %d; in the two episodes %d (%.1f %%); in episode 2 %d; absent from the order run %d (%.1f %%)" % (
    N, len(episode), 100.0 * len(episode) / N, len(ep2), len(absent), 100.0 * len(absent) / N))
P()
for a in LEAD5:
    r = R[a]
    disp, hold = extra(a)
    rows = {k: RC.turn_row(r, k) for k in shared}          # q_us, st_us, n_reads, n_qual, e_us
    q_sum = sum(v[0] for v in rows.values()) / 1e6
    st_sum = sum(v[1] for v in rows.values()) / 1e6
    reading = [k for k in shared if rows[k][2] > 0]
    io_after = 0
    for k in reading:
        e_us = rows[k][4]
        for end_us, pages, ms in r.s2h[k]:
            io_after += max(0, end_us - max(end_us - ms * 1000, e_us))
    parked = 0
    for k in reading:
        e_us = rows[k][4]
        for t_us, waited_ms, launched in disp.get(k, ()):
            if launched and t_us > e_us:
                parked += min(waited_ms * 1000, t_us - e_us)
    held = [k for k in reading if k in hold]
    held_st = sum(rows[k][1] for k in held) / 1e6
    P("%s" % a)
    P("  stall composition: reading turns %d, stall %.1f s, read I/O after arrival %.1f s (%.1f %%), parked before a launch %.1f s (%.1f %%); "
      "turns with ntp_hold %d carrying %.1f s (%.1f %%)" % (
          len(reading), st_sum, io_after / 1e6, 100.0 * io_after / 1e6 / st_sum, parked / 1e6, 100.0 * parked / 1e6 / st_sum,
          len(held), held_st, 100.0 * held_st / st_sum))
    eq = sum(rows[k][0] for k in episode) / 1e6
    es = sum(rows[k][1] for k in episode) / 1e6
    et = sum(r.cli[k]["ttft"] for k in episode)
    tt = sum(r.cli[k]["ttft"] for k in shared)
    e2q = sum(rows[k][0] for k in ep2) / 1e6
    e2s = sum(rows[k][1] for k in ep2) / 1e6
    P("  share in the two episodes: queue wait %.3f, stall %.3f, TTFT %.3f; in episode 2 alone: queue wait %.3f, stall %.3f" % (
        eq / q_sum, es / st_sum, et / tt, e2q / q_sum, e2s / st_sum))
    top = sorted(shared, key=lambda k: -rows[k][0])
    n1 = int(round(0.01 * N))
    n01 = int(round(0.001 * N))
    P("  the %d turns with the largest queue wait: share of queue wait %.3f, of stall %.3f, sessions %d; the %d largest: share of queue wait %.3f" % (
        n1, sum(rows[k][0] for k in top[:n1]) / 1e6 / q_sum, sum(rows[k][1] for k in top[:n1]) / 1e6 / st_sum,
        len(set(k[0] for k in top[:n1])), n01, sum(rows[k][0] for k in top[:n01]) / 1e6 / q_sum))
    P("  absent from the order run: mean TTFT %.4f s; all 'elsewhere' turns: mean TTFT %.4f s" % (
        sum(r.cli[k]["ttft"] for k in absent) / len(absent),
        (tt - et) / (N - len(episode))))
    gaps = []
    unpaired = 0
    for k, reads in r.s2h.items():
        la = sorted(t for t, w, launched in disp.get(k, ()) if launched)
        rd = sorted(reads)
        if len(la) != len(rd):
            unpaired += len(rd)
            continue
        for t0, (end_us, pages, ms) in zip(la, rd):
            gaps.append((end_us - ms * 1000 - t0) / 1e6)
    P("  read start minus launch line (whole log, %d reads paired, %d not): median %.4f s, over 1 s on %d (%.1f %%), max %.3f s, below -0.005 s on %d" % (
        len(gaps), unpaired, RC.pct(gaps, 0.5), sum(1 for g in gaps if g > 1.0), 100.0 * sum(1 for g in gaps if g > 1.0) / len(gaps),
        max(gaps), sum(1 for g in gaps if g < -0.005)))
    kr = r.kept_ret
    kept_reading = [k for k in r.kept_keys if k in r.s2h]
    P("  labels: ntp_stage lines whole log %d, on kept returning turns %d (turns %d); refused launches whole log %d, on kept returning turns %d (turns %d); "
      "kept turns that read %d, s2h_io lines on them %d, s2h_io lines whole log %d" % (
          r.n_ntp_all, sum(r.ntp.get(k, 0) for k in kr), sum(1 for k in kr if r.ntp.get(k, 0)),
          sum(r.refused.values()), sum(r.refused.get(k, 0) for k in kr), sum(1 for k in kr if r.refused.get(k, 0)),
          len(kept_reading), sum(len(r.s2h[k]) for k in kept_reading), r.n_s2h_all))
    rnr = [k for k in shared if r.refused.get(k, 0) and k not in r.s2h]
    P("  shared turns with a refused launch and no read: %d, queue wait %.1f s = %.4f s per shared turn" % (
        len(rnr), sum(rows[k][0] for k in rnr) / 1e6, sum(rows[k][0] for k in rnr) / 1e6 / N))
    pre = [rows[k][4] / 1e6 - (r.t_start + r.cli[k]["t_send"]) for k in shared]
    res = (tt - q_sum) / N
    P("  pre-entry delay (entry_time - client send): mean %.4f s = %.3f of mean(TTFT - queue wait) %.4f s; p99 %.3f s; max %.3f s; share over 0.5 s %.4f; "
      "mean in the episodes %.4f s, elsewhere %.4f s" % (
          sum(pre) / N, sum(pre) / N / res, res, RC.pct(pre, 0.99), max(pre), sum(1 for x in pre if x > 0.5) / N,
          sum(rows[k][4] / 1e6 - (r.t_start + r.cli[k]["t_send"]) for k in episode) / len(episode),
          (sum(pre) - sum(rows[k][4] / 1e6 - (r.t_start + r.cli[k]["t_send"]) for k in episode)) / (N - len(episode))))
    P("  reading turns with stall below 0 / 0.02 / 0.05 / 0.1 / 0.2 s threshold: %s; with stall exactly 0: %d; min of TTFT - queue wait - pre-entry: %.4f s" % (
        " / ".join(str(sum(1 for k in reading if rows[k][1] < th)) for th in (1, 20000, 50000, 100000, 200000)),
        sum(1 for k in reading if rows[k][1] == 0),
        min(r.cli[k]["ttft"] - rows[k][0] / 1e6 - p for k, p in zip(shared, pre))))
    P()
with open(os.path.join(OUT, "verify_concerns.out"), "w") as f:
    f.write("\n".join(out) + "\n")
