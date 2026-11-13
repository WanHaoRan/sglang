#!/usr/bin/env python3
"""Supplementary numbers for the concerns list (not items 1-5):

 - what the "read stall" of item 3 is made of: SSD read time, time parked before
   the launch (srs_dispatch waited_ms), deliberate holds (ntp_hold lines);
 - the delay between the client's send time and the server's entry_time, which
   sits inside (TTFT - queue wait);
 - the spread between repeats of the same configuration on the same turns.

Run as:  nice -n 19 python3 -I -B supp.py
"""
import json
import os
import re
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
RES = "/home/wanhr/sglang/agent_cache/results"
sys.path.insert(0, OUT)
import analyze as A  # noqa: E402

lines_out = []
NUMS = {}


def P(s=""):
    lines_out.append(s)
    print(s)


KEY_RE = re.compile(r"rid=\S*?-c(\d+)-t(\d+)(?=[\s,)]|$)")
KV_RE = re.compile(r"(\w+)=(\S+)")

with open(RES + "/.current_run") as f:
    current = f.read().strip()

labels = sorted(set(A.ALL11 + A.CUT7))
R = {lab: A.Run(lab) for lab in labels}
shared = {}
for sname, arms in A.SETS:
    s = None
    for a in arms:
        s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
    shared[sname] = s

# ---- ntp_hold lines from the raw logs (not in the cache)
hold = {}
for a in A.SEVEN:
    r = R[a]
    assert r.folder != current
    h = {}
    n = 0
    with open(os.path.join(RES, r.folder, "server.log"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "HICACHE_EVT ntp_hold " not in line:
                continue
            m = KEY_RE.search(line)
            if not m:
                continue
            kv = dict(KV_RE.findall(line))
            n += 1
            h.setdefault((int(m.group(1)), int(m.group(2))), []).append(float(kv.get("hold_ms", "nan")))
    hold[a] = h
    P("%-13s ntp_hold lines %d on %d keys" % (a, n, len(h)))
P()

P("WHAT THE READ STALL IS MADE OF, per arm, on the set's shared turns (slack 0.1, bound 0.1, threshold 0.05)")
P("%-6s %-13s %8s %9s %9s %9s %9s | %7s %9s %9s | %7s %9s %7s %9s" % (
    "set", "arm", "readTurn", "stallSum", "ioMsSum", "parkedSum", "io/stall", "heldTrn", "heldStall",
    "heldShare", "lw_held", "lwHeldSum", "lw_all", "lwSum"))
for sname in ("LEAD5", "SEVEN"):
    for a in dict(A.SETS)[sname]:
        r = R[a]
        sh = sorted(shared[sname])
        n = len(sh)
        stall_sum = io_sum = parked = 0.0
        held_turns = 0
        held_stall = 0.0
        lw_held_n = 0
        lw_held_sum = 0.0
        lw_n = 0
        lw_sum = 0.0
        reading = 0
        for k in sh:
            if k not in r.s2h:
                continue
            reading += 1
            st, has_q, qw = r.stall(k, 0.1)
            stall_sum += st
            arr = r.arrival(k)
            # read time that falls after arrival (reads staged before arrival cost the turn nothing)
            for y in r.s2h[k]:
                s0 = max(y[0] - y[5] / 1000.0, arr)
                if y[0] > s0:
                    io_sum += y[0] - s0
            for y in r.srs.get(k, ()):
                if y[7] == 1 and y[4] is not None and y[0] > arr:
                    parked += min(float(y[4]) / 1000.0, y[0] - arr)
            is_held = k in hold[a]
            if is_held:
                held_turns += 1
                held_stall += st
            if st >= 0.05 and (qw - st) < 0.1:
                lw_n += 1
                lw_sum += st
                if is_held:
                    lw_held_n += 1
                    lw_held_sum += st
        P("%-6s %-13s %8d %9.1f %9.1f %9.1f %9.3f | %7d %9.1f %9.3f | %7d %9.1f %7d %9.1f" % (
            sname, a, reading, stall_sum, io_sum, parked, io_sum / stall_sum if stall_sum else 0,
            held_turns, held_stall, held_stall / stall_sum if stall_sum else 0, lw_held_n, lw_held_sum,
            lw_n, lw_sum))
        NUMS[(sname, a)] = {
            "n_shared": n, "reading": reading, "stall_sum_s": stall_sum, "io_after_arrival_s": io_sum,
            "parked_before_launch_s": parked, "held_turns": held_turns, "held_stall_s": held_stall,
            "lastwait_held_turns": lw_held_n, "lastwait_held_stall_s": lw_held_sum,
            "lastwait_turns": lw_n, "lastwait_stall_s": lw_sum,
        }
P()

P("DELAY FROM CLIENT SEND TO SERVER entry_time, on the set's shared turns (it is inside TTFT - queue wait)")
P("%-6s %-13s %9s %9s %9s %9s %9s %12s" % ("set", "arm", "mean", "p50", "p90", "p99", "max", "share>0.5s"))
for sname in ("LEAD5", "SEVEN", "ALL11"):
    for a in dict(A.SETS)[sname]:
        r = R[a]
        dv = [r.arrival(k) - (r.t_start + r.cli[k]["t_send"]) for k in sorted(shared[sname])]
        P("%-6s %-13s %9.4f %9.4f %9.4f %9.4f %9.4f %12.5f" % (
            sname, a, A.mean(dv), A.pct(dv, 0.5), A.pct(dv, 0.9), A.pct(dv, 0.99), max(dv),
            sum(1 for z in dv if z > 0.5) / len(dv)))
        NUMS[("pre_entry", sname, a)] = {"mean": A.mean(dv), "p50": A.pct(dv, 0.5), "p99": A.pct(dv, 0.99)}
P()

P("FORWARD TIME TO FIRST TOKEN PROXY: TTFT - queue wait - pre-entry delay, mean on shared turns")
for sname in ("LEAD5", "SEVEN"):
    for a in dict(A.SETS)[sname]:
        r = R[a]
        dv = [r.cli[k]["ttft"] - r.qw(k) - (r.arrival(k) - (r.t_start + r.cli[k]["t_send"]))
              for k in sorted(shared[sname])]
        P("%-6s %-13s mean %.4f  p50 %.4f  p99 %.4f  min %.4f" % (
            sname, a, A.mean(dv), A.pct(dv, 0.5), A.pct(dv, 0.99), min(dv)))
P()

P("REPEAT PAIRS on ALL11's shared turns (same configuration run twice): second / first - 1")
pairs = [("baseline", "base_rep"), ("hintfree", "hintfree_ctl"), ("jit_p50", "jit_p50_ctl")]
sh = shared["ALL11"]
P("%-24s %9s %9s %9s %9s %9s %9s %9s %9s" % ("pair", "meanTTFT", "meanQW", "TTFT-QW", "stall", "readTurn",
                                              "pages", "lastWait", "gpuQueue"))
for a, b in pairs:
    oa = A.item23(R[a], sh)
    ob = A.item23(R[b], sh)
    ia = A.item1(R[a], sh)
    ib = A.item1(R[b], sh)

    def rel(x, y):
        return (y / x - 1.0) * 100.0

    P("%-24s %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%%" % (
        a + " -> " + b, rel(oa["mean_ttft"], ob["mean_ttft"]), rel(oa["mean_qw"], ob["mean_qw"]),
        rel(oa["mean_ttft_minus_qw"], ob["mean_ttft_minus_qw"]), rel(oa["mean_stall"], ob["mean_stall"]),
        rel(ia["reading_turns"], ib["reading_turns"]), rel(ia["ssd_pages"], ib["ssd_pages"]),
        rel(oa["lastwait_sum_per_turn"], ob["lastwait_sum_per_turn"]),
        rel(oa["gpuq_sum_per_turn"], ob["gpuq_sum_per_turn"])))
    P("   values: TTFT %.4f/%.4f  QW %.4f/%.4f  TTFT-QW %.4f/%.4f  stall %.4f/%.4f  readTurns %d/%d  pages %d/%d  lastWait %.4f/%.4f  gpuQueue %.4f/%.4f"
      % (oa["mean_ttft"], ob["mean_ttft"], oa["mean_qw"], ob["mean_qw"], oa["mean_ttft_minus_qw"],
         ob["mean_ttft_minus_qw"], oa["mean_stall"], ob["mean_stall"], ia["reading_turns"],
         ib["reading_turns"], ia["ssd_pages"], ib["ssd_pages"], oa["lastwait_sum_per_turn"],
         ob["lastwait_sum_per_turn"], oa["gpuq_sum_per_turn"], ob["gpuq_sum_per_turn"]))
P()
P("SAME PAIRS, evictions over each run's own kept window (tokens, and per kept turn)")
for a, b in pairs:
    for x in (a, b):
        r = R[x]
        evd, evh, _, _ = A.evict_window(r)
        P("  %-13s window %.0f s  kept turns %d  evict_device %d (%.0f/turn)  evict_host %d (%.0f/turn)" % (
            x, r.cut, len(r.kept), evd, evd / len(r.kept), evh, evh / len(r.kept)))
P()
P("LEAD5 + controls: evictions per kept turn over each run's own kept window")
for x in A.SEVEN:
    r = R[x]
    evd, evh, _, _ = A.evict_window(r)
    P("  %-13s window %.0f s  kept turns %d  evict_device %d (%.0f/turn)  evict_host %d (%.0f/turn)" % (
        x, r.cut, len(r.kept), evd, evd / len(r.kept), evh, evh / len(r.kept)))
P()
P("WHEN THE SHARED TURNS RAN: client t_send of the set's shared turns within each run (min / median / max, s)")
for sname in ("LEAD5", "SEVEN", "ALL11"):
    for a in dict(A.SETS)[sname]:
        r = R[a]
        ts = [r.cli[k]["t_send"] for k in shared[sname]]
        P("%-6s %-13s %9.1f %9.1f %9.1f   shared/keptReturning = %d/%d = %.4f" % (
            sname, a, min(ts), A.pct(ts, 0.5), max(ts), len(ts), len(r.kept_ret), len(ts) / len(r.kept_ret)))
with open(os.path.join(OUT, "supp.out"), "w") as f:
    f.write("\n".join(lines_out) + "\n")
with open(os.path.join(OUT, "supp.json"), "w") as f:
    json.dump({"|".join(k): v for k, v in NUMS.items()}, f, indent=1)
