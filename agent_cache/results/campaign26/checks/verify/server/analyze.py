#!/usr/bin/env python3
"""Independent recomputation of the campaign 26 server-side results (items 1-5).

Reads only cache/*.json written by parse_runs.py.  Writes
  results_items1to5.json   every number, as a flat list of records
  results_items1to5.txt    the same as readable tables
Standard library only.  Run as:  nice -n 19 python3 -I -B analyze.py
"""
import json
import os
import time
from collections import Counter, defaultdict

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
CACHE = OUT + "/cache"

ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
         "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
CUT7 = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13",
        "jit_exact"]
SETS = [("LEAD5", LEAD5), ("SEVEN", SEVEN), ("ALL11", ALL11), ("CUT7", CUT7)]
EPISODES = [("ep1", 5500, 6999), ("ep2", 8750, 10999)]

VALUES = []
TXT = []
EXTRA_ROWS = []


def P(s=""):
    TXT.append(s)


def V(item, arm, arm_set, metric, value, unit=""):
    VALUES.append({"item": item, "arm": arm, "arm_set": arm_set, "metric": metric,
                   "unit": unit, "value": value})


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


class Run:
    def __init__(self, label):
        with open(os.path.join(CACHE, label + ".json")) as f:
            d = json.load(f)
        self.label = label
        self.folder = d["folder"]
        self.t_start = d["t_start"]
        self.turns = d["turns"]
        ok = [t for t in self.turns if t["ttft"] is not None]
        self.ok_sorted = sorted(ok, key=lambda t: t["t_send"])
        self.max_t_send = self.ok_sorted[-1]["t_send"]
        self.cut = self.max_t_send - 300.0
        self.kept = [t for t in self.ok_sorted if t["t_send"] < self.cut]
        self.cli = {}
        self.cli_dups = 0
        for t in self.turns:
            k = (t["conv"], t["turn"])
            if k in self.cli:
                self.cli_dups += 1
            self.cli[k] = t
        self.kept_keys = set((t["conv"], t["turn"]) for t in self.kept)
        self.kept_ret = set(k for k in self.kept_keys if k[1] > 0)
        self.rts_last = {}
        self.rts_first = {}
        self.rts_n = Counter()
        for r in d["rts"]:
            k = (r[1], r[2])
            self.rts_last[k] = r
            self.rts_first.setdefault(k, r)
            self.rts_n[k] += 1
        self.s2h = defaultdict(list)
        for r in d["s2h"]:
            self.s2h[(r[1], r[2])].append(r)
        self.s2h = dict(self.s2h)
        self.s2h_all = d["s2h"]
        self.ntp = defaultdict(list)
        for r in d["ntp"]:
            self.ntp[(r[1], r[2])].append(r)
        self.ntp = dict(self.ntp)
        self.srs = defaultdict(list)
        for r in d["srs"]:
            self.srs[(r[1], r[2])].append(r)
        self.srs = dict(self.srs)
        self.evd = d["evd"]
        self.evh = d["evh"]
        self.dec = d["dec"]
        self.pre = d["pre"]

    def qw(self, k):
        return self.rts_last[k][3] / 1000.0

    def arrival(self, k):
        return self.rts_last[k][5]

    def stall(self, k, slack):
        """(stall seconds, has a read that ended by admission + slack, queue wait)."""
        r = self.rts_last[k]
        arrival = r[5]
        qw = r[3] / 1000.0
        adm = arrival + qw
        ends = [x[0] for x in self.s2h.get(k, ()) if x[0] <= adm + slack]
        if not ends:
            return 0.0, False, qw
        st = max(ends) - arrival
        if st < 0.0:
            st = 0.0
        if st > qw:
            st = qw
        return st, True, qw


def item1(run, keys):
    keys = sorted(keys)
    reading = [k for k in keys if k in run.s2h]
    lines = sum(len(run.s2h[k]) for k in reading)
    pages = sum(x[3] for k in reading for x in run.s2h[k])
    staged_lines = sum(len(run.ntp[k]) for k in keys if k in run.ntp)
    staged_turns = sum(1 for k in keys if k in run.ntp)
    ref_lines = sum(1 for k in keys for x in run.srs.get(k, ()) if x[7] == 0)
    ref_turns = sum(1 for k in keys if any(x[7] == 0 for x in run.srs.get(k, ())))
    return {
        "n": len(keys),
        "reading_turns": len(reading),
        "ssd_pages": pages,
        "s2h_lines": lines,
        "reads_per_reading_turn": (lines / len(reading)) if reading else float("nan"),
        "staged_reads": staged_lines,
        "staged_turns": staged_turns,
        "refused_launches": ref_lines,
        "refused_turns": ref_turns,
    }


def evict_window(run):
    lo, hi = 0.0, run.cut
    evd = sum(r[1] for r in run.evd if lo <= r[0] - run.t_start < hi)
    evh = sum(r[1] for r in run.evh if lo <= r[0] - run.t_start < hi)
    evd_n = sum(1 for r in run.evd if lo <= r[0] - run.t_start < hi)
    evh_n = sum(1 for r in run.evh if lo <= r[0] - run.t_start < hi)
    return evd, evh, evd_n, evh_n


def item23(run, keys, slack=0.1, bound=0.1, thr=0.05):
    keys = sorted(keys)
    n = len(keys)
    o = {"n": n, "missing_rts": 0}
    sq = 0.0
    sres = 0.0
    sttft = 0.0
    stot = 0.0
    landed = 0
    landed_qual = 0
    no_qual = 0
    landed_sum = 0.0
    lw_sum = 0.0
    lw_n = 0
    gq_sum = 0.0
    gq_n = 0
    qw_gt_ttft = 0
    qws = []
    sts = []
    rd_ttft = rd_qw = 0.0
    rd_n = 0
    before_arrival = 0
    for k in keys:
        if k not in run.rts_last:
            o["missing_rts"] += 1
            continue
        st, has_q, qw = run.stall(k, slack)
        ttft = run.cli[k]["ttft"]
        qws.append(qw)
        sts.append(st)
        if k in run.s2h:
            rd_ttft += ttft
            rd_qw += qw
            rd_n += 1
            ends = [x[0] for x in run.s2h[k] if x[0] <= run.arrival(k) + qw + slack]
            if ends and max(ends) <= run.arrival(k):
                before_arrival += 1
        sq += qw
        sres += ttft - qw
        sttft += ttft
        stot += st
        if qw > ttft + 0.05:
            qw_gt_ttft += 1
        if k not in run.s2h:
            continue
        if st < thr:
            landed += 1
            landed_sum += st
            if has_q:
                landed_qual += 1
            else:
                no_qual += 1
        elif (qw - st) < bound:
            lw_sum += st
            lw_n += 1
        else:
            gq_sum += st
            gq_n += 1
    o.update({
        "mean_ttft": sttft / n, "mean_qw": sq / n, "mean_ttft_minus_qw": sres / n,
        "mean_stall": stot / n, "landed_n": landed, "landed_qualifying_n": landed_qual,
        "reading_no_qualifying_read_n": no_qual, "landed_sum_per_turn": landed_sum / n,
        "lastwait_sum_per_turn": lw_sum / n, "lastwait_n": lw_n,
        "gpuq_sum_per_turn": gq_sum / n, "gpuq_n": gq_n,
        "qw_gt_ttft_share": qw_gt_ttft / n, "qw_gt_ttft_n": qw_gt_ttft,
        "sum_ttft": sttft, "sum_qw": sq, "sum_stall": stot,
        "lastwait_sum": lw_sum, "gpuq_sum": gq_sum,
        "last_read_ended_before_arrival_n": before_arrival,
        "qw_p50": pct(qws, 0.5) if qws else float("nan"),
        "qw_p90": pct(qws, 0.9) if qws else float("nan"),
        "qw_p99": pct(qws, 0.99) if qws else float("nan"),
        "qw_max": max(qws) if qws else float("nan"),
        "stall_p90": pct(sts, 0.9) if sts else float("nan"),
        "stall_p99": pct(sts, 0.99) if sts else float("nan"),
        "stall_max": max(sts) if sts else float("nan"),
        "reading_n": rd_n,
        "reading_mean_ttft": (rd_ttft / rd_n) if rd_n else float("nan"),
        "reading_mean_qw": (rd_qw / rd_n) if rd_n else float("nan"),
        "reading_mean_stall": (stot / rd_n) if rd_n else float("nan"),
        "nonreading_mean_ttft": ((sttft - rd_ttft) / (n - rd_n)) if n > rd_n else float("nan"),
        "nonreading_mean_qw": ((sq - rd_qw) / (n - rd_n)) if n > rd_n else float("nan"),
    })
    return o


def main():
    t0 = time.time()
    labels = sorted(set(ALL11 + CUT7 + ["order_run"]))
    R = {lab: Run(lab) for lab in labels}
    P("analyze.py  generated %s UTC" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    P("definitions: kept = turns with ttft, sorted by t_send, t_send < max t_send - 300;")
    P("shared = returning (turn>0) keys kept in every arm of the set; queue wait = queue_duration")
    P("of the key's last ReqTimeStats line; percentile q = sorted(v)[min(len-1,int(q*len))].")
    P()

    # ---------------- per-run basics
    P("PER-RUN BASICS")
    P("%-14s %7s %7s %7s %9s %9s %6s %6s %6s" % ("run", "turns", "kept", "keptRet", "max_tsend",
                                                 "window_s", "dupRTS", "noRTS", "dupCli"))
    for lab in labels:
        r = R[lab]
        dup = sum(1 for c in r.rts_n.values() if c > 1)
        nor = sum(1 for k in r.cli if k not in r.rts_last)
        P("%-14s %7d %7d %7d %9.3f %9.3f %6d %6d %6d" % (
            lab, len(r.turns), len(r.kept), len(r.kept_ret), r.max_t_send, r.cut, dup, nor,
            r.cli_dups))
        V("2", lab, "", "keys_with_more_than_one_ReqTimeStats_line", dup, "keys")
        V("2", lab, "", "client_turns_without_ReqTimeStats_line", nor, "keys")
        V("0", lab, "", "turn_records", len(r.turns), "turns")
        V("0", lab, "", "kept_turns", len(r.kept), "turns")
        V("0", lab, "", "kept_returning_turns", len(r.kept_ret), "turns")
        V("0", lab, "", "kept_window_length", r.cut, "s")
        if lab != "order_run":
            k1 = item1(r, r.kept_ret)
            evd, evh, evd_n, evh_n = evict_window(r)
            V("1", lab, "", "ssd_pages_kept_returning_turns", k1["ssd_pages"], "pages")
            V("1", lab, "", "turns_that_read_from_ssd_kept_returning", k1["reading_turns"], "turns")
            V("1", lab, "", "s2h_io_lines_kept_returning", k1["s2h_lines"], "lines")
            V("1", lab, "", "evict_device_tokens_kept_window", evd, "tokens")
            V("1", lab, "", "evict_host_tokens_kept_window", evh, "tokens")
            V("1", lab, "", "evict_device_lines_kept_window", evd_n, "lines")
            V("1", lab, "", "evict_host_lines_kept_window", evh_n, "lines")
    P()

    # ---------------- positions from the order run
    order = R["order_run"]
    pos = {}
    for i, t in enumerate(order.ok_sorted):
        pos[(t["conv"], t["turn"])] = i
    P("ORDER RUN: %d turns with ttft; positions 0..%d" % (len(pos), len(pos) - 1))

    def scope_of(k):
        p = pos.get(k)
        if p is None:
            return "else"
        for name, lo, hi in EPISODES:
            if lo <= p <= hi:
                return name
        return "else"

    shared = {}
    for sname, arms in SETS:
        s = None
        for a in arms:
            s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
        shared[sname] = s
        sc = Counter(scope_of(k) for k in s)
        absent = sum(1 for k in s if k not in pos)
        P("%s: shared returning turns = %d  (ep1 %d, ep2 %d, elsewhere %d of which absent from order run %d)"
          % (sname, len(s), sc["ep1"], sc["ep2"], sc["else"], absent))
        V("0", "all", sname, "shared_returning_turns", len(s), "turns")
        V("0", "all", sname, "shared_turns_in_episode1", sc["ep1"], "turns")
        V("0", "all", sname, "shared_turns_in_episode2", sc["ep2"], "turns")
        V("0", "all", sname, "shared_turns_elsewhere", sc["else"], "turns")
        V("0", "all", sname, "shared_turns_absent_from_order_run", absent, "turns")
    P()

    # ---------------- item 1
    for sname, arms in SETS:
        sh = shared[sname]
        P("ITEM 1  set %s  (shared turns %d)" % (sname, len(sh)))
        P("%-13s %7s %8s %9s %10s %8s %7s %7s %7s %7s %7s %12s %12s" % (
            "arm", "keptRet", "readTurn", "pagesShrd", "pagesKeptR", "s2hLines", "rd/turn",
            "staged", "stgTurn", "refused", "refTurn", "evDevTok", "evHostTok"))
        for a in arms:
            r = R[a]
            s1 = item1(r, sh)
            k1 = item1(r, r.kept_ret)
            evd, evh, evd_n, evh_n = evict_window(r)
            P("%-13s %7d %8d %9d %10d %8d %7.4f %7d %7d %7d %7d %12d %12d" % (
                a, len(r.kept_ret), s1["reading_turns"], s1["ssd_pages"], k1["ssd_pages"],
                s1["s2h_lines"], s1["reads_per_reading_turn"], s1["staged_reads"],
                s1["staged_turns"], s1["refused_launches"], s1["refused_turns"], evd, evh))
            V("1", a, sname, "turns_that_read_from_ssd", s1["reading_turns"], "turns")
            V("1", a, sname, "ssd_pages_shared_turns", s1["ssd_pages"], "pages")
            V("1", a, sname, "s2h_io_lines_shared_turns", s1["s2h_lines"], "lines")
            V("1", a, sname, "reads_per_reading_turn", s1["reads_per_reading_turn"], "reads/turn")
            V("1", a, sname, "staged_reads_ntp_stage_lines", s1["staged_reads"], "lines")
            V("1", a, sname, "turns_with_staged_read", s1["staged_turns"], "turns")
            V("1", a, sname, "refused_launches", s1["refused_launches"], "lines")
            V("1", a, sname, "turns_with_refused_launch", s1["refused_turns"], "turns")
        P()

    # ---------------- items 2 and 3
    for sname, arms in SETS:
        sh = shared[sname]
        P("ITEMS 2+3  set %s  (shared turns %d; slack 0.1 s, admitted-within bound 0.1 s, landed threshold 0.05 s)" % (sname, len(sh)))
        P("%-13s %9s %9s %9s %9s | %7s %7s %6s | %9s %6s | %9s %6s | %8s" % (
            "arm", "meanTTFT", "meanQW", "TTFT-QW", "meanStall", "landed", "landedQ", "noQual",
            "lastWait", "n", "gpuQueue", "n", "qw>ttft"))
        for a in arms:
            o = item23(R[a], sh)
            P("%-13s %9.4f %9.4f %9.4f %9.4f | %7d %7d %6d | %9.4f %6d | %9.4f %6d | %8.5f" % (
                a, o["mean_ttft"], o["mean_qw"], o["mean_ttft_minus_qw"], o["mean_stall"],
                o["landed_n"], o["landed_qualifying_n"], o["reading_no_qualifying_read_n"],
                o["lastwait_sum_per_turn"], o["lastwait_n"], o["gpuq_sum_per_turn"], o["gpuq_n"],
                o["qw_gt_ttft_share"]))
            V("2", a, sname, "mean_queue_wait", o["mean_qw"], "s")
            V("2", a, sname, "mean_ttft_minus_queue_wait", o["mean_ttft_minus_qw"], "s")
            V("2", a, sname, "mean_ttft", o["mean_ttft"], "s")
            V("3", a, sname, "mean_read_stall_per_shared_turn", o["mean_stall"], "s")
            V("3", a, sname, "reading_turns_landed_by_arrival", o["landed_n"], "turns")
            V("3", a, sname, "reading_turns_landed_with_a_qualifying_read", o["landed_qualifying_n"], "turns")
            V("3", a, sname, "reading_turns_without_a_qualifying_read", o["reading_no_qualifying_read_n"], "turns")
            V("3", a, sname, "stall_read_was_last_wait_per_shared_turn", o["lastwait_sum_per_turn"], "s")
            V("3", a, sname, "turns_read_was_last_wait", o["lastwait_n"], "turns")
            V("3", a, sname, "stall_gpu_queue_held_as_well_per_shared_turn", o["gpuq_sum_per_turn"], "s")
            V("3", a, sname, "turns_gpu_queue_held_as_well", o["gpuq_n"], "turns")
            V("3", a, sname, "stall_of_landed_turns_per_shared_turn", o["landed_sum_per_turn"], "s")
            V("3", a, sname, "reading_turns_last_read_ended_before_arrival", o["last_read_ended_before_arrival_n"], "turns")
            for m, u in (("qw_p50", "s"), ("qw_p90", "s"), ("qw_p99", "s"), ("qw_max", "s"),
                         ("stall_p90", "s"), ("stall_p99", "s"), ("stall_max", "s"),
                         ("reading_mean_ttft", "s"), ("reading_mean_qw", "s"),
                         ("reading_mean_stall", "s"), ("nonreading_mean_ttft", "s"),
                         ("nonreading_mean_qw", "s"), ("sum_qw", "s"), ("sum_stall", "s"),
                         ("sum_ttft", "s"), ("lastwait_sum", "s"), ("gpuq_sum", "s")):
                V("x-23", a, sname, m, o[m], u)
            EXTRA_ROWS.append((sname, a, o))
            V("5c", a, sname, "share_shared_turns_queue_wait_exceeds_ttft_by_0.05s", o["qw_gt_ttft_share"], "fraction")
            V("5c", a, sname, "turns_queue_wait_exceeds_ttft_by_0.05s", o["qw_gt_ttft_n"], "turns")
        P()

    P("EXTRA to items 2+3: percentiles and the reading / non-reading split (shared turns)")
    P("%-6s %-13s %8s %8s %8s %8s | %8s %8s %8s | %6s %9s %9s %9s | %9s %9s | %7s" % (
        "set", "arm", "qw p50", "qw p90", "qw p99", "qw max", "stl p90", "stl p99", "stl max",
        "readN", "rd TTFT", "rd QW", "rd stall", "nonrd TTFT", "nonrd QW", "preArr"))
    for sname, a, o in EXTRA_ROWS:
        P("%-6s %-13s %8.4f %8.4f %8.4f %8.3f | %8.4f %8.4f %8.3f | %6d %9.4f %9.4f %9.4f | %9.4f %9.4f | %7d" % (
            sname, a, o["qw_p50"], o["qw_p90"], o["qw_p99"], o["qw_max"], o["stall_p90"],
            o["stall_p99"], o["stall_max"], o["reading_n"], o["reading_mean_ttft"],
            o["reading_mean_qw"], o["reading_mean_stall"], o["nonreading_mean_ttft"],
            o["nonreading_mean_qw"], o["last_read_ended_before_arrival_n"]))
    P()

    # ---------------- item 4
    variants = [("bound", 0.05, 0.1, 0.05), ("bound", 0.2, 0.1, 0.2), ("bound", 0.5, 0.1, 0.5),
                ("slack", 0.0, 0.0, 0.1), ("slack", 0.5, 0.5, 0.1)]
    for sname, arms in SETS[:2]:
        sh = shared[sname]
        P("ITEM 4  set %s  robustness of the two sums (s per shared turn) and their counts" % sname)
        P("%-13s %-12s %9s %6s %9s %6s %9s %7s" % ("arm", "variant", "lastWait", "n", "gpuQueue",
                                                   "n", "meanStall", "landed"))
        for a in arms:
            base = item23(R[a], sh)
            P("%-13s %-12s %9.4f %6d %9.4f %6d %9.4f %7d" % (
                a, "base", base["lastwait_sum_per_turn"], base["lastwait_n"],
                base["gpuq_sum_per_turn"], base["gpuq_n"], base["mean_stall"], base["landed_n"]))
            for what, val, slack, bound in variants:
                o = item23(R[a], sh, slack=slack, bound=bound)
                tag = "%s=%g" % (what, val)
                P("%-13s %-12s %9.4f %6d %9.4f %6d %9.4f %7d" % (
                    a, tag, o["lastwait_sum_per_turn"], o["lastwait_n"], o["gpuq_sum_per_turn"],
                    o["gpuq_n"], o["mean_stall"], o["landed_n"]))
                V("4", a, sname, "stall_read_was_last_wait_per_shared_turn[%s]" % tag, o["lastwait_sum_per_turn"], "s")
                V("4", a, sname, "turns_read_was_last_wait[%s]" % tag, o["lastwait_n"], "turns")
                V("4", a, sname, "stall_gpu_queue_held_as_well_per_shared_turn[%s]" % tag, o["gpuq_sum_per_turn"], "s")
                V("4", a, sname, "turns_gpu_queue_held_as_well[%s]" % tag, o["gpuq_n"], "turns")
                if what == "slack":
                    V("4", a, sname, "mean_read_stall_per_shared_turn[%s]" % tag, o["mean_stall"], "s")
                    V("4", a, sname, "reading_turns_landed_by_arrival[%s]" % tag, o["landed_n"], "turns")
        P()

    # ---------------- item 5
    P("ITEM 5a  (read start - arrival) over every s2h_io line with a ReqTimeStats line for its key; read start = line time - ms")
    P("%-13s %7s %7s %10s %10s %10s %10s %10s %9s %9s" % (
        "arm", "lines", "noRTS", "min", "p01", "p50", "p99", "max", "sh<-0.05", "n<-0.05"))
    for a in sorted(set(ALL11 + CUT7), key=lambda x: (x not in ("nofc", "nofc_noend"), x not in LEAD5, x)):
        r = R[a]
        dv = []
        nor = 0
        for x in r.s2h_all:
            k = (x[1], x[2])
            if k not in r.rts_last:
                nor += 1
                continue
            dv.append((x[0] - x[5] / 1000.0) - r.rts_last[k][5])
        nneg = sum(1 for z in dv if z < -0.05)
        P("%-13s %7d %7d %10.4f %10.4f %10.4f %10.4f %10.4f %9.5f %9d" % (
            a, len(dv), nor, min(dv), pct(dv, 0.01), pct(dv, 0.5), pct(dv, 0.99), max(dv),
            nneg / len(dv), nneg))
        if a in LEAD5:
            V("5a", a, "", "s2h_io_lines_with_arrival", len(dv), "lines")
            V("5a", a, "", "s2h_io_lines_without_ReqTimeStats", nor, "lines")
            V("5a", a, "", "read_start_minus_arrival_min", min(dv), "s")
            V("5a", a, "", "read_start_minus_arrival_p01", pct(dv, 0.01), "s")
            V("5a", a, "", "read_start_minus_arrival_median", pct(dv, 0.5), "s")
            V("5a", a, "", "read_start_minus_arrival_share_below_-0.05s", nneg / len(dv), "fraction")
            V("5a", a, "", "read_start_minus_arrival_lines_below_-0.05s", nneg, "lines")
    P()
    P("ITEM 5a restricted to s2h_io lines of LEAD5 shared turns")
    for a in LEAD5:
        r = R[a]
        dv = [(x[0] - x[5] / 1000.0) - r.rts_last[k][5]
              for k in sorted(shared["LEAD5"]) for x in r.s2h.get(k, ())]
        nneg = sum(1 for z in dv if z < -0.05)
        P("%-13s %7d %10.4f %10.4f %10.4f %9.5f %9d" % (
            a, len(dv), min(dv), pct(dv, 0.01), pct(dv, 0.5), nneg / len(dv), nneg))
        V("5a", a, "LEAD5", "read_start_minus_arrival_min[shared turns only]", min(dv), "s")
        V("5a", a, "LEAD5", "read_start_minus_arrival_p01[shared turns only]", pct(dv, 0.01), "s")
        V("5a", a, "LEAD5", "read_start_minus_arrival_median[shared turns only]", pct(dv, 0.5), "s")
        V("5a", a, "LEAD5", "read_start_minus_arrival_share_below_-0.05s[shared turns only]", nneg / len(dv), "fraction")
    P()
    P("ITEM 5b  (ReqTimeStats line time - entry_time) - client latency, all client turns with a ReqTimeStats line; and LEAD5 shared turns")
    P("%-13s %7s %10s %10s %10s %10s %10s | %7s %10s %10s %10s" % (
        "arm", "n", "min", "p01", "p50", "p99", "max", "nShared", "p01", "p50", "p99"))
    for a in sorted(set(ALL11 + CUT7), key=lambda x: (x not in LEAD5, x)):
        r = R[a]
        dv = []
        for k, t in r.cli.items():
            if k in r.rts_last and t["latency"] is not None:
                x = r.rts_last[k]
                dv.append((x[0] - x[5]) - t["latency"])
        ds = []
        if a in LEAD5:
            for k in sorted(shared["LEAD5"]):
                x = r.rts_last[k]
                ds.append((x[0] - x[5]) - r.cli[k]["latency"])
        P("%-13s %7d %10.4f %10.4f %10.4f %10.4f %10.4f | %7d %10s %10s %10s" % (
            a, len(dv), min(dv), pct(dv, 0.01), pct(dv, 0.5), pct(dv, 0.99), max(dv), len(ds),
            ("%.4f" % pct(ds, 0.01)) if ds else "-", ("%.4f" % pct(ds, 0.5)) if ds else "-",
            ("%.4f" % pct(ds, 0.99)) if ds else "-"))
        if a in LEAD5:
            V("5b", a, "", "server_minus_client_latency_n", len(dv), "turns")
            V("5b", a, "", "server_minus_client_latency_median", pct(dv, 0.5), "s")
            V("5b", a, "", "server_minus_client_latency_p01", pct(dv, 0.01), "s")
            V("5b", a, "", "server_minus_client_latency_p99", pct(dv, 0.99), "s")
            V("5b", a, "", "server_minus_client_latency_min", min(dv), "s")
            V("5b", a, "", "server_minus_client_latency_max", max(dv), "s")
            V("5b", a, "LEAD5", "server_minus_client_latency_median[shared turns only]", pct(ds, 0.5), "s")
            V("5b", a, "LEAD5", "server_minus_client_latency_p01[shared turns only]", pct(ds, 0.01), "s")
            V("5b", a, "LEAD5", "server_minus_client_latency_p99[shared turns only]", pct(ds, 0.99), "s")
    P()
    P("EXTRA clock check: entry_time - (client t_start + t_send), all client turns")
    P("%-13s %7s %10s %10s %10s %10s %10s" % ("arm", "n", "min", "p01", "p50", "p99", "max"))
    for a in sorted(set(ALL11 + CUT7), key=lambda x: (x not in LEAD5, x)):
        r = R[a]
        dv = [r.rts_last[k][5] - (r.t_start + t["t_send"]) for k, t in r.cli.items() if k in r.rts_last]
        P("%-13s %7d %10.4f %10.4f %10.4f %10.4f %10.4f" % (
            a, len(dv), min(dv), pct(dv, 0.01), pct(dv, 0.5), pct(dv, 0.99), max(dv)))
        if a in LEAD5:
            V("5x", a, "", "entry_time_minus_client_send_time_median", pct(dv, 0.5), "s")
            V("5x", a, "", "entry_time_minus_client_send_time_p01", pct(dv, 0.01), "s")
            V("5x", a, "", "entry_time_minus_client_send_time_p99", pct(dv, 0.99), "s")
            V("5x", a, "", "entry_time_minus_client_send_time_min", min(dv), "s")
            V("5x", a, "", "entry_time_minus_client_send_time_max", max(dv), "s")
    P()
    P("ITEM 5c is the qw>ttft column of the ITEMS 2+3 tables (share of shared turns with queue wait > ttft + 0.05 s)")
    P()

    # ---------------- extras: episode splits for the comparison with split_ec_lead.out
    for sname, arms in SETS:
        sh = shared[sname]
        groups = defaultdict(list)
        for k in sh:
            groups[scope_of(k)].append(k)
        groups["episodes"] = groups["ep1"] + groups["ep2"]
        P("EXTRA split by position, set %s" % sname)
        P("%-13s %-9s %6s %9s %9s %9s %9s %9s %9s %8s %9s %9s %9s %9s" % (
            "arm", "scope", "n", "meanTTFT", "p50", "p90", "p99", "meanQW", "TTFT-QW", "readTurn",
            "pages", "meanStall", "lastWait", "gpuQueue"))
        for a in arms:
            r = R[a]
            for sc in ("ep1", "ep2", "episodes", "else"):
                ks = groups[sc]
                if not ks:
                    continue
                o = item23(r, ks)
                s1 = item1(r, ks)
                tt = [r.cli[k]["ttft"] for k in ks]
                rec = [max(r.cli[k]["reply_reprefilled"] or 0, 0) for k in ks]
                P("%-13s %-9s %6d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %8d %9d %9.4f %9.4f %9.4f" % (
                    a, sc, len(ks), o["mean_ttft"], pct(tt, 0.5), pct(tt, 0.9), pct(tt, 0.99),
                    o["mean_qw"], o["mean_ttft_minus_qw"], s1["reading_turns"], s1["ssd_pages"],
                    o["mean_stall"], o["lastwait_sum_per_turn"], o["gpuq_sum_per_turn"]))
                for m, val, u in (
                        ("n_turns", len(ks), "turns"), ("mean_ttft", o["mean_ttft"], "s"),
                        ("p50_ttft", pct(tt, 0.5), "s"), ("p90_ttft", pct(tt, 0.9), "s"),
                        ("p95_ttft", pct(tt, 0.95), "s"), ("p99_ttft", pct(tt, 0.99), "s"),
                        ("sum_ttft", o["sum_ttft"], "s"),
                        ("mean_queue_wait", o["mean_qw"], "s"),
                        ("mean_ttft_minus_queue_wait", o["mean_ttft_minus_qw"], "s"),
                        ("mean_read_stall", o["mean_stall"], "s"),
                        ("turns_that_read_from_ssd", s1["reading_turns"], "turns"),
                        ("ssd_pages", s1["ssd_pages"], "pages"),
                        ("s2h_io_lines", s1["s2h_lines"], "lines"),
                        ("recomputed_tokens_sum", sum(rec), "tokens"),
                        ("recomputed_tokens_mean", mean(rec), "tokens"),
                        ("sum_queue_wait", o["sum_qw"], "s"), ("sum_stall", o["sum_stall"], "s"),
                        ("lastwait_sum", o["lastwait_sum"], "s"), ("gpuq_sum", o["gpuq_sum"], "s"),
                        ("lastwait_n", o["lastwait_n"], "turns"), ("gpuq_n", o["gpuq_n"], "turns")):
                    V("x-split:" + sc, a, sname, m, val, u)
        P()

    # ---------------- extras: whole-set client-side figures on the shared turns
    for sname, arms in SETS:
        sh = sorted(shared[sname])
        P("EXTRA shared-turn TTFT and recomputed tokens, set %s (n=%d)" % (sname, len(sh)))
        P("%-13s %9s %9s %9s %9s %9s %12s %10s %12s %10s" % (
            "arm", "mean", "p50", "p90", "p95", "p99", "recompSum", "recompMean", "cachedSum", "promptSum"))
        for a in arms:
            r = R[a]
            tt = [r.cli[k]["ttft"] for k in sh]
            rec = [max(r.cli[k]["reply_reprefilled"] or 0, 0) for k in sh]
            cached = sum((r.cli[k]["cached_tokens"] or 0) for k in sh)
            prompt = sum((r.cli[k]["prompt_tokens"] or 0) for k in sh)
            P("%-13s %9.4f %9.4f %9.4f %9.4f %9.4f %12d %10.2f %12d %10d" % (
                a, mean(tt), pct(tt, 0.5), pct(tt, 0.9), pct(tt, 0.95), pct(tt, 0.99), sum(rec),
                mean(rec), cached, prompt))
            for m, val, u in (("mean_ttft", mean(tt), "s"), ("p50_ttft", pct(tt, 0.5), "s"),
                              ("p90_ttft", pct(tt, 0.9), "s"), ("p95_ttft", pct(tt, 0.95), "s"),
                              ("p99_ttft", pct(tt, 0.99), "s"),
                              ("recomputed_tokens_sum", sum(rec), "tokens"),
                              ("recomputed_tokens_mean", mean(rec), "tokens"),
                              ("cached_tokens_sum", cached, "tokens"),
                              ("prompt_tokens_sum", prompt, "tokens")):
                V("x-shared", a, sname, m, val, u)
        P()

    # ---------------- extras: whole-run server totals (no window, and by time window)
    P("EXTRA whole-run totals (all log lines) and kept-window-by-time totals")
    P("%-13s %8s %10s %10s %10s %12s %12s %8s %8s" % (
        "arm", "s2hAll", "pagesAll", "pagesKeptW", "pagesRet0", "evDevAll", "evHostAll", "ntpAll", "refAll"))
    for a in sorted(set(ALL11 + CUT7), key=lambda x: (x not in LEAD5, x)):
        r = R[a]
        pall = sum(x[3] for x in r.s2h_all)
        pwin = sum(x[3] for x in r.s2h_all if 0.0 <= x[0] - r.t_start < r.cut)
        p0 = sum(x[3] for x in r.s2h_all if x[2] == 0)
        evd_all = sum(x[1] for x in r.evd)
        evh_all = sum(x[1] for x in r.evh)
        ntp_all = sum(len(v) for v in r.ntp.values())
        ref_all = sum(1 for v in r.srs.values() for x in v if x[7] == 0)
        P("%-13s %8d %10d %10d %10d %12d %12d %8d %8d" % (
            a, len(r.s2h_all), pall, pwin, p0, evd_all, evh_all, ntp_all, ref_all))
        for m, val, u in (("s2h_io_lines_whole_log", len(r.s2h_all), "lines"),
                          ("ssd_pages_whole_log", pall, "pages"),
                          ("ssd_pages_lines_in_kept_window_by_time", pwin, "pages"),
                          ("ssd_pages_first_turn_keys", p0, "pages"),
                          ("evict_device_tokens_whole_log", evd_all, "tokens"),
                          ("evict_host_tokens_whole_log", evh_all, "tokens"),
                          ("ntp_stage_lines_whole_log", ntp_all, "lines"),
                          ("refused_launches_whole_log", ref_all, "lines")):
            V("x-run", a, "", m, val, u)
    P()
    # ---------------- extras: Decode / Prefill batch lines inside the kept window (by line time)
    P("EXTRA Decode batch and Prefill batch lines with 0 <= line time - t_start < kept window")
    P("%-13s %6s %8s %8s %9s %8s %8s %9s | %6s %7s %11s %11s %8s %9s" % (
        "arm", "decN", "meanRun", "meanQ", "shareQ>0", "meanUse", "maxUse", "meanThr",
        "preN", "newSeq", "newTok", "cachedTok", "meanQ", "shareQ>0"))
    for a in sorted(set(ALL11 + CUT7), key=lambda x: (x not in LEAD5, x)):
        r = R[a]
        dec = [x for x in r.dec if 0.0 <= x[0] - r.t_start < r.cut]
        pre = [x for x in r.pre if 0.0 <= x[0] - r.t_start < r.cut]
        preq = [x[5] for x in pre if x[5] is not None]
        o = {
            "decode_lines_kept_window": len(dec),
            "decode_mean_running_req": mean([x[1] for x in dec]),
            "decode_mean_queue_req": mean([x[5] for x in dec]),
            "decode_share_lines_queue_req_gt0": sum(1 for x in dec if x[5] > 0) / len(dec),
            "decode_mean_token_usage": mean([x[3] for x in dec]),
            "decode_max_token_usage": max(x[3] for x in dec),
            "decode_mean_gen_throughput": mean([x[4] for x in dec]),
            "prefill_lines_kept_window": len(pre),
            "prefill_new_seq_sum": sum(x[1] for x in pre),
            "prefill_new_token_sum": sum(x[2] for x in pre),
            "prefill_cached_token_sum": sum(x[3] for x in pre),
            "prefill_mean_queue_req": mean(preq),
            "prefill_share_lines_queue_req_gt0": (sum(1 for z in preq if z > 0) / len(preq)) if preq else float("nan"),
        }
        P("%-13s %6d %8.3f %8.3f %9.4f %8.4f %8.2f %9.2f | %6d %7d %11d %11d %8.3f %9.4f" % (
            a, o["decode_lines_kept_window"], o["decode_mean_running_req"], o["decode_mean_queue_req"],
            o["decode_share_lines_queue_req_gt0"], o["decode_mean_token_usage"],
            o["decode_max_token_usage"], o["decode_mean_gen_throughput"],
            o["prefill_lines_kept_window"], o["prefill_new_seq_sum"], o["prefill_new_token_sum"],
            o["prefill_cached_token_sum"], o["prefill_mean_queue_req"],
            o["prefill_share_lines_queue_req_gt0"]))
        for m, val in o.items():
            V("x-run", a, "", m, val, "")
    P()
    P("done in %.1f s; %d values" % (time.time() - t0, len(VALUES)))

    with open(os.path.join(OUT, "results_items1to5.json"), "w") as f:
        json.dump(VALUES, f, indent=0)
    with open(os.path.join(OUT, "results_items1to5.txt"), "w") as f:
        f.write("\n".join(TXT) + "\n")
    print("\n".join(TXT))


if __name__ == "__main__":
    main()
