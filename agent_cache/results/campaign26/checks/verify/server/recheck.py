#!/usr/bin/env python3
"""Second, from-scratch implementation of items 1-5 (campaign 26 server-side check).

Written after this check was restarted, before any existing *.out file, analysis
script or DECISIONS.md was opened in the restarted session.  It shares no code
with parse_runs.py / analyze.py: it reads the raw client.jsonl and server.log
again with plain string handling (no regex, no cache) and does all time
arithmetic of the stall definition in exact integer microseconds, so that the
threshold decisions (0.05 s, 0.1 s, ...) cannot depend on float rounding.

It then diffs every number it has in common with frozen_results_items1to5.json.

Run as:  nice -n 19 python3 -I -B recheck.py
Writes:  recheck.out, recheck.json, recheck_turns.json (per-turn table, LEAD5+SEVEN arms)
"""
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

RES = "/home/wanhr/sglang/agent_cache/results"
OUT = RES + "/campaign26/checks/verify/server"

RUNS = {
    "baseline": "20261101_082229_three_tier_wc_custom_L1nat_H70",
    "base_rep": "20261105_075438_three_tier_wc_custom_L1nat_H70",
    "hintfree": "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "hintfree_ctl": "20261111_045808_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "jit_p50": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_ctl": "20261111_024927_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_d_cut": "20261111_151101_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_exact": "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r13": "20261111_070012_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r1": "20261111_110558_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "nofc": "20261111_090311_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "nofc_noend": "20261111_130801_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "order_run": "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
}
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
         "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
CUT7 = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13",
        "jit_exact"]
SETS = {"LEAD5": LEAD5, "SEVEN": SEVEN, "ALL11": ALL11, "CUT7": CUT7}
US = 1000000

lines_out = []
values = []


def P(s=""):
    lines_out.append(s)


def V(item, arm, arm_set, metric, value):
    values.append((item, arm, arm_set, metric, value))


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def line_us(line):
    """'[YYYY-MM-DD HH:MM:SS.mmm] ' -> integer microseconds since the epoch (UTC), or None."""
    if len(line) < 26 or line[0] != "[" or line[24] != "]" or line[25] != " ":
        return None
    try:
        dt = datetime(int(line[1:5]), int(line[6:8]), int(line[9:11]), int(line[12:14]),
                      int(line[15:17]), int(line[18:20]), tzinfo=timezone.utc)
        ms = int(line[21:24])
    except ValueError:
        return None
    if line[5] != "-" or line[8] != "-" or line[11] != " " or line[14] != ":" or line[20] != ".":
        return None
    return int(dt.timestamp()) * US + ms * 1000


def tok_after(text, name):
    """Value of name=<value> in text, cut at a blank, comma, ')' or line end; None if absent."""
    i = text.find(name + "=")
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_"):
        i = text.find(name + "=", i + 1)
    if i < 0:
        return None
    j = i + len(name) + 1
    e = j
    n = len(text)
    while e < n and text[e] not in " ,)\n\r\t":
        e += 1
    return text[j:e]


def key_of(rid):
    """'...-c<conv>-t<turn>' -> (conv, turn) or None."""
    if rid is None:
        return None
    parts = rid.rsplit("-", 2)
    if len(parts) != 3:
        return None
    c, t = parts[1], parts[2]
    if len(c) < 2 or len(t) < 2 or c[0] != "c" or t[0] != "t":
        return None
    if not (c[1:].isdigit() and t[1:].isdigit()):
        return None
    return (int(c[1:]), int(t[1:]))


def dec_us(s, unit_us):
    """Decimal string times unit_us microseconds -> exact integer microseconds."""
    d = Decimal(s) * unit_us
    i = int(d)
    if d != i:
        raise ValueError("not a whole number of microseconds: %r" % s)
    return i


class Run:
    pass


def load(label, current):
    folder = RUNS[label]
    if folder == current or not current:
        raise SystemExit("refusing to read %s (current run %r)" % (folder, current))
    rd = os.path.join(RES, folder)
    cds = sorted(d for d in os.listdir(rd)
                 if d.startswith("client_") and os.path.isdir(os.path.join(rd, d)))
    r = Run()
    r.label = label
    r.n_client_dirs = len(cds)
    turns = []
    t_start = None
    with open(os.path.join(rd, cds[-1], "client.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "run":
                assert t_start is None
                t_start = rec["t_start"]
            elif kind == "turn":
                turns.append(rec)
    r.t_start = t_start
    r.turns = turns
    ok = sorted((t for t in turns if t.get("ttft") is not None), key=lambda t: t["t_send"])
    r.ok = ok
    r.null_ttft = len(turns) - len(ok)
    r.max_send_ok = max(t["t_send"] for t in ok)
    r.max_send_all = max(t["t_send"] for t in turns)
    r.cut = r.max_send_ok - 300.0
    r.kept = [t for t in ok if t["t_send"] < r.cut]
    r.cli = {}
    for t in turns:
        k = (t["conv"], t["turn"])
        assert k not in r.cli, "duplicate client key %r in %s" % (k, label)
        r.cli[k] = t
    r.kept_keys = set((t["conv"], t["turn"]) for t in r.kept)
    assert len(r.kept_keys) == len(r.kept)
    r.kept_ret = set(k for k in r.kept_keys if k[1] > 0)

    rts = {}      # key -> list of (line_us, queue_us, entry_us)
    s2h = {}      # key -> list of (end_us, pages, ms)
    ntp = {}      # key -> number of ntp_stage lines
    refused = {}  # key -> number of srs_dispatch lines with launched=0
    evd = []
    evh = []
    n_rts_all = n_rts_unkeyed = 0
    n_s2h_all = n_ntp_all = n_srs_all = 0
    with open(os.path.join(rd, "server.log"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            i_evt = line.find("HICACHE_EVT ")
            i_rts = line.find("ReqTimeStats(") if i_evt < 0 else -1
            if i_evt < 0 and i_rts < 0:
                continue
            t = line_us(line)
            if t is None:
                raise SystemExit("line of interest without a time stamp in %s: %r" % (label, line[:200]))
            if i_rts >= 0:
                n_rts_all += 1
                body = line[i_rts:]
                k = key_of(tok_after(body, "rid"))
                if k is None:
                    n_rts_unkeyed += 1
                    continue
                q = tok_after(body, "queue_duration")
                assert q.endswith("ms"), q
                e = tok_after(body, "entry_time")
                rts.setdefault(k, []).append((t, dec_us(q[:-2], 1000), dec_us(e, US)))
                continue
            body = line[i_evt + len("HICACHE_EVT "):]
            kind = body.split(" ", 1)[0].strip()
            if kind == "s2h_io":
                n_s2h_all += 1
                k = key_of(tok_after(body, "rid"))
                assert k is not None, line
                s2h.setdefault(k, []).append((t, int(tok_after(body, "pages")), int(tok_after(body, "ms"))))
            elif kind == "ntp_stage":
                n_ntp_all += 1
                k = key_of(tok_after(body, "rid"))
                assert k is not None, line
                ntp[k] = ntp.get(k, 0) + 1
            elif kind == "srs_dispatch":
                n_srs_all += 1
                k = key_of(tok_after(body, "rid"))
                assert k is not None, line
                la = tok_after(body, "launched")
                assert la in ("0", "1"), line
                if la == "0":
                    refused[k] = refused.get(k, 0) + 1
            elif kind == "evict_device":
                evd.append((t, int(tok_after(body, "tokens"))))
            elif kind == "evict_host":
                evh.append((t, int(tok_after(body, "tokens"))))
    r.rts, r.s2h, r.ntp, r.refused, r.evd, r.evh = rts, s2h, ntp, refused, evd, evh
    r.n_rts_all, r.n_rts_unkeyed = n_rts_all, n_rts_unkeyed
    r.n_s2h_all, r.n_ntp_all, r.n_srs_all = n_s2h_all, n_ntp_all, n_srs_all
    r.dup_rts = sum(1 for v in rts.values() if len(v) > 1)
    r.cli_no_rts = sum(1 for k in r.cli if k not in rts)
    return r


def turn_row(r, k, slack_us=100000):
    """Exact per-turn figures: (queue_us, stall_us, n_reads, n_qualifying, entry_us)."""
    line_t, q_us, e_us = r.rts[k][-1]
    reads = r.s2h.get(k, ())
    qual = [x[0] for x in reads if x[0] <= e_us + q_us + slack_us]
    if qual:
        st = max(qual) - e_us
        st = 0 if st < 0 else (q_us if st > q_us else st)
    else:
        st = 0
    return q_us, st, len(reads), len(qual), e_us


def classify(r, keys, slack_us=100000, bound_us=100000, thr_us=50000):
    n = len(keys)
    o = {"n": n}
    s_q = s_ttft = s_res = s_st = 0.0
    landed = lw_n = gq_n = 0
    lw = gq = 0.0
    over = 0
    edge_thr = edge_bound = 0
    for k in keys:
        q_us, st_us, n_reads, n_qual, e_us = turn_row(r, k, slack_us)
        ttft = r.cli[k]["ttft"]
        q = q_us / 1e6
        st = st_us / 1e6
        s_q += q
        s_ttft += ttft
        s_res += ttft - q
        s_st += st
        if q > ttft + 0.05:
            over += 1
        if n_reads == 0:
            continue
        if st_us < thr_us:
            landed += 1
        elif (q_us - st_us) < bound_us:
            lw += st
            lw_n += 1
        else:
            gq += st
            gq_n += 1
        if abs(st_us - thr_us) <= 1000:
            edge_thr += 1
        if st_us >= thr_us and abs((q_us - st_us) - bound_us) <= 1000:
            edge_bound += 1
    o.update(mean_ttft=s_ttft / n, mean_qw=s_q / n, mean_res=s_res / n, mean_stall=s_st / n,
             landed=landed, lw=lw / n, lw_n=lw_n, gq=gq / n, gq_n=gq_n, over=over,
             edge_thr=edge_thr, edge_bound=edge_bound, sum_ttft=s_ttft, sum_qw=s_q,
             sum_stall=s_st, lw_sum=lw, gq_sum=gq)
    return o


def main():
    t0 = time.time()
    with open(RES + "/.current_run") as f:
        current = f.read().strip()
    R = {lab: load(lab, current) for lab in RUNS}
    P("recheck.py  generated %s UTC   (current run, never read: %s)" % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), current))
    P()
    P("PER RUN")
    P("%-14s %5s %6s %6s %7s %8s %9s %7s %7s %6s %6s %6s" % (
        "run", "cdirs", "turns", "nullTT", "kept", "keptRet", "window_s", "rtsAll", "unkeyed",
        "dupRTS", "noRTS", "maxAll"))
    for lab in RUNS:
        r = R[lab]
        P("%-14s %5d %6d %6d %7d %8d %9.3f %7d %7d %6d %6d %6s" % (
            lab, r.n_client_dirs, len(r.turns), r.null_ttft, len(r.kept), len(r.kept_ret), r.cut,
            r.n_rts_all, r.n_rts_unkeyed, r.dup_rts, r.cli_no_rts,
            "same" if r.max_send_all == r.max_send_ok else "DIFF"))
        V("0", lab, "", "turn_records", len(r.turns))
        V("0", lab, "", "kept_turns", len(r.kept))
        V("0", lab, "", "kept_returning_turns", len(r.kept_ret))
        V("0", lab, "", "kept_window_length", r.cut)
        V("2", lab, "", "keys_with_more_than_one_ReqTimeStats_line", r.dup_rts)
        V("2", lab, "", "client_turns_without_ReqTimeStats_line", r.cli_no_rts)
    P()

    # positions in the order run; two tie-breaks to see whether ties matter
    orun = R["order_run"]
    pos = {(t["conv"], t["turn"]): i for i, t in enumerate(orun.ok)}
    alt = sorted(orun.ok, key=lambda t: (t["t_send"], t["conv"], t["turn"]))
    pos_alt = {(t["conv"], t["turn"]): i for i, t in enumerate(alt)}
    moved = sum(1 for k in pos if pos[k] != pos_alt[k])

    def scope(k, table=pos):
        p = table.get(k)
        if p is None:
            return "else"
        if 5500 <= p <= 6999:
            return "ep1"
        if 8750 <= p <= 10999:
            return "ep2"
        return "else"

    scope_flips = sum(1 for k in pos if scope(k) != scope(k, pos_alt))
    P("ORDER RUN: %d positions; keys whose position depends on the tie-break among equal t_send: %d;"
      " keys whose episode/elsewhere label depends on it: %d" % (len(pos), moved, scope_flips))

    shared = {}
    for sname, arms in SETS.items():
        s = None
        for a in arms:
            s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
        shared[sname] = sorted(s)
        n1 = sum(1 for k in s if scope(k) == "ep1")
        n2 = sum(1 for k in s if scope(k) == "ep2")
        ab = sum(1 for k in s if k not in pos)
        P("%s shared returning turns %d  (ep1 %d, ep2 %d, elsewhere %d, absent from order run %d)" % (
            sname, len(s), n1, n2, len(s) - n1 - n2, ab))
        V("0", "all", sname, "shared_returning_turns", len(s))
        V("0", "all", sname, "shared_turns_in_episode1", n1)
        V("0", "all", sname, "shared_turns_in_episode2", n2)
        V("0", "all", sname, "shared_turns_elsewhere", len(s) - n1 - n2)
        V("0", "all", sname, "shared_turns_absent_from_order_run", ab)
    P()

    # item 1
    for sname, arms in SETS.items():
        sh = shared[sname]
        P("ITEM 1  %s (%d shared turns)" % (sname, len(sh)))
        P("%-13s %8s %9s %10s %8s %7s %7s %7s %7s %7s %12s %12s" % (
            "arm", "readTurn", "pagesShrd", "pagesKeptR", "s2hLines", "rd/turn", "staged",
            "stgTurn", "refused", "refTurn", "evDevTok", "evHostTok"))
        for a in arms:
            r = R[a]
            rd = [k for k in sh if k in r.s2h]
            pages = sum(x[1] for k in rd for x in r.s2h[k])
            nlines = sum(len(r.s2h[k]) for k in rd)
            pages_kr = sum(x[1] for k in r.kept_ret for x in r.s2h.get(k, ()))
            st_lines = sum(r.ntp.get(k, 0) for k in sh)
            st_turns = sum(1 for k in sh if r.ntp.get(k, 0) > 0)
            rf_lines = sum(r.refused.get(k, 0) for k in sh)
            rf_turns = sum(1 for k in sh if r.refused.get(k, 0) > 0)
            evd = sum(tok for t, tok in r.evd if 0.0 <= t / 1e6 - r.t_start < r.cut)
            evh = sum(tok for t, tok in r.evh if 0.0 <= t / 1e6 - r.t_start < r.cut)
            P("%-13s %8d %9d %10d %8d %7.4f %7d %7d %7d %7d %12d %12d" % (
                a, len(rd), pages, pages_kr, nlines, nlines / len(rd), st_lines, st_turns,
                rf_lines, rf_turns, evd, evh))
            V("1", a, sname, "turns_that_read_from_ssd", len(rd))
            V("1", a, sname, "ssd_pages_shared_turns", pages)
            V("1", a, sname, "s2h_io_lines_shared_turns", nlines)
            V("1", a, sname, "reads_per_reading_turn", nlines / len(rd))
            V("1", a, sname, "staged_reads_ntp_stage_lines", st_lines)
            V("1", a, sname, "turns_with_staged_read", st_turns)
            V("1", a, sname, "refused_launches", rf_lines)
            V("1", a, sname, "turns_with_refused_launch", rf_turns)
            V("1", a, "", "ssd_pages_kept_returning_turns", pages_kr)
            V("1", a, "", "evict_device_tokens_kept_window", evd)
            V("1", a, "", "evict_host_tokens_kept_window", evh)
        P()

    # items 2, 3, 5c
    for sname, arms in SETS.items():
        sh = shared[sname]
        P("ITEMS 2+3+5c  %s (%d shared turns)" % (sname, len(sh)))
        P("%-13s %9s %9s %9s %9s | %6s | %9s %6s | %9s %6s | %7s | %6s %6s" % (
            "arm", "meanTTFT", "meanQW", "TTFT-QW", "meanStall", "landed", "lastWait", "n",
            "gpuQueue", "n", "qw>ttft", "edge05", "edgeBd"))
        for a in arms:
            o = classify(R[a], sh)
            P("%-13s %9.4f %9.4f %9.4f %9.4f | %6d | %9.4f %6d | %9.4f %6d | %7d | %6d %6d" % (
                a, o["mean_ttft"], o["mean_qw"], o["mean_res"], o["mean_stall"], o["landed"],
                o["lw"], o["lw_n"], o["gq"], o["gq_n"], o["over"], o["edge_thr"], o["edge_bound"]))
            V("2", a, sname, "mean_queue_wait", o["mean_qw"])
            V("2", a, sname, "mean_ttft_minus_queue_wait", o["mean_res"])
            V("2", a, sname, "mean_ttft", o["mean_ttft"])
            V("3", a, sname, "mean_read_stall_per_shared_turn", o["mean_stall"])
            V("3", a, sname, "reading_turns_landed_by_arrival", o["landed"])
            V("3", a, sname, "stall_read_was_last_wait_per_shared_turn", o["lw"])
            V("3", a, sname, "turns_read_was_last_wait", o["lw_n"])
            V("3", a, sname, "stall_gpu_queue_held_as_well_per_shared_turn", o["gq"])
            V("3", a, sname, "turns_gpu_queue_held_as_well", o["gq_n"])
            V("5c", a, sname, "share_shared_turns_queue_wait_exceeds_ttft_by_0.05s", o["over"] / o["n"])
            V("5c", a, sname, "turns_queue_wait_exceeds_ttft_by_0.05s", o["over"])
        P()

    # item 4
    variants = [("bound=0.05", 100000, 50000), ("bound=0.2", 100000, 200000),
                ("bound=0.5", 100000, 500000), ("slack=0", 0, 100000), ("slack=0.5", 500000, 100000)]
    for sname in ("LEAD5", "SEVEN"):
        sh = shared[sname]
        P("ITEM 4  %s" % sname)
        P("%-13s %-11s %9s %6s %9s %6s %6s" % ("arm", "variant", "lastWait", "n", "gpuQueue", "n", "edgeBd"))
        for a in SETS[sname]:
            for tag, slack_us, bound_us in variants:
                o = classify(R[a], sh, slack_us=slack_us, bound_us=bound_us)
                P("%-13s %-11s %9.4f %6d %9.4f %6d %6d" % (a, tag, o["lw"], o["lw_n"], o["gq"],
                                                         o["gq_n"], o["edge_bound"]))
                V("4", a, sname, "stall_read_was_last_wait_per_shared_turn[%s]" % tag, o["lw"])
                V("4", a, sname, "turns_read_was_last_wait[%s]" % tag, o["lw_n"])
                V("4", a, sname, "stall_gpu_queue_held_as_well_per_shared_turn[%s]" % tag, o["gq"])
                V("4", a, sname, "turns_gpu_queue_held_as_well[%s]" % tag, o["gq_n"])
        P()

    # item 5a, 5b
    P("ITEM 5a  read start - arrival, every s2h_io line whose key has a ReqTimeStats line")
    P("%-13s %7s %6s %11s %11s %9s %9s %7s" % ("arm", "lines", "noRTS", "min", "p01", "median",
                                                "sh<-0.05", "n<-0.05"))
    for a in LEAD5:
        r = R[a]
        dv = []
        nor = 0
        for k, reads in r.s2h.items():
            if k not in r.rts:
                nor += len(reads)
                continue
            e_us = r.rts[k][-1][2]
            for end_us, pages, ms in reads:
                dv.append((end_us - ms * 1000 - e_us) / 1e6)
        neg = sum(1 for z in dv if z < -0.05)
        P("%-13s %7d %6d %11.4f %11.4f %9.4f %9.5f %7d" % (
            a, len(dv), nor, min(dv), pct(dv, 0.01), pct(dv, 0.5), neg / len(dv), neg))
        V("5a", a, "", "s2h_io_lines_with_arrival", len(dv))
        V("5a", a, "", "s2h_io_lines_without_ReqTimeStats", nor)
        V("5a", a, "", "read_start_minus_arrival_min", min(dv))
        V("5a", a, "", "read_start_minus_arrival_p01", pct(dv, 0.01))
        V("5a", a, "", "read_start_minus_arrival_median", pct(dv, 0.5))
        V("5a", a, "", "read_start_minus_arrival_share_below_-0.05s", neg / len(dv))
        V("5a", a, "", "read_start_minus_arrival_lines_below_-0.05s", neg)
    P()
    P("ITEM 5b  (ReqTimeStats line time - entry_time) - client latency, every client turn")
    P("%-13s %7s %10s %10s %10s" % ("arm", "n", "p01", "median", "p99"))
    for a in LEAD5:
        r = R[a]
        dv = []
        for k, t in r.cli.items():
            if k in r.rts and t.get("latency") is not None:
                lt, q_us, e_us = r.rts[k][-1]
                dv.append((lt - e_us) / 1e6 - t["latency"])
        P("%-13s %7d %10.4f %10.4f %10.4f" % (a, len(dv), pct(dv, 0.01), pct(dv, 0.5), pct(dv, 0.99)))
        V("5b", a, "", "server_minus_client_latency_n", len(dv))
        V("5b", a, "", "server_minus_client_latency_median", pct(dv, 0.5))
        V("5b", a, "", "server_minus_client_latency_p01", pct(dv, 0.01))
        V("5b", a, "", "server_minus_client_latency_p99", pct(dv, 0.99))
    P()

    # split by position and whole-set client figures
    for sname, arms in SETS.items():
        sh = shared[sname]
        groups = {"ep1": [k for k in sh if scope(k) == "ep1"],
                  "ep2": [k for k in sh if scope(k) == "ep2"],
                  "else": [k for k in sh if scope(k) == "else"]}
        groups["episodes"] = groups["ep1"] + groups["ep2"]
        P("SPLIT BY POSITION  %s" % sname)
        P("%-13s %-9s %6s %9s %9s %9s %9s %9s %9s %8s %9s %9s %12s" % (
            "arm", "scope", "n", "meanTTFT", "p50", "p90", "p99", "meanQW", "TTFT-QW", "readTurn",
            "pages", "meanStall", "recompSum"))
        for a in arms:
            r = R[a]
            for sc in ("ep1", "ep2", "episodes", "else"):
                ks = groups[sc]
                if not ks:
                    continue
                o = classify(r, ks)
                tt = [r.cli[k]["ttft"] for k in ks]
                rec = [max(r.cli[k].get("reply_reprefilled") or 0, 0) for k in ks]
                rdt = sum(1 for k in ks if k in r.s2h)
                pg = sum(x[1] for k in ks for x in r.s2h.get(k, ()))
                P("%-13s %-9s %6d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %8d %9d %9.4f %12d" % (
                    a, sc, len(ks), o["mean_ttft"], pct(tt, 0.5), pct(tt, 0.9), pct(tt, 0.99),
                    o["mean_qw"], o["mean_res"], rdt, pg, o["mean_stall"], sum(rec)))
                for m, val in (("n_turns", len(ks)), ("mean_ttft", o["mean_ttft"]),
                               ("p50_ttft", pct(tt, 0.5)), ("p90_ttft", pct(tt, 0.9)),
                               ("p95_ttft", pct(tt, 0.95)), ("p99_ttft", pct(tt, 0.99)),
                               ("mean_queue_wait", o["mean_qw"]),
                               ("mean_ttft_minus_queue_wait", o["mean_res"]),
                               ("mean_read_stall", o["mean_stall"]),
                               ("turns_that_read_from_ssd", rdt), ("ssd_pages", pg),
                               ("recomputed_tokens_sum", sum(rec)),
                               ("lastwait_sum", o["lw_sum"]), ("gpuq_sum", o["gq_sum"]),
                               ("lastwait_n", o["lw_n"]), ("gpuq_n", o["gq_n"])):
                    V("x-split:" + sc, a, sname, m, val)
        P()
        P("SHARED-TURN TTFT  %s" % sname)
        for a in arms:
            r = R[a]
            tt = [r.cli[k]["ttft"] for k in sh]
            rec = [max(r.cli[k].get("reply_reprefilled") or 0, 0) for k in sh]
            P("%-13s mean %.4f p50 %.4f p90 %.4f p95 %.4f p99 %.4f recomputed %d" % (
                a, sum(tt) / len(tt), pct(tt, 0.5), pct(tt, 0.9), pct(tt, 0.95), pct(tt, 0.99), sum(rec)))
            for m, val in (("mean_ttft", sum(tt) / len(tt)), ("p50_ttft", pct(tt, 0.5)),
                           ("p90_ttft", pct(tt, 0.9)), ("p95_ttft", pct(tt, 0.95)),
                           ("p99_ttft", pct(tt, 0.99)), ("recomputed_tokens_sum", sum(rec)),
                           ("recomputed_tokens_mean", sum(rec) / len(rec))):
                V("x-shared", a, sname, m, val)
        P()

    # per-turn table for later comparisons
    table = {}
    for a in sorted(set(LEAD5 + SEVEN)):
        r = R[a]
        rows = {}
        for k in shared["LEAD5"] if a in LEAD5 else shared["SEVEN"]:
            q_us, st_us, n_reads, n_qual, e_us = turn_row(r, k)
            rows["%d-%d" % k] = [q_us, st_us, n_reads, r.cli[k]["ttft"]]
        table[a] = rows
    with open(os.path.join(OUT, "recheck_turns.json"), "w") as f:
        json.dump(table, f)

    # diff against the frozen values
    with open(os.path.join(OUT, "frozen_results_items1to5.json")) as f:
        frozen = json.load(f)
    fz = {}
    for rec in frozen:
        fz[(rec["item"], rec["arm"], rec["arm_set"], rec["metric"])] = rec["value"]
    P("DIFF AGAINST frozen_results_items1to5.json")
    n_cmp = n_exact = n_close = 0
    worst = 0.0
    bad = []
    missing = 0
    by_item = {}
    for item, arm, arm_set, metric, val in values:
        key = (item, arm, arm_set, metric)
        if key not in fz:
            missing += 1
            continue
        n_cmp += 1
        ref = fz[key]
        c = by_item.setdefault(item, [0, 0])
        c[0] += 1
        if isinstance(val, int) and isinstance(ref, int):
            if val == ref:
                n_exact += 1
                c[1] += 1
            else:
                bad.append((key, val, ref))
            continue
        d = abs(val - ref)
        rel = d / max(1.0, abs(ref))
        worst = max(worst, rel)
        if rel <= 1e-9:
            n_close += 1
            c[1] += 1
        else:
            bad.append((key, val, ref))
    P("values computed here: %d; with a frozen counterpart: %d (no counterpart: %d)" % (
        len(values), n_cmp, missing))
    P("integers equal: %d; floats within 1e-9 relative: %d; worst relative float difference: %.3g" % (
        n_exact, n_close, worst))
    P("per item (compared, agreeing): %s" % json.dumps({k: v for k, v in sorted(by_item.items())}))
    P("DISAGREEMENTS: %d" % len(bad))
    for key, val, ref in bad:
        P("   %s  recheck=%r  frozen=%r" % (key, val, ref))
    P()
    P("done in %.1f s" % (time.time() - t0))
    with open(os.path.join(OUT, "recheck.json"), "w") as f:
        json.dump([{"item": i, "arm": a, "arm_set": s, "metric": m, "value": v}
                   for i, a, s, m, v in values], f)
    with open(os.path.join(OUT, "recheck.out"), "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
