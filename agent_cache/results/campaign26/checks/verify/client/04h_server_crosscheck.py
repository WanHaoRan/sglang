#!/usr/bin/env python3
"""Item 5, part h: is the client's TTFT what the server saw?  Join client turns with the server's ReqTimeStats.

Read-only pass over server.log of the 13 finished runs (never the run named in results/.current_run).
Writes server_reqstats.pkl in this folder so later passes do not re-read the logs.
"""
import json
import math
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(RESULTS, ".current_run")) as f:
    CURRENT = f.read().strip()

RX = re.compile(r"ReqTimeStats\(rid=(\S*?)-c(\d+)-t(\d+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+), attempts=(\d+).*?"
                r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")
CACHE = os.path.join(HERE, "server_reqstats.pkl")


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        S = pickle.load(f)
else:
    S = {}
    for a, d in P.items():
        if CURRENT and d["folder"] == CURRENT:
            print("refusing to read the running run")
            sys.exit(2)
        path = os.path.join(RESULTS, d["folder"], "server.log")
        rows = {}
        nlines = 0
        nstats = 0
        dup = 0
        with open(path, "r", errors="replace") as f:
            for line in f:
                nlines += 1
                if "ReqTimeStats(" not in line:
                    continue
                nstats += 1
                m = RX.search(line)
                if not m:
                    continue
                k = (int(m.group(2)), int(m.group(3)))
                rec = (int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7)), float(m.group(8)),
                       float(m.group(9)), float(m.group(10)))
                if k in rows:
                    dup += 1
                rows[k] = rec
        S[a] = {"rows": rows, "lines": nlines, "reqstats_lines": nstats, "dup": dup}
    with open(CACHE, "wb") as f:
        pickle.dump(S, f, protocol=pickle.HIGHEST_PROTOCOL)

RES = {}
print("=== join of client turn records with server ReqTimeStats lines by (conv, turn)")
print("   %-14s %7s %8s %7s %5s %9s %9s %9s %8s %8s %8s %9s %9s %9s" % (
    "run", "client", "server", "joined", "dup", "in==prm", "cach==", "out==", "off_med", "off_p99", "off_max", "q>ttft", "resid_med", "resid_p99"))
for a, d in P.items():
    t0 = d["run"][0]["t_start"]
    rows = S[a]["rows"]
    turns = d["turns"]
    j = [(t, rows[(t["conv"], t["turn"])]) for t in turns if (t["conv"], t["turn"]) in rows]
    off = [r[6] - (t0 + t["t_send"]) for t, r in j]
    qgt = sum(1 for t, r in j if r[4] / 1000.0 > t["ttft"] + 0.005)
    resid = [t["latency"] - (r[4] + r[5]) / 1000.0 - (r[6] - (t0 + t["t_send"])) for t, r in j]
    after = [t["ttft"] - r[4] / 1000.0 - (r[6] - (t0 + t["t_send"])) for t, r in j]
    res = {"client_turns": len(turns), "server_request_lines_with_conv_turn": len(rows), "joined": len(j),
           "server_duplicate_keys": S[a]["dup"],
           "input_len_eq_prompt_tokens": sum(1 for t, r in j if r[0] == t["prompt_tokens"]),
           "cached_input_len_eq_cached_tokens": sum(1 for t, r in j if r[1] == t["cached_tokens"]),
           "output_len_eq_completion_tokens": sum(1 for t, r in j if r[2] == t["completion_tokens"]),
           "attempts_nonzero": sum(1 for t, r in j if r[3] != 0),
           "entry_minus_send_median_s": pct(off, 0.5), "entry_minus_send_p99_s": pct(off, 0.99), "entry_minus_send_max_s": max(off),
           "entry_minus_send_min_s": min(off), "entry_minus_send_over_1s": sum(1 for x in off if x > 1.0),
           "queue_longer_than_ttft": qgt, "latency_minus_server_total_median_s": pct(resid, 0.5),
           "latency_minus_server_total_p99_s": pct(resid, 0.99), "mean_entry_minus_send_s": mean(off)}
    RES[a] = res
    print("   %-14s %7d %8d %7d %5d %9d %9d %9d %8.4f %8.4f %8.3f %9d %9.4f %9.4f" % (
        a, len(turns), len(rows), len(j), S[a]["dup"], res["input_len_eq_prompt_tokens"], res["cached_input_len_eq_cached_tokens"],
        res["output_len_eq_completion_tokens"], res["entry_minus_send_median_s"], res["entry_minus_send_p99_s"],
        res["entry_minus_send_max_s"], qgt, res["latency_minus_server_total_median_s"], res["latency_minus_server_total_p99_s"]))
    print("        attempts != 0: %d   entry-send over 1 s: %d   mean entry-send %.4f s   min %.4f" % (
        res["attempts_nonzero"], res["entry_minus_send_over_1s"], res["mean_entry_minus_send_s"], res["entry_minus_send_min_s"]))

# mean queue wait on the ALL11 shared turns, to set against "mean TTFT = mean queue wait + after admission" of compare_ec.out
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]


def kept_map(turns):
    v = sorted((t for t in turns if t.get("ttft") is not None), key=lambda t: t["t_send"])
    cut = v[-1]["t_send"] - 300.0
    return {(t["conv"], t["turn"]): t for t in v if t["t_send"] < cut}


KM = {a: kept_map(P[a]["turns"]) for a in P}
keys = sorted(set.intersection(*[{k for k in KM[a] if k[1] > 0} for a in ALL11]))
print()
print("=== ALL11 shared turns (%d): mean TTFT, mean server queue wait, the rest; share of the mean TTFT that is queue wait;" % len(keys))
print("    mean (server entry - client send), which is in TTFT but in neither server number")
printed = {"baseline": "4.53", "base_rep": "5.50", "hintfree": "1.04", "jit_p50": "0.89", "jit_exact": "0.13", "jit_p50_ctl": "0.69",
           "hintfree_ctl": "1.29", "ec_r13": "0.25", "nofc": "0.31", "ec_r1": "0.27", "nofc_noend": "0.32"}
okc = 0
for a in ALL11:
    t0 = P[a]["run"][0]["t_start"]
    miss = sum(1 for k in keys if k not in S[a]["rows"])
    q = mean([S[a]["rows"][k][4] / 1000.0 for k in keys if k in S[a]["rows"]])
    mu = mean([KM[a][k]["ttft"] for k in keys])
    off = mean([S[a]["rows"][k][6] - (t0 + KM[a][k]["t_send"]) for k in keys if k in S[a]["rows"]])
    ok = ("%.2f" % q) == printed[a]
    okc += 1 if ok else 0
    RES[a].update({"all11_mean_ttft": mu, "all11_mean_queue_wait": q, "all11_queue_share": q / mu, "all11_mean_entry_minus_send": off,
                   "all11_keys_without_server_line": miss, "all11_queue_matches_compare_ec_out": ok})
    print("   %-14s %7.4f = %7.4f + %7.4f   queue share %.3f   entry-send %.4f   keys without a server line %d   compare_ec.out prints %s -> %s" % (
        a, mu, q, mu - q, q / mu, off, miss, printed[a], "match" if ok else "DIFFERENT"))
print("   queue-wait means equal to compare_ec.out at 2 decimals: %d of %d" % (okc, len(ALL11)))

with open(os.path.join(HERE, "item5h_server_crosscheck.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5h_server_crosscheck.json")
