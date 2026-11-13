#!/usr/bin/env python3
"""r2_07_readclass.py -- why my client-side proxy for "the turn read from SSD" differs from
split_ec_lead.out for ec_r13 and ec_r1 (8 and 5 turns).

Reads only the s2h_io lines of the five LEAD5 server logs.
Run as:  nice -n 19 python3 -I -B r2_07_readclass.py
"""
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"
L5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
PRINTED_READ = {"jit_exact": 1262, "ec_r13": 1582, "ec_r1": 1572, "nofc": 1581, "nofc_noend": 1898}
RX = re.compile(r"HICACHE_EVT s2h_io rid=\S*?-c(\d+)-t(\d+) pages=(\d+) tokens=(\d+) ms=([\d.]+)")

with open(os.path.join(RESULTS, ".current_run")) as f:
    current = f.read().strip()
with open(os.path.join(HERE, "r2_parsed.pkl"), "rb") as f:
    P = pickle.load(f)

KM = {}
for a in L5:
    t = sorted((r for r in P[a]["records"] if r.get("kind") == "turn" and r.get("ttft") is not None),
               key=lambda r: r["t_send"])
    thr = t[-1]["t_send"] - 300.0
    KM[a] = {(r["conv"], r["turn"]): r for r in t if r["t_send"] < thr}
keys = None
for a in L5:
    ks = set(k for k in KM[a] if k[1] > 0)
    keys = ks if keys is None else keys & ks
print("LEAD5 shared returning turns:", len(keys))

out = {}
for a in L5:
    folder = P[a]["folder"]
    if current and current in folder:
        print("refusing the running directory")
        sys.exit(2)
    reads = {}
    n_lines = 0
    with open(os.path.join(RESULTS, folder, "server.log"), "r", errors="replace") as f:
        for line in f:
            if "s2h_io" not in line:
                continue
            m = RX.search(line)
            if not m:
                continue
            n_lines += 1
            k = (int(m.group(1)), int(m.group(2)))
            e = reads.setdefault(k, [0, 0])
            e[0] += 1
            e[1] += int(m.group(3))
    srv = set(k for k in keys if k in reads)
    srv_pages = set(k for k in keys if k in reads and reads[k][1] > 0)
    cli = set(k for k in keys if ((KM[a][k].get("cached_details") or {}).get("storage") or 0) > 0)
    only_srv = sorted(srv - cli)
    only_cli = sorted(cli - srv)
    out[a] = {"s2h_io_lines": n_lines, "turns_with_s2h_io_line": len(srv),
              "turns_with_s2h_io_pages_gt0": len(srv_pages),
              "turns_with_client_storage_tokens": len(cli), "printed_read_turns": PRINTED_READ[a],
              "server_only": len(only_srv), "client_only": len(only_cli),
              "server_only_examples": [
                  {"key": list(k), "reads": reads[k][0], "pages": reads[k][1],
                   "cached_details": KM[a][k].get("cached_details"),
                   "recomputed": KM[a][k].get("reply_reprefilled"), "ttft": round(KM[a][k]["ttft"], 3)}
                  for k in only_srv[:10]]}
    print("%-11s s2h_io lines %5d | shared turns with an s2h_io line %4d (pages>0: %4d) | printed "
          "'read' %4d | client cached_details.storage>0 %4d | server only %d, client only %d"
          % (a, n_lines, len(srv), len(srv_pages), PRINTED_READ[a], len(cli), len(only_srv),
             len(only_cli)))
    for e in out[a]["server_only_examples"]:
        print("      server-only turn", e)
with open(os.path.join(HERE, "r2_readclass.json"), "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
