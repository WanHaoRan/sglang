#!/usr/bin/env python3
"""Second, regex-free pass over the raw logs to cross-check parse_runs.py.

Counts and sums the lines of interest with plain string splitting and compares
them with the cached parse.  Also checks that the server's ReqTimeStats token
counts agree with the client record of the same key (rid -> key mapping).

Run as:  nice -n 19 python3 -I -B check_raw.py
"""
import json
import os

RES = "/home/wanhr/sglang/agent_cache/results"
OUT = RES + "/campaign26/checks/verify/server"
CACHE = OUT + "/cache"
lines_out = []


def P(s=""):
    lines_out.append(s)
    print(s)


def field(body, name):
    i = body.find(name + "=")
    if i < 0:
        return None
    j = i + len(name) + 1
    e = j
    while e < len(body) and body[e] not in " ,)\n":
        e += 1
    return body[j:e]


with open(RES + "/.current_run") as f:
    current = f.read().strip()

LABELS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl",
          "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "order_run"]
allok = True
for lab in LABELS:
    with open(os.path.join(CACHE, lab + ".json")) as f:
        d = json.load(f)
    assert d["folder"] != current
    path = os.path.join(RES, d["folder"], "server.log")
    n_rts = n_rts_key = 0
    n_s2h = pages = s2h_ms = 0
    n_ntp = 0
    n_srs = n_ref = 0
    n_evd = evd_tok = 0
    n_evh = evh_tok = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "ReqTimeStats(" in line:
                n_rts += 1
                rid = field(line, "rid")
                parts = rid.rsplit("-", 2)
                if len(parts) == 3 and parts[1][:1] == "c" and parts[2][:1] == "t" \
                        and parts[1][1:].isdigit() and parts[2][1:].isdigit():
                    n_rts_key += 1
            elif "HICACHE_EVT s2h_io " in line:
                n_s2h += 1
                pages += int(field(line, "pages"))
                s2h_ms += int(field(line, "ms"))
            elif "HICACHE_EVT ntp_stage " in line:
                n_ntp += 1
            elif "HICACHE_EVT srs_dispatch " in line:
                n_srs += 1
                if field(line, "launched") == "0":
                    n_ref += 1
            elif "HICACHE_EVT evict_device " in line:
                n_evd += 1
                evd_tok += int(field(line, "tokens"))
            elif "HICACHE_EVT evict_host " in line:
                n_evh += 1
                evh_tok += int(field(line, "tokens"))
    c = {
        "rts_keyed": (n_rts_key, len(d["rts"])),
        "s2h_lines": (n_s2h, len(d["s2h"])),
        "s2h_pages": (pages, sum(r[3] for r in d["s2h"])),
        "s2h_ms": (s2h_ms, int(sum(r[5] for r in d["s2h"]))),
        "ntp_stage": (n_ntp, len(d["ntp"])),
        "srs_dispatch": (n_srs, len(d["srs"])),
        "srs_refused": (n_ref, sum(1 for r in d["srs"] if r[7] == 0)),
        "evict_device_lines": (n_evd, len(d["evd"])),
        "evict_device_tokens": (evd_tok, sum(r[1] for r in d["evd"])),
        "evict_host_lines": (n_evh, len(d["evh"])),
        "evict_host_tokens": (evh_tok, sum(r[1] for r in d["evh"])),
    }
    bad = {k: v for k, v in c.items() if v[0] != v[1]}
    # rid -> key mapping check against the client record
    cli = {(t["conv"], t["turn"]): t for t in d["turns"]}
    n = mism_in = mism_out = mism_cached = 0
    max_in = 0
    for r in d["rts"]:
        t = cli.get((r[1], r[2]))
        if t is None:
            continue
        n += 1
        if r[6] != t["prompt_tokens"]:
            mism_in += 1
            max_in = max(max_in, abs(r[6] - (t["prompt_tokens"] or 0)))
        if r[8] != t["completion_tokens"]:
            mism_out += 1
        if r[7] != t["cached_tokens"]:
            mism_cached += 1
    P("%-13s raw==cache: %s   %s" % (lab, "OK" if not bad else "MISMATCH " + json.dumps(bad),
                                     json.dumps({k: v[0] for k, v in c.items()})))
    P("%-13s ReqTimeStats vs client record, %d keys: input_len!=prompt_tokens %d (max diff %d), output_len!=completion_tokens %d, cached_input_len!=cached_tokens %d"
      % ("", n, mism_in, max_in, mism_out, mism_cached))
    if bad:
        allok = False
P("ALL RAW COUNTS EQUAL THE CACHED PARSE: %s" % allok)
with open(os.path.join(OUT, "check_raw.out"), "w") as f:
    f.write("\n".join(lines_out) + "\n")
