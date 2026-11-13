#!/usr/bin/env python3
"""Independent parser for the campaign 26 server-side check.

Reads client.jsonl and server.log of the thirteen runs named in the task and
writes one compact JSON cache per run into this folder (cache/<label>.json),
plus a diagnostics text (parse_diag.txt).  Standard library only.

Run as:  nice -n 19 python3 -I -B parse_runs.py
"""
import calendar
import json
import os
import re
import sys
import time

RES = "/home/wanhr/sglang/agent_cache/results"
OUT = RES + "/campaign26/checks/verify/server"
CACHE = OUT + "/cache"

RUNS = [
    ("baseline", "20261101_082229_three_tier_wc_custom_L1nat_H70"),
    ("base_rep", "20261105_075438_three_tier_wc_custom_L1nat_H70"),
    ("hintfree", "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70"),
    ("hintfree_ctl", "20261111_045808_three_tier_wc_dl_poll0_custom_L1nat_H70"),
    ("jit_p50", "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("jit_p50_ctl", "20261111_024927_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("jit_p50_d_cut", "20261111_151101_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("jit_exact", "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ec_r13", "20261111_070012_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ec_r1", "20261111_110558_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("nofc", "20261111_090311_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("nofc_noend", "20261111_130801_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("order_run", "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70"),
]

TS_RE = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] ")
KEY_RE = re.compile(r"rid=(\S*?)-c(\d+)-t(\d+)(?=[\s,)]|$)")
RID_RE = re.compile(r"rid=([^\s,)]+)")
NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
QD_RE = re.compile(r"queue_duration=" + NUM + r"(ms|s)\b")
FD_RE = re.compile(r"forward_duration=" + NUM + r"(ms|s)\b")
ET_RE = re.compile(r"entry_time=" + NUM)
KV_RE = re.compile(r"(\w+)=(\S+)")
EVT_RE = re.compile(r"HICACHE_EVT (\w+)")
DEC_RE = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+),"
    r".*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
PRE_RE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)"
)
PRE_Q_RE = re.compile(r"#queue-req: (\d+)")
PRE_RUN_RE = re.compile(r"#running-req: (\d+)")

_sec_cache = {}


def line_epoch(m):
    k = m.group(0)[1:20]
    base = _sec_cache.get(k)
    if base is None:
        base = calendar.timegm(
            (int(m.group(1)), int(m.group(2)), int(m.group(3)),
             int(m.group(4)), int(m.group(5)), int(m.group(6)), 0, 0, 0)
        )
        _sec_cache[k] = base
    return base + int(m.group(7)) / 1000.0


def last_client_dir(run_dir):
    ds = sorted(d for d in os.listdir(run_dir)
                if d.startswith("client_") and os.path.isdir(os.path.join(run_dir, d)))
    return ds, (ds[-1] if ds else None)


def parse_client(path, diag):
    t_start = None
    run_rec = None
    turns = []
    kinds = {}
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            k = r.get("kind")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "run":
                if run_rec is not None:
                    diag.append("  WARNING more than one run record")
                run_rec = r
                t_start = r.get("t_start")
            elif k == "turn":
                turns.append({
                    "i": len(turns),
                    "conv": r.get("conv"),
                    "turn": r.get("turn"),
                    "t_send": r.get("t_send"),
                    "ttft": r.get("ttft"),
                    "latency": r.get("latency"),
                    "prompt_tokens": r.get("prompt_tokens"),
                    "completion_tokens": r.get("completion_tokens"),
                    "cached_tokens": r.get("cached_tokens"),
                    "reply_reprefilled": r.get("reply_reprefilled"),
                    "cached_details": r.get("cached_details"),
                    "gap_slept": r.get("gap_slept"),
                    "error": r.get("error"),
                    "has_error_field": "error" in r,
                })
    diag.append("  client kinds: %s  bad_json_lines=%d" % (sorted(kinds.items(), key=lambda x: str(x[0])), bad))
    return t_start, run_rec, turns


def parse_server(path, diag):
    rts = []       # [line_t, conv, turn, queue_ms, forward_ms, entry_time, input_len, cached_input_len, output_len, attempts, lineno]
    s2h = []       # [line_t, conv, turn, pages, tokens, ms, lineno]
    ntp = []       # [line_t, conv, turn, tokens, matched, gate, inflight, lineno]
    srs = []       # [line_t, conv, turn, pages, waited_ms, inflight_pages, pinned_tokens, launched, lineno]
    evd = []       # [line_t, tokens, requested]
    evh = []       # [line_t, tokens, requested]
    dec = []       # [line_t, running, tokens, usage, thr, queue]
    pre = []       # [line_t, new_seq, new_tok, cached_tok, running, queue]
    evt_kinds = {}
    prefixes = {}
    c = {
        "lines": 0, "no_ts": 0, "rts_total": 0, "rts_nokey": 0, "rts_badfmt": 0,
        "rts_unit_s": 0, "s2h_total": 0, "s2h_nokey": 0, "s2h_bad": 0,
        "ntp_total": 0, "ntp_nokey": 0, "srs_total": 0, "srs_nokey": 0,
        "srs_bad": 0, "evd_total": 0, "evd_bad": 0, "evh_total": 0, "evh_bad": 0,
        "dec_total": 0, "dec_bad": 0, "pre_total": 0, "pre_bad": 0,
        "marker_no_ts": 0, "ts_backwards": 0,
    }
    samples = {"rts_nokey": [], "s2h_nokey": [], "ntp_nokey": [], "srs_nokey": [],
               "bad": [], "marker_no_ts": []}
    first_t = None
    last_t = None
    prev_t = None
    max_back = 0.0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f):
            c["lines"] += 1
            m = TS_RE.match(line)
            if not m:
                c["no_ts"] += 1
                if ("ReqTimeStats(" in line or "HICACHE_EVT" in line
                        or "Decode batch" in line or "Prefill batch" in line):
                    c["marker_no_ts"] += 1
                    if len(samples["marker_no_ts"]) < 5:
                        samples["marker_no_ts"].append(line[:300].rstrip())
                continue
            t = line_epoch(m)
            if first_t is None:
                first_t = t
            if prev_t is not None and t < prev_t:
                c["ts_backwards"] += 1
                if prev_t - t > max_back:
                    max_back = prev_t - t
            prev_t = t
            last_t = t
            body = line[m.end():]
            if "ReqTimeStats(" in body:
                c["rts_total"] += 1
                qd = QD_RE.search(body)
                fd = FD_RE.search(body)
                et = ET_RE.search(body)
                km = KEY_RE.search(body)
                if not (qd and et):
                    c["rts_badfmt"] += 1
                    if len(samples["bad"]) < 8:
                        samples["bad"].append(line[:400].rstrip())
                    continue
                if not km:
                    c["rts_nokey"] += 1
                    if len(samples["rts_nokey"]) < 4:
                        samples["rts_nokey"].append(line[:300].rstrip())
                    continue
                q = float(qd.group(1))
                if qd.group(2) == "s":
                    q *= 1000.0
                    c["rts_unit_s"] += 1
                fwd = None
                if fd:
                    fwd = float(fd.group(1))
                    if fd.group(2) == "s":
                        fwd *= 1000.0
                kv = dict(KV_RE.findall(body.replace(",", " ").replace(")", " ")))

                def gi(name):
                    v = kv.get(name)
                    try:
                        return int(v)
                    except Exception:
                        return None
                prefixes[km.group(1)] = prefixes.get(km.group(1), 0) + 1
                rts.append([t, int(km.group(2)), int(km.group(3)), q, fwd,
                            float(et.group(1)), gi("input_len"), gi("cached_input_len"),
                            gi("output_len"), gi("attempts"), lineno])
                continue
            em = EVT_RE.search(body)
            if em:
                kind = em.group(1)
                evt_kinds[kind] = evt_kinds.get(kind, 0) + 1
                if kind in ("s2h_io", "ntp_stage", "srs_dispatch"):
                    km = KEY_RE.search(body)
                    kv = dict(KV_RE.findall(body))
                    tag = {"s2h_io": "s2h", "ntp_stage": "ntp", "srs_dispatch": "srs"}[kind]
                    c[tag + "_total"] += 1
                    if not km:
                        c[tag + "_nokey"] += 1
                        if len(samples[tag + "_nokey"]) < 4:
                            samples[tag + "_nokey"].append(line[:300].rstrip())
                        continue
                    prefixes[km.group(1)] = prefixes.get(km.group(1), 0) + 1
                    conv, turn = int(km.group(2)), int(km.group(3))
                    try:
                        if kind == "s2h_io":
                            s2h.append([t, conv, turn, int(kv["pages"]), int(kv["tokens"]),
                                        float(kv["ms"]), lineno])
                        elif kind == "ntp_stage":
                            ntp.append([t, conv, turn, kv.get("tokens"), kv.get("matched"),
                                        kv.get("gate"), kv.get("inflight"), lineno])
                        else:
                            srs.append([t, conv, turn, int(kv["pages"]), kv.get("waited_ms"),
                                        kv.get("inflight_pages"), int(kv["pinned_tokens"]),
                                        int(kv["launched"]), lineno])
                    except Exception:
                        c[tag + "_bad"] = c.get(tag + "_bad", 0) + 1
                        if len(samples["bad"]) < 8:
                            samples["bad"].append(line[:400].rstrip())
                elif kind in ("evict_device", "evict_host"):
                    tag = "evd" if kind == "evict_device" else "evh"
                    c[tag + "_total"] += 1
                    kv = dict(KV_RE.findall(body))
                    try:
                        rec = [t, int(kv["tokens"]), kv.get("requested")]
                    except Exception:
                        c[tag + "_bad"] += 1
                        if len(samples["bad"]) < 8:
                            samples["bad"].append(line[:400].rstrip())
                        continue
                    (evd if kind == "evict_device" else evh).append(rec)
                continue
            if "Decode batch" in body:
                c["dec_total"] += 1
                dm = DEC_RE.search(body)
                if not dm:
                    c["dec_bad"] += 1
                    if len(samples["bad"]) < 8:
                        samples["bad"].append(line[:400].rstrip())
                    continue
                dec.append([t, int(dm.group(1)), int(dm.group(2)), float(dm.group(3)),
                            float(dm.group(4)), int(dm.group(5))])
                continue
            if "Prefill batch" in body:
                c["pre_total"] += 1
                pm = PRE_RE.search(body)
                if not pm:
                    c["pre_bad"] += 1
                    if len(samples["bad"]) < 8:
                        samples["bad"].append(line[:400].rstrip())
                    continue
                qm = PRE_Q_RE.search(body)
                rm = PRE_RUN_RE.search(body)
                pre.append([t, int(pm.group(1)), int(pm.group(2)), int(pm.group(3)),
                            int(rm.group(1)) if rm else None,
                            int(qm.group(1)) if qm else None])
                continue
    c["max_ts_backwards_s"] = round(max_back, 3)
    diag.append("  server counts: %s" % json.dumps(c, sort_keys=True))
    diag.append("  HICACHE_EVT kinds: %s" % json.dumps(evt_kinds, sort_keys=True))
    diag.append("  rid prefixes (keyed lines): %s" % json.dumps(prefixes, sort_keys=True))
    diag.append("  first/last log time: %s .. %s" % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(first_t)) if first_t else None,
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(last_t)) if last_t else None))
    for k, v in samples.items():
        for s in v:
            diag.append("    sample[%s]: %s" % (k, s))
    return {
        "rts": rts, "s2h": s2h, "ntp": ntp, "srs": srs, "evd": evd, "evh": evh,
        "dec": dec, "pre": pre, "counts": c, "evt_kinds": evt_kinds,
        "prefixes": prefixes, "first_t": first_t, "last_t": last_t,
    }


def main():
    with open(RES + "/.current_run") as f:
        current = f.read().strip()
    os.makedirs(CACHE, exist_ok=True)
    diag = ["parse_runs.py  generated %s UTC" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "current run (never read): %r" % current]
    only = set(sys.argv[1:])
    for label, folder in RUNS:
        if only and label not in only:
            continue
        if current and (folder == current or current in folder or folder in current):
            raise SystemExit("refusing to read the live run folder: " + folder)
        run_dir = os.path.join(RES, folder)
        ds, cd = last_client_dir(run_dir)
        diag.append("")
        diag.append("== %s  %s" % (label, folder))
        diag.append("  client dirs: %s  -> using %s" % (ds, cd))
        cpath = os.path.join(run_dir, cd, "client.jsonl")
        spath = os.path.join(run_dir, "server.log")
        t_start, run_rec, turns = parse_client(cpath, diag)
        srv = parse_server(spath, diag)
        diag.append("  client t_start=%r  turns=%d  args=%s" % (
            t_start, len(turns),
            json.dumps({k: (run_rec or {}).get("args", {}).get(k) for k in
                        ("num_users", "turns", "arrival_rate", "gap_scale", "oracle_hints",
                         "hints", "hint_source", "hint_point", "hint_end", "concurrency",
                         "timeout", "seed", "gap_cap")}, sort_keys=True)))
        out = {
            "label": label, "folder": folder, "client_dir": cd,
            "client_size": os.path.getsize(cpath), "server_size": os.path.getsize(spath),
            "client_mtime": os.path.getmtime(cpath), "server_mtime": os.path.getmtime(spath),
            "t_start": t_start, "run_args": (run_rec or {}).get("args"),
            "run_turns_field": (run_rec or {}).get("turns"),
            "run_users_field": (run_rec or {}).get("users"),
            "turns": turns,
        }
        out.update(srv)
        with open(os.path.join(CACHE, label + ".json"), "w") as f:
            json.dump(out, f)
        diag.append("  cached -> %s" % os.path.join(CACHE, label + ".json"))
    name = "parse_diag.txt" if not only else "parse_diag_partial.txt"
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(diag) + "\n")
    print("\n".join(diag))


if __name__ == "__main__":
    main()
