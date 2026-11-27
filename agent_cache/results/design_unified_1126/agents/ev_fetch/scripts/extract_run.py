"""Extract one run's server-log events and client records into a pickle (CPU, stdlib; streams the log once).

  python extract_run.py <label> <run folder name> <programs folder name>

Writes extracts/<label>.pkl with:
  evts   : kind -> list of (t_epoch, rid or None, fields dict) for the HICACHE_EVT kinds kept below
  prefill: list of (t, new_seq, new_token, cached_token, usage, running, queue)
  decode : list of (t, running, token, usage, queue)
  reqs   : rid -> the client's req record (trimmed), with t_send_epoch
  run    : the client's run record args, t_start; plays; summary
Log times are UTC wall-clock with ms resolution; the client's t_start is time.time() on the same host.
"""
import calendar
import json
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402

KEEP = {"prefetch_start", "s2h_query", "srs_park", "srs_dispatch", "s2h_io", "srs_pin", "srs_unpin", "srs_pin_skipped",
        "ntp_hold", "ntp_admit", "ntp_arrive", "ntp_gate", "ntp_due", "ntp_refetch", "ntp_reevict", "ntp_stage",
        "ntp_plan", "ntp_pick", "ec_start", "ec_release", "ec_bump", "ec_price", "ntp_room", "ntp_expire",
        "load_back_init", "srs_write_hold", "srs_rate", "host_census", "h2s_group", "h2s_io", "h2s_submit", "h2s_done",
        "nue_register", "evict_host"}
# per-op write events are many: keep only what the write-rate series needs
SLIM = {"h2s_io": ("pages", "tokens", "ms", "skipped"), "h2s_submit": ("tokens",), "h2s_done": ("tokens",),
        "nue_register": ("next_use_ms",), "evict_host": ("tokens",)}
TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)[.,](\d+)\] (.*)$")
EVT = re.compile(r"^HICACHE_EVT (\w+) ?(.*)$")
KV = re.compile(r"(\w+)=(\S*)")
PREFILL = re.compile(r"^Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), "
                     r"#running-req: (\d+), #queue-req: (\d+)")
DECODE = re.compile(r"^Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+),.*#queue-req: (\d+)")
CLEAN = re.compile(r"NIXL L3 cleanup")
RTS = re.compile(r"^ReqTimeStats\(rid=([^,]+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+), attempts=(\d+)"
                 r".*?queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")


def num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def epoch(m):
    y, mo, d, h, mi, s, frac = m.groups()[:7]
    return calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s))) + float("0." + frac)


def main():
    label, run, programs = sys.argv[1:4]
    log = os.path.join(C.RESULTS, run, "server.log")
    client = os.path.join(C.RESULTS, run, programs, "client.jsonl")
    evts, prefill, decode, cleanup, rts = {}, [], [], [], {}
    t0 = time.time()
    with open(log, errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            body = m.group(8)
            if body.startswith("HICACHE_EVT"):
                e = EVT.match(body)
                if not e or e.group(1) not in KEEP:
                    continue
                kind = e.group(1)
                fields = {k: num(v) for k, v in KV.findall(e.group(2))}
                rid = fields.pop("rid", None)
                if kind in SLIM:
                    fields = {k: fields.get(k) for k in SLIM[kind]}
                evts.setdefault(kind, []).append((epoch(m), rid, fields))
            elif body.startswith("Prefill batch"):
                p = PREFILL.match(body)
                if p:
                    prefill.append((epoch(m), int(p[1]), int(p[2]), int(p[3]), float(p[4]), int(p[5]), int(p[6])))
            elif body.startswith("Decode batch"):
                p = DECODE.match(body)
                if p:
                    decode.append((epoch(m), int(p[1]), int(p[2]), float(p[3]), int(p[4])))
            elif body.startswith("ReqTimeStats("):
                p = RTS.match(body)
                if p:
                    # rid -> (log time = completion, input, cached, output, queue s, forward s, entry epoch)
                    rts[p[1]] = (epoch(m), int(p[2]), int(p[3]), int(p[4]), float(p[6]) / 1e3, float(p[7]) / 1e3,
                                 float(p[8]))
            elif CLEAN.search(body):
                cleanup.append((epoch(m), body[:300]))
    reqs, plays, run_rec, summary = {}, [], None, None
    with open(client) as f:
        for line in f:
            d = json.loads(line)
            k = d.get("kind")
            if k == "run":
                run_rec = {"t_start": d["t_start"], "args": d["args"], "trees": d.get("trees"), "tag": d.get("tag")}
            elif k == "turn":
                # replay_agentic.py (campaigns up to 26): one conversation per user, rid <tag>-c<conv>-t<turn>
                d.pop("kind")
                d["rid"] = f"{run_rec['tag']}-c{d['conv']}-t{d['turn']}"
                d["k"], d["play"], d["stream"] = d["turn"], d["conv"], 0
                d["wait_s"] = d.get("gap_slept")
                d["hint"] = d.get("dist_hint")
                d["key"] = None
                d["t_send_epoch"] = run_rec["t_start"] + d["t_send"] if d.get("t_send") is not None else None
                reqs[d["rid"]] = d
            elif k == "req":
                d.pop("kind")
                d["t_send_epoch"] = run_rec["t_start"] + d["t_send"] if d.get("t_send") is not None else None
                reqs[d["rid"]] = d
            elif k == "play":
                plays.append(d)
            elif k == "summary":
                summary = d
    out = os.path.join(C.EXTRACTS, f"{label}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"label": label, "run": run, "programs": programs, "evts": evts, "prefill": prefill,
                     "decode": decode, "cleanup": cleanup, "rts": rts, "reqs": reqs, "plays": plays, "run_rec": run_rec,
                     "summary": summary}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(label, "events", {k: len(v) for k, v in sorted(evts.items())}, "prefill", len(prefill), "decode", len(decode),
          "cleanup", len(cleanup), "rts", len(rts), "reqs", len(reqs), f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
