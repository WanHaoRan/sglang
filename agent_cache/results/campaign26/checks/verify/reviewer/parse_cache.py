"""Reviewer pass over campaign 26: parse each finished run once and cache what later scripts need.
Read-only on the runs; writes cache_<label>.json beside this script.
  nice -n 19 python3 -I -B parse_cache.py
"""
import glob, json, os, re, sys, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
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
    "ORDER": "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
}
CURRENT = open(os.path.join(RES, ".current_run")).read().strip()
TS = re.compile(r"^\[(.{23})\] ")
RID = r"rid=\S+?-c(\d+)-t(\d+)"
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), .*queue_duration=([\d.]+)ms.*forward_duration=([\d.]+)ms.*entry_time=([\d.]+)")
IO = re.compile(r"HICACHE_EVT s2h_io " + RID + r" pages=(\d+) tokens=(\d+) ms=(\d+)")
STAGE = re.compile(r"HICACHE_EVT ntp_stage " + RID)
DISP = re.compile(r"HICACHE_EVT srs_dispatch " + RID + r" pages=(\d+) .*pinned_tokens=(\d+) launched=(\d)")
ECS = re.compile(r"HICACHE_EVT ec_start " + RID + r" decision=(\w+) tokens=(\d+) start_ms=(\S+) idle_ms=(\S+) read_ms=(\S+) wait_ms=(\S+)")
ARR = re.compile(r"HICACHE_EVT ntp_arrive " + RID + r" arr_err_ms=(\S+) delta_ms=(\S+) pos=(\d+) qlen=(\d+)")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), .*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
EVICT = re.compile(r"HICACHE_EVT (evict_device|evict_host) tokens=(\d+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()


def parse(label, run):
    assert run != CURRENT, "refusing to read the run being written"
    cdir = sorted(glob.glob(os.path.join(RES, run, "client_*")))[-1]
    t0 = None
    turns = []
    convs = {}
    n_err = 0
    for line in open(os.path.join(cdir, "client.jsonl")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        k = r.get("kind")
        if k == "run":
            t0 = r["t_start"]
        elif k == "conv_start":
            convs[r["conv"]] = [r.get("source"), r.get("turns"), r.get("start_s")]
        elif k == "turn":
            if r.get("error"):
                n_err += 1
            h = r.get("dist_hint") or {}
            cd = r.get("cached_details") or {}
            turns.append([
                r["conv"], r["turn"], r.get("t_send"), r.get("ttft"), r.get("latency"),
                r.get("prompt_tokens"), r.get("completion_tokens"), r.get("cached_tokens"),
                max(r.get("reply_reprefilled") or 0, 0), r.get("gap_slept"),
                cd.get("device"), cd.get("host"), cd.get("storage"),
                h.get("p10"), h.get("p50"), h.get("p90"), h.get("n"), h.get("matched"), h.get("true_gap_s"), h.get("reply_key"),
                r.get("source"),
            ])
    q, io, staged, refused, launched, ecs, arr, dec, ev = {}, [], {}, {}, {}, [], {}, [], {"evict_device": 0, "evict_host": 0}
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            body = line[m.end():]
            if body.startswith("ReqTimeStats"):
                x = RTS.match(body)
                if x:
                    q[f"{x[1]}-{x[2]}"] = [float(x[3]) / 1000, float(x[4]) / 1000, float(x[5])]
                continue
            if body.startswith("Decode batch"):
                x = DEC.match(body)
                if x:
                    dec.append([round(stamp(m[1]) - t0, 2), int(x[1]), float(x[3]), float(x[4]), int(x[5])])
                continue
            if not body.startswith("HICACHE_EVT"):
                continue
            x = IO.match(body)
            if x:
                io.append([int(x[1]), int(x[2]), round(stamp(m[1]), 3), int(x[3]), int(x[5])])
                continue
            x = STAGE.match(body)
            if x:
                key = f"{x[1]}-{x[2]}"
                staged.setdefault(key, []).append(round(stamp(m[1]), 3))
                continue
            x = DISP.match(body)
            if x:
                key = f"{x[1]}-{x[2]}"
                if x[5] == "0":
                    refused[key] = refused.get(key, 0) + 1
                else:
                    launched[key] = launched.get(key, 0) + 1
                continue
            x = ECS.match(body)
            if x:
                ecs.append([int(x[1]), int(x[2]), round(stamp(m[1]), 3), x[3], int(x[4]), x[5], x[6], x[7], x[8]])
                continue
            x = ARR.match(body)
            if x:
                arr[f"{x[1]}-{x[2]}"] = [round(stamp(m[1]), 3), x[3], int(x[5]), int(x[6])]
                continue
            x = EVICT.match(body)
            if x:
                ev[x[1]] += int(x[2])
    out = {"label": label, "run": run, "t0": t0, "errors": n_err, "turns": turns, "convs": convs, "queue": q, "io": io,
           "staged": staged, "refused": refused, "launched": launched, "ec_start": ecs, "arrive": arr, "decode": dec, "evict": ev}
    json.dump(out, open(os.path.join(HERE, f"cache_{label}.json"), "w"))
    print(f"{label:14s} turns {len(turns):6d} errors {n_err} queue {len(q):6d} s2h_io {len(io):5d} staged {len(staged):5d} refused {sum(refused.values()):5d} ec_start {len(ecs):5d}", flush=True)


if __name__ == "__main__":
    want = sys.argv[1:] or list(RUNS)
    for label in want:
        parse(label, RUNS[label])
