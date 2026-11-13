"""Shared loader and definitions for the confound checks of campaign 26.
Run every script as:  nice -n 19 python3 -I -B <script>   (standard library only).
parse_cache.py reads the raw logs once and writes cache/<label>.json; the tests read the cache.
Nothing here reads the run named in results/.current_run."""
import calendar, glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))  # agent_cache/results
CACHE = os.path.join(HERE, "cache")

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
    "order": "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
}
# Runs that finished after the task's list was written (2026-11-11 22:46Z and 2026-11-12 00:48Z, both on GPU-a7198925).
# Kept out of RUNS so the scripts that loop over RUNS read the same twelve arms as before; parse with their labels.
EXTRA = {
    "jit_p50_d": "20261111_204059_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r13_rep": "20261111_224624_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
}
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
CUT7 = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"]
EPISODES = ((5500, 7000), (8750, 11000))
MARGIN = 300.0

TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] ")
RID = r"rid=\S+?-c(\d+)-t(\d+)"
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), .*?queue_duration=([\d.]+)ms.*?forward_duration=([\d.]+)ms.*?entry_time=([\d.]+)")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), .*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PRE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
IO = re.compile(r"HICACHE_EVT s2h_io " + RID + r" pages=(\d+) tokens=(\d+) ms=(\d+)")
DISP = re.compile(r"HICACHE_EVT srs_dispatch " + RID + r" pages=(\d+) .*?pinned_tokens=(\d+) launched=(\d)")
STAGE = re.compile(r"HICACHE_EVT ntp_stage " + RID)
EVICT = re.compile(r"HICACHE_EVT (evict_device|evict_host) tokens=(\d+)")


def current_run():
    p = os.path.join(RES, ".current_run")
    return open(p).read().strip() if os.path.exists(p) else ""


def client_path(run):
    return os.path.join(sorted(glob.glob(os.path.join(RES, run, "client_*")))[-1], "client.jsonl")


def stamp(m):
    return calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000.0


def parse_run(label):
    run = RUNS[label] if label in RUNS else EXTRA[label]
    assert run != current_run(), "refusing to read the run being written"
    out = {"label": label, "run": run, "turns": [], "conv_start": {}, "conv_end": {}, "users": [], "bad_lines": 0, "other_kinds": {}}
    path = client_path(run)
    out["client"] = os.path.relpath(path, RES)
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                out["bad_lines"] += 1
                continue
            k = r.get("kind")
            if k == "run":
                out["t0"] = r["t_start"]
                out["args"] = r.get("args")
                out["run_rec"] = {x: r[x] for x in r if x != "args"}
            elif k == "turn":
                cd = r.get("cached_details") or {}
                dh = r.get("dist_hint") or {}
                out["turns"].append({
                    "conv": r["conv"], "turn": r["turn"], "t_send": r["t_send"], "ttft": r.get("ttft"), "latency": r.get("latency"),
                    "pt": r.get("prompt_tokens"), "ct": r.get("completion_tokens"), "cached": r.get("cached_tokens"),
                    "rr": r.get("reply_reprefilled"), "dev": cd.get("device"), "host": cd.get("host"), "sto": cd.get("storage"),
                    "gap": r.get("gap_slept"), "pre_gap": r.get("pre_gap"), "err": r.get("error"), "olen": r.get("output_length"),
                    "rpt": r.get("replay_prompt_tokens"), "delta": r.get("delta_tokens"), "true_gap": dh.get("true_gap_s"),
                    "point": dh.get("point_s"), "src": r.get("source")})
            elif k == "conv_start":
                out["conv_start"][str(r["conv"])] = {x: r[x] for x in r if x not in ("kind", "conv")}
            elif k == "conv_end":
                out["conv_end"][str(r["conv"])] = {x: r[x] for x in r if x not in ("kind", "conv")}
            elif k == "users":
                out["users"].append([r.get("t"), r.get("cap"), r.get("live"), r.get("waiting"), r.get("in_flight"), r.get("admitted"), r.get("finished")])
            elif k == "summary":
                out["summary"] = r
            else:
                out["other_kinds"][str(k)] = out["other_kinds"].get(str(k), 0) + 1
    req, s2h, stage, disp, evd, evh, dec, pre = {}, [], [], [], [], [], [], []
    n_lines = n_ts = 0
    first_t = last_t = None
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            n_lines += 1
            m = TS.match(line)
            if not m:
                continue
            n_ts += 1
            body = line[m.end():]
            c = body[:12]
            if c == "ReqTimeStats":
                q = RTS.match(body)
                if q:
                    t = stamp(m)
                    req[q[1] + "-" + q[2]] = [float(q[3]), float(q[4]), float(q[5]), t]
            elif c == "Decode batch":
                d = DEC.match(body)
                if d:
                    dec.append([stamp(m), int(d[1]), int(d[2]), float(d[3]), float(d[4]), int(d[5])])
            elif c == "Prefill batc":
                p = PRE.match(body)
                if p:
                    pre.append([stamp(m), int(p[1]), int(p[2]), int(p[3])])
            elif c == "HICACHE_EVT ":
                t = stamp(m)
                if first_t is None:
                    first_t = t
                last_t = t
                x = IO.match(body)
                if x:
                    s2h.append([t, int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])])
                    continue
                x = DISP.match(body)
                if x:
                    disp.append([t, int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])])
                    continue
                x = STAGE.match(body)
                if x:
                    stage.append([t, int(x[1]), int(x[2])])
                    continue
                x = EVICT.match(body)
                if x:
                    (evd if x[1] == "evict_device" else evh).append([t, int(x[2])])
    out.update(req=req, s2h=s2h, stage=stage, disp=disp, evict_device=evd, evict_host=evh, dec=dec, pre=pre,
               server_lines=n_lines, server_stamped=n_ts)
    return out


def load(label):
    with open(os.path.join(CACHE, label + ".json")) as f:
        return json.load(f)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def rec(t):
    return max(t.get("rr") or 0, 0)


def kept(d, margin=MARGIN):
    """Turns with a ttft, sorted by t_send, sent before (largest t_send of those - margin)."""
    turns = sorted((t for t in d["turns"] if t["ttft"] is not None), key=lambda t: t["t_send"])
    end = turns[-1]["t_send"] - margin
    if margin <= 0:
        return turns
    return [t for t in turns if t["t_send"] < end]


def by_key(turns):
    return {(t["conv"], t["turn"]): t for t in turns}


def order_rank():
    d = load("order")
    turns = sorted((t for t in d["turns"] if t["ttft"] is not None), key=lambda t: t["t_send"])
    return {(t["conv"], t["turn"]): i for i, t in enumerate(turns)}


def shared_keys(B, names, rank):
    """Returning-turn keys kept in every arm of names; B[label] = by_key(kept turns)."""
    s = set.intersection(*(set(k for k in B[a] if k[1] > 0) for a in names))
    return sorted(s, key=lambda k: (rank.get(k, 1 << 30), k))


def in_episode(rank, k, episodes=EPISODES):
    p = rank.get(k, -1)
    return any(lo <= p < hi for lo, hi in episodes)
