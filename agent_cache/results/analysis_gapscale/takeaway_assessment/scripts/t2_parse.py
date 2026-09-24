#!/usr/bin/env python3
"""Parse SSD-arm runs (c8, c9) into per-turn records keyed by rid; pickle for later analysis."""
import json, os, re, glob, pickle, datetime as dt, sys
R = "/home/wanhr/sglang/agent_cache/results"
RUNS = {
    ("x10", "three_tier_to"): "20260922_192912_three_tier_to_NAT160",
    ("x10", "three_tier_wc"): "20260922_220250_three_tier_wc_NAT160",
    ("x1", "three_tier_to"): "20260923_053724_three_tier_to_NAT160",
    ("x1", "three_tier_wc"): "20260923_080528_three_tier_wc_NAT160",
}
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] ")
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
KV = re.compile(r"(\w+)=(\S+)")
PS = re.compile(r"HiCache prefetch (success|dropped) req=(\S+) completed=(\d+) matched=(\d+) loaded=(\d+)")
TSL = re.compile(r"ReqTimeStats\(rid=([^,]+), input_len=(\d+), cached_input_len=(\d+).*?queue_duration=([0-9.]+)(ms|s), forward_duration=([0-9.]+)(ms|s), entry_time=([0-9.]+)")
BATCH = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([0-9.]+), #running-req: (\d+), #queue-req: (\d+)")

def parse(key, run):
    d = os.path.join(R, run)
    cl = glob.glob(os.path.join(d, "client_*", "client.jsonl"))[0]
    rows = [json.loads(l) for l in open(cl) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r.get("kind") == "run")
    turns = [r for r in rows if r.get("kind") == "turn"]
    ev = {}  # rid -> dict of lists
    glob_ev = {"evict_host": [], "evict_device": [], "d2h_done": [], "h2s_done": [], "batch": []}
    for line in open(os.path.join(d, "server.log"), errors="replace"):
        m = TS.match(line)
        if not m: continue
        t = None
        if "HICACHE_EVT" in line:
            t = ts(m[1])
            name = line.split("HICACHE_EVT ", 1)[1].split()[0]
            kv = dict(KV.findall(line.split("HICACHE_EVT ", 1)[1]))
            rid = kv.get("rid")
            if rid:
                ev.setdefault(rid, {}).setdefault(name, []).append((t, kv))
            elif name in glob_ev:
                glob_ev[name].append((t, int(kv.get("tokens", 0)), int(kv.get("requested", 0) or 0)))
        elif "HiCache prefetch " in line:
            mm = PS.search(line)
            if mm:
                ev.setdefault(mm[2], {}).setdefault("pf_result", []).append((ts(m[1]), dict(kind=mm[1], completed=int(mm[3]), matched=int(mm[4]), loaded=int(mm[5]))))
        elif "ReqTimeStats(" in line:
            mm = TSL.search(line)
            if mm:
                q = float(mm[4]) / (1000.0 if mm[5] == "ms" else 1.0)
                f = float(mm[6]) / (1000.0 if mm[7] == "ms" else 1.0)
                ev.setdefault(mm[1], {}).setdefault("rts", []).append((ts(m[1]), dict(input_len=int(mm[2]), cached=int(mm[3]), queue=q, forward=f, entry=float(mm[8]))))
        elif "Prefill batch" in line:
            mm = BATCH.search(line)
            if mm:
                glob_ev["batch"].append((ts(m[1]), int(mm[1]), int(mm[2]), int(mm[3]), float(mm[4]), int(mm[5]), int(mm[6])))
    return dict(t0=t0, turns=turns, ev=ev, glob=glob_ev, run=run)

if __name__ == "__main__":
    out = {}
    for k, run in RUNS.items():
        out[k] = parse(k, run)
        print(k, run, len(out[k]["turns"]), len(out[k]["ev"]), {n: len(v) for n, v in out[k]["glob"].items()}, file=sys.stderr)
    pickle.dump(out, open(os.path.join(os.path.dirname(__file__), "t2.pkl"), "wb"))
