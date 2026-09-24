import json, re, os, glob, datetime as dt, pickle, sys
R = "/home/wanhr/sglang/agent_cache/results"
RUNS = {"three_tier_to": "20260922_051155_three_tier_to_NAT160", "three_tier_wc": "20260922_064515_three_tier_wc_NAT160",
        "hbm_host": "20260922_023714_hbm_host_NAT160"}
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] ")
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
KV = re.compile(r"(\w+)=([^\s,]+)")
out = {}
for arm, run in RUNS.items():
    d = os.path.join(R, run)
    cl = glob.glob(os.path.join(d, "client_*", "client.jsonl"))[0]
    rows = [json.loads(l) for l in open(cl) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r.get("kind") == "run")
    turns = [r for r in rows if r.get("kind") == "turn"]
    ev = {}   # rid -> {evt: [ (ts, kv) ]}
    glob_ev = {"evict_host": [], "evict_device": [], "h2s_io": [], "d2h_done": [], "h2d_done": [], "batch": [], "s2h_io": []}
    for line in open(os.path.join(d, "server.log"), errors="replace"):
        m = TS.match(line)
        if not m: continue
        t = ts(m[1])
        if "HICACHE_EVT" in line:
            name = line.split("HICACHE_EVT ", 1)[1].split()[0]
            kv = dict(KV.findall(line.split("HICACHE_EVT ", 1)[1]))
            if name in glob_ev: glob_ev[name].append((t, kv))
            if "rid" in kv:
                ev.setdefault(kv["rid"], {}).setdefault(name, []).append((t, kv))
        elif "HiCache prefetch" in line and "req=" in line:
            kv = dict(KV.findall(line)); st = "success" if "prefetch success" in line else ("dropped" if "dropped" in line else "other")
            ev.setdefault(kv["req"], {}).setdefault("pf_" + st, []).append((t, kv))
        elif "Prefill batch" in line or "Decode batch" in line:
            m2 = re.search(r"token usage: ([0-9.]+).*?#running-req: (\d+).*?#queue-req: (\d+)", line)
            if m2: glob_ev["batch"].append((t, {"kind": "P" if "Prefill" in line else "D", "usage": float(m2[1]), "run": int(m2[2]), "q": int(m2[3])}))
    out[arm] = dict(t0=t0, turns=turns, ev=ev, g=glob_ev)
    print(arm, "turns", len(turns), "rids", len(ev), {k: len(v) for k, v in glob_ev.items()})
pickle.dump(out, open("/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/t1.pkl", "wb"))
