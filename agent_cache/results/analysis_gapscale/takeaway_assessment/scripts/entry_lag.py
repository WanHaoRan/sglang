import json, re, glob, sys
from datetime import datetime
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
for run in sys.argv[1:]:
    base = f"/home/wanhr/sglang/agent_cache/results/{run}"
    cj = glob.glob(base + "/client_*/client.jsonl")[0]
    T = {}; t0 = None
    for l in open(cj):
        r = json.loads(l)
        if r["kind"] == "run": t0 = r["t_start"]
        elif r["kind"] == "turn": T[(r["conv"], r["turn"])] = r
    ent = []; pfl = []; first_pf = {}
    for l in open(base + "/server.log", errors="ignore"):
        if "ReqTimeStats(rid=" in l:
            m = re.search(r"rid=\S+-c(\d+)-t(\d+),.*queue_duration=([\d.]+)ms.*entry_time=([\d.]+)", l)
            if m and (int(m.group(1)), int(m.group(2))) in T:
                r = T[(int(m.group(1)), int(m.group(2)))]
                ent.append(float(m.group(4)) - (t0 + r["t_send"]))
        elif "prefetch_start rid=" in l:
            m = re.search(r"^\[([^\]]+)\].*rid=\S+-c(\d+)-t(\d+) ", l)
            if m: first_pf.setdefault((int(m.group(2)), int(m.group(3))), ts(m.group(1)))
    for k, v in first_pf.items():
        if k in T: pfl.append(v - (t0 + T[k]["t_send"]))
    q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float("nan")
    print(run, "entry_time - send (s): n=%d p50=%.3f p90=%.3f max=%.2f" % (len(ent), q(ent,.5), q(ent,.9), max(ent) if ent else float('nan')),
          "| first prefetch_start - send: p50=%.3f p90=%.3f" % (q(pfl,.5), q(pfl,.9)))
