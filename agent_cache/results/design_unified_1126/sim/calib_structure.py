"""Calibration helper: who waits. For campaign 34's arms, measured (client.jsonl) and simulated (runs/calib, seeds
0-4): the share of returning requests sent in minutes 90-110 that read from the SSD, their mean TTFT against the
others', and mean TTFT by the request's own wait before it was sent. Writes measured_structure.json (read by
calib_table.py)."""
import csv, json, os, sys
import numpy as np
RESULTS = "/home/wanhr/sglang/agent_cache/results"
SIM = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "calibration_runs"
BINS = ((0, 1), (1, 10), (10, 60), (60, 300), (300, 1800))
def manifest(c):
    out = {}
    for line in open(os.path.join(RESULTS, c, "manifest.txt")):
        p = line.split()
        if len(p) >= 3: out[p[0]] = os.path.join(RESULTS, p[1], p[2], "client.jsonl")
    return out
def stats(rows):
    # rows: (t_send, ttft, ssd_tokens, first, wait)
    r = np.array(rows, float)
    ret = r[(r[:, 0] >= 5400) & (r[:, 0] < 6600) & (r[:, 3] == 0)]
    rd = ret[ret[:, 2] > 0]; ot = ret[ret[:, 2] == 0]
    out = {"reader_share": len(rd) / max(1, len(ret)), "ttft_readers": float(rd[:, 1].mean()) if len(rd) else None,
           "ttft_others": float(ot[:, 1].mean()) if len(ot) else None}
    cut = r[(r[:, 0] < 6600) & (r[:, 3] == 0)]
    for lo, hi in BINS:
        w = cut[(cut[:, 4] >= lo) & (cut[:, 4] < hi)]
        out[f"wait_{lo}_{hi}"] = float(w[:, 1].mean()) if len(w) else None
        out[f"n_wait_{lo}_{hi}"] = int(len(w))
    return out
res = {}
for arm, path in manifest("campaign34").items():
    rows = []
    for line in open(path):
        if line.startswith('{"kind":"req"'):
            d = json.loads(line)
            if d.get("ttft") is None: continue
            det = d.get("cached_details") or {}
            rows.append((d["t_send"], d["ttft"], det.get("storage", 0), 1 if d["k"] == 0 and d["stream"] == 0 else 0, d["wait_s"]))
    res[arm] = {"measured": stats(rows)}
    sims = []
    for k in range(5):
        rows = []
        with open(os.path.join(SIM, OUT, "calib", "default", arm, f"seed{k}", "requests.csv")) as f:
            for d in csv.DictReader(f):
                rows.append((float(d["t_send"]), float(d["ttft"]), float(d["tok_ssd"]), int(d["first"]), float(d["wait"])))
        sims.append(stats(rows))
    res[arm]["sim"] = sims
json.dump(res, open(os.path.join(SIM, "measured_structure.json"), "w"), indent=1)
for arm, v in res.items():
    m, s0 = v["measured"], v["sim"][0]
    print(arm, "measured readers %.3f TTFT readers %.1f others %.1f | sim %.3f %.1f %.1f" % (m["reader_share"], m["ttft_readers"], m["ttft_others"], s0["reader_share"], s0["ttft_readers"], s0["ttft_others"]))
    print("   by own wait measured:", [round(m[f"wait_{lo}_{hi}"], 1) for lo, hi in BINS], "sim:", [round(s0[f"wait_{lo}_{hi}"], 1) for lo, hi in BINS])
